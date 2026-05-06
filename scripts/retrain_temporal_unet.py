"""
Retrain TemporalUNet T=3, seed=42 for 30 epochs.

Resumes from the existing 5-epoch checkpoint (warm weights, fresh optimizer).
Uses BCE+Dice loss with capped pos_weight.
Early stopping on val positive_only_iou (patience=8).
Saves all artifacts to artifacts/.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# torch must come before matplotlib on Windows (shm.dll load order)
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.storage import ContrailSampleStore
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.models import build_model
from src.training.losses import CombinedLoss
from src.training.trainer import (
    BANDS_PER_FRAME,
    TrainingConfig,
    _load_and_stack,
    _select_frame_indices,
    build_dataloaders,
    compute_channel_stats,
    split_samples,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED          = 42
N_EPOCHS      = 30
START_EPOCH   = 6        # resumes from 5-epoch checkpoint
PATIENCE      = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 8
N_FRAMES      = 3
SPLIT_SEED    = 42
POS_WEIGHT_CAP = 10.0
EVAL_THRESHOLD = 0.5     # fixed threshold used during training for monitoring
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESUME_CKPT = Path("checkpoints/m3_temporal_unet_seed42_best.pt")

OUT_MODEL    = Path("artifacts/models/temporal_unet_t3_epoch30_seed42_best.pt")
OUT_HISTORY  = Path("artifacts/tables/training_history_temporal_unet_t3_epoch30_seed42.csv")
OUT_METRICS  = Path("artifacts/tables/corrected_metrics_temporal_unet_t3_epoch30_seed42.csv")
OUT_THRESH   = Path("artifacts/tables/threshold_scan_temporal_unet_t3_epoch30_seed42.csv")
OUT_PRED     = Path("artifacts/predictions/prediction_temporal_unet_t3_epoch30_seed42.png")

for p in [OUT_MODEL, OUT_HISTORY, OUT_METRICS, OUT_THRESH, OUT_PRED]:
    p.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_pos_weight(train_samples: list, cap: float = POS_WEIGHT_CAP) -> float:
    """Compute BCE pos_weight from training masks, capped to avoid instability.

    Args:
        train_samples: List of :class:`~src.data.storage.ContrailSample` objects.
        cap: Maximum allowed weight (default ``POS_WEIGHT_CAP``).

    Returns:
        Capped ratio of negative to positive pixels across all training samples.
    """
    total_pos = 0
    total_pix = 0
    for s in train_samples:
        mask = np.load(s.mask_path).squeeze(-1)
        total_pos += int(mask.sum())
        total_pix += mask.size
    total_neg = total_pix - total_pos
    w = total_neg / (total_pos + 1e-9)
    logger.info("Class balance: %d pos / %d neg  raw_weight=%.2f  capped=%.2f",
                total_pos, total_neg, w, min(w, cap))
    return float(min(w, cap))


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return mean batch loss.

    Args:
        model: Network to train.
        loader: Training DataLoader.
        optimizer: Parameter update rule.
        criterion: Loss function (BCE+Dice combined).
        device: Torch device to send tensors to.

    Returns:
        Mean loss averaged over batches.
    """
    model.train()
    total = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def val_epoch(model, loader, criterion, device, threshold=EVAL_THRESHOLD):
    """Returns (mean_loss, corrected_metrics_dict) on validation set."""
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            all_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    probs   = np.concatenate(all_probs,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    corrected = compute_corrected_metrics(probs, targets, threshold=threshold)
    return total_loss / max(len(loader), 1), corrected


def _monitor_val(corrected: dict) -> float:
    """Primary early-stopping signal: positive_only_iou, fall back to micro_iou."""
    v = corrected["positive_only_iou"]
    return corrected["micro_iou"] if math.isnan(v) else v


def save_prediction_vis(model, test_samples, ch_mean, ch_std, threshold, device, path):
    """Save a grid of input / GT / probability / binary-prediction panels.

    Args:
        model: Trained network in eval mode.
        test_samples: List of test :class:`~src.data.storage.ContrailSample` objects.
        ch_mean: Channel-wise mean for normalisation (from training set).
        ch_std: Channel-wise std for normalisation.
        threshold: Binary threshold applied to sigmoid probabilities.
        device: Torch device.
        path: Output PNG file path.
    """
    frame_indices = _select_frame_indices(N_FRAMES)
    pos_samples = [s for s in test_samples
                   if np.load(s.mask_path).squeeze(-1).sum() > 0]
    show = (pos_samples or test_samples)[:4]

    cols = 4
    fig, axes = plt.subplots(len(show), cols, figsize=(cols * 4, len(show) * 4))
    if len(show) == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, sample in enumerate(show):
            image, mask = _load_and_stack(sample, frame_indices, ch_mean, ch_std)
            # visualise band-11 middle frame
            mid_band = image[N_FRAMES // 2 * BANDS_PER_FRAME]
            vis = (mid_band - mid_band.min()) / (mid_band.max() - mid_band.min() + 1e-6)

            logits = model(torch.from_numpy(image).unsqueeze(0).to(device))
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_b = (prob >= threshold).astype(np.uint8)

            axes[row, 0].imshow(vis, cmap="gray"); axes[row, 0].set_title("Input (band 11 mid)")
            axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[row, 1].set_title("GT mask")
            axes[row, 2].imshow(prob, cmap="hot",  vmin=0, vmax=1)
            axes[row, 2].set_title(f"Pred prob  max={prob.max():.3f}")
            axes[row, 3].imshow(pred_b, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].set_title(f"Pred bin (t={threshold:.2f})")
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction visualisation → %s", path)


def save_csv(path, rows, fieldnames):
    """Write a list of dicts to a CSV file.

    Args:
        path: Destination file path.
        rows: List of dicts to write as rows.
        fieldnames: Column order for the CSV header.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Train TemporalUNet T=3 for up to 30 epochs and save all artifacts.

    Loads samples via ``CONTRAIL_DB_PATH`` env var, resumes from the existing
    5-epoch warm-start checkpoint, runs early stopping on val positive_only_iou,
    and writes model, metrics, threshold scan, training history, and prediction
    visualisation to ``artifacts/``.
    """
    t_start = time.time()
    set_seed(SEED)

    # ── Load samples ──────────────────────────────────────────────────────────
    _db = Path(os.environ.get("CONTRAIL_DB_PATH", "data/contrail_samples.db"))
    store = ContrailSampleStore(_db)
    samples = store.get_all()
    logger.info("Loaded %d samples from DB", len(samples))

    train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)

    # ── Class weight ──────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(train_s)

    # ── Channel stats ─────────────────────────────────────────────────────────
    logger.info("Computing channel stats …")
    ch_mean, ch_std = compute_channel_stats(train_s)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    cfg = TrainingConfig(
        model_name="temporal_unet",
        loss_name="bce_dice",
        n_frames=N_FRAMES,
        use_augmentation=True,
        batch_size=BATCH_SIZE,
        run_name="temporal_unet_t3_epoch30",
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_s, val_s, test_s, cfg, ch_mean, ch_std, seed=SEED
    )

    # ── Model: resume from 5-epoch checkpoint ─────────────────────────────────
    state = torch.load(str(RESUME_CKPT), map_location=DEVICE)
    base_ch = int(state["unet.enc1.conv.block.0.weight"].shape[0])
    logger.info("Resuming from %s  (BASE_CHANNELS=%d)", RESUME_CKPT.name, base_ch)
    model = build_model("temporal_unet", n_frames=N_FRAMES, base_channels=base_ch).to(DEVICE)
    model.load_state_dict(state)

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    criterion = CombinedLoss(pos_weight=pos_weight).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_monitor = -1.0
    best_state       = None
    best_epoch       = START_EPOCH - 1
    patience_ctr     = 0
    history_rows     = []

    logger.info("Training epochs %d–%d on %s  pos_weight=%.2f  patience=%d",
                START_EPOCH, N_EPOCHS, DEVICE, pos_weight, PATIENCE)

    for epoch in range(START_EPOCH, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss              = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_corrected = val_epoch(model, val_loader, criterion, DEVICE)
        monitor              = _monitor_val(vl_corrected)
        scheduler.step(monitor)
        elapsed = time.time() - t0

        pos_iou   = vl_corrected["positive_only_iou"]
        micro_iou = vl_corrected["micro_iou"]
        logger.info(
            "[epoch %02d/%02d]  tr=%.4f  vl=%.4f  "
            "val_pos_iou=%.4f  val_micro_iou=%.4f  monitor=%.4f  (%.1fs)",
            epoch, N_EPOCHS, tr_loss, vl_loss,
            pos_iou if not math.isnan(pos_iou) else -1,
            micro_iou, monitor, elapsed,
        )

        row = dict(
            epoch=epoch,
            tr_loss=round(tr_loss, 6),
            vl_loss=round(vl_loss, 6),
            val_positive_only_iou=round(pos_iou, 6) if not math.isnan(pos_iou) else "",
            val_micro_iou=round(micro_iou, 6),
            val_all_iou=round(vl_corrected["all_iou"], 6),
            monitor=round(monitor, 6),
            elapsed_s=round(elapsed, 1),
        )
        history_rows.append(row)

        if monitor > best_val_monitor:
            best_val_monitor = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
            logger.info("  ↑ new best  monitor=%.4f  epoch=%d", best_val_monitor, best_epoch)
        else:
            patience_ctr += 1
            logger.info("  patience %d/%d", patience_ctr, PATIENCE)
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Save training history ─────────────────────────────────────────────────
    save_csv(OUT_HISTORY, history_rows, list(history_rows[0].keys()))

    # ── Load best checkpoint ──────────────────────────────────────────────────
    logger.info("Loading best weights from epoch %d", best_epoch)
    model.load_state_dict(best_state)
    torch.save(best_state, str(OUT_MODEL))
    logger.info("Saved best model → %s", OUT_MODEL)

    # ── Threshold scan on val, apply to test ─────────────────────────────────
    logger.info("Threshold scan on validation set …")
    val_probs,  val_tgts  = collect_predictions(model, val_loader,  DEVICE)
    test_probs, test_tgts = collect_predictions(model, test_loader, DEVICE)

    thresh_rows = []
    for t in THRESHOLD_GRID:
        vm = compute_corrected_metrics(val_probs, val_tgts, threshold=t)
        thresh_rows.append(dict(
            threshold=t,
            val_positive_only_iou=round(vm["positive_only_iou"], 6) if not math.isnan(vm["positive_only_iou"]) else "",
            val_micro_iou=round(vm["micro_iou"], 6),
            val_all_iou=round(vm["all_iou"], 6),
            val_micro_precision=round(vm["micro_precision"], 6),
            val_micro_recall=round(vm["micro_recall"], 6),
        ))

    # Best threshold: prefer positive_only_iou, fall back to micro_iou
    def _thresh_score(r):
        v = r["val_positive_only_iou"]
        return float(r["val_micro_iou"]) if v == "" else float(v)

    best_row  = max(thresh_rows, key=_thresh_score)
    best_t    = float(best_row["threshold"])
    logger.info("Best threshold (val) = %.2f  val_pos_iou=%s  val_micro_iou=%s",
                best_t, best_row["val_positive_only_iou"], best_row["val_micro_iou"])

    save_csv(OUT_THRESH, thresh_rows, list(thresh_rows[0].keys()))

    # ── Final test metrics ────────────────────────────────────────────────────
    test_corrected = compute_corrected_metrics(test_probs, test_tgts, threshold=best_t)
    metrics_row = dict(
        experiment_name="temporal_unet_t3_epoch30",
        seed=SEED,
        threshold=best_t,
        **{k: round(v, 6) if isinstance(v, float) and not math.isnan(v) else v
           for k, v in test_corrected.items()},
    )
    save_csv(OUT_METRICS, [metrics_row], list(metrics_row.keys()))

    # ── Prediction visualisation ──────────────────────────────────────────────
    save_prediction_vis(model, test_s, ch_mean, ch_std, best_t, DEVICE, OUT_PRED)

    # ── Print results ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("  RESULTS: temporal_unet_t3  30 epochs  seed=42")
    print("=" * 60)
    print(f"  Best checkpoint epoch    : {best_epoch}")
    print(f"  Selected threshold (val) : {best_t:.2f}")
    print()
    print(f"  test positive_only_iou   : {test_corrected['positive_only_iou']:.4f}")
    print(f"  test micro_iou           : {test_corrected['micro_iou']:.4f}")
    print(f"  test all_iou (naive)     : {test_corrected['all_iou']:.4f}")
    print()
    print(f"  test micro_precision     : {test_corrected['micro_precision']:.4f}")
    print(f"  test micro_recall        : {test_corrected['micro_recall']:.4f}")
    print(f"  test micro_f1            : {test_corrected['micro_f1']:.4f}")
    print()
    print(f"  empty_count              : {test_corrected['empty_count']}")
    print(f"  empty_false_positive_rate: {test_corrected['empty_false_positive_rate']:.4f}")
    print(f"  positive sample count    : {test_corrected['target_positive_sample_count']} / {test_corrected['total_sample_count']}")
    print()
    print(f"  Prediction image saved   : {OUT_PRED.exists()}")
    print(f"  Total wall time          : {total_time/60:.1f} min")
    print("=" * 60)

    # ── Last 10 lines of history ──────────────────────────────────────────────
    print("\n--- Training history (last 10 rows) ---")
    for r in history_rows[-10:]:
        print(f"  epoch={r['epoch']:02d}  tr={r['tr_loss']:.4f}  vl={r['vl_loss']:.4f}"
              f"  pos_iou={r['val_positive_only_iou']}  micro_iou={r['val_micro_iou']}"
              f"  ({r['elapsed_s']}s)")


if __name__ == "__main__":
    main()
