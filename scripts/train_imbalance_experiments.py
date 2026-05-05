"""
Two imbalance-handling experiments on UNet (n_frames=1, base_ch=32, seed=42):

  exp1: UNet + BCE + pos_weight=136.0
  exp2: UNet + BCE+Dice (0.5/0.5 mix, no pos_weight) with user-specified dice formula

Both train from scratch for up to 30 epochs with early stopping (patience=8)
on val positive_only_iou.  All other hyper-parameters match the existing runs.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# torch MUST come before numpy / matplotlib on Windows (shm.dll load order)
import torch
import torch.nn as nn

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.storage import ContrailSampleStore
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.models import build_model
from src.training.trainer import (
    ContrailDataset, split_samples, compute_channel_stats,
    BATCH_SIZE, NUM_WORKERS, BANDS_PER_FRAME,
)
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
SEED           = 42
N_EPOCHS       = 30
PATIENCE       = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
BASE_CHANNELS  = 32
N_FRAMES       = 1
SPLIT_SEED     = 42
DEVICE         = torch.device("cpu")
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ARTIFACTS_MODELS  = Path("artifacts/models")
ARTIFACTS_TABLES  = Path("artifacts/tables")
ARTIFACTS_PREDS   = Path("artifacts/predictions")
for p in [ARTIFACTS_MODELS, ARTIFACTS_TABLES, ARTIFACTS_PREDS]:
    p.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── User-specified dice loss (exact formula from spec) ────────────────────────
def _dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """dice_loss as specified: flat-view per-sample, then mean."""
    pred = torch.sigmoid(logits)
    # flatten to (B, -1); works for (B,1,H,W) or (B,H,W)
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1).float()
    intersection = (pred * target).sum(dim=1)
    return 1 - ((2 * intersection + smooth) /
                (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()


class _BCEDiceLoss(nn.Module):
    """0.5 * BCE(logits) + 0.5 * dice_loss(logits), no pos_weight."""
    def __init__(self) -> None:
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        bce  = self._bce(logits, targets.float())
        dice = _dice_loss(logits, targets)
        return 0.5 * bce + 0.5 * dice


class _BCEPosWeightLoss(nn.Module):
    """BCE with logits, fixed pos_weight."""
    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self._bce(logits, targets.float())


# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
print(f"Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
print("Computing channel stats...")
ch_mean, ch_std = compute_channel_stats(train_s)

def _make_loader(split, shuffle: bool) -> DataLoader:
    ds = ContrailDataset(split, n_frames=N_FRAMES,
                         channel_mean=ch_mean, channel_std=ch_std,
                         augment=False)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)

train_loader = _make_loader(train_s, shuffle=True)
val_loader   = _make_loader(val_s,   shuffle=False)
test_loader  = _make_loader(test_s,  shuffle=False)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_probs, all_tgts = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            tgts  = masks.cpu().numpy()
            all_probs.append(probs)
            all_tgts.append(tgts)
    probs_np = np.concatenate(all_probs)
    tgts_np  = np.concatenate(all_tgts)
    metrics  = compute_corrected_metrics(probs_np, tgts_np, threshold=0.5)
    return total_loss / len(loader.dataset), metrics


def _monitor(m: dict) -> float:
    v = m.get("positive_only_iou", float("nan"))
    return v if not np.isnan(v) else m.get("micro_iou", 0.0)


# ── Single experiment runner ──────────────────────────────────────────────────
def run_experiment(label: str, display: str, criterion: nn.Module,
                   ckpt_path: Path, metrics_csv: Path, scan_csv: Path) -> dict:
    print(f"\n{'='*64}")
    print(f"  {display}")
    print(f"{'='*64}")

    torch.manual_seed(SEED)
    model = build_model("unet", n_frames=N_FRAMES,
                        base_channels=BASE_CHANNELS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = []
    best_monitor = -1.0
    patience_ctr = 0
    best_epoch   = 0
    t0_total = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vm = val_epoch(model, val_loader, criterion)
        monitor = _monitor(vm)
        scheduler.step(monitor)
        elapsed = time.time() - t0

        pos_iou   = vm["positive_only_iou"]
        micro_iou = vm["micro_iou"]
        print(f"  [epoch {epoch:02d}/{N_EPOCHS}]  tr={tr_loss:.4f}  vl={vl_loss:.4f}"
              f"  val_pos_iou={pos_iou:.4f}  val_micro_iou={micro_iou:.4f}"
              f"  monitor={monitor:.4f}  ({elapsed:.1f}s)")

        history.append(dict(epoch=epoch, tr=tr_loss, vl=vl_loss,
                            pos_iou=pos_iou, micro_iou=micro_iou))

        if monitor > best_monitor:
            best_monitor = monitor
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ** new best  monitor={monitor:.4f}  epoch={epoch}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(f"    patience {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # save training history CSV
    hist_csv = ARTIFACTS_TABLES / f"training_history_{label}.csv"
    with open(hist_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "tr", "vl", "pos_iou", "micro_iou"])
        w.writeheader(); w.writerows(history)
    print(f"  Saved training history -> {hist_csv}")

    # ── load best weights ─────────────────────────────────────────────────────
    print(f"  Loading best weights from epoch {best_epoch}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # ── val threshold scan ────────────────────────────────────────────────────
    print("  Threshold scan on validation set...")
    val_probs, val_targets = collect_predictions(model, val_loader, DEVICE)
    print(f"  val probs: min={val_probs.min():.4f}  mean={val_probs.mean():.4f}"
          f"  p95={np.percentile(val_probs,95):.4f}  max={val_probs.max():.4f}")

    scan_rows = []
    best_t, best_val_pos_iou = THRESHOLD_GRID[0], -1.0
    print(f"  {'thresh':>6}  {'val_pos_iou':>11}  {'val_micro_iou':>13}  {'recall':>8}")
    for t in THRESHOLD_GRID:
        m = compute_corrected_metrics(val_probs, val_targets, threshold=t)
        poi = m["positive_only_iou"]
        print(f"  {t:>6.1f}  {poi:>11.6f}  {m['micro_iou']:>13.6f}  {m['micro_recall']:>8.4f}")
        scan_rows.append(dict(
            threshold=t,
            val_positive_only_iou=poi,
            val_micro_iou=m["micro_iou"],
            val_all_iou=m["all_iou"],
            val_micro_precision=m["micro_precision"],
            val_micro_recall=m["micro_recall"],
        ))
        if not np.isnan(poi) and poi > best_val_pos_iou:
            best_val_pos_iou = poi; best_t = t
    print(f"  -> best_threshold={best_t}  val_pos_iou={best_val_pos_iou:.6f}")

    with open(scan_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        w.writeheader(); w.writerows(scan_rows)
    print(f"  Saved threshold scan -> {scan_csv}")

    # ── test corrected metrics ────────────────────────────────────────────────
    print("  Test evaluation...")
    test_probs, test_targets = collect_predictions(model, test_loader, DEVICE)
    print(f"  test probs: min={test_probs.min():.4f}  mean={test_probs.mean():.4f}"
          f"  p95={np.percentile(test_probs,95):.4f}  max={test_probs.max():.4f}")

    tm = compute_corrected_metrics(test_probs, test_targets, threshold=best_t)

    print(f"\n  -- Test results (threshold={best_t}) ----------------------")
    print(f"  positive_only_iou   : {tm['positive_only_iou']:.6f}")
    print(f"  micro_iou           : {tm['micro_iou']:.6f}")
    print(f"  micro_precision     : {tm['micro_precision']:.6f}")
    print(f"  micro_recall        : {tm['micro_recall']:.6f}")
    print(f"  micro_f1            : {tm['micro_f1']:.6f}")
    print(f"  empty_fpr           : {tm['empty_false_positive_rate']:.4f}")
    print(f"  pos samples         : {tm['target_positive_sample_count']} / {tm['total_sample_count']}")
    print(f"  Wall time           : {(time.time()-t0_total)/60:.1f} min")

    row = dict(
        experiment=label, seed=SEED, threshold=best_t,
        positive_only_iou=tm["positive_only_iou"],
        positive_only_precision=tm["positive_only_precision"],
        positive_only_recall=tm["positive_only_recall"],
        positive_only_f1=tm["positive_only_f1"],
        micro_iou=tm["micro_iou"],
        micro_precision=tm["micro_precision"],
        micro_recall=tm["micro_recall"],
        micro_f1=tm["micro_f1"],
        empty_false_positive_rate=tm["empty_false_positive_rate"],
        empty_predicted_positive_rate_mean=tm["empty_predicted_positive_rate_mean"],
        target_positive_sample_count=tm["target_positive_sample_count"],
        total_sample_count=tm["total_sample_count"],
        val_best_threshold=best_t,
        val_pos_iou=best_val_pos_iou,
        best_epoch=best_epoch,
    )
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"  Saved metrics -> {metrics_csv}")

    # ── prediction visualisation ──────────────────────────────────────────────
    pos_idx = [i for i in range(len(test_targets)) if test_targets[i].sum() > 0]
    vis_idx = pos_idx[:4]
    if vis_idx:
        n = len(vis_idx)
        fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        for row_i, si in enumerate(vis_idx):
            # band-11 is channel 0 of the middle frame (frame 0 for n_frames=1)
            ds = ContrailDataset(test_s, n_frames=N_FRAMES,
                                 channel_mean=ch_mean, channel_std=ch_std)
            img, msk = ds[si]
            band11 = img[0].numpy()
            gt     = test_targets[si]
            prob   = test_probs[si]
            pred   = (prob >= best_t).astype(np.uint8)

            axes[row_i, 0].imshow(band11, cmap="gray"); axes[row_i, 0].set_title("Band-11")
            axes[row_i, 1].imshow(gt,     cmap="gray"); axes[row_i, 1].set_title("GT mask")
            axes[row_i, 2].imshow(prob,   cmap="hot",  vmin=0, vmax=1); axes[row_i, 2].set_title("Probability")
            axes[row_i, 3].imshow(pred,   cmap="gray"); axes[row_i, 3].set_title(f"Pred @{best_t}")
            for ax in axes[row_i]: ax.axis("off")
        plt.tight_layout()
        vis_path = ARTIFACTS_PREDS / f"prediction_{label}.png"
        fig.savefig(vis_path, dpi=80)
        plt.close(fig)
        print(f"  Saved prediction vis -> {vis_path}")

    return {"display": display, **row}


# ── Run both experiments ───────────────────────────────────────────────────────
results = []

results.append(run_experiment(
    label       = "unet_bce_pw136",
    display     = "UNet + BCE pw=136",
    criterion   = _BCEPosWeightLoss(pos_weight=136.0),
    ckpt_path   = ARTIFACTS_MODELS / "unet_bce_pw136_seed42_best.pt",
    metrics_csv = ARTIFACTS_TABLES / "corrected_metrics_unet_bce_pw136.csv",
    scan_csv    = ARTIFACTS_TABLES / "threshold_scan_unet_bce_pw136.csv",
))

results.append(run_experiment(
    label       = "unet_dice_bce",
    display     = "UNet + Dice+BCE (0.5/0.5)",
    criterion   = _BCEDiceLoss(),
    ckpt_path   = ARTIFACTS_MODELS / "unet_dice_bce_seed42_best.pt",
    metrics_csv = ARTIFACTS_TABLES / "corrected_metrics_unet_dice_bce.csv",
    scan_csv    = ARTIFACTS_TABLES / "threshold_scan_unet_dice_bce.csv",
))

# ── Reload existing 3 results and build full comparison ───────────────────────
existing = [
    ("UNet + BCE (orig)",       "artifacts/tables/corrected_metrics_unet_bce.csv"),
    ("UNet + BCE-Dice (orig)",  "artifacts/tables/corrected_metrics_unet_bce_dice.csv"),
    ("TemporalUNet T=3",        "artifacts/tables/corrected_metrics_temporal_unet_t3.csv"),
]

all_rows = []
for display, csv_path in existing:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    all_rows.append({
        "model": display,
        "threshold":              float(row["threshold"]),
        "positive_only_iou":      float(row["positive_only_iou"]),
        "micro_iou":              float(row["micro_iou"]),
        "micro_precision":        float(row["micro_precision"]),
        "micro_recall":           float(row["micro_recall"]),
        "micro_f1":               float(row["micro_f1"]),
        "empty_false_positive_rate": float(row["empty_false_positive_rate"]),
    })

for r in results:
    all_rows.append({
        "model":                  r["display"],
        "threshold":              r["threshold"],
        "positive_only_iou":      r["positive_only_iou"],
        "micro_iou":              r["micro_iou"],
        "micro_precision":        r["micro_precision"],
        "micro_recall":           r["micro_recall"],
        "micro_f1":               r["micro_f1"],
        "empty_false_positive_rate": r["empty_false_positive_rate"],
    })

comp_path = ARTIFACTS_TABLES / "final_corrected_model_comparison.csv"
with open(comp_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
print(f"\nSaved full comparison -> {comp_path}")

# ── Print final table ──────────────────────────────────────────────────────────
print("\n" + "="*90)
print("  FULL CORRECTED MODEL COMPARISON  (test set, val-selected threshold)")
print("="*90)
print(f"  {'Model':<26} {'thr':>4}  {'pos_iou':>8}  {'micro_iou':>9}  "
      f"{'prec':>7}  {'recall':>7}  {'f1':>7}  {'empty_fpr':>9}")
print(f"  {'-'*26}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for r in all_rows:
    print(f"  {r['model']:<26} {r['threshold']:>4.1f}  "
          f"{r['positive_only_iou']:>8.4f}  {r['micro_iou']:>9.4f}  "
          f"{r['micro_precision']:>7.4f}  {r['micro_recall']:>7.4f}  "
          f"{r['micro_f1']:>7.4f}  {r['empty_false_positive_rate']:>9.4f}")
print("="*90)
