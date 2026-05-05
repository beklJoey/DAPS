"""Experiment 2: UNet + BCE-Dice combined loss (no pos_weight), seed=42.

Loss ablation experiment using equal-weight Dice+BCE without class-imbalance
correction.  Architecture: UNet(in_channels=5, base_channels=64).

Artifacts saved to:
    artifacts/models/unet_bce_dice_seed42_best.pt
    artifacts/tables/corrected_metrics_unet_bce_dice.csv
    artifacts/tables/threshold_scan_unet_bce_dice.csv
    artifacts/tables/training_history_exp2_unet_bce_dice.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.storage import ContrailSampleStore
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.models.unet import UNet
from src.training.trainer import (
    ContrailDataset, split_samples, compute_channel_stats,
    BATCH_SIZE, NUM_WORKERS,
)
from src.training.losses import CombinedLoss
from src.utils.seed import set_seed

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED: int = 42
N_EPOCHS: int = 30
PATIENCE: int = 8
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
BASE_CHANNELS: int = 64
IN_CHANNELS: int = 5
N_FRAMES: int = 1
SPLIT_SEED: int = 42
THRESHOLD_GRID: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEVICE: torch.device = torch.device("cpu")

LABEL: str = "unet_bce_dice"
DISPLAY: str = "UNet+BCE-Dice (orig)"

DB_PATH: Path = Path(os.environ.get("CONTRAIL_DB_PATH", "data/contrail_samples.db"))
CKPT_PATH: Path = Path("artifacts/models/unet_bce_dice_seed42_best.pt")
METRICS_CSV: Path = Path("artifacts/tables/corrected_metrics_unet_bce_dice.csv")
SCAN_CSV: Path = Path("artifacts/tables/threshold_scan_unet_bce_dice.csv")
HIST_CSV: Path = Path("artifacts/tables/training_history_exp2_unet_bce_dice.csv")

for _p in [CKPT_PATH.parent, METRICS_CSV.parent]:
    _p.mkdir(parents=True, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────────
set_seed(SEED)
store = ContrailSampleStore(DB_PATH)
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
ch_mean, ch_std = compute_channel_stats(train_s)


def _make_loader(split: list, shuffle: bool) -> DataLoader:
    """Build a DataLoader from a sample split.

    Args:
        split: List of :class:`~src.data.storage.ContrailSample` objects.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        Configured :class:`~torch.utils.data.DataLoader`.
    """
    ds = ContrailDataset(split, n_frames=N_FRAMES,
                         channel_mean=ch_mean, channel_std=ch_std)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


train_loader = _make_loader(train_s, shuffle=True)
val_loader = _make_loader(val_s, shuffle=False)
test_loader = _make_loader(test_s, shuffle=False)


def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Run one training epoch and return mean loss.

    Args:
        model: Network to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Parameter update rule.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / max(len(loader.dataset), 1)


def val_epoch(model: nn.Module, loader: DataLoader,
              criterion: nn.Module) -> tuple[float, dict]:
    """Run one validation epoch.

    Args:
        model: Network to evaluate.
        loader: Validation data loader.
        criterion: Loss function.

    Returns:
        Tuple of ``(val_loss, metrics_dict)``.
    """
    model.eval()
    total = 0.0
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            total += criterion(logits, masks).item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    m = compute_corrected_metrics(probs, targets, threshold=0.5)
    return total / max(len(loader.dataset), 1), m


def _monitor(m: dict) -> float:
    """Return ``positive_only_iou`` or fall back to ``micro_iou`` if NaN.

    Args:
        m: Metrics dictionary from :func:`compute_corrected_metrics`.

    Returns:
        Scalar monitor value.
    """
    v = m.get("positive_only_iou", float("nan"))
    return v if not np.isnan(v) else m.get("micro_iou", 0.0)


# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  {DISPLAY}")
print(f"{'='*64}")

torch.manual_seed(SEED)
model = UNet(in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
criterion = CombinedLoss(alpha=0.5)

history: list[dict] = []
best_monitor = -1.0
patience_ctr = 0
best_epoch = 0
t0_total = time.time()

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vm = val_epoch(model, val_loader, criterion)
    monitor = _monitor(vm)
    scheduler.step(monitor)

    print(f"  [epoch {epoch:02d}/{N_EPOCHS}]  tr={tr_loss:.4f}  vl={vl_loss:.4f}"
          f"  val_pos_iou={vm['positive_only_iou']:.4f}"
          f"  monitor={monitor:.4f}  ({time.time()-t0:.1f}s)")
    history.append(dict(epoch=epoch, tr=tr_loss, vl=vl_loss,
                        pos_iou=vm["positive_only_iou"],
                        micro_iou=vm["micro_iou"]))

    if monitor > best_monitor:
        best_monitor = monitor
        best_epoch = epoch
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"    ** new best  monitor={monitor:.4f}  epoch={epoch}")
        patience_ctr = 0
    else:
        patience_ctr += 1
        print(f"    patience {patience_ctr}/{PATIENCE}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

with open(HIST_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "tr", "vl", "pos_iou", "micro_iou"])
    writer.writeheader()
    writer.writerows(history)

print(f"  Loading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

print("  Threshold scan on validation set...")
val_probs, val_targets = collect_predictions(model, val_loader, DEVICE)
best_t, best_val_pos_iou = THRESHOLD_GRID[0], -1.0
scan_rows: list[dict] = []
for t in THRESHOLD_GRID:
    m = compute_corrected_metrics(val_probs, val_targets, threshold=t)
    poi = m.get("positive_only_iou", float("nan"))
    scan_rows.append(dict(threshold=t, **m))
    if not np.isnan(poi) and poi > best_val_pos_iou:
        best_val_pos_iou = poi
        best_t = t
print(f"  -> best_threshold={best_t}  val_pos_iou={best_val_pos_iou:.6f}")

with open(SCAN_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
    writer.writeheader()
    writer.writerows(scan_rows)

print("  Test evaluation...")
test_probs, test_targets = collect_predictions(model, test_loader, DEVICE)
tm = compute_corrected_metrics(test_probs, test_targets, threshold=best_t)
wall_min = (time.time() - t0_total) / 60

for k, v in tm.items():
    print(f"  {k:<25}: {v:.6f}")
print(f"  Wall time: {wall_min:.1f} min")

with open(METRICS_CSV, "w", newline="") as f:
    row = dict(experiment=LABEL, seed=SEED, threshold=best_t,
               val_best_threshold=best_t, val_pos_iou=best_val_pos_iou, **tm)
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
print(f"  Saved metrics -> {METRICS_CSV}")
