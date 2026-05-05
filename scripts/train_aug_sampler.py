"""
UNet + BCE + pos_weight=136 + augmentation + WeightedRandomSampler (50% pos/batch).

Changes vs train_imbalance_experiments.py exp1:
  1. augment=True on training set (HFlip / VFlip / Rotate90, via existing ContrailDataset)
  2. WeightedRandomSampler: positive samples weighted so expected batch ratio = 50%

Everything else (architecture, LR, epochs, patience, evaluation) is identical.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.storage import ContrailSampleStore
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.models import build_model
from src.training.trainer import (
    ContrailDataset, split_samples, compute_channel_stats,
    BATCH_SIZE, NUM_WORKERS,
)

# ── Config (identical to pw=136 experiment) ───────────────────────────────────
SEED          = 42
N_EPOCHS      = 30
PATIENCE      = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
BASE_CHANNELS = 32
N_FRAMES      = 1
SPLIT_SEED    = 42
POS_WEIGHT    = 136.0
DEVICE        = torch.device("cpu")
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DB_PATH     = Path(os.environ.get("CONTRAIL_DB_PATH", "data/contrail_samples.db"))
LABEL       = "unet_bce_pw136_aug_sampler"
DISPLAY     = "UNet+BCE pw=136 +aug+sampler"
CKPT_PATH   = Path("artifacts/models/unet_bce_pw136_aug_sampler_seed42_best.pt")
METRICS_CSV = Path("artifacts/tables/corrected_metrics_unet_bce_pw136_aug_sampler.csv")
SCAN_CSV    = Path("artifacts/tables/threshold_scan_unet_bce_pw136_aug_sampler.csv")
HIST_CSV    = Path("artifacts/tables/training_history_unet_bce_pw136_aug_sampler.csv")
VIS_PATH    = Path("artifacts/predictions/prediction_unet_bce_pw136_aug_sampler.png")

for p in [CKPT_PATH.parent, METRICS_CSV.parent, VIS_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
store = ContrailSampleStore(DB_PATH)
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
print(f"Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
print("Computing channel stats...")
ch_mean, ch_std = compute_channel_stats(train_s)

# ── WeightedRandomSampler ────────────────────────────────────────────────────
print("Scanning training masks for sample-level pos/neg labels...")
train_has_pos = [
    bool(np.load(s.mask_path).sum() > 0) for s in train_s
]
n_pos = sum(train_has_pos)
n_neg = len(train_s) - n_pos
print(f"  {n_pos} positive-mask samples / {n_neg} empty-mask samples "
      f"(ratio {n_pos/len(train_s):.3f})")

# weight each sample so class totals are equal -> expected 50% positive per batch
w_pos = 1.0 / n_pos if n_pos > 0 else 0.0
w_neg = 1.0 / n_neg if n_neg > 0 else 0.0
sample_weights = torch.tensor(
    [w_pos if has_pos else w_neg for has_pos in train_has_pos],
    dtype=torch.float32,
)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_s),
    replacement=True,
)

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_ds = ContrailDataset(train_s, n_frames=N_FRAMES,
                           channel_mean=ch_mean, channel_std=ch_std,
                           augment=True)   # <-- augmentation ON
val_ds   = ContrailDataset(val_s,   n_frames=N_FRAMES,
                           channel_mean=ch_mean, channel_std=ch_std,
                           augment=False)
test_ds  = ContrailDataset(test_s,  n_frames=N_FRAMES,
                           channel_mean=ch_mean, channel_std=ch_std,
                           augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler,          # <-- weighted sampler
                          num_workers=NUM_WORKERS,
                          pin_memory=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ── Loss ──────────────────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32)
)

# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(images), masks.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def val_epoch(model) -> tuple[float, dict]:
    model.eval()
    total_loss, all_probs, all_tgts = 0.0, [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            logits = model(images)
            total_loss += criterion(logits, masks.float()).item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
            all_tgts.append(masks.squeeze(1).cpu().numpy())
    probs_np = np.concatenate(all_probs)
    tgts_np  = np.concatenate(all_tgts)
    m = compute_corrected_metrics(probs_np, tgts_np, threshold=0.5)
    return total_loss / len(val_loader.dataset), m


def _monitor(m: dict) -> float:
    v = m.get("positive_only_iou", float("nan"))
    return v if not np.isnan(v) else m.get("micro_iou", 0.0)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  {DISPLAY}")
print(f"{'='*64}")

torch.manual_seed(SEED)
model = build_model("unet", n_frames=N_FRAMES, base_channels=BASE_CHANNELS).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

history = []
best_monitor = -1.0
patience_ctr = 0
best_epoch   = 0
t0_total = time.time()

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_epoch(model, optimizer)
    vl_loss, vm = val_epoch(model)
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
    w = csv.DictWriter(f, fieldnames=["epoch","tr","vl","pos_iou","micro_iou"])
    w.writeheader(); w.writerows(history)
print(f"  Saved training history -> {HIST_CSV}")

# ── Load best, threshold scan, test eval ─────────────────────────────────────
print(f"  Loading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

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
    scan_rows.append(dict(threshold=t, val_positive_only_iou=poi,
                          val_micro_iou=m["micro_iou"], val_all_iou=m["all_iou"],
                          val_micro_precision=m["micro_precision"],
                          val_micro_recall=m["micro_recall"]))
    if not np.isnan(poi) and poi > best_val_pos_iou:
        best_val_pos_iou = poi; best_t = t
print(f"  -> best_threshold={best_t}  val_pos_iou={best_val_pos_iou:.6f}")

with open(SCAN_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
    w.writeheader(); w.writerows(scan_rows)
print(f"  Saved threshold scan -> {SCAN_CSV}")

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
    experiment=LABEL, seed=SEED, threshold=best_t,
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
with open(METRICS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    w.writeheader(); w.writerow(row)
print(f"  Saved metrics -> {METRICS_CSV}")

# ── Prediction visualisation ──────────────────────────────────────────────────
pos_idx = [i for i in range(len(test_targets)) if test_targets[i].sum() > 0]
vis_idx = pos_idx[:4]
if vis_idx:
    n = len(vis_idx)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for ri, si in enumerate(vis_idx):
        img, _ = test_ds[si]
        band11 = img[0].numpy()
        axes[ri, 0].imshow(band11,          cmap="gray"); axes[ri, 0].set_title("Band-11")
        axes[ri, 1].imshow(test_targets[si], cmap="gray"); axes[ri, 1].set_title("GT mask")
        axes[ri, 2].imshow(test_probs[si],   cmap="hot", vmin=0, vmax=1)
        axes[ri, 2].set_title("Probability")
        axes[ri, 3].imshow((test_probs[si] >= best_t).astype(np.uint8), cmap="gray")
        axes[ri, 3].set_title(f"Pred @{best_t}")
        for ax in axes[ri]: ax.axis("off")
    plt.tight_layout()
    fig.savefig(VIS_PATH, dpi=80); plt.close(fig)
    print(f"  Saved prediction vis -> {VIS_PATH}")

# ── Updated comparison table (6 models) ─────────────────────────────────────
prev_csvs = [
    ("UNet + BCE (orig)",          "artifacts/tables/corrected_metrics_unet_bce.csv"),
    ("UNet + BCE-Dice (orig)",     "artifacts/tables/corrected_metrics_unet_bce_dice.csv"),
    ("TemporalUNet T=3",           "artifacts/tables/corrected_metrics_temporal_unet_t3.csv"),
    ("UNet + BCE pw=136",          "artifacts/tables/corrected_metrics_unet_bce_pw136.csv"),
    ("UNet + Dice+BCE (0.5/0.5)",  "artifacts/tables/corrected_metrics_unet_dice_bce.csv"),
]

all_rows = []
for display, csv_path in prev_csvs:
    with open(csv_path) as f:
        r = next(csv.DictReader(f))
    all_rows.append(dict(
        model=display, threshold=float(r["threshold"]),
        positive_only_iou=float(r["positive_only_iou"]),
        micro_iou=float(r["micro_iou"]),
        micro_precision=float(r["micro_precision"]),
        micro_recall=float(r["micro_recall"]),
        micro_f1=float(r["micro_f1"]),
        empty_false_positive_rate=float(r["empty_false_positive_rate"]),
    ))

all_rows.append(dict(
    model=DISPLAY, threshold=best_t,
    positive_only_iou=tm["positive_only_iou"],
    micro_iou=tm["micro_iou"],
    micro_precision=tm["micro_precision"],
    micro_recall=tm["micro_recall"],
    micro_f1=tm["micro_f1"],
    empty_false_positive_rate=tm["empty_false_positive_rate"],
))

comp_path = Path("artifacts/tables/final_corrected_model_comparison.csv")
with open(comp_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
print(f"\nSaved updated comparison -> {comp_path}")

print("\n" + "="*94)
print("  FULL CORRECTED MODEL COMPARISON  (test set, val-selected threshold)")
print("="*94)
print(f"  {'Model':<30} {'thr':>4}  {'pos_iou':>8}  {'micro_iou':>9}  "
      f"{'prec':>7}  {'recall':>7}  {'f1':>7}  {'empty_fpr':>9}")
print(f"  {'-'*30}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for r in all_rows:
    print(f"  {r['model']:<30} {r['threshold']:>4.1f}  "
          f"{r['positive_only_iou']:>8.4f}  {r['micro_iou']:>9.4f}  "
          f"{r['micro_precision']:>7.4f}  {r['micro_recall']:>7.4f}  "
          f"{r['micro_f1']:>7.4f}  {r['empty_false_positive_rate']:>9.4f}")
print("="*94)
