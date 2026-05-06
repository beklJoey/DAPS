"""
Retrain UNet+BCE+pw=136 with expanded dataset:
  - 121 existing DB samples (band_11/14/15 at middle frame → 3 channels)
  - 500 new samples from data/extra/contrails/*.npy (ch0/1/2 → 3 channels)
  - Total: 621 samples, fresh 80/10/10 split
  - UNet(in_channels=3, base_channels=32)
  - BCE + pos_weight=136
  - 30 epochs max, early stopping patience=8 on val positive_only_iou
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# torch MUST come before numpy / matplotlib on Windows (shm.dll load order)
import torch
import torch.nn as nn

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.data.storage import ContrailSampleStore, ContrailSample
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.models.unet import UNet, BASE_CHANNELS

# ── Config ────────────────────────────────────────────────────────────────────
SEED           = 42
N_NEW_SAMPLES  = 500
N_EPOCHS       = 30
PATIENCE       = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
BASE_CH        = 32
IN_CHANNELS    = 3    # band_11/14/15 from DB; ch0/1/2 from extra npy
POS_WEIGHT     = 136.0
BATCH_SIZE     = 8
NUM_WORKERS    = 0
SPLIT_SEED     = 42
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
MIDDLE_FRAME   = 4    # frame index in the 8-frame band files
DEVICE         = torch.device("cpu")
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

EXTRA_DIR   = Path("data/extra/contrails")
TRAIN_DF    = Path("data/extra/train_df.csv")
DB_PATH     = Path(os.environ.get("CONTRAIL_DB_PATH", "data/contrail_samples.db"))

ARTIFACTS_MODELS  = Path("artifacts/models")
ARTIFACTS_TABLES  = Path("artifacts/tables")
ARTIFACTS_PREDS   = Path("artifacts/predictions")
for p in [ARTIFACTS_MODELS, ARTIFACTS_TABLES, ARTIFACTS_PREDS]:
    p.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Loss ──────────────────────────────────────────────────────────────────────
class _BCEPosWeightLoss(nn.Module):
    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self._bce(logits, targets.float())


# ── Dataset ───────────────────────────────────────────────────────────────────
# Each item is ('db', ContrailSample) or ('extra', Path)
Item = Tuple[str, object]

class CombinedDataset(Dataset):
    """Handles both existing DB samples (5-band → 3ch) and extra 4-ch npy samples."""

    def __init__(self, items: List[Item], augment: bool = False) -> None:
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        kind, data = self.items[idx]
        if kind == 'db':
            img, mask = self._load_db(data)
        else:
            img, mask = self._load_extra(data)
        if self.augment:
            img, mask = self._aug(img, mask)
        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())

    def _load_db(self, sample: ContrailSample) -> Tuple[np.ndarray, np.ndarray]:
        # frame_paths = [band_11, band_13, band_14, band_15, band_16]
        # use indices 0 (band_11), 2 (band_14), 3 (band_15)
        channels = []
        for bi in [0, 2, 3]:
            arr = np.load(sample.frame_paths[bi]).astype(np.float32)  # (H,W,8)
            frame = arr[:, :, MIDDLE_FRAME]  # (H,W)
            b_min, b_max = frame.min(), frame.max()
            frame = (frame - b_min) / (b_max - b_min + 1e-6)
            channels.append(frame)
        img = np.stack(channels, axis=0)  # (3,H,W) float32
        mask = np.load(sample.mask_path).squeeze(-1).astype(np.float32)  # (H,W)
        return img, mask

    def _load_extra(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.load(path).astype(np.float32)  # (H,W,4)
        img = arr[:, :, :3].transpose(2, 0, 1)  # (3,H,W) already in [0,1]
        mask = arr[:, :, 3]                      # (H,W) binary float
        return img, mask

    def _aug(self, img: np.ndarray, mask: np.ndarray):
        if np.random.random() > 0.5:
            img  = img[:, :, ::-1]
            mask = mask[:, ::-1]
        if np.random.random() > 0.5:
            img  = img[:, ::-1, :]
            mask = mask[::-1, :]
        k = np.random.randint(0, 4)
        if k:
            img  = np.rot90(img,  k, axes=(1, 2))
            mask = np.rot90(mask, k)
        return img, mask


# ── Data preparation ──────────────────────────────────────────────────────────
print("Loading DB samples...")
store = ContrailSampleStore(DB_PATH)
db_samples = store.get_all()
db_ids = {s.sample_id for s in db_samples}
print(f"  DB samples : {len(db_samples)}")

print("Loading train_df.csv...")
with open(TRAIN_DF) as f:
    train_ids = [row['record_id'] for row in csv.DictReader(f)]
print(f"  train_df   : {len(train_ids)} records")

# Candidates: in train_df, have extra npy, NOT in DB
candidates = []
for rid in train_ids:
    npy_path = EXTRA_DIR / f"{rid}.npy"
    if npy_path.exists() and rid not in db_ids:
        candidates.append(npy_path)
print(f"  Candidates : {len(candidates)} (in train_df, has npy, not in DB)")

rng = random.Random(SEED)
rng.shuffle(candidates)
selected_extra = candidates[:N_NEW_SAMPLES]
print(f"  Selected   : {len(selected_extra)} new extra samples")

# Count positives in new samples
pos_new = sum(np.load(p).astype(np.float32)[:, :, 3].sum() > 0 for p in selected_extra)
print(f"  Positive (mask>0): {pos_new} / {len(selected_extra)} new samples")

# Build combined item list
all_items: List[Item] = (
    [('db', s) for s in db_samples] +
    [('extra', p) for p in selected_extra]
)
print(f"\nTotal items: {len(all_items)}")

# 80/10/10 split
rng2 = np.random.default_rng(SPLIT_SEED)
idx = rng2.permutation(len(all_items)).tolist()
n_train = int(len(all_items) * TRAIN_RATIO)
n_val   = int(len(all_items) * VAL_RATIO)
train_items = [all_items[i] for i in idx[:n_train]]
val_items   = [all_items[i] for i in idx[n_train:n_train + n_val]]
test_items  = [all_items[i] for i in idx[n_train + n_val:]]
print(f"Split: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

# Count positives per split
def _count_pos(items):
    """Count samples that contain at least one positive (contrail) pixel.

    Args:
        items: List of ``(kind, data)`` tuples where *kind* is ``'db'`` or
            ``'extra'`` and *data* is a :class:`~src.data.storage.ContrailSample`
            or Path to the combined ``.npy`` file.

    Returns:
        Number of items whose mask contains any positive pixel.
    """
    n = 0
    for kind, data in items:
        if kind == 'db':
            mask = np.load(data.mask_path).squeeze(-1)
        else:
            mask = np.load(data).astype(np.float32)[:, :, 3]
        if mask.sum() > 0:
            n += 1
    return n

pos_train = _count_pos(train_items)
pos_val   = _count_pos(val_items)
pos_test  = _count_pos(test_items)
print(f"  Positives: {pos_train}/{len(train_items)} train, "
      f"{pos_val}/{len(val_items)} val, {pos_test}/{len(test_items)} test")

def _make_loader(items, shuffle, augment=False):
    """Wrap a list of items in a DataLoader.

    Args:
        items: List of ``(kind, data)`` tuples for :class:`CombinedDataset`.
        shuffle: Whether to shuffle every epoch.
        augment: Whether to apply random horizontal/vertical flips.

    Returns:
        Configured :class:`~torch.utils.data.DataLoader`.
    """
    ds = CombinedDataset(items, augment=augment)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)

train_loader = _make_loader(train_items, shuffle=True,  augment=True)
val_loader   = _make_loader(val_items,   shuffle=False, augment=False)
test_loader  = _make_loader(test_items,  shuffle=False, augment=False)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer) -> float:
    """Run one training epoch and return mean loss.

    Args:
        model: Network to train.
        loader: DataLoader over the training split.
        criterion: Loss function (BCE with pos_weight).
        optimizer: Parameter update rule.

    Returns:
        Mean training loss normalised by dataset size.
    """
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
    return total_loss / max(len(loader.dataset), 1)


def val_epoch(model, loader, criterion) -> tuple:
    """Run one validation epoch.

    Args:
        model: Network to evaluate.
        loader: DataLoader over the validation split.
        criterion: Loss function.

    Returns:
        Tuple of ``(mean_val_loss, metrics_dict)`` from
        :func:`~src.evaluation.metrics.compute_corrected_metrics` at threshold=0.5.
    """
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
    return total_loss / max(len(loader.dataset), 1), metrics


def _monitor(m: dict) -> float:
    """Return ``positive_only_iou``, falling back to ``micro_iou`` if NaN.

    Args:
        m: Metrics dict from :func:`~src.evaluation.metrics.compute_corrected_metrics`.

    Returns:
        Scalar early-stopping signal.
    """
    v = m.get("positive_only_iou", float("nan"))
    return v if not np.isnan(v) else m.get("micro_iou", 0.0)


# ── Training loop ─────────────────────────────────────────────────────────────
LABEL   = "unet_bce_pw136_expanded500"
DISPLAY = "UNet+BCE pw=136 +500 expanded (3ch)"
CKPT    = ARTIFACTS_MODELS / f"{LABEL}_seed{SEED}_best.pt"
METRICS_CSV = ARTIFACTS_TABLES / f"corrected_metrics_{LABEL}.csv"
SCAN_CSV    = ARTIFACTS_TABLES / f"threshold_scan_{LABEL}.csv"

print(f"\n{'='*64}")
print(f"  {DISPLAY}")
print(f"{'='*64}")

torch.manual_seed(SEED)
criterion = _BCEPosWeightLoss(POS_WEIGHT)
model = UNet(in_channels=IN_CHANNELS, base_channels=BASE_CH).to(DEVICE)
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

    pos_iou   = vm["positive_only_iou"]
    micro_iou = vm["micro_iou"]
    elapsed   = time.time() - t0
    print(f"  [epoch {epoch:02d}/{N_EPOCHS}]  tr={tr_loss:.4f}  vl={vl_loss:.4f}"
          f"  val_pos_iou={pos_iou:.4f}  val_micro={micro_iou:.4f}"
          f"  monitor={monitor:.4f}  ({elapsed:.1f}s)")

    history.append(dict(epoch=epoch, tr=tr_loss, vl=vl_loss,
                        pos_iou=pos_iou, micro_iou=micro_iou))

    if monitor > best_monitor:
        best_monitor = monitor
        best_epoch   = epoch
        torch.save(model.state_dict(), CKPT)
        print(f"    ** new best  monitor={monitor:.4f}  epoch={epoch}")
        patience_ctr = 0
    else:
        patience_ctr += 1
        print(f"    patience {patience_ctr}/{PATIENCE}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

hist_csv = ARTIFACTS_TABLES / f"training_history_{LABEL}.csv"
with open(hist_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["epoch","tr","vl","pos_iou","micro_iou"])
    w.writeheader(); w.writerows(history)
print(f"  Saved training history -> {hist_csv}")

# ── Load best & evaluate ──────────────────────────────────────────────────────
print(f"  Loading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

print("  Threshold scan on validation set...")
val_probs, val_targets = collect_predictions(model, val_loader, DEVICE)
print(f"  val probs: min={val_probs.min():.4f}  mean={val_probs.mean():.4f}"
      f"  p95={np.percentile(val_probs,95):.4f}  max={val_probs.max():.4f}")

scan_rows = []
best_t, best_val_pos_iou = THRESHOLD_GRID[0], -1.0
print(f"  {'thresh':>6}  {'val_pos_iou':>11}  {'micro_iou':>9}  {'recall':>8}")
for t in THRESHOLD_GRID:
    m = compute_corrected_metrics(val_probs, val_targets, threshold=t)
    poi = m["positive_only_iou"]
    print(f"  {t:>6.1f}  {poi:>11.6f}  {m['micro_iou']:>9.6f}  {m['micro_recall']:>8.4f}")
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

# ── Append to comparison table ────────────────────────────────────────────────
comp_path = ARTIFACTS_TABLES / "final_corrected_model_comparison.csv"
if comp_path.exists():
    with open(comp_path, newline="") as f:
        existing_rows = list(csv.DictReader(f))
    # Remove old expanded row if re-run
    existing_rows = [r for r in existing_rows if r.get("model") != DISPLAY]
    new_row = {
        "model": DISPLAY,
        "threshold": best_t,
        "positive_only_iou": tm["positive_only_iou"],
        "micro_iou": tm["micro_iou"],
        "micro_precision": tm["micro_precision"],
        "micro_recall": tm["micro_recall"],
        "micro_f1": tm["micro_f1"],
        "empty_false_positive_rate": tm["empty_false_positive_rate"],
    }
    all_comp_rows = existing_rows + [new_row]
    with open(comp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_comp_rows[0].keys()))
        w.writeheader(); w.writerows(all_comp_rows)
    print(f"\nUpdated comparison table -> {comp_path}")

    print("\n" + "="*90)
    print("  FULL MODEL COMPARISON  (test set, val-selected threshold)")
    print("="*90)
    print(f"  {'Model':<36} {'thr':>4}  {'pos_iou':>8}  {'micro_iou':>9}  "
          f"{'prec':>7}  {'recall':>7}  {'f1':>7}  {'empty_fpr':>9}")
    print(f"  {'-'*36}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
    for r in all_comp_rows:
        print(f"  {r['model']:<36} {float(r['threshold']):>4.1f}  "
              f"{float(r['positive_only_iou']):>8.4f}  {float(r['micro_iou']):>9.4f}  "
              f"{float(r['micro_precision']):>7.4f}  {float(r['micro_recall']):>7.4f}  "
              f"{float(r['micro_f1']):>7.4f}  {float(r['empty_false_positive_rate']):>9.4f}")
    print("="*90)
