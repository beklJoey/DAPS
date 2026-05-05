"""
UNet + BCE(pw=136) + image-level classification head.

Architecture addition (bottleneck output -> GAP -> Linear(1)):
  img_logit = Linear(c*16, 1)(GAP(bottleneck_features))

Joint loss:
  total_loss = seg_loss(seg_logits, mask) + 0.5 * bce(img_logit, image_label)
  where image_label = 1 if mask.sum() > 0 else 0

Inference gate:
  if sigmoid(img_logit) < 0.5: output all-zero mask
  else: output sigmoid(seg_logits) >= PIXEL_THRESHOLD

All other settings identical to UNet+BCE+pw=136 experiment.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.storage import ContrailSampleStore
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from src.training.trainer import (
    ContrailDataset, split_samples, compute_channel_stats,
    BATCH_SIZE, NUM_WORKERS, BANDS_PER_FRAME,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
N_EPOCHS      = 30
PATIENCE      = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
BASE_CHANNELS = 32
N_FRAMES      = 1
SPLIT_SEED    = 42
POS_WEIGHT    = 136.0
CLS_LOSS_W    = 0.5       # weight of classification loss
IMG_GATE      = 0.5       # inference gate threshold on img_prob
PIXEL_THRESH  = 0.6       # val-selected pixel-level threshold (reused from pw=136)
DEVICE        = torch.device("cpu")
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

LABEL       = "unet_bce_pw136_cls_head"
DISPLAY     = "UNet+BCE pw=136 +cls_head"
CKPT_PATH   = Path("artifacts/models/unet_bce_pw136_cls_head_seed42_best.pt")
METRICS_CSV = Path("artifacts/tables/corrected_metrics_unet_bce_pw136_cls_head.csv")
SCAN_CSV    = Path("artifacts/tables/threshold_scan_unet_bce_pw136_cls_head.csv")
HIST_CSV    = Path("artifacts/tables/training_history_unet_bce_pw136_cls_head.csv")
VIS_PATH    = Path("artifacts/predictions/prediction_unet_bce_pw136_cls_head.png")

for p in [CKPT_PATH.parent, METRICS_CSV.parent, VIS_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── UNet building blocks (copied from src/models/unet.py) ────────────────────
DROPOUT_P = 0.1

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
    def forward(self, x): return self.block(x)

class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── UNet with classification head ─────────────────────────────────────────────
class UNetWithClsHead(nn.Module):
    """U-Net with an image-level classification head on the bottleneck.

    forward() returns (seg_logits, img_logit):
      seg_logits : (B, 1, H, W)  — raw segmentation logits
      img_logit  : (B, 1)        — raw image-level logit (has contrail?)
    """
    def __init__(self, in_channels: int = BANDS_PER_FRAME,
                 base_channels: int = BASE_CHANNELS) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = _EncoderBlock(in_channels, c)
        self.enc2 = _EncoderBlock(c, c * 2)
        self.enc3 = _EncoderBlock(c * 2, c * 4)
        self.enc4 = _EncoderBlock(c * 4, c * 8)

        self.bottleneck = _ConvBlock(c * 8, c * 16, dropout_p=DROPOUT_P)

        # image-level classification head
        self.gap     = nn.AdaptiveAvgPool2d(1)          # (B, c*16, 1, 1)
        self.cls_head = nn.Linear(c * 16, 1)            # (B, 1)

        self.dec4 = _DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = _DecoderBlock(c * 8,  c * 4, c * 4)
        self.dec2 = _DecoderBlock(c * 4,  c * 2, c * 2)
        self.dec1 = _DecoderBlock(c * 2,  c,     c)
        self.seg_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        bn    = self.bottleneck(x)

        # classification branch
        img_logit = self.cls_head(self.gap(bn).flatten(1))  # (B, 1)

        # segmentation branch
        x = self.dec4(bn, s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)
        seg_logits = self.seg_head(x)                       # (B, 1, H, W)

        return seg_logits, img_logit


# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
print(f"Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")
print("Computing channel stats...")
ch_mean, ch_std = compute_channel_stats(train_s)

def _make_loader(split, shuffle):
    ds = ContrailDataset(split, n_frames=N_FRAMES,
                         channel_mean=ch_mean, channel_std=ch_std, augment=False)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)

train_loader = _make_loader(train_s, shuffle=True)
val_loader   = _make_loader(val_s,   shuffle=False)
test_loader  = _make_loader(test_s,  shuffle=False)

# ── Loss ──────────────────────────────────────────────────────────────────────
seg_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32)
)
cls_criterion = nn.BCEWithLogitsLoss()   # no pos_weight: cls labels are ~50/50

# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)                  # (B,1,H,W)

        image_labels = (masks.sum(dim=(1,2,3)) > 0).float().unsqueeze(1)  # (B,1)

        optimizer.zero_grad()
        seg_logits, img_logit = model(images)

        seg_loss = seg_criterion(seg_logits, masks.float())
        cls_loss = cls_criterion(img_logit, image_labels)
        loss = seg_loss + CLS_LOSS_W * cls_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def val_epoch(model) -> tuple[float, dict, float]:
    """Returns (loss, corrected_metrics_at_0.5, mean_cls_acc)."""
    model.eval()
    total_loss = 0.0
    all_seg_probs, all_img_probs, all_tgts = [], [], []
    cls_correct, cls_total = 0, 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            image_labels = (masks.sum(dim=(1,2,3)) > 0).float().unsqueeze(1)

            seg_logits, img_logit = model(images)
            seg_loss = seg_criterion(seg_logits, masks.float())
            cls_loss = cls_criterion(img_logit, image_labels)
            total_loss += (seg_loss + CLS_LOSS_W * cls_loss).item() * images.size(0)

            # collect raw probs; apply gate after full concatenation
            all_seg_probs.append(torch.sigmoid(seg_logits).squeeze(1).cpu().numpy())  # (B,H,W)
            all_img_probs.append(torch.sigmoid(img_logit).squeeze(1).cpu().numpy())   # (B,)
            all_tgts.append(masks.squeeze(1).cpu().numpy())                           # (B,H,W)

            cls_pred = (torch.sigmoid(img_logit) >= 0.5).float()
            cls_correct += (cls_pred == image_labels).sum().item()
            cls_total   += image_labels.size(0)

    seg_probs_np = np.concatenate(all_seg_probs, axis=0)   # (N,H,W)
    tgts_np      = np.concatenate(all_tgts,      axis=0)   # (N,H,W)
    img_probs_np = np.concatenate(all_img_probs,  axis=0)  # (N,)
    gate         = (img_probs_np >= IMG_GATE)[:, np.newaxis, np.newaxis]  # (N,1,1)
    gated_probs  = seg_probs_np * gate
    m = compute_corrected_metrics(gated_probs, tgts_np, threshold=PIXEL_THRESH)
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
    return total_loss / len(val_loader.dataset), m, cls_acc


def _monitor(m):
    v = m.get("positive_only_iou", float("nan"))
    return v if not np.isnan(v) else m.get("micro_iou", 0.0)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  {DISPLAY}")
print(f"  seg_loss(pw={POS_WEIGHT}) + {CLS_LOSS_W}*cls_bce  |  gate@{IMG_GATE}  |  pixel@{PIXEL_THRESH}")
print(f"{'='*64}")

torch.manual_seed(SEED)
model     = UNetWithClsHead(in_channels=BANDS_PER_FRAME, base_channels=BASE_CHANNELS).to(DEVICE)
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
    vl_loss, vm, cls_acc = val_epoch(model)
    monitor = _monitor(vm)
    scheduler.step(monitor)
    elapsed = time.time() - t0

    pos_iou   = vm["positive_only_iou"]
    micro_iou = vm["micro_iou"]
    print(f"  [epoch {epoch:02d}/{N_EPOCHS}]  tr={tr_loss:.4f}  vl={vl_loss:.4f}"
          f"  val_pos_iou={pos_iou:.4f}  val_micro_iou={micro_iou:.4f}"
          f"  cls_acc={cls_acc:.3f}  monitor={monitor:.4f}  ({elapsed:.1f}s)")

    history.append(dict(epoch=epoch, tr=tr_loss, vl=vl_loss,
                        pos_iou=pos_iou, micro_iou=micro_iou, cls_acc=cls_acc))

    if monitor > best_monitor:
        best_monitor = monitor
        best_epoch   = epoch
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
    w = csv.DictWriter(f, fieldnames=["epoch","tr","vl","pos_iou","micro_iou","cls_acc"])
    w.writeheader(); w.writerows(history)
print(f"  Saved training history -> {HIST_CSV}")

# ── Load best weights ─────────────────────────────────────────────────────────
print(f"  Loading best weights from epoch {best_epoch}")
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ── Collect gated test predictions ───────────────────────────────────────────
def collect_gated(model, loader):
    """Run inference; returns (gated_seg_probs, targets, img_probs) all as (N,...) arrays."""
    model.eval()
    all_seg_probs, all_tgts, all_img_probs = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            seg_logits, img_logit = model(images)
            all_seg_probs.append(torch.sigmoid(seg_logits).squeeze(1).cpu().numpy())  # (B,H,W)
            all_img_probs.append(torch.sigmoid(img_logit).squeeze(1).cpu().numpy())   # (B,)
            tgts = masks.squeeze(1).cpu().numpy() if masks.dim() == 4 else masks.cpu().numpy()
            all_tgts.append(tgts)

    seg_probs_np = np.concatenate(all_seg_probs, axis=0)
    img_probs_np = np.concatenate(all_img_probs, axis=0)
    tgts_np      = np.concatenate(all_tgts,      axis=0)
    gate         = (img_probs_np >= IMG_GATE)[:, np.newaxis, np.newaxis]
    return seg_probs_np * gate, tgts_np, img_probs_np

# ── Val threshold scan (on gated probs) ─────────────────────────────────────
print("  Threshold scan on validation set (with gate)...")
val_probs_g, val_targets_g, val_img_probs = collect_gated(model, val_loader)
print(f"  val img_prob: min={val_img_probs.min():.4f}  mean={val_img_probs.mean():.4f}"
      f"  max={val_img_probs.max():.4f}")
print(f"  val seg probs (after gate): min={val_probs_g.min():.4f}"
      f"  mean={val_probs_g.mean():.4f}  max={val_probs_g.max():.4f}")

scan_rows = []
best_t, best_val_pos_iou = THRESHOLD_GRID[0], -1.0
print(f"  {'thresh':>6}  {'val_pos_iou':>11}  {'val_micro_iou':>13}  {'recall':>8}")
for t in THRESHOLD_GRID:
    m = compute_corrected_metrics(val_probs_g, val_targets_g, threshold=t)
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

# ── Test evaluation ───────────────────────────────────────────────────────────
print("  Test evaluation (with gate)...")
test_probs_g, test_targets_g, test_img_probs = collect_gated(model, test_loader)

print(f"\n  Per-sample img_prob on test set:")
for i in range(len(test_img_probs)):
    has_mask = "pos" if test_targets_g[i].sum() > 0 else "empty"
    gate_str = "PASS" if test_img_probs[i] >= IMG_GATE else "BLOCKED"
    print(f"    sample {i:2d}  img_prob={test_img_probs[i]:.4f}  [{has_mask}]  -> {gate_str}")

tm = compute_corrected_metrics(test_probs_g, test_targets_g, threshold=best_t)
print(f"\n  -- Test results (pixel_thresh={best_t}, gate@{IMG_GATE}) --------")
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
pos_idx = [i for i in range(len(test_targets_g)) if test_targets_g[i].sum() > 0]
vis_idx = pos_idx[:4]
if vis_idx:
    test_ds = ContrailDataset(test_s, n_frames=N_FRAMES,
                              channel_mean=ch_mean, channel_std=ch_std)
    n = len(vis_idx)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for ri, si in enumerate(vis_idx):
        img, _ = test_ds[si]
        axes[ri,0].imshow(img[0].numpy(),                              cmap="gray"); axes[ri,0].set_title("Band-11")
        axes[ri,1].imshow(test_targets_g[si],                         cmap="gray"); axes[ri,1].set_title("GT mask")
        axes[ri,2].imshow(test_probs_g[si],   cmap="hot",vmin=0,vmax=1);            axes[ri,2].set_title(f"Prob (gated, img_p={test_img_probs[si]:.2f})")
        axes[ri,3].imshow((test_probs_g[si]>=best_t).astype(np.uint8), cmap="gray"); axes[ri,3].set_title(f"Pred @{best_t}")
        for ax in axes[ri]: ax.axis("off")
    plt.tight_layout()
    fig.savefig(VIS_PATH, dpi=80); plt.close(fig)
    print(f"  Saved prediction vis -> {VIS_PATH}")

# ── Updated comparison table ─────────────────────────────────────────────────
prev_csvs = [
    ("UNet+BCE (orig)",           "artifacts/tables/corrected_metrics_unet_bce.csv"),
    ("UNet+BCE-Dice (orig)",      "artifacts/tables/corrected_metrics_unet_bce_dice.csv"),
    ("TemporalUNet T=3",          "artifacts/tables/corrected_metrics_temporal_unet_t3.csv"),
    ("UNet+BCE pw=136",           "artifacts/tables/corrected_metrics_unet_bce_pw136.csv"),
    ("UNet+Dice+BCE (0.5/0.5)",   "artifacts/tables/corrected_metrics_unet_dice_bce.csv"),
    ("UNet+BCE pw=136 +aug+samp", "artifacts/tables/corrected_metrics_unet_bce_pw136_aug_sampler.csv"),
]
all_rows = []
for display, csv_path in prev_csvs:
    with open(csv_path) as f:
        r = next(csv.DictReader(f))
    all_rows.append(dict(model=display, threshold=float(r["threshold"]),
        positive_only_iou=float(r["positive_only_iou"]), micro_iou=float(r["micro_iou"]),
        micro_precision=float(r["micro_precision"]),    micro_recall=float(r["micro_recall"]),
        micro_f1=float(r["micro_f1"]),
        empty_false_positive_rate=float(r["empty_false_positive_rate"])))

all_rows.append(dict(model=DISPLAY, threshold=best_t,
    positive_only_iou=tm["positive_only_iou"], micro_iou=tm["micro_iou"],
    micro_precision=tm["micro_precision"],     micro_recall=tm["micro_recall"],
    micro_f1=tm["micro_f1"],
    empty_false_positive_rate=tm["empty_false_positive_rate"]))

comp_path = Path("artifacts/tables/final_corrected_model_comparison.csv")
with open(comp_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
print(f"\nSaved updated comparison -> {comp_path}")

print("\n" + "="*96)
print("  FULL CORRECTED MODEL COMPARISON  (test set, val-selected threshold)")
print("="*96)
print(f"  {'Model':<30} {'thr':>4}  {'pos_iou':>8}  {'micro_iou':>9}  "
      f"{'prec':>7}  {'recall':>7}  {'f1':>7}  {'empty_fpr':>9}")
print(f"  {'-'*30}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for r in all_rows:
    print(f"  {r['model']:<30} {r['threshold']:>4.1f}  "
          f"{r['positive_only_iou']:>8.4f}  {r['micro_iou']:>9.4f}  "
          f"{r['micro_precision']:>7.4f}  {r['micro_recall']:>7.4f}  "
          f"{r['micro_f1']:>7.4f}  {r['empty_false_positive_rate']:>9.4f}")
print("="*96)
