"""
Prediction comparison: same positive test sample, 3 models side-by-side.
  Col 1: input image (band_11/14/15 as RGB)
  Col 2: ground truth mask
  Col 3: Exp1 pred (UNet+BCE, thr=0.1)
  Col 4: Exp5 pred (UNet+BCE+pw=136, thr=0.6)
  Col 5: Exp8 pred (UNet+BCE+pw=136+500exp, thr=0.8)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.storage import ContrailSampleStore
from src.models import build_model
from src.models.unet import UNet
from src.training.trainer import (
    split_samples, compute_channel_stats,
    _load_and_stack, _select_frame_indices,
)

MIDDLE_FRAME = 4
SMOOTH = 1e-6

DB_PATH   = Path("data/contrail_samples.db")
CKPT1     = Path("checkpoints/m1_unet_bce_seed42_best.pt")
CKPT5     = Path("artifacts/models/unet_bce_pw136_seed42_best.pt")
CKPT8     = Path("artifacts/models/unet_bce_pw136_expanded500_seed42_best.pt")
OUT_PATH  = Path("artifacts/plots/prediction_comparison.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Reproduce Exp1/5 split ────────────────────────────────────────────────────
print("Loading DB samples and reproducing split...")
store = ContrailSampleStore(DB_PATH)
db_samples = store.get_all()
train_s, val_s, test_s = split_samples(db_samples, seed=42)
print(f"  test set: {len(test_s)} samples")

ch_mean, ch_std = compute_channel_stats(train_s)
frame_indices = _select_frame_indices(1)

# Find positive test samples
pos_test = []
for s in test_s:
    mask = np.load(s.mask_path).squeeze(-1).astype(np.float32)
    if mask.sum() > 0:
        pos_test.append((s, mask))
print(f"  positive test samples: {len(pos_test)}")

# Pick sample with the largest mask (most visible contrails)
pos_test.sort(key=lambda x: x[1].sum(), reverse=True)
sample, gt_mask = pos_test[0]
print(f"  chosen sample: {sample.sample_id}  (mask pixels: {int(gt_mask.sum())})")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_display_rgb(s):
    """3-ch RGB from band_11/14/15 for visualization."""
    chans = []
    for bi in [0, 2, 3]:
        arr = np.load(s.frame_paths[bi]).astype(np.float32)
        f = arr[:, :, MIDDLE_FRAME]
        chans.append((f - f.min()) / (f.max() - f.min() + 1e-6))
    return np.stack(chans, axis=-1)  # (H,W,3)

def load_exp15_tensor(s, ch_mean, ch_std, frame_indices):
    img, _ = _load_and_stack(s, frame_indices, ch_mean, ch_std)
    return torch.from_numpy(img).unsqueeze(0)  # (1,5,H,W)

def load_exp8_tensor(s):
    chans = []
    for bi in [0, 2, 3]:
        arr = np.load(s.frame_paths[bi]).astype(np.float32)
        f = arr[:, :, MIDDLE_FRAME]
        chans.append((f - f.min()) / (f.max() - f.min() + 1e-6))
    img = np.stack(chans, axis=0)  # (3,H,W)
    return torch.from_numpy(img).unsqueeze(0)  # (1,3,H,W)

def infer(model, tensor, threshold):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()
    pred = (probs >= threshold).astype(np.float32)
    return pred

def sample_iou(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    if union < SMOOTH:
        return float("nan")
    return float((inter + SMOOTH) / (union + SMOOTH))


# ── Load tensors ──────────────────────────────────────────────────────────────
display_rgb = load_display_rgb(sample)
tensor_15   = load_exp15_tensor(sample, ch_mean, ch_std, frame_indices)
tensor_8    = load_exp8_tensor(sample)

# ── Load models ───────────────────────────────────────────────────────────────
def load_unet(ckpt_path, in_channels):
    state = torch.load(ckpt_path, map_location="cpu")
    # infer base_channels from enc1 first-layer weight shape
    bc = int(state["enc1.conv.block.0.weight"].shape[0])
    model = UNet(in_channels=in_channels, base_channels=bc)
    model.load_state_dict(state)
    return model

print("Loading models...")
model1 = load_unet(CKPT1, in_channels=5)
model5 = load_unet(CKPT5, in_channels=5)
model8 = load_unet(CKPT8, in_channels=3)

# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference...")
pred1 = infer(model1, tensor_15, threshold=0.1)
pred5 = infer(model5, tensor_15, threshold=0.6)
pred8 = infer(model8, tensor_8,  threshold=0.8)

iou1 = sample_iou(pred1, gt_mask)
iou5 = sample_iou(pred5, gt_mask)
iou8 = sample_iou(pred8, gt_mask)
print(f"  Exp1 IoU={iou1:.4f}  Exp5 IoU={iou5:.4f}  Exp8 IoU={iou8:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("Qualitative Prediction Comparison on a Positive Test Sample",
             fontsize=13, fontweight="bold", y=1.01)

panels = [
    (display_rgb, "Input Image\n(band 11/14/15)", None,   None),
    (gt_mask,     "Ground Truth Mask",             "gray", None),
    (pred1,       "Exp 1: UNet+BCE",               "gray", f"pos_iou = {iou1:.4f}"),
    (pred5,       "Exp 5: UNet+BCE+pw=136",        "gray", f"pos_iou = {iou5:.4f}"),
    (pred8,       "Exp 7: UNet+BCE+pw=136\n+500 samples", "gray", f"pos_iou = {iou8:.4f}"),
]

for ax, (img, col_title, cmap, subtitle) in zip(axes, panels):
    if img.ndim == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9, labelpad=5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_PATH}")
