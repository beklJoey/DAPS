"""Val probability diagnostic — evaluation only, no training."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # must be first on Windows
import matplotlib; matplotlib.use("Agg")

from src.data.storage import ContrailSampleStore
from src.training.trainer import split_samples, compute_channel_stats, build_dataloaders, TrainingConfig
from src.models import build_model
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics

DEVICE = torch.device("cpu")
SPLIT_SEED = 42

store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
ch_mean, ch_std = compute_channel_stats(train_s)

cfg = TrainingConfig(model_name="temporal_unet", loss_name="bce_dice",
                     n_frames=3, use_augmentation=False, run_name="diag")
_, val_loader, _ = build_dataloaders(train_s, val_s, test_s, cfg, ch_mean, ch_std, seed=SPLIT_SEED)

ckpt = Path("artifacts/models/temporal_unet_t3_epoch30_seed42_best.pt")
state = torch.load(str(ckpt), map_location=DEVICE)
base_ch = int(state["unet.enc1.conv.block.0.weight"].shape[0])
model = build_model("temporal_unet", n_frames=3, base_channels=base_ch).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded: {ckpt.name}  base_ch={base_ch}")

val_probs, val_targets = collect_predictions(model, val_loader, DEVICE)
print(f"val_probs shape: {val_probs.shape}  val_targets shape: {val_targets.shape}")

pos_idx   = [i for i in range(len(val_probs)) if val_targets[i].sum() > 0]
empty_idx = [i for i in range(len(val_probs)) if val_targets[i].sum() == 0]
print(f"Val: {len(pos_idx)} positive-mask, {len(empty_idx)} empty-mask")

def stats(arr, label):
    flat = arr.ravel()
    print(f"  {label}:")
    print(f"    min={flat.min():.4f}  mean={flat.mean():.4f}  median={np.median(flat):.4f}")
    print(f"    p95={np.percentile(flat,95):.4f}  p99={np.percentile(flat,99):.4f}  max={flat.max():.4f}")

print("\n=== 1. All val samples (probs) ===")
stats(val_probs, "all")

print("\n=== 2. Positive-mask val samples only ===")
if pos_idx:
    pos_probs   = val_probs[pos_idx]
    pos_targets = val_targets[pos_idx]
    stats(pos_probs, "positive-only probs")
    fg_ratio = pos_targets.mean()
    print(f"  target_foreground_ratio (positive samples): {fg_ratio:.4f}")
    pos_px = pos_probs[pos_targets.astype(bool)]
    neg_px = pos_probs[~pos_targets.astype(bool)]
    print(f"  GT-positive pixels  n={len(pos_px):,}  mean={pos_px.mean():.4f}  median={np.median(pos_px):.4f}  p99={np.percentile(pos_px,99):.4f}  max={pos_px.max():.4f}")
    print(f"  GT-negative pixels  n={len(neg_px):,}  mean={neg_px.mean():.4f}  median={np.median(neg_px):.4f}  p99={np.percentile(neg_px,99):.4f}  max={neg_px.max():.4f}")

print("\n=== 3. Threshold scan on val ===")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
print(f"  {'thresh':>6}  {'pos_iou':>8}  {'micro_iou':>9}  {'precision':>9}  {'recall':>8}  {'f1':>8}  {'pred_pos_rate':>13}")
for t in thresholds:
    m = compute_corrected_metrics(val_probs, val_targets, threshold=t)
    ppr = float((val_probs >= t).mean())
    poi = m["positive_only_iou"]
    poi_s = f"{poi:.4f}" if poi == poi else "   nan"
    print(f"  {t:>6.2f}  {poi_s:>8}  {m['micro_iou']:>9.4f}  "
          f"{m['micro_precision']:>9.4f}  {m['micro_recall']:>8.4f}  {m['micro_f1']:>8.4f}  {ppr:>13.4f}")
