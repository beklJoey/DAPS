"""
Image-level filter on UNet+BCE+pw=136.

Pipeline per sample:
  1. pixel_pred = (probs >= PIXEL_THRESHOLD)       [val-selected = 0.6]
  2. if max(probs) < filter_threshold: pred = zeros [image-level gate]

Scan filter_threshold in [0.30, 0.35, ..., 0.90] and report
precision, recall, F1, empty_fpr at each value.
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib; matplotlib.use("Agg")
import numpy as np

from src.data.storage import ContrailSampleStore
from src.training.trainer import split_samples, compute_channel_stats, ContrailDataset, BATCH_SIZE
from src.models import build_model
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from torch.utils.data import DataLoader

DEVICE        = torch.device("cpu")
SPLIT_SEED    = 42
CKPT          = Path("artifacts/models/unet_bce_pw136_seed42_best.pt")
PIXEL_THRESH  = 0.6   # val-selected threshold from previous experiment
TABLES_DIR    = Path("artifacts/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FILTER_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.95, 0.05)]

# ── Data ──────────────────────────────────────────────────────────────────────
store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
ch_mean, ch_std = compute_channel_stats(train_s)

test_loader = DataLoader(
    ContrailDataset(test_s, n_frames=1, channel_mean=ch_mean,
                    channel_std=ch_std, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)

# ── Load model & run inference once ──────────────────────────────────────────
state   = torch.load(str(CKPT), map_location=DEVICE)
base_ch = next(v.shape[0] for k, v in state.items()
               if "enc1" in k and "weight" in k and v.dim() == 4)
model   = build_model("unet", n_frames=1, base_channels=base_ch).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded {CKPT.name}  base_ch={base_ch}")

test_probs, test_targets = collect_predictions(model, test_loader, DEVICE)
N = len(test_probs)
max_probs = test_probs.max(axis=(1, 2))   # (N,) per-sample max probability

n_pos_mask = int((test_targets.sum(axis=(1,2)) > 0).sum())
print(f"Test: {N} samples ({n_pos_mask} positive-mask, {N-n_pos_mask} empty-mask)")
print(f"max_prob stats: min={max_probs.min():.4f}  mean={max_probs.mean():.4f}"
      f"  median={np.median(max_probs):.4f}  max={max_probs.max():.4f}")
print(f"Pixel threshold (val-selected): {PIXEL_THRESH}")

# per-sample max_prob breakdown
print(f"\nSample max_prob distribution:")
for s_i in range(N):
    has_mask = "pos" if test_targets[s_i].sum() > 0 else "empty"
    print(f"  sample {s_i:2d}  max_prob={max_probs[s_i]:.4f}  [{has_mask}]")

# ── Metrics helpers ───────────────────────────────────────────────────────────
SMOOTH = 1e-6

def metrics_from_binary(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute corrected metrics directly from binary pred/target arrays."""
    N = len(preds)
    tgts  = targets.astype(bool)
    preds = preds.astype(bool)

    pos_idx   = [i for i in range(N) if tgts[i].any()]
    empty_idx = [i for i in range(N) if not tgts[i].any()]

    # pixel-level micro aggregation
    p_flat = preds.ravel()
    t_flat = tgts.ravel()
    tp = int((p_flat &  t_flat).sum())
    fp = int((p_flat & ~t_flat).sum())
    fn = int((~p_flat & t_flat).sum())

    prec = (tp + SMOOTH) / (tp + fp + SMOOTH)
    rec  = (tp + SMOOTH) / (tp + fn + SMOOTH)
    f1   = 2 * prec * rec / (prec + rec + SMOOTH)

    # positive-only iou (macro mean)
    def _iou(i):
        inter = int((preds[i] & tgts[i]).sum())
        union = int((preds[i] | tgts[i]).sum())
        return (inter + SMOOTH) / (union + SMOOTH)

    pos_iou = float(np.mean([_iou(i) for i in pos_idx])) if pos_idx else float("nan")

    # empty-sample false positive rate
    if empty_idx:
        total_px = preds.shape[1] * preds.shape[2]
        rates    = np.array([preds[i].sum() / total_px for i in empty_idx])
        empty_fpr = float((rates > 0).mean())
    else:
        empty_fpr = 0.0

    return dict(precision=float(prec), recall=float(rec), f1=float(f1),
                pos_iou=pos_iou, empty_fpr=empty_fpr,
                tp=tp, fp=fp, fn=fn,
                n_filtered=0)   # filled by caller


# ── Baseline: no filter (pixel threshold only) ───────────────────────────────
base_preds = (test_probs >= PIXEL_THRESH)
base_m = metrics_from_binary(base_preds, test_targets)
print(f"\nBaseline (no filter, pixel_thresh={PIXEL_THRESH}):")
print(f"  precision={base_m['precision']:.4f}  recall={base_m['recall']:.4f}"
      f"  f1={base_m['f1']:.4f}  pos_iou={base_m['pos_iou']:.4f}"
      f"  empty_fpr={base_m['empty_fpr']:.4f}")

# ── Filter scan ───────────────────────────────────────────────────────────────
rows = []
print(f"\n  {'filter_t':>8}  {'n_filtered':>10}  {'precision':>10}  "
      f"{'recall':>8}  {'f1':>8}  {'pos_iou':>8}  {'empty_fpr':>10}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

for ft in FILTER_THRESHOLDS:
    # apply filter: zero out predictions where max_prob < ft
    filtered_probs = test_probs.copy()
    filtered_mask  = max_probs < ft          # (N,) bool
    filtered_probs[filtered_mask] = 0.0
    n_filtered = int(filtered_mask.sum())

    preds = (filtered_probs >= PIXEL_THRESH)
    m = metrics_from_binary(preds, test_targets)
    m["n_filtered"] = n_filtered
    m["filter_threshold"] = ft
    rows.append(m)

    poi_s = f"{m['pos_iou']:.4f}" if not np.isnan(m["pos_iou"]) else "   nan"
    print(f"  {ft:>8.2f}  {n_filtered:>10d}  {m['precision']:>10.4f}  "
          f"{m['recall']:>8.4f}  {m['f1']:>8.4f}  {poi_s:>8}  {m['empty_fpr']:>10.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = TABLES_DIR / "image_filter_scan_unet_bce_pw136.csv"
fields = ["filter_threshold", "n_filtered", "precision", "recall",
          "f1", "pos_iou", "empty_fpr", "tp", "fp", "fn"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"\nSaved -> {csv_path}")

# best F1
best = max(rows, key=lambda r: r["f1"])
print(f"Best F1={best['f1']:.4f} @ filter_threshold={best['filter_threshold']}"
      f"  precision={best['precision']:.4f}  recall={best['recall']:.4f}"
      f"  empty_fpr={best['empty_fpr']:.4f}  n_filtered={best['n_filtered']}")
