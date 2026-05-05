"""Corrected-metrics evaluation for UNet+BCE, UNet+BCE-Dice, TemporalUNet T=3."""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# torch must come before numpy/matplotlib on Windows (shm.dll load order)
import torch
import matplotlib; matplotlib.use("Agg")
import numpy as np

from src.data.storage import ContrailSampleStore
from src.training.trainer import (
    split_samples, compute_channel_stats, ContrailDataset, BATCH_SIZE
)
from torch.utils.data import DataLoader
from src.models import build_model
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics

DEVICE = torch.device("cpu")
SPLIT_SEED = 42
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ARTIFACTS = Path("artifacts/tables")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────────────
store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
ch_mean, ch_std = compute_channel_stats(train_s)
print(f"Split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")

def make_loader(split_samples, n_frames):
    ds = ContrailDataset(split_samples, n_frames=n_frames,
                         channel_mean=ch_mean, channel_std=ch_std, augment=False)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── experiments ───────────────────────────────────────────────────────────────
EXPERIMENTS = [
    dict(
        label="unet_bce",
        display="UNet + BCE",
        ckpt="checkpoints/m1_unet_bce_seed42_best.pt",
        model_name="unet",
        n_frames=1,
        out_csv="artifacts/tables/corrected_metrics_unet_bce.csv",
    ),
    dict(
        label="unet_bce_dice",
        display="UNet + BCE-Dice",
        ckpt="checkpoints/m2_unet_bce_dice_seed42_best.pt",
        model_name="unet",
        n_frames=1,
        out_csv="artifacts/tables/corrected_metrics_unet_bce_dice.csv",
    ),
    dict(
        label="temporal_unet_t3",
        display="TemporalUNet T=3",
        ckpt="artifacts/models/temporal_unet_t3_epoch30_seed42_best.pt",
        model_name="temporal_unet",
        n_frames=3,
        out_csv="artifacts/tables/corrected_metrics_temporal_unet_t3.csv",
    ),
]

# ── metric field names ────────────────────────────────────────────────────────
FIELDS = [
    "experiment", "seed", "threshold",
    "positive_only_iou", "positive_only_precision", "positive_only_recall", "positive_only_f1",
    "micro_iou", "micro_precision", "micro_recall", "micro_f1",
    "empty_false_positive_rate", "empty_predicted_positive_rate_mean",
    "target_positive_sample_count", "total_sample_count",
    "val_best_threshold", "val_pos_iou",
]

comparison_rows = []

for exp in EXPERIMENTS:
    ckpt = Path(exp["ckpt"])
    print(f"\n{'='*60}")
    print(f"  {exp['display']}  ({ckpt.name})")
    print(f"{'='*60}")

    state = torch.load(str(ckpt), map_location=DEVICE)

    # auto-detect base_channels
    base_ch = None
    for k, v in state.items():
        if "enc1" in k and "weight" in k and v.dim() == 4:
            base_ch = v.shape[0]
            break
    if base_ch is None:
        base_ch = 32
    print(f"  base_channels={base_ch}")

    model = build_model(exp["model_name"], n_frames=exp["n_frames"],
                        base_channels=base_ch).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    val_loader  = make_loader(val_s,  exp["n_frames"])
    test_loader = make_loader(test_s, exp["n_frames"])

    # ── val threshold scan ────────────────────────────────────────────────────
    print("  Running val inference...")
    val_probs, val_targets = collect_predictions(model, val_loader, DEVICE)
    print(f"  val probs: min={val_probs.min():.4f}  mean={val_probs.mean():.4f}  "
          f"p95={np.percentile(val_probs,95):.4f}  max={val_probs.max():.4f}")

    best_t, best_val_pos_iou = THRESHOLD_GRID[0], -1.0
    print(f"  {'thresh':>6}  {'val_pos_iou':>11}  {'val_micro_iou':>13}  {'recall':>8}")
    for t in THRESHOLD_GRID:
        m = compute_corrected_metrics(val_probs, val_targets, threshold=t)
        poi = m["positive_only_iou"]
        print(f"  {t:>6.1f}  {poi:>11.6f}  {m['micro_iou']:>13.6f}  {m['micro_recall']:>8.4f}")
        if not np.isnan(poi) and poi > best_val_pos_iou:
            best_val_pos_iou = poi
            best_t = t
    print(f"  -> best_threshold={best_t}  val_pos_iou={best_val_pos_iou:.6f}")

    # ── test evaluation ───────────────────────────────────────────────────────
    print("  Running test inference...")
    test_probs, test_targets = collect_predictions(model, test_loader, DEVICE)
    print(f"  test probs: min={test_probs.min():.4f}  mean={test_probs.mean():.4f}  "
          f"p95={np.percentile(test_probs,95):.4f}  max={test_probs.max():.4f}")

    tm = compute_corrected_metrics(test_probs, test_targets, threshold=best_t)

    # print summary
    print(f"\n  ── Test results (threshold={best_t}) ──────────────")
    print(f"  positive_only_iou   : {tm['positive_only_iou']:.6f}")
    print(f"  micro_iou           : {tm['micro_iou']:.6f}")
    print(f"  micro_precision     : {tm['micro_precision']:.6f}")
    print(f"  micro_recall        : {tm['micro_recall']:.6f}")
    print(f"  micro_f1            : {tm['micro_f1']:.6f}")
    print(f"  empty_fpr           : {tm['empty_false_positive_rate']:.4f}")
    print(f"  pos samples         : {tm['target_positive_sample_count']} / {tm['total_sample_count']}")

    # ── save individual CSV ───────────────────────────────────────────────────
    row = {
        "experiment": exp["label"],
        "seed": 42,
        "threshold": best_t,
        **{k: tm[k] for k in [
            "positive_only_iou", "positive_only_precision",
            "positive_only_recall", "positive_only_f1",
            "micro_iou", "micro_precision", "micro_recall", "micro_f1",
            "empty_false_positive_rate", "empty_predicted_positive_rate_mean",
            "target_positive_sample_count", "total_sample_count",
        ]},
        "val_best_threshold": best_t,
        "val_pos_iou": best_val_pos_iou,
    }
    out = Path(exp["out_csv"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"  Saved -> {out}")

    comparison_rows.append({
        "model": exp["display"],
        **row,
    })

# ── final comparison table ────────────────────────────────────────────────────
comp_path = ARTIFACTS / "final_corrected_model_comparison.csv"
comp_fields = [
    "model", "threshold",
    "positive_only_iou", "micro_iou",
    "micro_precision", "micro_recall", "micro_f1",
    "empty_false_positive_rate",
    "target_positive_sample_count", "total_sample_count",
]
with open(comp_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=comp_fields, extrasaction="ignore")
    w.writeheader()
    for r in comparison_rows:
        w.writerow(r)
print(f"\nSaved comparison -> {comp_path}")

# ── print comparison ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  FINAL CORRECTED MODEL COMPARISON (test set, val-selected threshold)")
print("="*80)
print(f"  {'Model':<22} {'thresh':>6}  {'pos_iou':>8}  {'micro_iou':>9}  "
      f"{'prec':>7}  {'recall':>7}  {'f1':>7}  {'empty_fpr':>9}")
print(f"  {'-'*22}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for r in comparison_rows:
    print(f"  {r['model']:<22} {r['threshold']:>6.1f}  "
          f"{r['positive_only_iou']:>8.4f}  {r['micro_iou']:>9.4f}  "
          f"{r['micro_precision']:>7.4f}  {r['micro_recall']:>7.4f}  "
          f"{r['micro_f1']:>7.4f}  {r['empty_false_positive_rate']:>9.4f}")
print("="*80)
