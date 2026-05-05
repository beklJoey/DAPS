"""Fine-grained threshold scan on test set for UNet+BCE+pw=136."""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.storage import ContrailSampleStore
from src.training.trainer import split_samples, compute_channel_stats, ContrailDataset, BATCH_SIZE
from src.models import build_model
from src.evaluation.metrics import collect_predictions, compute_corrected_metrics
from torch.utils.data import DataLoader

DEVICE     = torch.device("cpu")
SPLIT_SEED = 42
CKPT       = Path("artifacts/models/unet_bce_pw136_seed42_best.pt")
PLOTS_DIR  = Path("artifacts/plots")
TABLES_DIR = Path("artifacts/tables")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]

# ── Data ──────────────────────────────────────────────────────────────────────
store = ContrailSampleStore(Path("data/contrail_samples.db"))
samples = store.get_all()
train_s, val_s, test_s = split_samples(samples, seed=SPLIT_SEED)
ch_mean, ch_std = compute_channel_stats(train_s)

test_ds     = ContrailDataset(test_s, n_frames=1, channel_mean=ch_mean,
                              channel_std=ch_std, augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Load model ────────────────────────────────────────────────────────────────
state = torch.load(str(CKPT), map_location=DEVICE)
base_ch = next(v.shape[0] for k, v in state.items()
               if "enc1" in k and "weight" in k and v.dim() == 4)
model = build_model("unet", n_frames=1, base_channels=base_ch).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Loaded {CKPT.name}  base_ch={base_ch}")

# ── Inference ─────────────────────────────────────────────────────────────────
test_probs, test_targets = collect_predictions(model, test_loader, DEVICE)
print(f"test probs: min={test_probs.min():.4f}  mean={test_probs.mean():.4f}"
      f"  p95={np.percentile(test_probs,95):.4f}  max={test_probs.max():.4f}")
print(f"test samples: {len(test_targets)} total, "
      f"{sum(1 for t in test_targets if t.sum()>0)} positive-mask")

# ── Threshold scan ────────────────────────────────────────────────────────────
rows = []
print(f"\n  {'thresh':>6}  {'precision':>10}  {'recall':>8}  {'f1':>8}  "
      f"{'pos_iou':>8}  {'empty_fpr':>10}")
print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

for t in THRESHOLDS:
    m = compute_corrected_metrics(test_probs, test_targets, threshold=t)
    row = dict(
        threshold=t,
        precision=m["micro_precision"],
        recall=m["micro_recall"],
        f1=m["micro_f1"],
        pos_iou=m["positive_only_iou"],
        empty_fpr=m["empty_false_positive_rate"],
    )
    rows.append(row)
    print(f"  {t:>6.2f}  {row['precision']:>10.4f}  {row['recall']:>8.4f}"
          f"  {row['f1']:>8.4f}  {row['pos_iou']:>8.4f}  {row['empty_fpr']:>10.4f}")

# save CSV
scan_path = TABLES_DIR / "threshold_scan_fine_unet_bce_pw136.csv"
with open(scan_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nSaved -> {scan_path}")

# ── Precision-Recall curve ────────────────────────────────────────────────────
precisions = [r["precision"] for r in rows]
recalls    = [r["recall"]    for r in rows]
f1s        = [r["f1"]        for r in rows]
thresholds = [r["threshold"] for r in rows]

best_f1_idx = int(np.argmax(f1s))
best_t  = thresholds[best_f1_idx]
best_p  = precisions[best_f1_idx]
best_r  = recalls[best_f1_idx]
best_f1 = f1s[best_f1_idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Left: P-R curve --
ax = axes[0]
ax.plot(recalls, precisions, marker="o", markersize=4,
        linewidth=2, color="#4C72B0", label="Precision-Recall")
ax.scatter([best_r], [best_p], color="#C44E52", zorder=5, s=80)
ax.annotate(f"  best F1={best_f1:.4f}\n  @t={best_t}",
            xy=(best_r, best_p), xytext=(best_r + 0.04, best_p - 0.002),
            fontsize=8, color="#C44E52")

# annotate a few threshold labels
for idx in range(0, len(rows), 3):
    ax.annotate(f"{thresholds[idx]:.2f}",
                xy=(recalls[idx], precisions[idx]),
                xytext=(2, 4), textcoords="offset points", fontsize=7, color="gray")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curve\n(UNet+BCE+pw=136, test set)", fontsize=11, fontweight="bold")
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.002, max(precisions) * 1.4)
ax.grid(linestyle="--", alpha=0.5)
ax.legend(fontsize=9)

# -- Right: metrics vs threshold --
ax2 = axes[1]
ax2.plot(thresholds, precisions, marker="s", markersize=4, linewidth=2,
         color="#DD8452", label="Precision")
ax2.plot(thresholds, recalls,    marker="^", markersize=4, linewidth=2,
         color="#55A868", label="Recall")
ax2.plot(thresholds, f1s,        marker="o", markersize=4, linewidth=2,
         color="#4C72B0", label="F1")
ax2.axvline(x=best_t, color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"best F1 @t={best_t}")

ax2.set_xlabel("Threshold", fontsize=11)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_title("Precision / Recall / F1 vs Threshold\n(UNet+BCE+pw=136, test set)",
              fontsize=11, fontweight="bold")
ax2.set_xticks([round(t, 2) for t in thresholds[::2]])
ax2.set_xlim(0.02, 0.98)
ax2.grid(linestyle="--", alpha=0.5)
ax2.legend(fontsize=9)

plt.tight_layout()
plot_path = PLOTS_DIR / "pr_curve_unet_bce_pw136.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)
print(f"Saved plot -> {plot_path}")

print(f"\nBest F1={best_f1:.4f} at threshold={best_t}"
      f"  precision={best_p:.4f}  recall={best_r:.4f}")
