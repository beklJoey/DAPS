"""Generate two comparison plots and save to artifacts/plots/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

comp = read_csv("artifacts/tables/final_corrected_model_comparison.csv")
scan = read_csv("artifacts/tables/threshold_scan_unet_bce_pw136.csv")

# ── Plot 1: Model comparison bar chart ───────────────────────────────────────
labels = [r["model"] for r in comp]
label_map = {
    "UNet + BCE (orig)":          "UNet+BCE\n(orig)",
    "UNet + BCE-Dice (orig)":     "UNet+BCE-Dice\n(orig)",
    "TemporalUNet T=3":           "TemporalUNet\nT=3",
    "UNet + BCE pw=136":          "UNet+BCE\npw=136",
    "UNet + Dice+BCE (0.5/0.5)":  "UNet+Dice+BCE\n(0.5/0.5)",
}
short_labels = [label_map.get(l, l) for l in labels]
pos_iou   = [float(r["positive_only_iou"]) for r in comp]
precision = [float(r["micro_precision"])    for r in comp]
f1        = [float(r["micro_f1"])           for r in comp]

x = np.arange(len(labels))
width = 0.25

fig1, ax1 = plt.subplots(figsize=(11, 5))
bars_iou  = ax1.bar(x - width, pos_iou,   width, label="pos_only_iou", color="#4C72B0")
bars_prec = ax1.bar(x,         precision, width, label="precision",    color="#DD8452")
bars_f1   = ax1.bar(x + width, f1,        width, label="F1",           color="#55A868")

ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, fontsize=9)
ax1.set_ylabel("Score")
ax1.set_title("Model Comparison on Test Set", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(0, max(max(pos_iou), max(precision), max(f1)) * 1.35)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# value labels on bars
for bar in [*bars_iou, *bars_prec, *bars_f1]:
    h = bar.get_height()
    if h > 0.001:
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.0003,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

fig1.tight_layout()
out1 = PLOTS_DIR / "model_comparison.png"
fig1.savefig(out1, dpi=120)
plt.close(fig1)
print(f"Saved -> {out1}")

# ── Plot 2: Threshold scan line chart ─────────────────────────────────────────
thresholds = [float(r["threshold"])           for r in scan]
val_pos    = [float(r["val_positive_only_iou"]) for r in scan]
best_t = 0.6
best_v = val_pos[thresholds.index(best_t)]

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(thresholds, val_pos, marker="o", linewidth=2,
         color="#4C72B0", markersize=6, label="val_pos_iou")
ax2.axvline(x=best_t, color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"best threshold = {best_t}")
ax2.scatter([best_t], [best_v], color="#C44E52", zorder=5, s=80)
ax2.annotate(f"  {best_v:.4f}", xy=(best_t, best_v),
             xytext=(best_t + 0.03, best_v + 0.003),
             fontsize=9, color="#C44E52")

ax2.set_xlabel("Threshold", fontsize=11)
ax2.set_ylabel("val positive_only_iou", fontsize=11)
ax2.set_title("Threshold Selection (UNet+BCE+pw=136)", fontsize=13, fontweight="bold")
ax2.set_xticks(thresholds)
ax2.set_xlim(0.05, 0.95)
ax2.set_ylim(0, max(val_pos) * 1.25)
ax2.legend(fontsize=9)
ax2.grid(linestyle="--", alpha=0.5)

fig2.tight_layout()
out2 = PLOTS_DIR / "threshold_scan_pw136.png"
fig2.savefig(out2, dpi=120)
plt.close(fig2)
print(f"Saved -> {out2}")
