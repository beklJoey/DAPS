import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = [
    "UNet+BCE\n(orig)",
    "UNet+BCE-Dice\n(orig)",
    "TemporalUNet\nT=3",
    "UNet+BCE\npw=136",
    "UNet+Dice+BCE\n(0.5/0.5)",
    "UNet+BCE\n+aug+samp",
    "UNet+BCE\n+500 samples",
]

pos_iou   = [0.010651, 0.010651, 0.011609, 0.031421, 0.009160, 0.028291, 0.080325]
precision = [0.003277, 0.003277, 0.003646, 0.020781, 0.002234, 0.014246, 0.053757]
f1        = [0.006533, 0.006533, 0.007266, 0.039880, 0.004242, 0.026027, 0.097148]
empty_fpr = [1.0000,   1.0000,   1.0000,   1.0000,   1.0000,   1.0000,   0.7674]

x = np.arange(len(models))
width = 0.25

# ── Plot 1: pos_iou / precision / F1 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width, pos_iou,   width, label="pos_iou",   color="#4C72B0")
bars2 = ax.bar(x,         precision, width, label="precision", color="#DD8452")
bars3 = ax.bar(x + width, f1,        width, label="F1",        color="#55A868")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — pos_iou / Precision / F1 (test set, val-selected threshold)")
ax.legend()
ax.set_ylim(0, 0.13)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

for bar in [*bars1, *bars2, *bars3]:
    h = bar.get_height()
    if h > 0.002:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

plt.tight_layout()
plt.savefig("artifacts/plots/model_comparison_all7.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model_comparison_all7.png")

# ── Plot 2: empty_fpr ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#d62728" if v == 1.0 else "#ff7f0e" if v > 0.8 else "#2ca02c" for v in empty_fpr]

bars = ax.bar(x, empty_fpr, color=colors, width=0.55)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5)
ax.set_ylabel("Empty False Positive Rate")
ax.set_title("Empty FPR per Model (fraction of empty-mask test samples with any predicted positive)")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

for bar, v in zip(bars, empty_fpr):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("artifacts/plots/empty_fpr_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved empty_fpr_comparison.png")
