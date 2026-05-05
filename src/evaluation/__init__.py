"""Evaluation utilities: metrics, threshold scanning, and robustness analysis."""

from .metrics import (
    collect_predictions,
    compute_corrected_metrics,
    compute_metrics,
    dice_score,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
)
from .robustness import evaluate_by_group, plot_group_metrics, plot_pr_curve
from .threshold import (
    evaluate_with_best_threshold,
    find_best_threshold,
    plot_threshold_curves,
    scan_thresholds,
)

__all__ = [
    "collect_predictions",
    "compute_corrected_metrics",
    "compute_metrics",
    "dice_score",
    "f1_score",
    "iou_score",
    "precision_score",
    "recall_score",
    "evaluate_by_group",
    "plot_group_metrics",
    "plot_pr_curve",
    "evaluate_with_best_threshold",
    "find_best_threshold",
    "plot_threshold_curves",
    "scan_thresholds",
]
