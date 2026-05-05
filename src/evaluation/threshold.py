"""Threshold scanning: find the optimal binarisation threshold on validation data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import torch  # pylint: disable=wrong-import-position

from .metrics import collect_predictions, compute_metrics  # pylint: disable=wrong-import-position
from ..utils.logger import get_logger  # pylint: disable=wrong-import-position

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
THRESHOLD_LOW: float = 0.30
THRESHOLD_HIGH: float = 0.70
THRESHOLD_STEP: float = 0.05
FIGURES_DIR: Path = Path("figures/evaluation")


def _build_threshold_grid() -> List[float]:
    """Return the canonical list of thresholds to evaluate."""
    n_steps = round((THRESHOLD_HIGH - THRESHOLD_LOW) / THRESHOLD_STEP) + 1
    return [round(THRESHOLD_LOW + i * THRESHOLD_STEP, 6) for i in range(n_steps)]


def scan_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: List[float] | None = None,
) -> Dict[float, Dict[str, float]]:
    """Evaluate all metrics at each candidate threshold.

    Args:
        probs: Predicted probability maps ``(N, H, W)``.
        targets: Ground-truth binary masks ``(N, H, W)``.
        thresholds: List of threshold values to try.  Defaults to the
            canonical grid from 0.30 to 0.70 in steps of 0.05.

    Returns:
        Dict mapping each threshold to a metrics dict
        (``iou``, ``dice``, ``f1``, ``precision``, ``recall``).
    """
    if thresholds is None:
        thresholds = _build_threshold_grid()

    results: Dict[float, Dict[str, float]] = {}
    for t in thresholds:
        results[t] = compute_metrics(probs, targets, threshold=t)
        logger.debug("threshold=%.2f  iou=%.4f", t, results[t]["iou"])

    return results


def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    metric: str = "iou",
) -> Tuple[float, Dict[str, float]]:
    """Find the threshold that maximises *metric* on the provided data.

    Args:
        probs: Predicted probability maps ``(N, H, W)``.
        targets: Ground-truth binary masks ``(N, H, W)``.
        metric: Metric to optimise (default ``'iou'``).

    Returns:
        ``(best_threshold, metrics_at_best_threshold)``
    """
    results = scan_thresholds(probs, targets)
    best_t = max(results, key=lambda t: results[t][metric])
    logger.info(
        "Best threshold = %.2f  (%s = %.4f)",
        best_t, metric, results[best_t][metric],
    )
    return best_t, results[best_t]


def plot_threshold_curves(
    results: Dict[float, Dict[str, float]],
    run_name: str = "model",
) -> None:
    """Plot metric vs threshold curves and save to ``figures/evaluation/``.

    Args:
        results: Output of :func:`scan_thresholds`.
        run_name: Experiment label used in the filename.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(results)
    metrics = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        values = [results[t][metric] for t in thresholds]
        ax.plot(thresholds, values, marker="o", label=metric)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold sensitivity — {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = FIGURES_DIR / f"{run_name}_threshold_curve.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved threshold curve → %s", save_path)


def evaluate_with_best_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    run_name: str = "model",
) -> Tuple[float, Dict[str, float]]:
    """Select best threshold on val set, then report test-set metrics.

    Args:
        model: Trained model (set to eval internally).
        val_loader: Validation data loader for threshold selection.
        test_loader: Test data loader for final evaluation.
        device: Computation device.
        run_name: Experiment label for figure filenames.

    Returns:
        ``(best_threshold, test_metrics_dict)``
    """
    val_probs, val_targets = collect_predictions(model, val_loader, device)
    threshold_results = scan_thresholds(val_probs, val_targets)
    best_t, _ = find_best_threshold(val_probs, val_targets)

    plot_threshold_curves(threshold_results, run_name=run_name)

    test_probs, test_targets = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_probs, test_targets, threshold=best_t)

    logger.info(
        "[%s] Test metrics (threshold=%.2f): %s",
        run_name, best_t,
        {k: f"{v:.4f}" for k, v in test_metrics.items()},
    )
    return best_t, test_metrics
