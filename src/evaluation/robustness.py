"""Robustness evaluation: per-complexity-group metrics and visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import torch  # pylint: disable=wrong-import-position
from torch.utils.data import DataLoader, Subset  # pylint: disable=wrong-import-position

from .metrics import collect_predictions, compute_metrics  # pylint: disable=wrong-import-position
from ..data.storage import ContrailSample  # pylint: disable=wrong-import-position
from ..training.trainer import ContrailDataset  # pylint: disable=wrong-import-position
from ..utils.logger import get_logger  # pylint: disable=wrong-import-position

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
COMPLEXITY_GROUPS: Tuple[str, ...] = ("low", "mid", "high")
FIGURES_DIR: Path = Path("figures/evaluation")
DEFAULT_THRESHOLD: float = 0.5
BAR_ALPHA: float = 0.8


# ── Grouping helpers ──────────────────────────────────────────────────────────
def group_samples_by_complexity(
    samples: List[ContrailSample],
) -> Dict[str, List[int]]:
    """Return a mapping from complexity group to sample indices.

    Args:
        samples: List of :class:`~src.data.storage.ContrailSample` with
            ``complexity_group`` already assigned.

    Returns:
        Dict ``{'low': [idx, ...], 'mid': [...], 'high': [...]}``.
    """
    groups: Dict[str, List[int]] = {g: [] for g in COMPLEXITY_GROUPS}
    for idx, sample in enumerate(samples):
        group = sample.complexity_group
        if group in groups:
            groups[group].append(idx)
        else:
            logger.warning("Sample %s has unknown complexity_group=%r", sample.sample_id, group)
    return groups


def _make_group_loader(
    dataset: ContrailDataset, indices: List[int], batch_size: int = 8
) -> DataLoader:
    """Build a DataLoader for a subset of samples.

    Args:
        dataset: Full dataset.
        indices: Indices of samples to include.
        batch_size: Batch size for the loader.

    Returns:
        DataLoader wrapping the requested subset.
    """
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_by_group(
    model: torch.nn.Module,
    test_samples: List[ContrailSample],
    dataset: ContrailDataset,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 8,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics separately for each complexity group on the test set.

    Args:
        model: Trained model (eval mode set internally).
        test_samples: Flat list of test-split :class:`ContrailSample` objects.
        dataset: :class:`~src.training.trainer.ContrailDataset` backed by
            *test_samples* (same ordering).
        device: Computation device.
        threshold: Binarisation threshold for all metrics.
        batch_size: Batch size for per-group inference.

    Returns:
        Nested dict ``{group: {metric: value}}``.
    """
    groups = group_samples_by_complexity(test_samples)
    results: Dict[str, Dict[str, float]] = {}

    for group_name, indices in groups.items():
        if not indices:
            logger.warning("No samples in complexity group '%s' — skipping", group_name)
            continue
        loader = _make_group_loader(dataset, indices, batch_size)
        probs, targets = collect_predictions(model, loader, device)
        metrics = compute_metrics(probs, targets, threshold=threshold)
        results[group_name] = metrics
        logger.info(
            "Group %-4s (%3d samples):  %s",
            group_name, len(indices),
            {k: f"{v:.4f}" for k, v in metrics.items()},
        )

    return results


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_group_metrics(
    group_results: Dict[str, Dict[str, float]],
    run_name: str = "model",
) -> None:
    """Save a grouped bar chart of per-complexity-group metrics.

    Args:
        group_results: Output of :func:`evaluate_by_group`.
        run_name: Experiment label used in the filename.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    available_groups = [g for g in COMPLEXITY_GROUPS if g in group_results]
    if not available_groups:
        logger.warning("No groups to plot for '%s'", run_name)
        return

    metric_names = list(next(iter(group_results.values())).keys())
    x = np.arange(len(metric_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, group in enumerate(available_groups):
        values = [group_results[group][m] for m in metric_names]
        offset = (i - len(available_groups) / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=group, alpha=BAR_ALPHA)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-complexity-group performance — {run_name}")
    ax.legend(title="complexity")
    ax.grid(axis="y", alpha=0.3)

    save_path = FIGURES_DIR / f"{run_name}_group_metrics.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved group metrics chart → %s", save_path)


def plot_pr_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    run_name: str = "model",
) -> None:
    """Plot precision-recall curve and save to ``figures/evaluation/``.

    Args:
        probs: Predicted probability maps ``(N, H, W)``.
        targets: Ground-truth binary masks ``(N, H, W)``.
        run_name: Experiment label used in the filename.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    flat_probs = probs.ravel()
    flat_targets = targets.ravel().astype(int)
    precision, recall, _ = precision_recall_curve(flat_targets, flat_probs)
    ap = average_precision_score(flat_targets, flat_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {run_name}  (AP={ap:.3f})")
    ax.grid(True, alpha=0.3)

    save_path = FIGURES_DIR / f"{run_name}_pr_curve.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved PR curve → %s", save_path)
