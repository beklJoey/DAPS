"""Segmentation evaluation metrics: IoU, Dice, F1, Precision, Recall."""

from typing import Dict

import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────────
SMOOTH: float = 1e-6   # numerical stability in denominators
DEFAULT_THRESHOLD: float = 0.5


# ── Array-level helpers ───────────────────────────────────────────────────────
def _to_binary(
    probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Binarise a probability array.

    Args:
        probs: Float array of probabilities in [0, 1].
        threshold: Cutoff above which a pixel is predicted positive.

    Returns:
        Boolean array of the same shape.
    """
    return probs >= threshold


def iou_score(
    preds: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Compute Intersection-over-Union (Jaccard index).

    Args:
        preds: Predicted probabilities or binary mask ``(N,)`` or ``(H, W, ...)``.
        targets: Ground-truth binary mask, same shape as *preds*.
        threshold: Binarisation cutoff applied to *preds*.

    Returns:
        IoU score in [0, 1].
    """
    pred_bin = _to_binary(preds, threshold).astype(bool).ravel()
    tgt_bin = targets.astype(bool).ravel()
    intersection = float((pred_bin & tgt_bin).sum())
    union = float((pred_bin | tgt_bin).sum())
    return (intersection + SMOOTH) / (union + SMOOTH)


def dice_score(
    preds: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Compute Dice coefficient (F1 for binary segmentation).

    Args:
        preds: Predicted probabilities or binary mask.
        targets: Ground-truth binary mask.
        threshold: Binarisation cutoff.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred_bin = _to_binary(preds, threshold).astype(bool).ravel()
    tgt_bin = targets.astype(bool).ravel()
    tp = float((pred_bin & tgt_bin).sum())
    fp = float((pred_bin & ~tgt_bin).sum())
    fn = float((~pred_bin & tgt_bin).sum())
    return (2.0 * tp + SMOOTH) / (2.0 * tp + fp + fn + SMOOTH)


def precision_score(
    preds: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Compute binary precision: TP / (TP + FP).

    Args:
        preds: Predicted probabilities or binary mask.
        targets: Ground-truth binary mask.
        threshold: Binarisation cutoff.

    Returns:
        Precision in [0, 1].
    """
    pred_bin = _to_binary(preds, threshold).astype(bool).ravel()
    tgt_bin = targets.astype(bool).ravel()
    tp = float((pred_bin & tgt_bin).sum())
    fp = float((pred_bin & ~tgt_bin).sum())
    return (tp + SMOOTH) / (tp + fp + SMOOTH)


def recall_score(
    preds: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Compute binary recall (sensitivity): TP / (TP + FN).

    Args:
        preds: Predicted probabilities or binary mask.
        targets: Ground-truth binary mask.
        threshold: Binarisation cutoff.

    Returns:
        Recall in [0, 1].
    """
    pred_bin = _to_binary(preds, threshold).astype(bool).ravel()
    tgt_bin = targets.astype(bool).ravel()
    tp = float((pred_bin & tgt_bin).sum())
    fn = float((~pred_bin & tgt_bin).sum())
    return (tp + SMOOTH) / (tp + fn + SMOOTH)


def f1_score(
    preds: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    For binary segmentation this is equivalent to :func:`dice_score`.

    Args:
        preds: Predicted probabilities or binary mask.
        targets: Ground-truth binary mask.
        threshold: Binarisation cutoff.

    Returns:
        F1 score in [0, 1].
    """
    prec = precision_score(preds, targets, threshold)
    rec = recall_score(preds, targets, threshold)
    return (2.0 * prec * rec + SMOOTH) / (prec + rec + SMOOTH)


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """Compute the full suite of evaluation metrics in one call.

    Args:
        preds: Predicted probabilities array ``(N, H, W)`` or ``(H, W)``.
        targets: Ground-truth binary array, same shape.
        threshold: Binarisation cutoff.

    Returns:
        Dict with keys ``iou``, ``dice``, ``f1``, ``precision``, ``recall``.
    """
    return {
        "iou": iou_score(preds, targets, threshold),
        "dice": dice_score(preds, targets, threshold),
        "f1": f1_score(preds, targets, threshold),
        "precision": precision_score(preds, targets, threshold),
        "recall": recall_score(preds, targets, threshold),
    }


# ── Corrected metrics (empty-mask aware) ─────────────────────────────────────
def compute_corrected_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute per-group metrics that do not inflate IoU via empty-mask samples.

    Partitions the N-sample batch into positive-mask samples (target has at
    least one foreground pixel) and empty-mask samples (target is all-zero),
    then reports separate statistics for each group plus pixel-level micro
    aggregates.

    Args:
        probs: Sigmoid probability maps ``(N, H, W)`` in [0, 1].
        targets: Ground-truth binary masks ``(N, H, W)`` with values in {0, 1}.
        threshold: Binarisation cutoff applied to *probs*.

    Returns:
        Flat dict with keys: ``all_iou``, ``all_precision``, ``all_recall``,
        ``all_f1``, ``positive_only_iou``, ``positive_only_precision``,
        ``positive_only_recall``, ``positive_only_f1``, ``micro_iou``,
        ``micro_precision``, ``micro_recall``, ``micro_f1``,
        ``empty_count``, ``empty_false_positive_rate``,
        ``empty_predicted_positive_rate_mean``,
        ``target_positive_sample_count``, ``total_sample_count``.
    """
    n_samples = len(probs)
    preds = probs >= threshold                              # (N, H, W) bool
    tgts  = targets.astype(bool)                           # (N, H, W) bool

    # ── Identify positive and empty samples ───────────────────────────────────
    pos_mask_idx   = [i for i in range(n_samples) if tgts[i].any()]
    empty_mask_idx = [i for i in range(n_samples) if not tgts[i].any()]

    # ── Per-sample IoU (used for macro averages) ───────────────────────────────
    def _sample_iou(i: int) -> float:
        inter = int((preds[i] & tgts[i]).sum())
        union = int((preds[i] | tgts[i]).sum())
        return (inter + SMOOTH) / (union + SMOOTH)

    per_iou = [_sample_iou(i) for i in range(n_samples)]

    # ── All-samples (naive macro mean + pixel-level P/R/F1) ───────────────────
    all_iou = float(np.mean(per_iou))
    p_flat  = preds.ravel()
    t_flat  = tgts.ravel()
    _tp = int((p_flat & t_flat).sum())
    _fp = int((p_flat & ~t_flat).sum())
    _fn = int((~p_flat & t_flat).sum())
    all_prec = (_tp + SMOOTH) / (_tp + _fp + SMOOTH)
    all_rec  = (_tp + SMOOTH) / (_tp + _fn + SMOOTH)
    all_f1   = 2 * all_prec * all_rec / (all_prec + all_rec + SMOOTH)

    # ── Positive-only (exclude empty-mask samples) ────────────────────────────
    if pos_mask_idx:
        pos_iou  = float(np.mean([per_iou[i] for i in pos_mask_idx]))
        pp = preds[pos_mask_idx].ravel()
        pt = tgts[pos_mask_idx].ravel()
        _ptp = int((pp & pt).sum())
        _pfp = int((pp & ~pt).sum())
        _pfn = int((~pp & pt).sum())
        pos_prec = (_ptp + SMOOTH) / (_ptp + _pfp + SMOOTH)
        pos_rec  = (_ptp + SMOOTH) / (_ptp + _pfn + SMOOTH)
        pos_f1   = 2 * pos_prec * pos_rec / (pos_prec + pos_rec + SMOOTH)
    else:
        pos_iou = pos_prec = pos_rec = pos_f1 = float("nan")

    # ── Empty-mask stats ───────────────────────────────────────────────────────
    empty_count = len(empty_mask_idx)
    if empty_count:
        ep = preds[empty_mask_idx]                           # (E, H, W)
        total_px = ep.shape[1] * ep.shape[2]
        rates = ep.sum(axis=(1, 2)) / total_px               # (E,)
        empty_fpr  = float((rates > 0).mean())
        empty_rate = float(rates.mean())
    else:
        empty_fpr  = 0.0
        empty_rate = 0.0

    # ── Micro pixel aggregation (all N*H*W pixels) ────────────────────────────
    micro_iou  = (_tp + SMOOTH) / (_tp + _fp + _fn + SMOOTH)
    micro_prec = (_tp + SMOOTH) / (_tp + _fp + SMOOTH)
    micro_rec  = (_tp + SMOOTH) / (_tp + _fn + SMOOTH)
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + SMOOTH)

    return {
        "all_iou":            all_iou,
        "all_precision":      float(all_prec),
        "all_recall":         float(all_rec),
        "all_f1":             float(all_f1),
        "positive_only_iou":      pos_iou,
        "positive_only_precision": float(pos_prec),
        "positive_only_recall":    float(pos_rec),
        "positive_only_f1":        float(pos_f1),
        "micro_iou":       float(micro_iou),
        "micro_precision": float(micro_prec),
        "micro_recall":    float(micro_rec),
        "micro_f1":        float(micro_f1),
        "empty_count":                         empty_count,
        "empty_false_positive_rate":           empty_fpr,
        "empty_predicted_positive_rate_mean":  empty_rate,
        "target_positive_sample_count":        len(pos_mask_idx),
        "total_sample_count":                  n_samples,
    }


# ── Tensor-level helpers (used during inference) ──────────────────────────────
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and collect all predictions and targets.

    Args:
        model: Trained segmentation model (eval mode is set internally).
        loader: DataLoader yielding ``(images, masks)`` batches.
        device: Computation device.

    Returns:
        ``(probs, targets)`` where *probs* is sigmoid-activated and
        both arrays are shaped ``(N, H, W)`` in float32.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            tgts = masks.squeeze(1).cpu().numpy() if masks.dim() == 4 else masks.cpu().numpy()
            all_probs.append(probs)
            all_targets.append(tgts)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)
