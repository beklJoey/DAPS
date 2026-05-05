"""Segmentation loss functions: BCE, Dice, and combined BCE+Dice."""

import torch
import torch.nn as nn

# ── Constants ────────────────────────────────────────────────────────────────
ALPHA: float = 0.5        # Dice weight in combined loss  (1-ALPHA → BCE)
DICE_SMOOTH: float = 1.0  # Laplace smoothing to avoid division by zero


# ── Individual losses ─────────────────────────────────────────────────────────
class BCELoss(nn.Module):
    """Binary cross-entropy with logits.

    Accepts raw logits and applies sigmoid internally via
    :class:`torch.nn.BCEWithLogitsLoss`.

    Args:
        pos_weight: Scalar to up-weight the positive class.  ``None`` → equal weight.
    """

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Float binary mask ``(B, 1, H, W)`` or ``(B, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self._bce(logits, targets.float())


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Applies sigmoid to logits internally; operates on continuous probabilities.

    Args:
        smooth: Laplace smoothing constant added to numerator and denominator.
    """

    def __init__(self, smooth: float = DICE_SMOOTH) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Float binary mask ``(B, 1, H, W)`` or ``(B, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of Dice and BCE losses.

    ``L = α · L_Dice + (1 − α) · L_BCE``

    Args:
        alpha: Weight for the Dice component (0 → pure BCE, 1 → pure Dice).
        pos_weight: Positive-class weight forwarded to :class:`BCELoss`.
        smooth: Laplace smoothing forwarded to :class:`DiceLoss`.
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        pos_weight: float | None = None,
        smooth: float = DICE_SMOOTH,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        self.bce = BCELoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Raw model output ``(B, 1, H, W)``.
            targets: Float binary mask.

        Returns:
            Scalar combined loss tensor.
        """
        return (
            self.alpha * self.dice(logits, targets)
            + (1.0 - self.alpha) * self.bce(logits, targets)
        )


# ── Factory ───────────────────────────────────────────────────────────────────
def build_loss(name: str) -> nn.Module:
    """Instantiate a loss function by name.

    Args:
        name: One of ``'bce'``, ``'dice'``, or ``'bce_dice'``.

    Returns:
        Instantiated loss :class:`~torch.nn.Module`.

    Raises:
        ValueError: For unrecognised loss names.
    """
    registry = {
        "bce": BCELoss,
        "dice": DiceLoss,
        "bce_dice": CombinedLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Choose from {sorted(registry)}.")
    return registry[name]()
