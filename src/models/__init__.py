"""Model definitions: single-frame U-Net and temporal multi-frame variant."""

from typing import Union

import torch.nn as nn

from .temporal_unet import BANDS_PER_FRAME, TemporalUNet
from .unet import UNet

__all__ = ["UNet", "TemporalUNet", "build_model", "BANDS_PER_FRAME"]


def build_model(
    model_name: str,
    n_frames: int = 1,
    base_channels: Union[int, None] = None,
) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: ``'unet'`` for single-frame U-Net or
            ``'temporal_unet'`` for the multi-frame variant.
        n_frames: Number of temporal frames (only used for ``'temporal_unet'``).
        base_channels: Override the default BASE_CHANNELS.  Pass the value
            inferred from a saved checkpoint to load weights correctly when
            the global constant has since changed.

    Returns:
        Uninitialised (random weights) model instance.

    Raises:
        ValueError: For unrecognised model names.
    """
    from .unet import BASE_CHANNELS as _DEFAULT_CH
    bc = base_channels if base_channels is not None else _DEFAULT_CH
    if model_name == "unet":
        return UNet(in_channels=BANDS_PER_FRAME, base_channels=bc)
    if model_name == "temporal_unet":
        return TemporalUNet(n_frames=n_frames, base_channels=bc)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from ['unet', 'temporal_unet']."
    )
