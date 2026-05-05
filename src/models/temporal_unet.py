"""Temporal U-Net: multi-frame stacked input variant for contrail segmentation."""

import torch
import torch.nn as nn

from .unet import UNet, BASE_CHANNELS

# ── Constants ────────────────────────────────────────────────────────────────
BANDS_PER_FRAME: int = 5   # band_11, band_13, band_14, band_15, band_16
T_VALUES = (1, 3, 5)       # supported frame counts for ablation study


class TemporalUNet(nn.Module):
    """U-Net variant that accepts *T* temporally stacked satellite frames.

    All *T* frames are concatenated along the channel dimension before
    being fed into an otherwise standard U-Net encoder-decoder.  No
    temporal convolutions are used; the backbone is identical to
    :class:`~src.models.unet.UNet`.

    Input shape:  ``(B, T * bands_per_frame, H, W)``
    Output shape: ``(B, 1, H, W)`` — raw logits.

    Args:
        n_frames: Number of temporal frames to stack (e.g. 1, 3, 5).
        bands_per_frame: Spectral bands per frame (default 3).
        base_channels: Base feature-map count of the U-Net backbone.
    """

    def __init__(
        self,
        n_frames: int = 3,
        bands_per_frame: int = BANDS_PER_FRAME,
        base_channels: int = BASE_CHANNELS,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.bands_per_frame = bands_per_frame
        in_channels = n_frames * bands_per_frame
        self.unet = UNet(in_channels=in_channels, base_channels=base_channels)

    @property
    def in_channels(self) -> int:
        """Total input channel count (T × bands_per_frame)."""
        return self.n_frames * self.bands_per_frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Stacked frame tensor ``(B, T * bands_per_frame, H, W)``.

        Returns:
            Logit tensor ``(B, 1, H, W)``.
        """
        return self.unet(x)
