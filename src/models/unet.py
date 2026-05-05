"""Single-frame U-Net for binary segmentation."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture constants ────────────────────────────────────────────────────
BASE_CHANNELS: int = 32
DROPOUT_P: float = 0.1
KERNEL_SIZE: int = 3
PAD: int = 1


# ── Building blocks ───────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    """Two successive Conv3×3 → BN → ReLU layers (standard U-Net block).

    Args:
        in_ch: Number of input feature maps.
        out_ch: Number of output feature maps.
        dropout_p: Spatial dropout probability applied after the second ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, KERNEL_SIZE, padding=PAD, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, KERNEL_SIZE, padding=PAD, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two Conv→BN→ReLU layers and optional dropout.

        Args:
            x: Input feature map ``(B, in_ch, H, W)``.

        Returns:
            Output feature map ``(B, out_ch, H, W)``.
        """
        return self.block(x)


class _EncoderBlock(nn.Module):
    """Encoder stage: ConvBlock → MaxPool2×2.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count (after conv).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pooled_features, skip_connection)``."""
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    """Decoder stage: bilinear upsample → skip concat → ConvBlock.

    Args:
        in_ch: Channels coming from the layer below (after upsampling).
        skip_ch: Channels from the matching encoder skip connection.
        out_ch: Output channel count.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample *x*, align to *skip*, concatenate, then apply conv block."""
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ── U-Net ─────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """Standard four-level U-Net for single-frame binary segmentation.

    Input shape:  ``(B, in_channels, H, W)``
    Output shape: ``(B, 1, H, W)`` — raw logits (apply sigmoid externally).

    Args:
        in_channels: Number of input channels (e.g. 3 for three satellite bands).
        base_channels: Feature-map count at the shallowest encoder level.
    """

    def __init__(
        self, in_channels: int = 3, base_channels: int = BASE_CHANNELS
    ) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = _EncoderBlock(in_channels, c)
        self.enc2 = _EncoderBlock(c, c * 2)
        self.enc3 = _EncoderBlock(c * 2, c * 4)
        self.enc4 = _EncoderBlock(c * 4, c * 8)

        self.bottleneck = _ConvBlock(c * 8, c * 16, dropout_p=DROPOUT_P)

        self.dec4 = _DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = _DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = _DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = _DecoderBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass.

        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            Logit tensor ``(B, 1, H, W)``.
        """
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)
