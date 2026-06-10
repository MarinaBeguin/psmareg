"""3D Attention U-Net for deformable medical image registration."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two-layer 3D convolution block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        num_groups = min(8, out_channels)

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block."""
        return self.block(x)


class AttentionGate3D(nn.Module):
    """Attention gate for 3D U-Net skip connections."""

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        intermediate_channels: int,
    ) -> None:
        super().__init__()

        self.skip_projection = nn.Sequential(
            nn.Conv3d(
                skip_channels,
                intermediate_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, intermediate_channels),
                num_channels=intermediate_channels,
            ),
        )

        self.gating_projection = nn.Sequential(
            nn.Conv3d(
                gating_channels,
                intermediate_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, intermediate_channels),
                num_channels=intermediate_channels,
            ),
        )

        self.attention = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """Apply attention to the skip feature map."""
        if gating.shape[2:] != skip.shape[2:]:
            gating = F.interpolate(
                gating,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        skip_proj = self.skip_projection(skip)
        gating_proj = self.gating_projection(gating)
        attention_map = self.attention(skip_proj + gating_proj)

        return skip * attention_map


class UpBlock3D(nn.Module):
    """Upsampling block with attention-gated skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.attention_gate = AttentionGate3D(
            skip_channels=skip_channels,
            gating_channels=in_channels,
            intermediate_channels=max(out_channels // 2, 8),
        )

        self.conv = ConvBlock3D(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features and fuse them with an attended skip."""
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        attended_skip = self.attention_gate(skip=skip, gating=x)
        x = torch.cat([x, attended_skip], dim=1)

        return self.conv(x)


class AttentionUNet3DRegistration(nn.Module):
    """3D Attention U-Net predicting a dense displacement field."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Sequence[int] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()

        if len(features) < 2:
            raise ValueError("features must contain at least two levels.")

        self.encoder_0 = ConvBlock3D(in_channels, features[0])
        self.encoder_1 = ConvBlock3D(features[0], features[1])
        self.encoder_2 = ConvBlock3D(features[1], features[2])
        self.encoder_3 = ConvBlock3D(features[2], features[3])

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(features[3], features[3] * 2)

        self.decoder_3 = UpBlock3D(
            in_channels=features[3] * 2,
            skip_channels=features[3],
            out_channels=features[3],
        )
        self.decoder_2 = UpBlock3D(
            in_channels=features[3],
            skip_channels=features[2],
            out_channels=features[2],
        )
        self.decoder_1 = UpBlock3D(
            in_channels=features[2],
            skip_channels=features[1],
            out_channels=features[1],
        )
        self.decoder_0 = UpBlock3D(
            in_channels=features[1],
            skip_channels=features[0],
            out_channels=features[0],
        )

        self.output_layer = nn.Conv3d(
            features[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self._initialize_output_layer()

    def _initialize_output_layer(self) -> None:
        """Initialize displacement prediction close to zero."""
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-5)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """Predict displacement from fixed and moving images."""
        x = torch.cat([fixed, moving], dim=1)

        enc_0 = self.encoder_0(x)
        enc_1 = self.encoder_1(self.pool(enc_0))
        enc_2 = self.encoder_2(self.pool(enc_1))
        enc_3 = self.encoder_3(self.pool(enc_2))

        bottleneck = self.bottleneck(self.pool(enc_3))

        dec_3 = self.decoder_3(bottleneck, enc_3)
        dec_2 = self.decoder_2(dec_3, enc_2)
        dec_1 = self.decoder_1(dec_2, enc_1)
        dec_0 = self.decoder_0(dec_1, enc_0)

        displacement = self.output_layer(dec_0)

        return displacement
