"""Simple 3D registration network."""

from __future__ import annotations

import torch
from torch import nn


class SmallRegistrationUNet(nn.Module):
    """Small 3D CNN predicting a dense displacement field."""

    def __init__(self, in_channels: int = 4, base_channels: int = 16) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """Predict moving-to-fixed displacement."""
        network_input = torch.cat([fixed, moving], dim=1)
        return self.decoder(self.encoder(network_input))
