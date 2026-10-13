"""Reusable building blocks for the retained TransMorph registration models."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def conv_norm_activation(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> list[nn.Module]:
    """Return flat layers, preserving ``nn.Sequential`` checkpoint indices.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Edge length of the convolution kernel.
        stride: Stride of the convolution.
        padding: Zero padding of the convolution.

    Returns:
        Convolution, instance normalization, and leaky ReLU, to be unpacked into
        an ``nn.Sequential``.
    """
    return [
        nn.Conv3d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.InstanceNorm3d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]


def pairwise_ct_pet_features(x: torch.Tensor) -> torch.Tensor:
    """Build the ten CT/PET pair channels in their historical order."""
    fixed_ct, fixed_pet, moving_ct, moving_pet = x.chunk(4, dim=1)
    return torch.cat(
        (
            fixed_ct,
            moving_ct,
            fixed_ct - moving_ct,
            (fixed_ct - moving_ct).abs(),
            fixed_ct * moving_ct,
            fixed_pet,
            moving_pet,
            fixed_pet - moving_pet,
            (fixed_pet - moving_pet).abs(),
            fixed_pet * moving_pet,
        ),
        dim=1,
    )


class ConvBlock3D(nn.Module):
    """Small 3D convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            *conv_norm_activation(in_channels, out_channels),
            *conv_norm_activation(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolution-normalization-activation groups.

        Args:
            x: Input features of shape ``[B, C, D, H, W]``.

        Returns:
            Features with the configured output channel count.
        """
        return self.block(x)


class TransformerBlock(nn.Module):
    """Global self-attention block used only at low resolution."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and an MLP with residual connections.

        Args:
            x: Feature tokens of shape ``[B, N, C]``.

        Returns:
            Updated tokens with the same shape.
        """
        attended, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attended
        return x + self.mlp(self.norm2(x))


class LowResolutionTransMorph(nn.Module):
    """Predict a coarse stationary velocity field from low-resolution CT/PET."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        max_velocity: float = 8.0,
    ) -> None:
        super().__init__()
        self.max_velocity = max_velocity

        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=4,
            stride=4,
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)])

        self.decoder = nn.Sequential(
            ConvBlock3D(embed_dim, 64),
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor,
                output_shape: tuple[int, int, int]) -> torch.Tensor:
        """Predict a bounded coarse velocity at the requested image size.

        Args:
            x: Concatenated CT/PET pair of shape ``[B, 4, D, H, W]``.
            output_shape: Requested ``(D, H, W)`` output size.

        Returns:
            Velocity field of shape ``[B, 3, D, H, W]`` in voxel units.
        """
        x_low = F.interpolate(
            x,
            scale_factor=0.25,
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        features = self.patch_embed(x_low)
        batch_size, channels, depth, height, width = features.shape

        tokens = features.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)

        features = tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        )

        velocity_low = self.decoder(features)
        velocity = F.interpolate(
            velocity_low,
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )

        return self.max_velocity * torch.tanh(velocity)


def _window_partition_3d(
    x: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    """Partition a ``[B, C, D, H, W]`` tensor into cubic windows."""
    batch_size, channels, depth, height, width = x.shape
    pad_d = (window_size - depth % window_size) % window_size
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    padded_depth, padded_height, padded_width = x.shape[-3:]
    x = x.view(
        batch_size,
        channels,
        padded_depth // window_size,
        window_size,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
    )
    windows = (x.permute(0, 2, 4, 6, 3, 5, 7,
                         1).contiguous().view(-1, window_size**3, channels))
    return windows, (padded_depth, padded_height, padded_width), (pad_d, pad_h,
                                                                  pad_w)


def _window_unpartition_3d(
    windows: torch.Tensor,
    *,
    batch_size: int,
    channels: int,
    padded_shape: tuple[int, int, int],
    padding: tuple[int, int, int],
    window_size: int,
) -> torch.Tensor:
    """Reverse ``_window_partition_3d``."""
    padded_depth, padded_height, padded_width = padded_shape
    pad_d, pad_h, pad_w = padding
    x = windows.view(
        batch_size,
        padded_depth // window_size,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        window_size,
        channels,
    )
    x = (x.permute(0, 7, 1, 4, 2, 5, 3,
                   6).contiguous().view(batch_size, channels, padded_depth,
                                        padded_height, padded_width))
    if pad_d or pad_h or pad_w:
        x = x[:, :, :padded_depth - pad_d, :padded_height -
              pad_h, :padded_width - pad_w]
    return x


class WindowTransformerBlock3D(nn.Module):
    """Small Swin-style 3D window-attention block.

    This is intentionally lighter than the full TransMorph Swin encoder, but it
    keeps the useful inductive bias: local window attention, shifted windows for
    cross-window communication, and channel-wise MLP refinement.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if shift_size >= window_size:
            raise ValueError("shift_size must be smaller than window_size.")
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local shifted-window attention and residual refinement.

        Args:
            x: Feature map of shape ``[B, C, D, H, W]``.

        Returns:
            Updated feature map with the same shape.
        """
        batch_size, channels, depth, height, width = x.shape
        residual = x
        shifted = (torch.roll(
            x,
            shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
            dims=(2, 3, 4),
        ) if self.shift_size > 0 else x)
        windows, padded_shape, padding = _window_partition_3d(
            shifted,
            window_size=self.window_size,
        )
        attended, _ = self.attn(
            self.norm1(windows),
            self.norm1(windows),
            self.norm1(windows),
            need_weights=False,
        )
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        x = _window_unpartition_3d(
            windows,
            batch_size=batch_size,
            channels=channels,
            padded_shape=padded_shape,
            padding=padding,
            window_size=self.window_size,
        )
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(2, 3, 4),
            )
        if x.shape[-3:] != (depth, height, width):
            x = x[:, :, :depth, :height, :width]
        return residual + x


class LocalResidualRefiner(nn.Module):
    """Predict a small full-resolution stationary residual velocity."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 24,
        max_residual_disp: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_residual_disp = max_residual_disp

        self.net = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.Conv3d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a bounded local residual velocity field.

        Args:
            x: Concatenated fixed and warped moving images.

        Returns:
            Residual velocity of shape ``[B, 3, D, H, W]``.
        """
        residual_velocity = self.net(x)
        return self.max_residual_disp * torch.tanh(residual_velocity)


class PairwiseFusionStem3D(nn.Module):
    """Build fixed/moving CT/PET pair features for a registration encoder."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *conv_norm_activation(10, out_channels),
            ConvBlock3D(out_channels, out_channels),
        )

    @staticmethod
    def pairwise_features(x: torch.Tensor) -> torch.Tensor:
        """Create the ten historically ordered CT/PET pair channels.

        Args:
            x: CT/PET pair of shape ``[B, 4, D, H, W]``.

        Returns:
            Pair features of shape ``[B, 10, D, H, W]``.
        """
        return pairwise_ct_pet_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse CT/PET pair features into encoder input channels.

        Args:
            x: Concatenated CT/PET pair of shape ``[B, 4, D, H, W]``.

        Returns:
            Fused feature map at the input spatial resolution.
        """
        return self.net(self.pairwise_features(x))


class TransMorphV2CoarseBackbone(nn.Module):
    """Closer TransMorph-style hierarchical Swin/UNet coarse SVF predictor."""

    def __init__(
        self,
        base_channels: int = 16,
        embed_dim: int = 96,
        num_heads: int = 4,
        window_size: int = 4,
        max_velocity: float = 8.0,
    ) -> None:
        super().__init__()
        self.max_velocity = max_velocity
        self.stem = PairwiseFusionStem3D(base_channels)
        self.down1 = nn.Sequential(
            *conv_norm_activation(base_channels, base_channels * 2, stride=2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
        )
        self.swin1 = nn.Sequential(
            WindowTransformerBlock3D(base_channels * 2, max(1, num_heads // 2),
                                     window_size, 0),
            WindowTransformerBlock3D(base_channels * 2, max(1, num_heads // 2),
                                     window_size, window_size // 2),
        )
        self.down2 = nn.Sequential(
            *conv_norm_activation(base_channels * 2,
                                  base_channels * 4,
                                  stride=2),
            ConvBlock3D(base_channels * 4, base_channels * 4),
        )
        self.swin2 = nn.Sequential(
            WindowTransformerBlock3D(base_channels * 4, num_heads, window_size,
                                     0),
            WindowTransformerBlock3D(base_channels * 4, num_heads, window_size,
                                     window_size // 2),
        )
        self.down3 = nn.Sequential(
            *conv_norm_activation(base_channels * 4, embed_dim, stride=2),
            ConvBlock3D(embed_dim, embed_dim),
        )
        self.swin3 = nn.Sequential(
            WindowTransformerBlock3D(embed_dim, num_heads, window_size, 0),
            WindowTransformerBlock3D(embed_dim, num_heads, window_size,
                                     window_size // 2),
        )
        self.up2 = nn.Sequential(
            *conv_norm_activation(embed_dim + base_channels * 4,
                                  base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
        )
        self.up1 = nn.Sequential(
            *conv_norm_activation(base_channels * 6, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
        )
        self.up0 = nn.Sequential(
            *conv_norm_activation(base_channels * 3, base_channels),
            ConvBlock3D(base_channels, base_channels),
        )
        self.head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor,
                output_shape: tuple[int, int, int]) -> torch.Tensor:
        """Predict a bounded multiscale coarse velocity field.

        Args:
            x: Concatenated CT/PET pair of shape ``[B, 4, D, H, W]``.
            output_shape: Requested ``(D, H, W)`` output size.

        Returns:
            Coarse velocity of shape ``[B, 3, D, H, W]``.
        """
        x_low = F.interpolate(
            x,
            scale_factor=0.5,
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        skip0 = self.stem(x_low)
        skip1 = self.swin1(self.down1(skip0))
        skip2 = self.swin2(self.down2(skip1))
        bottleneck = self.swin3(self.down3(skip2))
        up2 = F.interpolate(bottleneck,
                            size=skip2.shape[-3:],
                            mode="trilinear",
                            align_corners=False)
        up2 = self.up2(torch.cat((up2, skip2), dim=1))
        up1 = F.interpolate(up2,
                            size=skip1.shape[-3:],
                            mode="trilinear",
                            align_corners=False)
        up1 = self.up1(torch.cat((up1, skip1), dim=1))
        up0 = F.interpolate(up1,
                            size=skip0.shape[-3:],
                            mode="trilinear",
                            align_corners=False)
        up0 = self.up0(torch.cat((up0, skip0), dim=1))
        velocity_low = self.head(up0)
        velocity = F.interpolate(
            velocity_low,
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )
        return self.max_velocity * torch.tanh(velocity)


class MultiScaleResidualRefiner(nn.Module):
    """Deeper local refiner with a small U-Net and skip connections."""

    def __init__(
        self,
        base_channels: int = 16,
        max_residual_disp: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_residual_disp = max_residual_disp
        self.stem = PairwiseFusionStem3D(base_channels)
        self.down1 = nn.Sequential(
            *conv_norm_activation(base_channels, base_channels * 2, stride=2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
        )
        self.down2 = nn.Sequential(
            *conv_norm_activation(base_channels * 2,
                                  base_channels * 4,
                                  stride=2),
            ConvBlock3D(base_channels * 4, base_channels * 4),
        )
        self.up1 = nn.Sequential(
            *conv_norm_activation(base_channels * 6, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
        )
        self.up0 = nn.Sequential(
            *conv_norm_activation(base_channels * 3, base_channels),
            ConvBlock3D(base_channels, base_channels),
        )
        self.head = nn.Conv3d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a bounded U-Net residual velocity field.

        Args:
            x: Concatenated fixed and warped moving images.

        Returns:
            Residual velocity of shape ``[B, 3, D, H, W]``.
        """
        skip0 = self.stem(x)
        skip1 = self.down1(skip0)
        bottleneck = self.down2(skip1)
        up1 = F.interpolate(bottleneck,
                            size=skip1.shape[-3:],
                            mode="trilinear",
                            align_corners=False)
        up1 = self.up1(torch.cat((up1, skip1), dim=1))
        up0 = F.interpolate(up1,
                            size=skip0.shape[-3:],
                            mode="trilinear",
                            align_corners=False)
        up0 = self.up0(torch.cat((up0, skip0), dim=1))
        residual_velocity = self.head(up0)
        return self.max_residual_disp * torch.tanh(residual_velocity)
