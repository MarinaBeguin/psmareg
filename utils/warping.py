"""3D image warping utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def make_identity_grid(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a normalized identity grid for 3D grid sampling."""
    z_coordinates = torch.linspace(-1.0, 1.0, depth, device=device)
    y_coordinates = torch.linspace(-1.0, 1.0, height, device=device)
    x_coordinates = torch.linspace(-1.0, 1.0, width, device=device)

    zz_grid, yy_grid, xx_grid = torch.meshgrid(
        z_coordinates,
        y_coordinates,
        x_coordinates,
        indexing="ij",
    )

    grid = torch.stack((xx_grid, yy_grid, zz_grid), dim=-1)
    return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)


def normalize_displacement(displacement: torch.Tensor) -> torch.Tensor:
    """Convert a voxel-space displacement field to normalized grid coordinates."""
    _, channels, depth, height, width = displacement.shape

    if channels != 3:
        raise ValueError(f"Expected 3 displacement channels, got {channels}")

    normalized_x = 2.0 * displacement[:, 0] / max(width - 1, 1)
    normalized_y = 2.0 * displacement[:, 1] / max(height - 1, 1)
    normalized_z = 2.0 * displacement[:, 2] / max(depth - 1, 1)

    return torch.stack((normalized_x, normalized_y, normalized_z), dim=-1)


def warp_image_3d(
    image: torch.Tensor,
    displacement: torch.Tensor,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """Warp a 3D image using a dense voxel-space displacement field."""
    batch_size, _, depth, height, width = image.shape

    identity_grid = make_identity_grid(
        batch_size=batch_size,
        depth=depth,
        height=height,
        width=width,
        device=image.device,
    )
    normalized_displacement = normalize_displacement(displacement)

    sampling_grid = identity_grid + normalized_displacement

    return functional.grid_sample(
        image,
        sampling_grid,
        mode=interpolation_mode,
        padding_mode="border",
        align_corners=True,
    )

def downsample_displacement_by_2(displacement: torch.Tensor) -> torch.Tensor:
    """Downsample a full-resolution displacement field by a factor of 2."""
    return functional.interpolate(
        displacement,
        scale_factor=0.5,
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )


def upsample_displacement_by_2(displacement: torch.Tensor) -> torch.Tensor:
    """Upsample a low-resolution displacement field by a factor of 2."""
    return functional.interpolate(
        displacement,
        scale_factor=2.0,
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
