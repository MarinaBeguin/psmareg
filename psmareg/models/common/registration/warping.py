"""Shared 3D warping and displacement-field operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_identity_grid(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a normalized identity grid for 3D grid sampling.

    Args:
        batch_size: Number of grids.
        depth: Number of voxels along the D axis.
        height: Number of voxels along the H axis.
        width: Number of voxels along the W axis.
        device: Device of the grid.

    Returns:
        Grid of shape ``[B, D, H, W, 3]`` with coordinates in ``[-1, 1]``.
    """
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
    """Convert a voxel displacement field to normalized grid coordinates.

    Raises:
        ValueError: If the field does not have three channels.
    """
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
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp a 3D image using a dense voxel-space displacement field.

    Args:
        image: Image of shape ``[B, C, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]`` in x/y/z
            voxels.
        interpolation_mode: ``bilinear`` or ``nearest``.
        padding_mode: ``border`` or ``zeros`` for samples outside the image.

    Returns:
        Image sampled at each voxel plus its displacement.
    """
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

    return F.grid_sample(
        image,
        sampling_grid,
        mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def valid_sampling_mask(displacement: torch.Tensor) -> torch.Tensor:
    """Return the fraction of valid source samples for a pull-back field."""
    ones = torch.ones(
        (displacement.shape[0], 1, *displacement.shape[-3:]),
        device=displacement.device,
        dtype=displacement.dtype,
    )
    return warp_image_3d(
        ones,
        displacement,
        interpolation_mode="bilinear",
        padding_mode="zeros",
    ).clamp_(0.0, 1.0)


def compose_displacements(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Compose pull-back fields applied first and then second.

    With ``warp(image, field)(x) = image(x + field(x))``, this returns the
    field equivalent to ``warp(warp(image, first), second)``.

    Raises:
        ValueError: If the two fields differ in shape.
    """
    if first.shape != second.shape:
        raise ValueError("Displacement fields must have the same shape, "
                         f"got {tuple(first.shape)} and {tuple(second.shape)}.")
    return second + warp_image_3d(first, second)


def integrate_stationary_velocity(
    velocity: torch.Tensor,
    scaling_steps: int = 7,
) -> torch.Tensor:
    """Integrate a stationary velocity field using scaling-and-squaring."""
    displacement = velocity / float(2**scaling_steps)

    for _ in range(scaling_steps):
        displacement = displacement + warp_image_3d(displacement, displacement)

    return displacement
