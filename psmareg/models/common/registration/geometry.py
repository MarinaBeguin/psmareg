"""Voxel-coordinate and Euler-rotation primitives shared across pipelines."""

from __future__ import annotations

import torch


def voxel_grid_xyz(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create an ``[x, y, z]`` voxel grid of shape ``[3, D, H, W]``.

    Args:
        shape: Spatial dimensions in depth, height, width order.
        device: Device holding the grid.
        dtype: Coordinate precision.

    Returns:
        Tensor in displacement-channel order, not array-axis order.
    """
    depth, height, width = shape
    z = torch.arange(depth, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=0)


def euler_rotation_matrix_xyz(angles: torch.Tensor) -> torch.Tensor:
    """Return ``Rz @ Ry @ Rx`` for radians in ``[..., 3]`` xyz order.

    Args:
        angles: One or more Euler-angle triplets.

    Returns:
        Rotation matrices of shape ``[..., 3, 3]``.
    """
    rx, ry, rz = angles.unbind(dim=-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    rotation_x = torch.stack((
        torch.stack((one, zero, zero), dim=-1),
        torch.stack((zero, cx, -sx), dim=-1),
        torch.stack((zero, sx, cx), dim=-1),
    ),
                             dim=-2)
    rotation_y = torch.stack((
        torch.stack((cy, zero, sy), dim=-1),
        torch.stack((zero, one, zero), dim=-1),
        torch.stack((-sy, zero, cy), dim=-1),
    ),
                             dim=-2)
    rotation_z = torch.stack((
        torch.stack((cz, -sz, zero), dim=-1),
        torch.stack((sz, cz, zero), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    ),
                             dim=-2)
    return rotation_z @ rotation_y @ rotation_x
