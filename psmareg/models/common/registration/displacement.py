"""Voxel-displacement resizing with an explicit sampling convention."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def resize_voxel_displacement(
    displacement: torch.Tensor,
    shape: Sequence[int],
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Interpolate a displacement and convert its voxel units to ``shape``.

    Args:
        displacement: Field in xyz channel order, ``[B, 3, D, H, W]``.
        shape: Target spatial shape in depth, height, width order.
        align_corners: Required interpolation and voxel-scaling convention.

    Returns:
        Resized displacement in target-grid voxel units.
    """
    source_shape = displacement.shape[-3:]
    resized = F.interpolate(displacement,
                            size=shape,
                            mode="trilinear",
                            align_corners=align_corners)
    if align_corners:
        scale = displacement.new_tensor([
            (shape[2] - 1) / max(source_shape[2] - 1, 1),
            (shape[1] - 1) / max(source_shape[1] - 1, 1),
            (shape[0] - 1) / max(source_shape[0] - 1, 1),
        ]).view(1, 3, 1, 1, 1)
        return resized * scale
    resized = resized.clone()
    resized[:, 0] *= shape[-1] / source_shape[-1]
    resized[:, 1] *= shape[-2] / source_shape[-2]
    resized[:, 2] *= shape[-3] / source_shape[-3]
    return resized


def resize_displacement(
    displacement: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """Resize using the Stage A/B/C ``align_corners=True`` convention."""
    return resize_voxel_displacement(displacement, shape, align_corners=True)
