"""Online spatial and intensity augmentations for real longitudinal pairs."""

from __future__ import annotations

import math

import torch

from psmareg.models.common.registration.geometry import (
    euler_rotation_matrix_xyz,
    voxel_grid_xyz,
)


def random_affine_augmentation_field(image: torch.Tensor,
                                     max_translation: float,
                                     max_rotation_degrees: float,
                                     max_scaling: float) -> torch.Tensor:
    """Sample a mild pull-back affine field in voxel coordinates.

    Args:
        image: Batched 3D image with shape ``[B, C, D, H, W]``.
        max_translation: Maximum absolute translation per axis, in voxels.
        max_rotation_degrees: Maximum absolute rotation per axis, in degrees.
        max_scaling: Maximum relative scale change around one.

    Returns:
        Dense displacement field with shape ``[B, 3, D, H, W]``.
    """
    batch_size, _, depth, height, width = image.shape
    device, dtype = (image.device, image.dtype)
    coordinates = voxel_grid_xyz((depth, height, width),
                                 device=device,
                                 dtype=dtype)
    center = torch.tensor([(width - 1) / 2, (height - 1) / 2, (depth - 1) / 2],
                          device=device,
                          dtype=dtype).view(3, 1, 1, 1)
    centered = coordinates - center
    fields: list[torch.Tensor] = []
    for _ in range(batch_size):
        angles = torch.empty(3, device=device, dtype=dtype).uniform_(
            -math.radians(max_rotation_degrees),
            math.radians(max_rotation_degrees))
        scale = torch.empty(3, device=device,
                            dtype=dtype).uniform_(1.0 - max_scaling,
                                                  1.0 + max_scaling)
        translation = torch.empty(3, device=device,
                                  dtype=dtype).uniform_(-max_translation,
                                                        max_translation)
        affine = euler_rotation_matrix_xyz(angles) @ torch.diag(scale)
        transformed = (affine @ centered.view(3, -1)).view_as(centered)
        fields.append(transformed - centered + translation.view(3, 1, 1, 1))
    return torch.stack(fields)


def independent_intensity_augmentation(image: torch.Tensor, *,
                                       is_pet: bool) -> torch.Tensor:
    """Apply mild independent intensity changes to normalized CT or PET.

    Args:
        image: Normalized image tensor with values expected in ``[0, 1]``.
        is_pet: Whether to use PET-specific scale and gamma perturbations.

    Returns:
        Augmented tensor with the original shape and dtype. Zero background is
        preserved and output intensities are clipped to ``[0, 1]``.
    """
    shape = (image.shape[0],) + (1,) * (image.ndim - 1)
    augmented = image.float().clamp(0.0, 1.0)
    foreground = (augmented > 0.0).to(augmented.dtype)
    if is_pet:
        scale = torch.exp(torch.randn(shape, device=image.device) * 0.12)
        gamma = torch.empty(shape, device=image.device).uniform_(0.9, 1.1)
        bias = torch.empty(shape, device=image.device).uniform_(-0.015, 0.015)
        augmented = augmented.mul(scale).clamp(0.0, 1.0).pow(gamma).add(
            bias * foreground)
    else:
        contrast = torch.empty(shape, device=image.device).uniform_(0.92, 1.08)
        bias = torch.empty(shape, device=image.device).uniform_(-0.025, 0.025)
        augmented = augmented.mul(contrast).add(bias * foreground)
        augmented = augmented.add(
            torch.randn_like(augmented) * 0.005 * foreground)
    return (augmented.clamp(0.0, 1.0) * foreground).to(image.dtype)
