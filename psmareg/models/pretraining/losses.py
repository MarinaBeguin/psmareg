"""Losses and metrics used by synthetic registration pretraining."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from psmareg.models.common.losses import (
    label_group_mask,
    lncc_loss,
    soft_dice_loss,
)
from psmareg.models.common.registration.displacement import resize_displacement
from psmareg.models.common.registration.jacobian import jacobian_determinant
from psmareg.models.common.registration.warping import warp_image_3d

Metrics = dict[str, float]


def stable_lncc_loss(moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Compute LNCC in float32 to avoid AMP half-precision NaNs."""
    with torch.amp.autocast(moving.device.type, enabled=False):
        return lncc_loss(moving.float(), fixed.float())


def robust_endpoint_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier endpoint loss, less dominated by outliers than field MSE.

    Args:
        predicted: Predicted field of shape ``[B, 3, D, H, W]``.
        target: Target field with the same shape.
        epsilon: Smoothing constant of the Charbonnier norm.

    Returns:
        Mean smoothed endpoint error.
    """
    squared_error = (predicted.float() - target.float()).square().sum(dim=1)
    return torch.sqrt(squared_error + epsilon**2).mean()


def multiscale_field_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        factors: tuple[int, ...] = (2, 4),
) -> torch.Tensor:
    """Supervise low-frequency displacement components at multiple scales.

    Args:
        predicted: Predicted field of shape ``[B, 3, D, H, W]``.
        target: Target field with the same shape.
        factors: Downsampling factors at which the fields are compared.

    Returns:
        Mean squared error averaged over the scales.
    """
    losses: list[torch.Tensor] = []
    for factor in factors:
        shape = tuple(max(2, size // factor) for size in predicted.shape[-3:])
        predicted_low = F.interpolate(
            predicted,
            size=shape,
            mode="trilinear",
            align_corners=True,
        )
        target_low = F.interpolate(
            target,
            size=shape,
            mode="trilinear",
            align_corners=True,
        )
        losses.append(F.mse_loss(predicted_low, target_low))
    return torch.stack(losses).mean()


def bending_energy(displacement: torch.Tensor) -> torch.Tensor:
    """Penalize high-frequency curvature in a displacement field."""
    losses: list[torch.Tensor] = []
    for spatial_dim in (2, 3, 4):
        first = torch.diff(displacement, dim=spatial_dim)
        second = torch.diff(first, dim=spatial_dim)
        losses.append(second.square().mean())
    return torch.stack(losses).mean()


def jacobian_hinge_loss(
    displacement: torch.Tensor,
    margin: float = 0.05,
    sampling_stride: int = 4,
) -> torch.Tensor:
    """Penalize non-positive and near-singular local transformations.

    Args:
        displacement: Field of shape ``[B, 3, D, H, W]``.
        margin: Smallest Jacobian determinant left unpenalized.
        sampling_stride: Stride of the finite-difference grid.

    Returns:
        Mean squared shortfall of the determinant below the margin.
    """
    determinants = jacobian_determinant(
        displacement.float(),
        sampling_stride=sampling_stride,
    )
    return F.relu(margin - determinants).square().mean()


def sampled_semantic_dice_loss(
    moving_label: torch.Tensor,
    target_displacement: torch.Tensor,
    predicted_displacement: torch.Tensor,
    downsample_factor: int,
    max_classes: int,
    binary: bool = False,
) -> torch.Tensor:
    """Compute low-resolution Dice on a small sample of foreground classes.

    Args:
        moving_label: Moving labels of shape ``[B, 1, D, H, W]``.
        target_displacement: Field that produces the reference warp.
        predicted_displacement: Field being supervised.
        downsample_factor: Spatial reduction applied before warping.
        max_classes: Largest number of classes sampled per batch.
        binary: Whether to merge all foreground labels into one class.

    Returns:
        Mean Dice loss between the two warped label maps.
    """
    shape = tuple(
        max(2, size // downsample_factor) for size in moving_label.shape[-3:])
    label_low = F.interpolate(
        moving_label.float(),
        size=shape,
        mode="nearest",
    ).long()
    target_low = resize_displacement(target_displacement, shape)
    predicted_low = resize_displacement(predicted_displacement, shape)

    if binary:
        classes = torch.ones(1, device=moving_label.device, dtype=torch.long)
    else:
        classes = torch.unique(label_low)
        classes = classes[classes > 0]
        if classes.numel() > max_classes:
            indices = torch.randperm(classes.numel(),
                                     device=classes.device)[:max_classes]
            classes = classes[indices]

    losses: list[torch.Tensor] = []
    for class_index in classes:
        if binary:
            moving_mask = (label_low > 0).float()
        else:
            moving_mask = (label_low == class_index).float()
        target_mask = warp_image_3d(
            moving_mask,
            target_low,
            interpolation_mode="bilinear",
        ).detach()
        predicted_mask = warp_image_3d(
            moving_mask,
            predicted_low,
            interpolation_mode="bilinear",
        )
        losses.append(soft_dice_loss(predicted_mask, target_mask, 1e-6))

    if not losses:
        return torch.zeros((), device=moving_label.device)
    return torch.stack(losses).mean()


def macro_ct_label_dice_loss(
    moving_label: torch.Tensor,
    target_displacement: torch.Tensor,
    predicted_displacement: torch.Tensor,
    groups: list[tuple[str, tuple[int, ...]]],
    downsample_factor: int,
    padding_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice on stable grouped anatomy rather than small noisy labels.

    Args:
        moving_label: Moving labels of shape ``[B, 1, D, H, W]``.
        target_displacement: Field that produces the reference warp.
        predicted_displacement: Field being supervised.
        groups: Names and class identifiers of the anatomical groups.
        downsample_factor: Spatial reduction applied before warping.
        padding_mode: Padding used when warping the group masks.
        eps: Numerical stabilizer of the Dice ratio.

    Returns:
        Mean Dice loss over the groups present in the labels.
    """
    shape = tuple(
        max(2, size // downsample_factor) for size in moving_label.shape[-3:])
    label_low = F.interpolate(
        moving_label.float(),
        size=shape,
        mode="nearest",
    ).long()
    target_low = resize_displacement(target_displacement, shape)
    predicted_low = resize_displacement(predicted_displacement, shape)

    losses: list[torch.Tensor] = []
    for _, class_ids in groups:
        moving_mask = label_group_mask(label_low, class_ids)
        if moving_mask.sum() < eps:
            continue
        target_mask = warp_image_3d(
            moving_mask,
            target_low,
            interpolation_mode="bilinear",
            padding_mode=padding_mode,
        ).detach()
        predicted_mask = warp_image_3d(
            moving_mask,
            predicted_low,
            interpolation_mode="bilinear",
            padding_mode=padding_mode,
        )
        losses.append(soft_dice_loss(predicted_mask, target_mask, eps))

    if not losses:
        return torch.zeros((), device=moving_label.device)
    return torch.stack(losses).mean()


def pet_biomarker_preservation_loss(
    moving_pet: torch.Tensor,
    moving_pet_label: torch.Tensor,
    target_displacement: torch.Tensor,
    predicted_displacement: torch.Tensor,
    downsample_factor: int,
    padding_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match PET hot-region volume and uptake for predicted vs target warp.

    This is deliberately not a fixed-vs-moving PET intensity loss. For synthetic
    pretraining we know the true deformation. The loss asks whether the
    predicted deformation preserves the moving hot-region biomarker measurements
    similarly to the target deformation, even when the synthetic fixed PET image
    contains independent uptake/dropout/false-hot changes.

    Args:
        moving_pet: Moving PET of shape ``[B, 1, D, H, W]``.
        moving_pet_label: Moving lesion labels with the same shape.
        target_displacement: Field that produces the reference warp.
        predicted_displacement: Field being supervised.
        downsample_factor: Spatial reduction applied before warping.
        padding_mode: Padding used when warping.
        eps: Numerical stabilizer of the relative errors.

    Returns:
        Relative difference of lesion volume and lesion uptake between the
        predicted and the reference warp.
    """
    if moving_pet_label.numel() == 0:
        return torch.zeros((), device=moving_pet.device, dtype=moving_pet.dtype)

    factor = max(1, int(downsample_factor))
    if factor > 1:
        shape = tuple(
            max(2, math.ceil(size / factor)) for size in moving_pet.shape[-3:])
        pet = F.interpolate(
            moving_pet.float(),
            size=shape,
            mode="trilinear",
            align_corners=True,
        )
        hot = F.interpolate(
            (moving_pet_label.float() > 0).float(),
            size=shape,
            mode="nearest",
        )
        target_disp = resize_displacement(target_displacement, shape)
        pred_disp = resize_displacement(predicted_displacement, shape)
    else:
        pet = moving_pet.float()
        hot = (moving_pet_label.float() > 0).float()
        target_disp = target_displacement
        pred_disp = predicted_displacement

    if float(hot.detach().sum().cpu()) <= 0.0:
        return torch.zeros((), device=moving_pet.device, dtype=moving_pet.dtype)

    target_hot = warp_image_3d(
        hot,
        target_disp,
        interpolation_mode="bilinear",
        padding_mode=padding_mode,
    ).clamp(0.0, 1.0)
    pred_hot = warp_image_3d(
        hot,
        pred_disp,
        interpolation_mode="bilinear",
        padding_mode=padding_mode,
    ).clamp(0.0, 1.0)
    target_pet = warp_image_3d(pet, target_disp,
                               padding_mode=padding_mode).clamp_min(0.0)
    pred_pet = warp_image_3d(pet, pred_disp,
                             padding_mode=padding_mode).clamp_min(0.0)

    reduce_dims = tuple(range(1, target_hot.ndim))
    target_volume = target_hot.sum(dim=reduce_dims)
    pred_volume = pred_hot.sum(dim=reduce_dims)
    target_tlg = (target_pet * target_hot).sum(dim=reduce_dims)
    pred_tlg = (pred_pet * pred_hot).sum(dim=reduce_dims)

    volume_error = (pred_volume - target_volume).abs() / (target_volume.abs() +
                                                          eps)
    tlg_error = (pred_tlg - target_tlg).abs() / (target_tlg.abs() + eps)
    # Robust cap: the objective is to discourage biomarker destruction, not let
    # rare tiny lesions dominate the whole synthetic field-learning objective.
    return 0.5 * (volume_error.clamp_max(2.0) + tlg_error.clamp_max(2.0)).mean()
