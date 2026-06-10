"""Loss functions for PSMAReg registration."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def image_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return F.mse_loss(prediction, target)


def image_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute error."""
    return F.l1_loss(prediction, target)


def smoothness_loss(displacement: torch.Tensor) -> torch.Tensor:
    """Compute first-order displacement smoothness loss."""
    depth_gradient = displacement[:, :, 1:, :, :] - displacement[:, :, :-1, :, :]
    height_gradient = displacement[:, :, :, 1:, :] - displacement[:, :, :, :-1, :]
    width_gradient = displacement[:, :, :, :, 1:] - displacement[:, :, :, :, :-1]

    return (
        depth_gradient.pow(2).mean()
        + height_gradient.pow(2).mean()
        + width_gradient.pow(2).mean()
    )


def multiclass_dice_loss(
    warped_label: torch.Tensor,
    fixed_label: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute multi-class Dice loss, excluding background class 0."""
    losses: list[torch.Tensor] = []

    for class_idx in range(1, num_classes):
        pred_mask = warped_label[:, class_idx]
        target_mask = fixed_label[:, class_idx]

        if target_mask.sum() < eps and pred_mask.sum() < eps:
            continue

        intersection = (pred_mask * target_mask).sum()
        denominator = pred_mask.sum() + target_mask.sum()

        dice = (2.0 * intersection + eps) / (denominator + eps)
        losses.append(1.0 - dice)

    if not losses:
        return torch.zeros((), device=warped_label.device)

    return torch.stack(losses).mean()



def lncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    window_size: int = 9,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute local normalized cross-correlation loss for 3D images."""
    if moving.shape != fixed.shape:
        raise ValueError(
            f"moving and fixed must have the same shape, got "
            f"{moving.shape} and {fixed.shape}."
        )

    if moving.ndim != 5:
        raise ValueError(f"Expected tensors with shape [B, C, D, H, W], got {moving.shape}.")

    channels = moving.shape[1]
    padding = window_size // 2

    kernel = torch.ones(
        (channels, 1, window_size, window_size, window_size),
        dtype=moving.dtype,
        device=moving.device,
    )

    window_volume = float(window_size**3)

    moving_sum = F.conv3d(moving, kernel, padding=padding, groups=channels)
    fixed_sum = F.conv3d(fixed, kernel, padding=padding, groups=channels)

    moving_sq_sum = F.conv3d(moving * moving, kernel, padding=padding, groups=channels)
    fixed_sq_sum = F.conv3d(fixed * fixed, kernel, padding=padding, groups=channels)
    cross_sum = F.conv3d(moving * fixed, kernel, padding=padding, groups=channels)

    moving_mean = moving_sum / window_volume
    fixed_mean = fixed_sum / window_volume

    cross = cross_sum - fixed_mean * moving_sum - moving_mean * fixed_sum + moving_mean * fixed_mean * window_volume
    moving_var = moving_sq_sum - 2.0 * moving_mean * moving_sum + moving_mean * moving_mean * window_volume
    fixed_var = fixed_sq_sum - 2.0 * fixed_mean * fixed_sum + fixed_mean * fixed_mean * window_volume

    lncc = (cross * cross) / (moving_var * fixed_var + eps)

    return -lncc.mean()