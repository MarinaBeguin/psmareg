"""Loss functions for PSMAReg registration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def image_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return functional.mse_loss(prediction, target)


def image_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute error."""
    return functional.l1_loss(prediction, target)


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
