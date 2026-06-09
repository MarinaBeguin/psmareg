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
