"""Global masked similarity scores for CT pre-alignment."""

from __future__ import annotations

import torch


def masked_ncc_score(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return global masked NCC per batch item (not local LNCC).

    Args:
        fixed: Fixed image of shape ``[B, C, D, H, W]``.
        moving: Moving image with the same shape.
        mask: Mask of the voxels entering the correlation.
        eps: Numerical stabilizer of the variance product.

    Returns:
        Correlation of shape ``[B]``, in ``[-1, 1]``.
    """
    mask_sum = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    fixed_mean = (fixed * mask).sum(dim=(2, 3, 4)) / mask_sum
    moving_mean = (moving * mask).sum(dim=(2, 3, 4)) / mask_sum
    fixed_centered = (fixed - fixed_mean[:, :, None, None, None]) * mask
    moving_centered = (moving - moving_mean[:, :, None, None, None]) * mask
    numerator = (fixed_centered * moving_centered).sum(dim=(2, 3, 4))
    denominator = torch.sqrt(fixed_centered.square().sum(dim=(2, 3, 4)) *
                             moving_centered.square().sum(dim=(2, 3, 4)) + eps)
    return (numerator / denominator.clamp_min(eps)).squeeze(1)
