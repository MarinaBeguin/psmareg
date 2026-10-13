"""Jacobian diagnostics for dense voxel-displacement fields."""

from __future__ import annotations

import torch


def jacobian_determinant(
    displacement: torch.Tensor,
    sampling_stride: int = 4,
) -> torch.Tensor:
    """Compute Jacobian determinants on a strided interior grid.

    Raises:
        ValueError: If the stride is below one or the field is too small for
            finite differences.
    """
    if sampling_stride < 1:
        raise ValueError("sampling_stride must be at least 1.")

    field = displacement[:, :, ::sampling_stride, ::sampling_stride, ::
                         sampling_stride]
    if min(field.shape[-3:]) < 2:
        raise ValueError("Displacement is too small for Jacobian computation.")

    base = field[:, :, :-1, :-1, :-1]
    derivative_z = (field[:, :, 1:, :-1, :-1] - base) / float(sampling_stride)
    derivative_y = (field[:, :, :-1, 1:, :-1] - base) / float(sampling_stride)
    derivative_x = (field[:, :, :-1, :-1, 1:] - base) / float(sampling_stride)

    j00 = 1.0 + derivative_x[:, 0]
    j01 = derivative_y[:, 0]
    j02 = derivative_z[:, 0]
    j10 = derivative_x[:, 1]
    j11 = 1.0 + derivative_y[:, 1]
    j12 = derivative_z[:, 1]
    j20 = derivative_x[:, 2]
    j21 = derivative_y[:, 2]
    j22 = 1.0 + derivative_z[:, 2]

    return (j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) +
            j02 * (j10 * j21 - j11 * j20))


def jacobian_summary(displacement: torch.Tensor) -> dict[str, float]:
    """Return compact folding and determinant statistics."""
    determinants = jacobian_determinant(
        displacement.detach().float(),
        sampling_stride=4,
    )
    return {
        "ndv_percent":
            float((determinants <= 0.0).float().mean().cpu() * 100.0),
        "jac_min":
            float(determinants.min().cpu()),
        "jac_p01":
            float(torch.quantile(determinants.flatten(), 0.01).cpu()),
    }
