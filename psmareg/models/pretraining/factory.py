"""Model construction and execution for synthetic pretraining stages."""

from __future__ import annotations

import torch
from torch import nn

from psmareg.config.synthetic_pretraining_config import ArchitectureConfig
from psmareg.models.common.registration.cascades import (
    AffineCascadeSettings,
    AffineTransMorphDiffCascadeRegistration,
    AffineTransMorphV2CascadeRegistration,
    TransMorphDiffCascadeRegistration,
    TransMorphV2CascadeRegistration,
)


def build_model(architecture: ArchitectureConfig) -> nn.Module:
    """Build a retained Stage A, Stage B, or Stage C registration model.

    Args:
        architecture: Model name and affine/deformable bounds.

    Returns:
        Freshly initialized registration model.

    Raises:
        ValueError: If the model name is unknown.
    """
    if architecture.model in ("transmorph_diff_cascade",
                              "transmorph_v2_cascade"):
        cascade = (TransMorphDiffCascadeRegistration
                   if architecture.model == "transmorph_diff_cascade" else
                   TransMorphV2CascadeRegistration)
        return cascade(max_velocity=architecture.max_velocity,
                       max_residual_disp=architecture.max_residual_disp,
                       scaling_steps=architecture.scaling_steps)

    settings = AffineCascadeSettings(
        max_velocity=architecture.max_velocity,
        max_residual_disp=architecture.max_residual_disp,
        scaling_steps=architecture.scaling_steps,
        max_affine_translation=architecture.max_affine_translation,
        max_affine_rotation_degrees=architecture.max_affine_rotation_degrees,
        max_affine_scaling=architecture.max_affine_scaling,
        body_center_initialization=architecture.body_center_initialization,
        body_mask_threshold=architecture.body_mask_threshold,
        max_body_translation=architecture.max_body_translation,
        zero_padding=architecture.zero_padding,
    )
    if architecture.model == "affine_transmorph_diff_cascade":
        return AffineTransMorphDiffCascadeRegistration(settings)
    if architecture.model == "affine_transmorph_v2_cascade":
        return AffineTransMorphV2CascadeRegistration(settings)
    raise ValueError(f"Unknown model: {architecture.model}")


def set_effective_max_velocity(model: nn.Module, value: float) -> None:
    """Update the TransMorph output cap without changing learned parameters."""
    if isinstance(
            model,
        (
            TransMorphDiffCascadeRegistration,
            AffineTransMorphDiffCascadeRegistration,
        ),
    ):
        model.set_max_velocity(value)


def run_registration_model(
    model: nn.Module,
    fixed_ct: torch.Tensor,
    fixed_pet: torch.Tensor,
    moving_ct: torch.Tensor,
    moving_pet: torch.Tensor,
) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    """Run a retained cascade and normalize its stage outputs.

    Args:
        model: Stage A/B deformable or Stage C affine cascade.
        fixed_ct: Fixed CT image of shape ``[B, 1, D, H, W]``.
        fixed_pet: Fixed PET image with the same shape.
        moving_ct: Moving CT image with the same shape.
        moving_pet: Moving PET image with the same shape.

    Returns:
        Final displacement, coarse displacement, residual displacement,
        and optional affine displacement, all in voxel units.

    Raises:
        TypeError: If ``model`` is not a retained cascade.
    """
    if isinstance(model, AffineTransMorphDiffCascadeRegistration):
        displacement, stage1, residual, affine = model(
            fixed_ct=fixed_ct,
            fixed_pet=fixed_pet,
            moving_ct=moving_ct,
            moving_pet=moving_pet,
        )
        return displacement, stage1, residual, affine
    if isinstance(model, TransMorphDiffCascadeRegistration):
        displacement, stage1, residual = model(
            fixed_ct=fixed_ct,
            fixed_pet=fixed_pet,
            moving_ct=moving_ct,
            moving_pet=moving_pet,
        )
        return displacement, stage1, residual, None

    raise TypeError(
        "run_registration_model expects a retained TransMorph cascade, "
        f"got {type(model).__name__}.")
