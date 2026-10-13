"""Retained Stage A, Stage B, and Stage C registration cascades."""

from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple

import torch
from torch import nn

from psmareg.models.common.registration.affine import (
    AffineHeadBase,
    AffineRegistrationHead,
    StrongAffineRegistrationHead,
    body_center_translation,
)
from psmareg.models.common.registration.blocks import (
    LocalResidualRefiner,
    LowResolutionTransMorph,
    MultiScaleResidualRefiner,
    TransMorphV2CoarseBackbone,
)
from psmareg.models.common.registration.warping import (
    compose_displacements,
    integrate_stationary_velocity,
    warp_image_3d,
)


class TransMorphDiffCascadeRegistration(nn.Module):
    """TransMorph-like diffeomorphic coarse stage plus local residual stage."""

    def __init__(
        self,
        max_velocity: float = 8.0,
        max_residual_disp: float = 2.0,
        scaling_steps: int = 7,
    ) -> None:
        super().__init__()
        self.scaling_steps = scaling_steps
        self.max_residual_disp = max_residual_disp

        self.coarse: nn.Module = LowResolutionTransMorph(
            in_channels=4,
            embed_dim=96,
            depth=4,
            num_heads=4,
            max_velocity=max_velocity,
        )

        self.refiner: nn.Module = LocalResidualRefiner(
            in_channels=4,
            base_channels=24,
            max_residual_disp=self.max_residual_disp,
        )

    def set_max_velocity(self, value: float) -> None:
        """Update the coarse velocity cap without changing learned weights."""
        setattr(self.coarse, "max_velocity", value)

    def forward(
        self,
        fixed_ct: torch.Tensor,
        fixed_pet: torch.Tensor,
        moving_ct: torch.Tensor,
        moving_pet: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Register a moving CT/PET pair to its fixed counterpart.

        Args:
            fixed_ct: Fixed CT of shape ``[B, 1, D, H, W]``.
            fixed_pet: Fixed PET with the same shape.
            moving_ct: Moving CT with the same shape.
            moving_pet: Moving PET with the same shape.

        Returns:
            Final, coarse, and residual displacements in voxel units.
        """
        _, _, depth, height, width = fixed_ct.shape

        coarse_input = torch.cat(
            [fixed_ct, fixed_pet, moving_ct, moving_pet],
            dim=1,
        )

        velocity = self.coarse(
            coarse_input,
            output_shape=(depth, height, width),
        )
        disp_stage1 = integrate_stationary_velocity(
            velocity,
            scaling_steps=self.scaling_steps,
        )

        warped_moving_ct = warp_image_3d(moving_ct, disp_stage1)
        warped_moving_pet = warp_image_3d(moving_pet, disp_stage1)

        residual_input = torch.cat(
            [fixed_ct, fixed_pet, warped_moving_ct, warped_moving_pet],
            dim=1,
        )
        residual_velocity = self.refiner(residual_input)
        residual_displacement = integrate_stationary_velocity(
            residual_velocity,
            scaling_steps=self.scaling_steps,
        )

        displacement = compose_displacements(
            first=disp_stage1,
            second=residual_displacement,
        )
        return displacement, disp_stage1, residual_displacement


class TransMorphV2CascadeRegistration(TransMorphDiffCascadeRegistration):
    """Higher-capacity TransMorph v2 cascade with true multi-scale coarse stage.

    The V1 modules built by ``super().__init__`` are intentionally replaced:
    this preserves the random-number stream of the archived Stage A/B code.
    """

    def __init__(
        self,
        max_velocity: float = 8.0,
        max_residual_disp: float = 2.0,
        scaling_steps: int = 7,
    ) -> None:
        super().__init__(
            max_velocity=max_velocity,
            max_residual_disp=max_residual_disp,
            scaling_steps=scaling_steps,
        )
        self.coarse = TransMorphV2CoarseBackbone(
            base_channels=16,
            embed_dim=96,
            num_heads=4,
            window_size=4,
            max_velocity=max_velocity,
        )
        self.refiner = MultiScaleResidualRefiner(
            base_channels=16,
            max_residual_disp=max_residual_disp,
        )


@dataclass(frozen=True)
class AffineCascadeSettings:
    """Construction bounds shared by the two affine cascade generations."""

    max_velocity: float = 8.0
    max_residual_disp: float = 2.0
    scaling_steps: int = 7
    max_affine_translation: float = 30.0
    max_affine_rotation_degrees: float = 15.0
    max_affine_scaling: float = 0.10
    body_center_initialization: bool = False
    body_mask_threshold: float = 0.02
    max_body_translation: float = 64.0
    zero_padding: bool = False


class AffineCascadeOutput(NamedTuple):
    """Displacements and affine diagnostics of one affine-cascade pass."""

    displacement: torch.Tensor
    affine_and_coarse: torch.Tensor
    residual: torch.Tensor
    affine: torch.Tensor
    body_translation: torch.Tensor
    affine_parameters: torch.Tensor


class AffineTransMorphDiffCascadeRegistration(nn.Module):
    """Learned affine alignment followed by diffeomorphic coarse/fine stages."""

    def __init__(self, settings: AffineCascadeSettings | None = None) -> None:
        super().__init__()
        settings = settings or AffineCascadeSettings()
        self.scaling_steps = settings.scaling_steps
        self.max_residual_disp = settings.max_residual_disp
        self.body_center_initialization = settings.body_center_initialization
        self.body_mask_threshold = settings.body_mask_threshold
        self.max_body_translation = settings.max_body_translation
        self.padding_mode = "zeros" if settings.zero_padding else "border"
        self.affine: nn.Module = AffineRegistrationHead(
            max_translation=settings.max_affine_translation,
            max_rotation_degrees=settings.max_affine_rotation_degrees,
            max_scaling=settings.max_affine_scaling,
        )
        self.coarse: nn.Module = LowResolutionTransMorph(
            in_channels=4,
            embed_dim=96,
            depth=4,
            num_heads=4,
            max_velocity=settings.max_velocity,
        )
        self.refiner: nn.Module = LocalResidualRefiner(
            in_channels=4,
            base_channels=24,
            max_residual_disp=settings.max_residual_disp,
        )

    def set_max_velocity(self, value: float) -> None:
        """Update the coarse velocity cap without changing learned weights."""
        setattr(self.coarse, "max_velocity", value)

    def reset_affine_head(self) -> None:
        """Reinitialize the affine head so that it predicts the identity.

        Raises:
            TypeError: If the affine stage was replaced by a module that cannot
                be reset.
        """
        if not isinstance(self.affine, AffineHeadBase):
            raise TypeError("The affine stage cannot be reset to identity.")
        self.affine.reset_to_identity()

    def forward(
        self,
        fixed_ct: torch.Tensor,
        fixed_pet: torch.Tensor,
        moving_ct: torch.Tensor,
        moving_pet: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
          AffineCascadeOutput):
        """Register a moving pair with affine, coarse, and residual stages.

        Args:
            fixed_ct: Fixed CT of shape ``[B, 1, D, H, W]``.
            fixed_pet: Fixed PET with the same shape.
            moving_ct: Moving CT with the same shape.
            moving_pet: Moving PET with the same shape.
            return_diagnostics: Whether to also return the body-center
                translation and the affine head parameters.

        Returns:
            Final displacement, affine-plus-coarse displacement, residual
            displacement, and affine displacement, all in voxel units. With
            ``return_diagnostics``, an ``AffineCascadeOutput`` holding these
            four fields followed by the two diagnostics.
        """
        output_shape = fixed_ct.shape[-3:]
        initial_translation = torch.zeros(fixed_ct.shape[0],
                                          3,
                                          device=fixed_ct.device,
                                          dtype=fixed_ct.dtype)
        if self.body_center_initialization:
            initial_translation = body_center_translation(
                fixed_ct,
                moving_ct,
                threshold=self.body_mask_threshold,
                max_translation=self.max_body_translation,
            )
        initial_displacement = initial_translation[:, :, None, None,
                                                   None].expand(
                                                       -1, -1, *output_shape)
        centered_moving_ct = warp_image_3d(moving_ct,
                                           initial_displacement,
                                           padding_mode=self.padding_mode)
        centered_moving_pet = warp_image_3d(moving_pet,
                                            initial_displacement,
                                            padding_mode=self.padding_mode)
        original_input = torch.cat(
            (fixed_ct, fixed_pet, centered_moving_ct, centered_moving_pet),
            dim=1,
        )
        residual_affine, affine_parameters = self.affine(
            original_input, output_shape)
        affine_displacement = compose_displacements(
            first=initial_displacement,
            second=residual_affine,
        )
        affine_moving_ct = warp_image_3d(moving_ct,
                                         affine_displacement,
                                         padding_mode=self.padding_mode)
        affine_moving_pet = warp_image_3d(moving_pet,
                                          affine_displacement,
                                          padding_mode=self.padding_mode)

        coarse_input = torch.cat(
            (fixed_ct, fixed_pet, affine_moving_ct, affine_moving_pet),
            dim=1,
        )
        coarse_velocity = self.coarse(coarse_input, output_shape=output_shape)
        coarse_displacement = integrate_stationary_velocity(
            coarse_velocity,
            scaling_steps=self.scaling_steps,
        )
        affine_and_coarse = compose_displacements(
            first=affine_displacement,
            second=coarse_displacement,
        )

        coarse_moving_ct = warp_image_3d(moving_ct,
                                         affine_and_coarse,
                                         padding_mode=self.padding_mode)
        coarse_moving_pet = warp_image_3d(moving_pet,
                                          affine_and_coarse,
                                          padding_mode=self.padding_mode)
        residual_input = torch.cat(
            (fixed_ct, fixed_pet, coarse_moving_ct, coarse_moving_pet),
            dim=1,
        )
        residual_velocity = self.refiner(residual_input)
        residual_displacement = integrate_stationary_velocity(
            residual_velocity,
            scaling_steps=self.scaling_steps,
        )
        displacement = compose_displacements(
            first=affine_and_coarse,
            second=residual_displacement,
        )
        output = AffineCascadeOutput(displacement, affine_and_coarse,
                                     residual_displacement, affine_displacement,
                                     initial_translation, affine_parameters)
        return output if return_diagnostics else output[:4]


class AffineTransMorphV2CascadeRegistration(
        AffineTransMorphDiffCascadeRegistration):
    """TransMorph v2: strong affine, hierarchical Swin coarse, deep refiner.

    The V1 modules built by ``super().__init__`` are intentionally replaced:
    this preserves the random-number stream of the archived Stage C/final code.
    """

    def __init__(self, settings: AffineCascadeSettings | None = None) -> None:
        settings = settings or AffineCascadeSettings()
        super().__init__(settings)
        self.affine = StrongAffineRegistrationHead(
            max_translation=settings.max_affine_translation,
            max_rotation_degrees=settings.max_affine_rotation_degrees,
            max_scaling=settings.max_affine_scaling,
        )
        self.coarse = TransMorphV2CoarseBackbone(
            base_channels=16,
            embed_dim=96,
            num_heads=4,
            window_size=4,
            max_velocity=settings.max_velocity,
        )
        self.refiner = MultiScaleResidualRefiner(
            base_channels=16,
            max_residual_disp=settings.max_residual_disp,
        )
