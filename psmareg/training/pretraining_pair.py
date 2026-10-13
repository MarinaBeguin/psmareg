"""Build synthetic fixed/moving pairs and model predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from psmareg.config.synthetic_pretraining_config import (
    SyntheticPretrainingConfig,)
from psmareg.data.synthetic_augmentations import (
    independent_pet_intensity_augmentation,
    synthmorph_like_pair_from_labels,
)
from psmareg.data.synthetic_deformations import (
    SyntheticTargetSettings,
    PoseAwareSettings,
    random_pose_aware_displacement_from_labels,
    safe_random_synthetic_displacement_components,
)
from psmareg.models.common.registration.jacobian import jacobian_determinant
from psmareg.models.common.registration.warping import (
    compose_displacements,
    valid_sampling_mask,
    warp_image_3d,
)
from psmareg.models.pretraining.factory import run_registration_model

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionOptions:
    """Per-pass settings that differ between training and validation.

    Attributes:
        deformation_scale: Curriculum multiplier of deformation magnitudes.
        similarity_downsample_factor: Downsampling of the image losses.
        collect_expensive_metrics: Whether to log before-registration
            similarity and Jacobian statistics.
        metric_sampling_stride: Stride of the endpoint-error metrics.
    """

    deformation_scale: float
    similarity_downsample_factor: int
    collect_expensive_metrics: bool
    metric_sampling_stride: int


@dataclass(frozen=True)
class SyntheticPair:
    """Synthetic fixed/moving pair and the deformation that relates them."""

    moving_ct: torch.Tensor
    moving_pet: torch.Tensor
    fixed_ct: torch.Tensor
    fixed_pet: torch.Tensor
    moving_ct_label: torch.Tensor | None
    moving_pet_label: torch.Tensor | None
    target_displacement: torch.Tensor
    target_affine: torch.Tensor
    pose_aware_applied: bool
    synthmorph_fov_augmented: bool
    pet_intensity_augmented: bool


@dataclass(frozen=True)
class Prediction:
    """Model outputs and the moving images warped by the final field."""

    displacement: torch.Tensor
    stage1: torch.Tensor | None
    residual: torch.Tensor | None
    affine: torch.Tensor | None
    warped_ct: torch.Tensor
    warped_pet: torch.Tensor
    valid_overlap: torch.Tensor


@dataclass(frozen=True)
class SimilarityImages:
    """Images at the resolution used by the image-similarity losses."""

    fixed_ct: torch.Tensor
    fixed_pet: torch.Tensor
    warped_ct: torch.Tensor
    warped_pet: torch.Tensor
    moving_ct: torch.Tensor | None
    moving_pet: torch.Tensor | None


def resolve_padding_mode(config: SyntheticPretrainingConfig) -> str:
    """Return the warping padding mode selected by the configuration."""
    return "zeros" if config.architecture.zero_padding else "border"


def _sample_target_displacement(
    moving: torch.Tensor,
    moving_ct_label: torch.Tensor | None,
    config: SyntheticPretrainingConfig,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Sample the known deformation of one synthetic pair.

    Args:
        moving: Moving CT/PET stack of shape ``[B, 2, D, H, W]``.
        moving_ct_label: Optional CT labels enabling pose-aware motion.
        config: Synthetic pretraining configuration.
        scale: Curriculum multiplier of deformation magnitudes.

    Returns:
        Target displacement, its affine component, and whether pose-aware
        articulated motion was composed into the target.
    """
    deformation, augmentation = config.deformation, config.augmentation
    jacobian_stride = config.loss.jacobian_loss_stride
    target, target_affine = safe_random_synthetic_displacement_components(
        image=moving,
        settings=SyntheticTargetSettings(
            max_elastic_displacement=(deformation.max_elastic_displacement *
                                      scale),
            elastic_control_point_spacing=(
                deformation.elastic_control_point_spacing),
            max_translation=deformation.max_translation * scale,
            max_rotation_degrees=deformation.max_rotation_degrees * scale,
            max_scaling=deformation.max_scaling * scale,
            minimum_jacobian=deformation.target_min_jacobian,
            max_attempts=deformation.target_generation_attempts,
            jacobian_stride=jacobian_stride,
        ),
    )
    if not (moving_ct_label is not None and augmentation.pose_aware_probability
            > 0.0 and float(torch.rand((), device=moving.device))
            < augmentation.pose_aware_probability):
        return target, target_affine, False

    # Keep the first fold-free candidate, or else the least folded one.
    best_candidate: torch.Tensor | None = None
    best_minimum = -float("inf")
    pose_settings = PoseAwareSettings(
        max_translation=augmentation.pose_max_translation * scale,
        max_rotation_degrees=augmentation.pose_max_rotation_degrees * scale,
        max_scaling=augmentation.pose_max_scaling * scale,
        region_blur_downsample=augmentation.pose_region_blur_downsample,
        min_region_fraction=augmentation.pose_min_region_fraction,
    )
    for _ in range(max(1, augmentation.pose_aware_attempts)):
        pose_displacement = random_pose_aware_displacement_from_labels(
            moving_ct_label, pose_settings)
        candidate = compose_displacements(first=target,
                                          second=pose_displacement)
        minimum = float(
            jacobian_determinant(candidate.detach().float(),
                                 sampling_stride=jacobian_stride).min().cpu())
        if minimum >= deformation.target_min_jacobian:
            return candidate, target_affine, True
        if minimum > best_minimum:
            best_candidate, best_minimum = candidate, minimum
    if best_candidate is None:
        return target, target_affine, False
    return best_candidate, target_affine, True


def synthesize_pair(
    batch: Batch,
    device: torch.device,
    config: SyntheticPretrainingConfig,
    options: BatchExecutionOptions,
) -> SyntheticPair:
    """Create a fixed/moving pair by deforming single scans.

    Args:
        batch: Single-scan CT/PET tensors and optional labels.
        device: Device receiving the batch.
        config: Synthetic pretraining configuration.
        options: Per-pass execution settings.

    Returns:
        Synthetic pair with its known target deformation.

    Raises:
        ValueError: If SynthMorph-like synthesis is requested without labels.
    """
    augmentation = config.augmentation
    moving_ct = batch["ct"].to(device, non_blocking=True)
    moving_pet = batch["pet"].to(device, non_blocking=True)
    moving_ct_label = (batch["ct_label"].to(device, non_blocking=True)
                       if "ct_label" in batch else None)
    moving_pet_label = (batch["pet_label"].to(device, non_blocking=True)
                        if "pet_label" in batch else None)
    target, target_affine, pose_aware_applied = _sample_target_displacement(
        torch.cat([moving_ct, moving_pet], dim=1), moving_ct_label, config,
        options.deformation_scale)

    sampling_padding = resolve_padding_mode(config)
    synthmorph_fov_augmented = False
    if augmentation.synthmorph_like:
        if moving_ct_label is None:
            raise ValueError(
                "synthmorph_like requires CT labels in the manifest.")
        (moving_ct, moving_pet, fixed_ct, fixed_pet,
         synthmorph_fov_augmented) = synthmorph_like_pair_from_labels(
             moving_ct_label=moving_ct_label,
             moving_pet_label=moving_pet_label,
             target_displacement=target,
             augmentation=augmentation,
             padding_mode=sampling_padding,
         )
    else:
        fixed_ct = warp_image_3d(moving_ct,
                                 target,
                                 padding_mode=sampling_padding)
        fixed_pet = warp_image_3d(moving_pet,
                                  target,
                                  padding_mode=sampling_padding)

    pet_intensity_augmented = (
        augmentation.pet_independent_intensity_probability > 0.0 and
        float(torch.rand((), device=device))
        < augmentation.pet_independent_intensity_probability)
    if pet_intensity_augmented:
        intensity_options = {
            "scale_log_std": augmentation.pet_independent_scale_log_std,
            "gamma_range": augmentation.pet_independent_gamma_range,
            "bias_range": augmentation.pet_independent_bias_range,
        }
        moving_pet = independent_pet_intensity_augmentation(
            moving_pet, **intensity_options)
        fixed_pet = independent_pet_intensity_augmentation(
            fixed_pet, **intensity_options)

    return SyntheticPair(
        moving_ct=moving_ct,
        moving_pet=moving_pet,
        fixed_ct=fixed_ct,
        fixed_pet=fixed_pet,
        moving_ct_label=moving_ct_label,
        moving_pet_label=moving_pet_label,
        target_displacement=target,
        target_affine=target_affine,
        pose_aware_applied=pose_aware_applied,
        synthmorph_fov_augmented=synthmorph_fov_augmented,
        pet_intensity_augmented=pet_intensity_augmented,
    )


def predict_pair(model: nn.Module, pair: SyntheticPair,
                 padding_mode: str) -> Prediction:
    """Run the model and warp the moving images with its final field.

    Args:
        model: Registration cascade of the current stage.
        pair: Synthetic fixed/moving pair.
        padding_mode: Padding used to warp the moving images.

    Returns:
        Stage fields, warped images, and the valid-overlap mask.
    """
    displacement, stage1, residual, affine = run_registration_model(
        model=model,
        fixed_ct=pair.fixed_ct,
        fixed_pet=pair.fixed_pet,
        moving_ct=pair.moving_ct,
        moving_pet=pair.moving_pet,
    )
    return Prediction(
        displacement=displacement,
        stage1=stage1,
        residual=residual,
        affine=affine,
        warped_ct=warp_image_3d(pair.moving_ct,
                                displacement,
                                padding_mode=padding_mode),
        warped_pet=warp_image_3d(pair.moving_pet,
                                 displacement,
                                 padding_mode=padding_mode),
        valid_overlap=valid_sampling_mask(displacement),
    )


def similarity_images(pair: SyntheticPair, prediction: Prediction,
                      options: BatchExecutionOptions) -> SimilarityImages:
    """Resize the images to the resolution of the similarity losses.

    The moving images are only needed, and only resized, for the optional
    before-registration metrics.

    Args:
        pair: Synthetic fixed/moving pair.
        prediction: Model outputs and warped images.
        options: Per-pass similarity resolution and metric settings.

    Returns:
        Images resized for the similarity losses; the moving images are included
        only for the expensive metrics.
    """
    factor = options.similarity_downsample_factor
    if factor <= 1:
        return SimilarityImages(pair.fixed_ct, pair.fixed_pet,
                                prediction.warped_ct, prediction.warped_pet,
                                pair.moving_ct, pair.moving_pet)
    shape = tuple(
        max(2, math.ceil(size / factor)) for size in pair.fixed_ct.shape[-3:])

    def resize(image: torch.Tensor) -> torch.Tensor:
        """Resample an image to the similarity resolution."""
        return F.interpolate(image,
                             size=shape,
                             mode="trilinear",
                             align_corners=True)

    return SimilarityImages(
        fixed_ct=resize(pair.fixed_ct),
        fixed_pet=resize(pair.fixed_pet),
        warped_ct=resize(prediction.warped_ct),
        warped_pet=resize(prediction.warped_pet),
        moving_ct=(resize(pair.moving_ct)
                   if options.collect_expensive_metrics else None),
        moving_pet=(resize(pair.moving_pet)
                    if options.collect_expensive_metrics else None),
    )
