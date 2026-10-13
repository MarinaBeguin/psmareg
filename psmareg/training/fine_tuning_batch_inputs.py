"""Prepare fine-tuning batches and run registration in the original frame."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from psmareg.config.fine_tuning_config import FineTuningConfig
from psmareg.data.augmentations import (
    independent_intensity_augmentation,
    random_affine_augmentation_field,
)
from psmareg.models.common.registration.cascades import (
    AffineTransMorphDiffCascadeRegistration,)
from psmareg.models.common.registration.displacement import (
    resize_voxel_displacement,)
from psmareg.models.common.registration.prealignment.deterministic import (
    TranslationPrealignSettings,
    estimate_lowres_ct_translation_field,
)
from psmareg.models.common.registration.warping import (
    compose_displacements,
    valid_sampling_mask,
    warp_image_3d,
)

TensorBatch = dict[str, torch.Tensor | list[str]]


def _get_tensor(batch: TensorBatch, key: str,
                device: torch.device) -> torch.Tensor:
    """Return one tensor batch entry on the requested device."""
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor for key '{key}', got {type(value)}")
    return value.to(device)


@dataclass(frozen=True)
class BatchData:
    """Images, optional labels, and augmentation validity for one pair."""

    fixed_ct: torch.Tensor
    fixed_pet: torch.Tensor
    moving_ct: torch.Tensor
    moving_pet: torch.Tensor
    fixed_ct_label: torch.Tensor | None
    moving_ct_label: torch.Tensor | None
    fixed_pet_label: torch.Tensor | None
    moving_pet_label: torch.Tensor | None
    augmentation_validity: torch.Tensor | None


def prepare_batch(
    batch: TensorBatch,
    device: torch.device,
    config: FineTuningConfig,
    apply_training_augmentation: bool,
) -> BatchData:
    """Move a pair to the device and apply the configured training augmentation.

    The spatial augmentation warps the moving scan and its labels with one
    random affine field; intensity augmentations are drawn independently for
    each of the four images.

    Args:
        batch: Paired PET/CT tensors and optional labels.
        device: Device receiving the batch.
        config: Validated training configuration.
        apply_training_augmentation: Whether to augment the pair.

    Returns:
        Images, optional labels, and the validity mask of the spatial
        augmentation when one was applied.
    """
    augmentation = config.augmentation

    def load_label(key: str) -> torch.Tensor | None:
        """Move an optional label map to the device as ``[B, D, H, W]``."""
        if key not in batch:
            return None
        return _get_tensor(batch, key, device).squeeze(1).long()

    def augmented(image: torch.Tensor, is_pet: bool) -> torch.Tensor:
        """Apply the intensity augmentation with its probability."""
        if torch.rand(
            (), device=device) < augmentation.train_intensity_aug_probability:
            return independent_intensity_augmentation(image, is_pet=is_pet)
        return image

    fixed_ct = _get_tensor(batch, "fixed_ct", device)
    fixed_pet = _get_tensor(batch, "fixed_pet", device)
    moving_ct = _get_tensor(batch, "moving_ct", device)
    moving_pet = _get_tensor(batch, "moving_pet", device)
    moving_ct_label = load_label("moving_ct_label")
    moving_pet_label = load_label("moving_pet_label")
    augmentation_validity: torch.Tensor | None = None
    if apply_training_augmentation:
        if torch.rand(
            (), device=device) < augmentation.train_spatial_aug_probability:
            field = random_affine_augmentation_field(
                moving_ct,
                max_translation=augmentation.train_aug_max_translation,
                max_rotation_degrees=(
                    augmentation.train_aug_max_rotation_degrees),
                max_scaling=augmentation.train_aug_max_scaling)

            def warp_label(label: torch.Tensor | None) -> torch.Tensor | None:
                """Warp an optional label map with nearest interpolation."""
                if label is None:
                    return None
                return warp_image_3d(label.unsqueeze(1).float(),
                                     field,
                                     interpolation_mode="nearest",
                                     padding_mode="zeros").squeeze(1).long()

            augmentation_validity = warp_image_3d(torch.ones_like(
                moving_ct[:, :1]),
                                                  field,
                                                  padding_mode="zeros")
            moving_ct = warp_image_3d(moving_ct, field, padding_mode="zeros")
            moving_pet = warp_image_3d(moving_pet, field, padding_mode="zeros")
            moving_ct_label = warp_label(moving_ct_label)
            moving_pet_label = warp_label(moving_pet_label)
        fixed_ct = augmented(fixed_ct, is_pet=False)
        fixed_pet = augmented(fixed_pet, is_pet=True)
        moving_ct = augmented(moving_ct, is_pet=False)
        moving_pet = augmented(moving_pet, is_pet=True)
        # The original run drew once for disabled FOV dropout. Preserve its
        # RNG stream so subsequent augmentations and model dropout match.
        _ = torch.rand((), device=device)
    return BatchData(
        fixed_ct=fixed_ct,
        fixed_pet=fixed_pet,
        moving_ct=moving_ct,
        moving_pet=moving_pet,
        fixed_ct_label=load_label("fixed_ct_label"),
        moving_ct_label=moving_ct_label,
        fixed_pet_label=load_label("fixed_pet_label"),
        moving_pet_label=moving_pet_label,
        augmentation_validity=augmentation_validity,
    )


@dataclass(frozen=True)
class RegistrationState:
    """Prediction and warped images in the original moving-image frame."""

    displacement: torch.Tensor
    stage1_displacement: torch.Tensor
    residual_displacement: torch.Tensor
    warped_ct: torch.Tensor
    warped_pet: torch.Tensor
    reference_overlap: torch.Tensor
    overlap: torch.Tensor
    prealign_translation: torch.Tensor
    body_translation: torch.Tensor | None = None
    affine_parameters: torch.Tensor | None = None


def _load_external_prealignment(batch: TensorBatch, device: torch.device,
                                reference: torch.Tensor) -> torch.Tensor:
    """Load a precomputed pre-alignment field on the grid of the images.

    Args:
        batch: Batch holding a ``prealign_displacement`` entry.
        device: Device receiving the field.
        reference: Image of shape ``[B, C, D, H, W]`` defining the grid.

    Returns:
        Displacement field of shape ``[B, 3, D, H, W]`` in image voxels.
    """
    field = _get_tensor(batch, "prealign_displacement", device)
    if field.ndim == 4:
        field = field.unsqueeze(0)
    target_shape = reference.shape[-3:]
    if field.shape != (reference.shape[0], 3, *target_shape):
        field = resize_voxel_displacement(field,
                                          target_shape,
                                          align_corners=False)
    return field


@dataclass(frozen=True)
class _PreparedMoving:
    """Moving images and field after optional external and CT pre-alignment."""

    ct: torch.Tensor
    pet: torch.Tensor
    displacement: torch.Tensor
    translation: torch.Tensor
    active: bool


def _warp_moving_pair(
    data: BatchData,
    displacement: torch.Tensor,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp the original moving CT and PET using the same field."""
    return (warp_image_3d(data.moving_ct,
                          displacement,
                          padding_mode=padding_mode),
            warp_image_3d(data.moving_pet,
                          displacement,
                          padding_mode=padding_mode))


def _prepare_moving_inputs(
    model: nn.Module,
    batch: TensorBatch,
    data: BatchData,
    device: torch.device,
    config: FineTuningConfig,
) -> _PreparedMoving:
    """Apply optional pre-alignment before the registration network."""
    prealignment = config.prealignment
    fixed_ct = data.fixed_ct
    model_input_padding_mode = "border"

    def resolve(padding_mode: str) -> str:
        """Replace the ``auto`` padding mode with the model's own."""
        return (model_input_padding_mode
                if padding_mode == "auto" else padding_mode)

    prealign_displacement = torch.zeros(fixed_ct.shape[0],
                                        3,
                                        *fixed_ct.shape[-3:],
                                        device=device,
                                        dtype=fixed_ct.dtype)
    prealign_translation = torch.zeros(fixed_ct.shape[0],
                                       3,
                                       device=device,
                                       dtype=fixed_ct.dtype)
    model_moving_ct, model_moving_pet = data.moving_ct, data.moving_pet
    external_prealign: torch.Tensor | None = None
    if "prealign_displacement" in batch:
        external_prealign = _load_external_prealignment(batch, device, fixed_ct)
        prealign_displacement = external_prealign.to(dtype=fixed_ct.dtype)
        prealign_translation = prealign_displacement.mean(dim=(2, 3, 4))
        model_moving_ct, model_moving_pet = _warp_moving_pair(
            data, prealign_displacement, model_input_padding_mode)
    if prealignment.deterministic_translation_prealign:
        prealign_displacement, prealign_translation = (
            estimate_lowres_ct_translation_field(
                fixed_ct, model_moving_ct,
                TranslationPrealignSettings(
                    downsample_factor=(
                        prealignment.deterministic_translation_downsample),
                    search_radius=(
                        prealignment.deterministic_translation_search_radius),
                    search_step=(
                        prealignment.deterministic_translation_search_step),
                    body_threshold=model.body_mask_threshold if isinstance(
                        model,
                        AffineTransMorphDiffCascadeRegistration) else 0.02,
                    max_translation=(
                        prealignment.deterministic_translation_max),
                    padding_mode=resolve(
                        prealignment.deterministic_translation_padding_mode),
                    feature_mode=(
                        prealignment.deterministic_translation_feature_mode))))
        if external_prealign is not None:
            prealign_displacement = compose_displacements(
                external_prealign, prealign_displacement)
            prealign_translation = prealign_displacement.mean(dim=(2, 3, 4))
        model_moving_ct, model_moving_pet = _warp_moving_pair(
            data, prealign_displacement,
            resolve(prealignment.deterministic_translation_input_padding_mode))
    prealign_active = (external_prealign is not None or
                       prealignment.deterministic_translation_prealign)
    return _PreparedMoving(model_moving_ct, model_moving_pet,
                           prealign_displacement, prealign_translation,
                           prealign_active)


def predict_registration(
    model: nn.Module,
    batch: TensorBatch,
    data: BatchData,
    device: torch.device,
    config: FineTuningConfig,
) -> RegistrationState:
    """Pre-align the pair, run the model, and warp the original moving images.

    The model only sees pre-aligned moving images. Its output is composed with
    the pre-alignment so that the returned field maps the original moving
    images, which are warped once with zero padding.

    Args:
        model: Registration model.
        batch: Batch, optionally holding a precomputed pre-alignment field.
        data: Images prepared by ``prepare_batch``.
        device: Device on which the model runs.
        config: Validated training configuration.

    Returns:
        Composed displacement, warped images, and valid-overlap masks.
    """
    moving = _prepare_moving_inputs(model, batch, data, device, config)

    model_inputs = {
        "fixed_ct": data.fixed_ct,
        "fixed_pet": data.fixed_pet,
        "moving_ct": moving.ct,
        "moving_pet": moving.pet,
    }
    body_translation = affine_parameters = None
    if isinstance(model, AffineTransMorphDiffCascadeRegistration):
        model_output = model(**model_inputs, return_diagnostics=True)
        body_translation = model_output.body_translation
        affine_parameters = model_output.affine_parameters
    else:
        model_output = model(**model_inputs)
    model_displacement, disp_stage1, residual = model_output[:3]
    displacement = compose_displacements(
        moving.displacement,
        model_displacement) if moving.active else model_displacement
    warped_ct, warped_pet = _warp_moving_pair(data, displacement, "zeros")
    reference_overlap = valid_sampling_mask(moving.displacement)
    overlap = valid_sampling_mask(displacement)
    if data.augmentation_validity is not None:
        reference_overlap = reference_overlap * warp_image_3d(
            data.augmentation_validity,
            moving.displacement,
            padding_mode="zeros")
        overlap = overlap * warp_image_3d(
            data.augmentation_validity, displacement, padding_mode="zeros")
    return RegistrationState(
        displacement=displacement,
        stage1_displacement=disp_stage1,
        residual_displacement=residual,
        warped_ct=warped_ct,
        warped_pet=warped_pet,
        reference_overlap=reference_overlap,
        overlap=overlap,
        prealign_translation=moving.translation,
        body_translation=body_translation,
        affine_parameters=affine_parameters,
    )
