"""Single-batch computation for lesion-aware PET/CT fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from psmareg.config.fine_tuning_config import FineTuningConfig
from psmareg.models.common.anatomy import macro_ct_label_groups
from psmareg.models.common.losses import (
    masked_lncc_loss,
    mind_ssc_loss,
    multiscale_correlation_loss,
)
from psmareg.models.fine_tuning.losses import (
    adaptive_pet_correspondence_weight,
    capped_pet_mse_loss,
    displacement_smoothness_loss,
    macro_label_dice_loss,
    multiclass_label_dice_loss,
    pet_hot_region_mse_loss,
    valid_overlap_hinge_loss,
    downsample_labels_and_displacement,
)
from psmareg.models.fine_tuning.lesion_losses import (
    mask_pet_similarity_around_moving_lesions,
    pet_lesion_occupancy_and_translation_losses,
    pet_lesion_strain_loss,
)
from psmareg.evaluation.challenge_metrics import (
    compute_challenge_like_binary_pet_scores,
    compute_challenge_like_ct_dsc_score,
    compute_challenge_like_pet_volume_errors,
    compute_multiclass_dsc_score,
)
from psmareg.training.fine_tuning_batch_inputs import (
    BatchData,
    RegistrationState,
    TensorBatch,
    predict_registration,
    prepare_batch,
)
from psmareg.training.fine_tuning_batch_metrics import (
    summarize_batch_metrics,)

Metrics = dict[str, float]


@dataclass(frozen=True)
class EpochWeights:
    """Loss weights after applying the current epoch's ramps."""

    lambda_macro_ct_labels: float
    lambda_pet_hot_region: float
    lambda_pet_lesion_occupancy: float
    lambda_pet_lesion_translation: float
    lambda_pet_lesion_strain: float


@dataclass(frozen=True)
class CtLabelResults:
    """CT-label loss and optional validation-only challenge measurements."""

    loss_ct_dice: torch.Tensor
    loss_ct_dice_log: float
    ct_dsc_log: float
    challenge_ct_dsc_log: float
    challenge_pet_dsc_log: float
    challenge_mtv_percent_error_log: float
    challenge_tlg_percent_error_log: float


def _compute_ct_label_results(
    data: BatchData,
    prediction: RegistrationState,
    config: FineTuningConfig,
    device: torch.device,
    compute_challenge_metrics: bool,
) -> CtLabelResults:
    """Compute CT-label supervision and optional challenge validation scores.

    Args:
        data: Images and optional labels of the pair.
        prediction: Predicted displacement.
        config: Validated training configuration.
        device: Device on which the loss is created.
        compute_challenge_metrics: Whether to compute challenge-like scores.

    Returns:
        Differentiable CT Dice loss and logged scores, which are NaN when the
        required labels are missing or the score was not requested.
    """
    nan = float("nan")
    if data.fixed_ct_label is None or data.moving_ct_label is None:
        return CtLabelResults(loss_ct_dice=torch.zeros((), device=device),
                              loss_ct_dice_log=nan,
                              ct_dsc_log=nan,
                              challenge_ct_dsc_log=nan,
                              challenge_pet_dsc_log=nan,
                              challenge_mtv_percent_error_log=nan,
                              challenge_tlg_percent_error_log=nan)
    displacement = prediction.displacement
    num_classes = config.architecture.num_ct_classes
    (dice_fixed_label, dice_moving_label,
     dice_displacement) = downsample_labels_and_displacement(
         data.fixed_ct_label, data.moving_ct_label, displacement,
         config.anatomical_loss.ct_dice_loss_downsample)
    loss_ct_dice = multiclass_label_dice_loss(fixed_label=dice_fixed_label,
                                              moving_label=dice_moving_label,
                                              displacement=dice_displacement,
                                              num_classes=num_classes,
                                              padding_mode="zeros")
    ct_dsc = compute_multiclass_dsc_score(data.fixed_ct_label,
                                          data.moving_ct_label,
                                          displacement,
                                          num_classes,
                                          padding_mode="zeros")
    challenge_ct_dsc = challenge_pet_dsc = nan
    mtv_percent_error = tlg_percent_error = nan
    if compute_challenge_metrics:
        challenge_ct_dsc = compute_challenge_like_ct_dsc_score(
            data.fixed_ct_label, data.moving_ct_label, displacement,
            num_classes)
        if (data.fixed_pet_label is not None and
                data.moving_pet_label is not None):
            challenge_pet_dsc = compute_challenge_like_binary_pet_scores(
                fixed_pet_label=data.fixed_pet_label,
                moving_pet_label=data.moving_pet_label,
                displacement=displacement)
        if data.moving_pet_label is not None:
            mtv_percent_error, tlg_percent_error = (
                compute_challenge_like_pet_volume_errors(
                    moving_pet=data.moving_pet,
                    moving_pet_label=data.moving_pet_label,
                    displacement=displacement))
    return CtLabelResults(loss_ct_dice=loss_ct_dice,
                          loss_ct_dice_log=float(loss_ct_dice.detach().cpu()),
                          ct_dsc_log=ct_dsc,
                          challenge_ct_dsc_log=challenge_ct_dsc,
                          challenge_pet_dsc_log=challenge_pet_dsc,
                          challenge_mtv_percent_error_log=mtv_percent_error,
                          challenge_tlg_percent_error_log=tlg_percent_error)


@dataclass(frozen=True)
class PetResults:
    """PET correspondence, hot-region and lesion-preservation losses."""

    similarity_mask: torch.Tensor | None
    similarity_removed_fraction: torch.Tensor
    loss_pet: torch.Tensor
    adaptive_weight: torch.Tensor
    hot_correspondence: float
    loss_hot_region: torch.Tensor
    loss_lesion_occupancy: torch.Tensor
    loss_lesion_translation: torch.Tensor
    loss_lesion_strain: torch.Tensor
    lesion_occupancy_components: torch.Tensor


def _compute_pet_lesion_losses(
    data: BatchData,
    prediction: RegistrationState,
    config: FineTuningConfig,
    weights: EpochWeights,
    loss_mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute active lesion losses in occupancy, translation, strain order."""
    loss_occupancy = torch.zeros((), device=device)
    loss_translation = torch.zeros((), device=device)
    loss_strain = torch.zeros((), device=device)
    components = torch.zeros((), device=device)
    if (weights.lambda_pet_lesion_occupancy > 0.0 or
            weights.lambda_pet_lesion_translation > 0.0 or
            weights.lambda_pet_lesion_strain
            > 0.0) and data.moving_pet_label is not None:
        (loss_occupancy, loss_translation,
         components) = pet_lesion_occupancy_and_translation_losses(
             moving_pet=data.moving_pet,
             moving_pet_label=data.moving_pet_label,
             displacement=prediction.displacement,
             valid_mask=loss_mask,
             config=config.lesion_loss)
        if weights.lambda_pet_lesion_strain > 0.0:
            loss_strain = pet_lesion_strain_loss(
                moving_pet_label=data.moving_pet_label,
                displacement=prediction.displacement,
                valid_mask=loss_mask,
                downsample_factor=config.lesion_loss.
                pet_lesion_occupancy_downsample,
                min_voxels=config.lesion_loss.pet_lesion_occupancy_min_voxels)
    return loss_occupancy, loss_translation, loss_strain, components


def _compute_pet_results(
    data: BatchData,
    prediction: RegistrationState,
    config: FineTuningConfig,
    weights: EpochWeights,
    loss_mask: torch.Tensor | None,
    device: torch.device,
) -> PetResults:
    """Compute the uncombined PET similarity and lesion-preservation terms."""
    padding_mode = "zeros"
    pet_similarity_mask = loss_mask
    pet_lesion_similarity_removed_fraction = torch.zeros((), device=device)
    if (config.pet_loss.pet_mask_similarity_near_lesions and
            data.moving_pet_label is not None):
        (pet_similarity_mask, pet_lesion_similarity_removed_fraction
        ) = mask_pet_similarity_around_moving_lesions(
            base_mask=loss_mask,
            moving_pet_label=data.moving_pet_label,
            displacement=prediction.displacement,
            dilate=config.pet_loss.pet_lesion_similarity_mask_dilate,
            downsample=config.pet_loss.pet_lesion_similarity_mask_downsample,
            padding_mode=padding_mode)
    if pet_similarity_mask is not None:
        loss_pet = capped_pet_mse_loss(
            warped_pet=prediction.warped_pet,
            fixed_pet=data.fixed_pet,
            valid_mask=pet_similarity_mask,
            upper_quantile=config.pet_loss.pet_loss_upper_quantile,
            quantile_downsample=config.pet_loss.pet_loss_quantile_downsample)
    else:
        loss_pet = capped_pet_mse_loss(
            warped_pet=prediction.warped_pet,
            fixed_pet=data.fixed_pet,
            valid_mask=None,
            upper_quantile=config.pet_loss.pet_loss_upper_quantile,
            quantile_downsample=config.pet_loss.pet_loss_quantile_downsample)
    adaptive_pet_weight = torch.ones((),
                                     device=device,
                                     dtype=data.fixed_ct.dtype)
    pet_hot_correspondence = float("nan")
    if config.pet_loss.adaptive_pet_loss and (
            config.similarity_loss.lambda_pet > 0.0 or
            config.similarity_loss.lambda_mind_pet > 0.0 or
            config.similarity_loss.lambda_multiscale_corr_pet > 0.0 or
        (weights.lambda_pet_hot_region > 0.0)):
        (adaptive_pet_weight,
         pet_hot_correspondence) = adaptive_pet_correspondence_weight(
             fixed_pet=data.fixed_pet,
             warped_pet=prediction.warped_pet,
             valid_mask=pet_similarity_mask,
             settings=config.pet_loss)
    loss_pet_hot_region = torch.zeros((), device=device)
    if weights.lambda_pet_hot_region > 0.0:
        loss_pet_hot_region = pet_hot_region_mse_loss(
            fixed_pet=data.fixed_pet,
            warped_pet=prediction.warped_pet,
            valid_mask=pet_similarity_mask,
            quantile=config.pet_loss.pet_hot_quantile,
            upper_quantile=config.pet_loss.pet_hot_upper_quantile,
            warped_weight=config.pet_loss.pet_hot_warped_weight,
            smooth_kernel=config.pet_loss.pet_hot_smooth_kernel)
    (loss_pet_lesion_occupancy, loss_pet_lesion_translation,
     loss_pet_lesion_strain,
     pet_lesion_occupancy_components) = _compute_pet_lesion_losses(
         data, prediction, config, weights, loss_mask, device)
    return PetResults(
        similarity_mask=pet_similarity_mask,
        similarity_removed_fraction=pet_lesion_similarity_removed_fraction,
        loss_pet=loss_pet,
        adaptive_weight=adaptive_pet_weight,
        hot_correspondence=pet_hot_correspondence,
        loss_hot_region=loss_pet_hot_region,
        loss_lesion_occupancy=loss_pet_lesion_occupancy,
        loss_lesion_translation=loss_pet_lesion_translation,
        loss_lesion_strain=loss_pet_lesion_strain,
        lesion_occupancy_components=pet_lesion_occupancy_components,
    )


@dataclass(frozen=True)
class DescriptorResults:
    """MIND and multiscale-correlation similarity losses for CT and PET."""

    mind_ct: torch.Tensor
    mind_pet: torch.Tensor
    multiscale_corr_ct: torch.Tensor
    multiscale_corr_pet: torch.Tensor


def _compute_descriptor_results(
    data: BatchData,
    prediction: RegistrationState,
    pet_results: PetResults,
    config: FineTuningConfig,
    loss_mask: torch.Tensor | None,
    device: torch.device,
) -> DescriptorResults:
    """Compute optional MIND and multiscale correlation losses.

    Args:
        data: Images of the pair.
        prediction: Warped moving images.
        pet_results: Provides the PET similarity mask excluding lesions.
        config: Validated training configuration.
        loss_mask: Optional valid-overlap mask used for the CT terms.
        device: Device on which disabled terms are created.

    Returns:
        The four descriptor losses; disabled terms are zero.
    """
    similarity = config.similarity_loss
    factor = similarity.mind_loss_downsample
    pet_mask = pet_results.similarity_mask
    zero = torch.zeros((), device=device)

    def pooled(image: torch.Tensor) -> torch.Tensor:
        """Average-pool an image by the MIND downsampling factor."""
        if factor <= 1:
            return image
        return F.avg_pool3d(image.float(),
                            kernel_size=factor,
                            stride=factor,
                            ceil_mode=True)

    def pooled_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Pool a mask and keep it in ``[0, 1]``."""
        if mask is None or factor <= 1:
            return mask
        return pooled(mask).clamp(0.0, 1.0)

    def mind_loss(warped: torch.Tensor, fixed: torch.Tensor,
                  mask: torch.Tensor | None) -> torch.Tensor:
        """Compute the MIND-SSC loss with the configured descriptor.

        Args:
            warped: Warped moving image.
            fixed: Fixed image.
            mask: Optional validity mask.

        Returns:
            MIND-SSC loss of the pair.
        """
        return mind_ssc_loss(warped,
                             fixed,
                             mask=mask,
                             radius=similarity.mind_radius,
                             dilation=similarity.mind_dilation)

    mind_fixed_ct, mind_warped_ct = pooled(data.fixed_ct), pooled(
        prediction.warped_ct)
    mind_fixed_pet, mind_warped_pet = pooled(data.fixed_pet), pooled(
        prediction.warped_pet)
    mind_ct_mask, mind_pet_mask = pooled_mask(loss_mask), pooled_mask(pet_mask)
    return DescriptorResults(
        mind_ct=mind_loss(mind_warped_ct, mind_fixed_ct, mind_ct_mask)
        if similarity.lambda_mind_ct > 0.0 else zero,
        mind_pet=mind_loss(mind_warped_pet, mind_fixed_pet, mind_pet_mask)
        if similarity.lambda_mind_pet > 0.0 else zero,
        multiscale_corr_ct=multiscale_correlation_loss(
            prediction.warped_ct,
            data.fixed_ct,
            mask=loss_mask,
            scales=similarity.multiscale_corr_scales)
        if similarity.lambda_multiscale_corr_ct > 0.0 else zero,
        multiscale_corr_pet=multiscale_correlation_loss(
            prediction.warped_pet,
            data.fixed_pet,
            mask=pet_mask,
            scales=similarity.multiscale_corr_scales)
        if similarity.lambda_multiscale_corr_pet > 0.0 else zero,
    )


@dataclass(frozen=True)
class BatchLossSummary:
    """Combined objective and the losses needed for epoch logging."""

    loss: torch.Tensor
    loss_ct: torch.Tensor
    loss_smooth: torch.Tensor
    loss_overlap: torch.Tensor
    loss_macro_ct_labels: torch.Tensor


@dataclass(frozen=True)
class BatchObjectiveInputs:
    """Intermediate outputs consumed together by the batch objective."""

    data: BatchData
    prediction: RegistrationState
    ct_labels: CtLabelResults
    pet_results: PetResults
    descriptors: DescriptorResults


def _compute_batch_objective(
    inputs: BatchObjectiveInputs,
    config: FineTuningConfig,
    weights: EpochWeights,
    device: torch.device,
    loss_overlap: torch.Tensor,
    loss_ct: torch.Tensor,
) -> BatchLossSummary:
    """Combine CT, PET, anatomy, overlap, and smoothness terms."""
    data = inputs.data
    prediction = inputs.prediction
    ct_labels = inputs.ct_labels
    pet_results = inputs.pet_results
    descriptors = inputs.descriptors
    loss_smooth = displacement_smoothness_loss(
        prediction.stage1_displacement,
        downsample_factor=config.regularization.smoothness_downsample,
    ) + 0.5 * displacement_smoothness_loss(
        prediction.residual_displacement,
        downsample_factor=config.regularization.smoothness_downsample,
    )

    loss_macro_ct_labels = torch.zeros((), device=device)
    macro_groups = macro_ct_label_groups(
        config.anatomical_loss.macro_ct_label_preset)
    if (weights.lambda_macro_ct_labels > 0.0 and macro_groups is not None and
            data.fixed_ct_label is not None and
            data.moving_ct_label is not None):
        fixed_label, moving_label, displacement = (
            downsample_labels_and_displacement(
                data.fixed_ct_label, data.moving_ct_label,
                prediction.displacement,
                config.anatomical_loss.macro_ct_label_downsample))
        loss_macro_ct_labels = macro_label_dice_loss(fixed_label=fixed_label,
                                                     moving_label=moving_label,
                                                     displacement=displacement,
                                                     groups=macro_groups,
                                                     padding_mode="zeros")

    ct_anatomy_alignment_loss = (
        loss_ct +
        config.anatomical_loss.lambda_ct_dice * ct_labels.loss_ct_dice +
        weights.lambda_macro_ct_labels * loss_macro_ct_labels)
    ct_robust_alignment_loss = (
        ct_anatomy_alignment_loss +
        config.similarity_loss.lambda_mind_ct * descriptors.mind_ct +
        config.similarity_loss.lambda_multiscale_corr_ct *
        descriptors.multiscale_corr_ct)
    loss = (ct_robust_alignment_loss + pet_results.adaptive_weight *
            config.similarity_loss.lambda_pet * pet_results.loss_pet +
            pet_results.adaptive_weight *
            config.similarity_loss.lambda_mind_pet * descriptors.mind_pet +
            pet_results.adaptive_weight *
            config.similarity_loss.lambda_multiscale_corr_pet *
            descriptors.multiscale_corr_pet + pet_results.adaptive_weight *
            weights.lambda_pet_hot_region * pet_results.loss_hot_region +
            weights.lambda_pet_lesion_occupancy *
            pet_results.loss_lesion_occupancy +
            weights.lambda_pet_lesion_translation *
            pet_results.loss_lesion_translation +
            weights.lambda_pet_lesion_strain * pet_results.loss_lesion_strain +
            config.regularization.lambda_smooth * loss_smooth +
            config.regularization.lambda_overlap * loss_overlap)
    return BatchLossSummary(
        loss=loss,
        loss_ct=loss_ct,
        loss_smooth=loss_smooth,
        loss_overlap=loss_overlap,
        loss_macro_ct_labels=loss_macro_ct_labels,
    )


def compute_batch_metrics(
    model: nn.Module,
    batch: TensorBatch,
    device: torch.device,
    config: FineTuningConfig,
    weights: EpochWeights,
    apply_training_augmentation: bool = False,
    compute_challenge_metrics: bool = False,
) -> tuple[torch.Tensor, Metrics]:
    """Compute the final objective and all logged batch metrics.

    Args:
        model: Registration model.
        batch: Paired PET/CT tensors and optional labels.
        device: Device receiving the batch.
        config: Validated, immutable training configuration.
        weights: Loss weights after the epoch ramps.
        apply_training_augmentation: Whether to augment the moving images.
        compute_challenge_metrics: Whether to calculate challenge metrics.

    Returns:
        Differentiable objective and a dictionary of scalar metrics.
    """
    data = prepare_batch(batch, device, config, apply_training_augmentation)
    prediction = predict_registration(model, batch, data, device, config)
    loss_overlap = valid_overlap_hinge_loss(
        prediction.overlap, prediction.reference_overlap,
        config.regularization.minimum_valid_overlap)
    ct_labels = _compute_ct_label_results(data, prediction, config, device,
                                          compute_challenge_metrics)
    loss_ct = masked_lncc_loss(prediction.warped_ct, data.fixed_ct,
                               prediction.overlap)
    pet_results = _compute_pet_results(data, prediction, config, weights,
                                       prediction.overlap, device)
    descriptors = _compute_descriptor_results(data, prediction, pet_results,
                                              config, prediction.overlap,
                                              device)
    inputs = BatchObjectiveInputs(data, prediction, ct_labels, pet_results,
                                  descriptors)
    losses = _compute_batch_objective(inputs, config, weights, device,
                                      loss_overlap, loss_ct)
    metrics = summarize_batch_metrics(inputs, losses, config, weights)
    return losses.loss, metrics
