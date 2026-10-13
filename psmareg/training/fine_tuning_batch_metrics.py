"""Ordered fine-tuning metric schema and batch diagnostic summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from psmareg.config.fine_tuning_config import FineTuningConfig
from psmareg.models.common.losses import image_mse_loss, lncc_loss
from psmareg.models.fine_tuning.losses import downsample_image_for_metric

if TYPE_CHECKING:
    from psmareg.training.fine_tuning_batch import (
        BatchLossSummary,
        BatchObjectiveInputs,
        EpochWeights,
    )

Metrics = dict[str, float]

# The historical CSV column order is part of the resume contract.
METRIC_KEYS = (
    "loss",
    "loss_ct",
    "loss_pet",
    "loss_mind_ct",
    "loss_mind_pet",
    "loss_multiscale_corr_ct",
    "loss_multiscale_corr_pet",
    "loss_pet_hot_region",
    "loss_pet_lesion_occupancy",
    "loss_pet_lesion_translation",
    "loss_pet_lesion_strain",
    "pet_lesion_occupancy_components",
    "pet_lesion_similarity_removed_fraction",
    "mind_ct_weight",
    "mind_pet_weight",
    "multiscale_corr_ct_weight",
    "multiscale_corr_pet_weight",
    "pet_hot_region_weight",
    "pet_lesion_occupancy_weight",
    "pet_lesion_translation_weight",
    "pet_lesion_strain_weight",
    "adaptive_pet_weight",
    "pet_hot_correspondence",
    "loss_ct_dice",
    "ct_dsc",
    "challenge_ct_dsc",
    "challenge_pet_dsc",
    "challenge_mtv_percent_error",
    "challenge_tlg_percent_error",
    "loss_smooth",
    "lncc_before_ct",
    "lncc_after_ct",
    "mse_before_pet",
    "mse_after_pet",
    "mean_disp",
    "max_disp",
    "disp_std",
    "disp_p95",
    "disp_p99",
    "valid_overlap",
    "loss_overlap",
    "body_translation_mean",
    "affine_translation_mean",
    "det_translation_mean",
    "det_translation_max",
    "det_translation_x",
    "det_translation_y",
    "det_translation_z",
    "loss_macro_ct_labels",
    "macro_ct_labels_weight",
)


def _scalar(value: torch.Tensor) -> float:
    """Detach a scalar tensor into a Python float."""
    return float(value.detach().cpu())


def _optional_mean(value: torch.Tensor | None) -> float:
    """Return the mean magnitude, or NaN when the model has no such stage."""
    return float("nan") if value is None else _scalar(value.abs().mean())


def _loss_metrics(inputs: BatchObjectiveInputs,
                  losses: BatchLossSummary) -> Metrics:
    """Collect the logged value of every loss term."""
    pet = inputs.pet_results
    descriptors = inputs.descriptors
    ct_labels = inputs.ct_labels
    return {
        "loss": _scalar(losses.loss),
        "loss_ct": _scalar(losses.loss_ct),
        "loss_pet": _scalar(pet.loss_pet),
        "loss_mind_ct": _scalar(descriptors.mind_ct),
        "loss_mind_pet": _scalar(descriptors.mind_pet),
        "loss_multiscale_corr_ct": _scalar(descriptors.multiscale_corr_ct),
        "loss_multiscale_corr_pet": _scalar(descriptors.multiscale_corr_pet),
        "loss_pet_hot_region": _scalar(pet.loss_hot_region),
        "loss_pet_lesion_occupancy": _scalar(pet.loss_lesion_occupancy),
        "loss_pet_lesion_translation": _scalar(pet.loss_lesion_translation),
        "loss_pet_lesion_strain": _scalar(pet.loss_lesion_strain),
        "loss_ct_dice": ct_labels.loss_ct_dice_log,
        "loss_smooth": _scalar(losses.loss_smooth),
        "loss_overlap": _scalar(losses.loss_overlap),
        "loss_macro_ct_labels": _scalar(losses.loss_macro_ct_labels),
    }


def _weight_metrics(inputs: BatchObjectiveInputs, config: FineTuningConfig,
                    weights: EpochWeights) -> Metrics:
    """Collect the loss weights and PET masking diagnostics of the batch."""
    pet = inputs.pet_results
    similarity = config.similarity_loss
    return {
        "pet_lesion_occupancy_components":
            _scalar(pet.lesion_occupancy_components),
        "pet_lesion_similarity_removed_fraction":
            _scalar(pet.similarity_removed_fraction),
        "mind_ct_weight":
            similarity.lambda_mind_ct,
        "mind_pet_weight":
            similarity.lambda_mind_pet,
        "multiscale_corr_ct_weight":
            similarity.lambda_multiscale_corr_ct,
        "multiscale_corr_pet_weight":
            similarity.lambda_multiscale_corr_pet,
        "pet_hot_region_weight":
            weights.lambda_pet_hot_region,
        "pet_lesion_occupancy_weight":
            weights.lambda_pet_lesion_occupancy,
        "pet_lesion_translation_weight":
            weights.lambda_pet_lesion_translation,
        "pet_lesion_strain_weight":
            weights.lambda_pet_lesion_strain,
        "adaptive_pet_weight":
            _scalar(pet.adaptive_weight),
        "pet_hot_correspondence":
            pet.hot_correspondence,
        "macro_ct_labels_weight":
            weights.lambda_macro_ct_labels,
    }


def _label_metrics(inputs: BatchObjectiveInputs) -> Metrics:
    """Collect label overlap and PET biomarker scores."""
    ct_labels = inputs.ct_labels
    return {
        "ct_dsc":
            ct_labels.ct_dsc_log,
        "challenge_ct_dsc":
            ct_labels.challenge_ct_dsc_log,
        "challenge_pet_dsc":
            ct_labels.challenge_pet_dsc_log,
        "challenge_mtv_percent_error":
            ct_labels.challenge_mtv_percent_error_log,
        "challenge_tlg_percent_error":
            ct_labels.challenge_tlg_percent_error_log,
    }


@torch.no_grad()
def _similarity_metrics(inputs: BatchObjectiveInputs,
                        losses: BatchLossSummary) -> Metrics:
    """Compare image similarity before and after registration."""
    data = inputs.data
    fixed_ct, moving_ct, fixed_pet, moving_pet = (downsample_image_for_metric(
        image.detach(),
        factor=4) for image in (data.fixed_ct, data.moving_ct, data.fixed_pet,
                                data.moving_pet))
    return {
        "lncc_before_ct": _scalar(lncc_loss(moving_ct, fixed_ct)),
        "lncc_after_ct": _scalar(losses.loss_ct),
        "mse_before_pet": _scalar(image_mse_loss(moving_pet, fixed_pet)),
        "mse_after_pet": _scalar(inputs.pet_results.loss_pet),
    }


@torch.no_grad()
def _displacement_metrics(inputs: BatchObjectiveInputs) -> Metrics:
    """Summarize the displacement field and the pre-alignment translation."""
    prediction = inputs.prediction
    sample = prediction.displacement.detach()[:, :, ::8, ::8, ::8]
    magnitudes = sample.abs().reshape(-1)
    translation = prediction.prealign_translation
    affine_translation = (None if prediction.affine_parameters is None else
                          prediction.affine_parameters[:, :3])
    return {
        "mean_disp": _scalar(sample.abs().mean()),
        "max_disp": _scalar(sample.abs().max()),
        "disp_std": _scalar(sample.std()),
        "disp_p95": _scalar(torch.quantile(magnitudes, 0.95)),
        "disp_p99": _scalar(torch.quantile(magnitudes, 0.99)),
        "valid_overlap": _scalar(prediction.overlap.mean()),
        "body_translation_mean": _optional_mean(prediction.body_translation),
        "affine_translation_mean": _optional_mean(affine_translation),
        "det_translation_mean": _scalar(translation.abs().mean()),
        "det_translation_max": _scalar(translation.abs().max()),
        "det_translation_x": _scalar(translation[:, 0].mean()),
        "det_translation_y": _scalar(translation[:, 1].mean()),
        "det_translation_z": _scalar(translation[:, 2].mean()),
    }


def summarize_batch_metrics(
    inputs: BatchObjectiveInputs,
    losses: BatchLossSummary,
    config: FineTuningConfig,
    weights: EpochWeights,
) -> Metrics:
    """Collect detached scalar losses, challenge scores, and diagnostics.

    Args:
        inputs: Prepared images, prediction, and intermediate loss results.
        losses: Combined objective and shared regularization terms.
        config: Validated fine-tuning configuration.
        weights: Loss weights after the current epoch's ramps.

    Returns:
        Metrics in the stable ``METRIC_KEYS`` order.

    Raises:
        RuntimeError: If the collected metrics do not cover the CSV schema.
    """
    values = {
        **_loss_metrics(inputs, losses),
        **_weight_metrics(inputs, config, weights),
        **_label_metrics(inputs),
        **_similarity_metrics(inputs, losses),
        **_displacement_metrics(inputs),
    }
    if len(values) != len(METRIC_KEYS):
        raise RuntimeError("Batch metrics no longer match the CSV schema.")
    return {key: values[key] for key in METRIC_KEYS}
