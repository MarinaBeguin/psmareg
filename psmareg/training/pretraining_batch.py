"""Per-batch synthetic pair generation, objectives, and diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from psmareg.config.synthetic_pretraining_config import (
    LossConfig,
    SyntheticPretrainingConfig,
)
from psmareg.models.common.anatomy import macro_ct_label_groups
from psmareg.models.common.losses import (
    image_mse_loss,
    masked_lncc_loss,
    smoothness_loss,
)
from psmareg.data.synthetic_deformations import (
    smooth_coarse_target,)
from psmareg.models.common.registration.displacement import resize_displacement
from psmareg.models.common.registration.jacobian import jacobian_summary
from psmareg.models.common.registration.warping import (
    compose_displacements,
    warp_image_3d,
)
from psmareg.models.pretraining.factory import run_registration_model
from psmareg.models.pretraining.losses import (
    bending_energy,
    jacobian_hinge_loss,
    macro_ct_label_dice_loss,
    multiscale_field_loss,
    pet_biomarker_preservation_loss,
    robust_endpoint_loss,
    sampled_semantic_dice_loss,
    stable_lncc_loss,
)

from psmareg.training.pretraining_pair import (
    BatchExecutionOptions,
    Prediction,
    SimilarityImages,
    SyntheticPair,
    resolve_padding_mode,
    predict_pair,
    similarity_images,
    synthesize_pair,
)

Batch = dict[str, torch.Tensor]
Metrics = dict[str, float]
Losses = dict[str, torch.Tensor]
# Loss terms and the ``LossConfig`` field weighting each of them, in the order
# in which they are summed and logged.
_LOSS_TERMS = (
    ("ct", "image_similarity_weight"),
    ("pet", "lambda_pet"),
    ("field", "lambda_field"),
    ("stage1_field", "lambda_stage1_field"),
    ("affine_field", "lambda_affine_field"),
    ("residual_field", "lambda_residual_field"),
    ("smooth", "lambda_smooth"),
    ("epe", "lambda_epe"),
    ("multiscale_field", "lambda_multiscale_field"),
    ("field_gradient", "lambda_field_gradient"),
    ("jacobian", "lambda_jacobian"),
    ("residual_bending", "lambda_residual_bending"),
    ("global_residual_penalty", "lambda_global_residual_penalty"),
    ("ct_label_dice", "lambda_ct_label_dice"),
    ("pet_label_dice", "lambda_pet_label_dice"),
    ("pet_biomarker", "lambda_pet_biomarker"),
    ("macro_ct_label_dice", "lambda_macro_ct_label_dice"),
    ("symmetric_field", "lambda_symmetric_field"),
)
_UNAVAILABLE_JACOBIAN = {
    "ndv_percent": float("nan"),
    "jac_min": float("nan"),
    "jac_p01": float("nan"),
}


def _image_losses(
    pair: SyntheticPair,
    prediction: Prediction,
    images: SimilarityImages,
    config: SyntheticPretrainingConfig,
) -> Losses:
    """Compute the CT and PET image-similarity losses."""
    if config.architecture.zero_padding:
        overlap = F.interpolate(prediction.valid_overlap,
                                size=images.fixed_ct.shape[-3:],
                                mode="trilinear",
                                align_corners=True)
        with torch.amp.autocast(prediction.warped_ct.device.type,
                                enabled=False):
            loss_ct = masked_lncc_loss(images.warped_ct.float(),
                                       images.fixed_ct.float(), overlap.float())
    else:
        loss_ct = stable_lncc_loss(images.warped_ct, images.fixed_ct)

    foreground_weight = config.loss.pet_foreground_weight
    if pair.moving_pet_label is not None and foreground_weight > 0.0:
        label_shape = images.fixed_pet.shape[-3:]
        fixed_pet_label = warp_image_3d(
            F.interpolate(pair.moving_pet_label.float(),
                          size=label_shape,
                          mode="nearest"),
            resize_displacement(pair.target_displacement, label_shape),
            interpolation_mode="nearest",
        )
        loss_pet = ((images.warped_pet - images.fixed_pet).square() *
                    (1.0 + foreground_weight *
                     (fixed_pet_label > 0).float())).mean()
    else:
        loss_pet = image_mse_loss(images.warped_pet, images.fixed_pet)
    return {"ct": loss_ct, "pet": loss_pet}


def _global_residual_penalty(
    residual: torch.Tensor,
    target_affine_mean_abs: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize low-frequency residual motion when the target is affine.

    This discourages the residual stage from absorbing global motion when the
    target contains a large affine component.

    Args:
        residual: Predicted residual displacement.
        target_affine_mean_abs: Mean magnitude of the target affine field.
        loss_config: Threshold, scale, and downsampling of the penalty.

    Returns:
        Weighted penalty, its detached weight, and the mean magnitude of the
        low-frequency residual.
    """
    residual_low = residual
    factor = loss_config.global_residual_downsample_factor
    if factor > 1:
        residual_low = F.interpolate(residual_low,
                                     size=tuple(
                                         max(2, math.ceil(size / factor))
                                         for size in residual_low.shape[-3:]),
                                     mode="trilinear",
                                     align_corners=False)
    weight = ((target_affine_mean_abs -
               loss_config.global_residual_target_threshold) /
              max(loss_config.global_residual_target_scale, 1e-6)).clamp(
                  0.0, 1.0).detach()
    lowfreq_mean_abs = residual_low.abs().mean()
    return weight * residual_low.square().mean(), weight, lowfreq_mean_abs


def _field_losses(
    pair: SyntheticPair,
    prediction: Prediction,
    config: SyntheticPretrainingConfig,
) -> tuple[Losses, Losses]:
    """Supervise the predicted fields with the known target deformation.

    Args:
        pair: Synthetic pair and its target deformation.
        prediction: Model outputs.
        config: Synthetic pretraining configuration.

    Returns:
        Field-supervision losses and detached diagnostics of the global
        residual penalty.
    """
    loss_config = config.loss
    predicted, target = prediction.displacement, pair.target_displacement
    zero = torch.zeros((), device=predicted.device)
    losses = {
        "field":
            F.mse_loss(predicted, target),
        "epe":
            robust_endpoint_loss(predicted, target),
        "multiscale_field":
            multiscale_field_loss(predicted, target),
        "field_gradient":
            smoothness_loss(predicted - target),
        "jacobian":
            jacobian_hinge_loss(
                predicted,
                margin=loss_config.jacobian_margin,
                sampling_stride=loss_config.jacobian_loss_stride),
        "stage1_field":
            zero,
        "affine_field":
            zero,
        "residual_field":
            zero,
        "residual_bending":
            zero,
        "global_residual_penalty":
            zero,
    }
    diagnostics = {
        "global_residual_weight": zero,
        "target_affine_mean_abs": zero,
        "pred_residual_lowfreq_mean_abs": zero,
    }
    if prediction.stage1 is not None:
        losses["stage1_field"] = F.mse_loss(
            prediction.stage1,
            smooth_coarse_target(
                target,
                downsample_factor=(
                    config.deformation.coarse_target_downsample_factor)))
    if prediction.affine is not None:
        # Smooth-L1 keeps rare 30-voxel translations from overwhelming every
        # deformable and image-similarity objective early in training.
        losses["affine_field"] = F.smooth_l1_loss(prediction.affine,
                                                  pair.target_affine,
                                                  beta=1.0)
        diagnostics["target_affine_mean_abs"] = pair.target_affine.abs().mean()
    if prediction.residual is None or prediction.stage1 is None:
        return losses, diagnostics

    losses["residual_field"] = F.mse_loss(
        compose_displacements(first=prediction.stage1.detach(),
                              second=prediction.residual), target)
    losses["residual_bending"] = bending_energy(prediction.residual)
    if (loss_config.lambda_global_residual_penalty > 0.0 and
            prediction.affine is not None):
        penalty, weight, lowfreq_mean_abs = _global_residual_penalty(
            prediction.residual, diagnostics["target_affine_mean_abs"],
            loss_config)
        diagnostics["pred_residual_lowfreq_mean_abs"] = lowfreq_mean_abs
        diagnostics["global_residual_weight"] = weight
        losses["global_residual_penalty"] = penalty
    return losses, diagnostics


def _label_losses(
    pair: SyntheticPair,
    prediction: Prediction,
    config: SyntheticPretrainingConfig,
) -> Losses:
    """Compute the label-overlap and PET-biomarker losses that are enabled."""
    loss_config = config.loss
    zero = torch.zeros((), device=prediction.displacement.device)
    losses = dict.fromkeys(("ct_label_dice", "pet_label_dice", "pet_biomarker",
                            "macro_ct_label_dice"), zero)
    if (pair.moving_ct_label is not None and
            loss_config.lambda_ct_label_dice > 0.0):
        losses["ct_label_dice"] = sampled_semantic_dice_loss(
            moving_label=pair.moving_ct_label,
            downsample_factor=loss_config.label_loss_downsample_factor,
            max_classes=loss_config.max_ct_label_classes_per_batch,
            target_displacement=pair.target_displacement,
            predicted_displacement=prediction.displacement)
    if (pair.moving_pet_label is not None and
            loss_config.lambda_pet_label_dice > 0.0):
        losses["pet_label_dice"] = sampled_semantic_dice_loss(
            moving_label=pair.moving_pet_label,
            downsample_factor=loss_config.label_loss_downsample_factor,
            max_classes=1,
            binary=True,
            target_displacement=pair.target_displacement,
            predicted_displacement=prediction.displacement)
    if (pair.moving_pet_label is not None and
            loss_config.lambda_pet_biomarker > 0.0):
        losses["pet_biomarker"] = pet_biomarker_preservation_loss(
            moving_pet=pair.moving_pet,
            moving_pet_label=pair.moving_pet_label,
            downsample_factor=loss_config.pet_biomarker_downsample_factor,
            padding_mode=resolve_padding_mode(config),
            target_displacement=pair.target_displacement,
            predicted_displacement=prediction.displacement)
    if (pair.moving_ct_label is not None and
            loss_config.lambda_macro_ct_label_dice > 0.0):
        losses["macro_ct_label_dice"] = macro_ct_label_dice_loss(
            moving_label=pair.moving_ct_label,
            groups=macro_ct_label_groups(loss_config.macro_ct_label_preset),
            downsample_factor=loss_config.macro_ct_label_downsample_factor,
            padding_mode=resolve_padding_mode(config),
            target_displacement=pair.target_displacement,
            predicted_displacement=prediction.displacement)
    return losses


def _symmetric_field_loss(
    model: nn.Module,
    pair: SyntheticPair,
    config: SyntheticPretrainingConfig,
) -> tuple[torch.Tensor, bool]:
    """Randomly supervise the backward registration of the pair.

    The negated target is only an approximate inverse, so this term is kept
    weak: the forward field loss remains the main teacher while this term
    discourages one-way shortcuts and unstable asymmetric fields.

    Returns:
        The loss and whether the backward pass was run for this batch.
    """
    device = pair.target_displacement.device
    if not (config.loss.lambda_symmetric_field > 0.0 and
            config.loss.symmetric_probability > 0.0 and
            float(torch.rand(
                (), device=device)) < config.loss.symmetric_probability):
        return torch.zeros((), device=device), False
    backward_displacement = run_registration_model(
        model=model,
        fixed_ct=pair.moving_ct,
        fixed_pet=pair.moving_pet,
        moving_ct=pair.fixed_ct,
        moving_pet=pair.fixed_pet,
    )[0]
    return F.smooth_l1_loss(backward_displacement,
                            -pair.target_displacement,
                            beta=1.0), True


def _smoothness_loss(prediction: Prediction) -> torch.Tensor:
    """Regularize the staged fields, or the final field of a single stage."""
    if prediction.stage1 is not None and prediction.residual is not None:
        return smoothness_loss(
            prediction.stage1) + 0.5 * smoothness_loss(prediction.residual)
    return smoothness_loss(prediction.displacement)


def _weighted_total(losses: Losses,
                    config: SyntheticPretrainingConfig) -> torch.Tensor:
    """Sum the loss terms with their configured weights."""
    weighted = [
        getattr(config.loss, weight_name) * losses[name]
        for name, weight_name in _LOSS_TERMS
    ]
    total = weighted[0]
    for term in weighted[1:]:
        total = total + term
    return total


@dataclass(frozen=True)
class _SyntheticMetricInputs:
    """Batch losses and predictions needed for synthetic metric logging."""

    loss: torch.Tensor
    losses: Losses
    diagnostics: Losses
    pair: SyntheticPair
    prediction: Prediction
    images: SimilarityImages
    symmetric_applied: bool


def _scalar(value: torch.Tensor) -> float:
    """Detach a scalar tensor into a Python float."""
    return float(value.detach().cpu())


def _loss_metrics(inputs: _SyntheticMetricInputs,
                  config: SyntheticPretrainingConfig) -> Metrics:
    """Collect the total, every loss term, and the penalty diagnostics."""
    metrics = {
        "loss": _scalar(inputs.loss),
        "image_similarity_weight": float(config.loss.image_similarity_weight),
    }
    for name, _ in _LOSS_TERMS:
        metrics[f"loss_{name}"] = _scalar(inputs.losses[name])
        if name == "global_residual_penalty":
            metrics.update({
                key: _scalar(value) for key, value in inputs.diagnostics.items()
            })
    return metrics


def _pair_metrics(inputs: _SyntheticMetricInputs,
                  config: SyntheticPretrainingConfig,
                  collect_before: bool) -> Metrics:
    """Collect augmentation flags and before/after image similarity."""
    pair, images = inputs.pair, inputs.images
    lncc_ct_before = mse_pet_before = float("nan")
    if collect_before:
        if images.moving_ct is None or images.moving_pet is None:
            raise ValueError(
                "Expensive metrics require the resized moving images.")
        lncc_ct_before = _scalar(
            stable_lncc_loss(images.moving_ct, images.fixed_ct))
        mse_pet_before = _scalar(
            image_mse_loss(images.moving_pet, images.fixed_pet))
    return {
        "symmetric_applied": float(inputs.symmetric_applied),
        "pet_intensity_augmented": float(pair.pet_intensity_augmented),
        "synthmorph_like": float(config.augmentation.synthmorph_like),
        "synthmorph_fov_augmented": float(pair.synthmorph_fov_augmented),
        "pose_aware_applied": float(pair.pose_aware_applied),
        "mse_pet_before": mse_pet_before,
        "mse_pet_after": _scalar(inputs.losses["pet"]),
        "lncc_ct_before": lncc_ct_before,
        "lncc_ct_after": _scalar(inputs.losses["ct"]),
    }


def _displacement_metrics(predicted: torch.Tensor, target: torch.Tensor,
                          options: BatchExecutionOptions) -> Metrics:
    """Compare the predicted and target displacement magnitudes."""
    stride = options.metric_sampling_stride
    predicted_sample = predicted[:, :, ::stride, ::stride, ::stride]
    target_sample = target[:, :, ::stride, ::stride, ::stride]

    def p95(field: torch.Tensor) -> float:
        """Return the 95th percentile of a subsampled field magnitude."""
        return _scalar(
            torch.quantile(field.abs()[:, :, ::8, ::8, ::8].flatten(), 0.95))

    return {
        "endpoint_error":
            _scalar(
                torch.linalg.vector_norm(predicted_sample - target_sample,
                                         dim=1).mean()),
        "target_endpoint":
            _scalar(torch.linalg.vector_norm(target_sample, dim=1).mean()),
        "pred_mean_abs":
            _scalar(predicted.abs().mean()),
        "target_mean_abs":
            _scalar(target.abs().mean()),
        "pred_p95":
            p95(predicted),
        "target_p95":
            p95(target),
        "deformation_scale":
            options.deformation_scale,
    }


@torch.no_grad()
def _summarize_metrics(
    inputs: _SyntheticMetricInputs,
    config: SyntheticPretrainingConfig,
    options: BatchExecutionOptions,
) -> Metrics:
    """Collect detached scalar losses and displacement diagnostics."""
    predicted = inputs.prediction.displacement
    target = inputs.pair.target_displacement
    metrics = {
        **_loss_metrics(inputs, config),
        **_pair_metrics(inputs, config, options.collect_expensive_metrics),
        **_displacement_metrics(predicted, target, options),
    }
    predicted_jacobian = target_jacobian = _UNAVAILABLE_JACOBIAN
    if options.collect_expensive_metrics:
        predicted_jacobian = jacobian_summary(predicted)
        target_jacobian = jacobian_summary(target)
    for prefix, summary in (("pred", predicted_jacobian), ("target",
                                                           target_jacobian)):
        for key, value in summary.items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def compute_batch_loss_and_metrics(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
    config: SyntheticPretrainingConfig,
    options: BatchExecutionOptions,
) -> tuple[torch.Tensor, Metrics]:
    """Synthesize a pair, register it, and compute the objective.

    Args:
        model: Registration model.
        batch: Single-scan CT/PET tensors and optional labels.
        device: Device receiving the batch.
        config: Validated synthetic pretraining configuration.
        options: Per-pass execution settings.

    Returns:
        Differentiable objective and a dictionary of scalar metrics.
    """
    pair = synthesize_pair(batch, device, config, options)
    prediction = predict_pair(model, pair, resolve_padding_mode(config))
    images = similarity_images(pair, prediction, options)

    losses = _image_losses(pair, prediction, images, config)
    field_losses, diagnostics = _field_losses(pair, prediction, config)
    losses.update(field_losses)
    losses.update(_label_losses(pair, prediction, config))
    losses["symmetric_field"], symmetric_applied = _symmetric_field_loss(
        model, pair, config)
    losses["smooth"] = _smoothness_loss(prediction)
    loss = _weighted_total(losses, config)

    inputs = _SyntheticMetricInputs(loss, losses, diagnostics, pair, prediction,
                                    images, symmetric_applied)
    metrics = _summarize_metrics(inputs, config, options)
    return loss, metrics
