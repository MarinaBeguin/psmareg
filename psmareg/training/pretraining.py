"""Epoch execution, checkpointing, and metrics for synthetic pretraining."""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from psmareg.config.synthetic_pretraining_config import (
    SyntheticPretrainingConfig,)
from psmareg.data.synthetic_registration_dataset import (
    SingleScanRegistrationDataset,)
from psmareg.data.synthetic_deformations import (
    CurriculumSettings,
    curriculum_deformation_scale,
)
from psmareg.training.checkpoint_state import checkpoint_metadata
from psmareg.training.metrics_csv import (
    append_metrics_row,
    format_metric_line,
)
from psmareg.training.selection import LossSelectionState
from psmareg.training.pretraining_batch import (
    BatchExecutionOptions,
    Batch,
    Metrics,
    compute_batch_loss_and_metrics,
)


def finite_mean(values: list[float]) -> float:
    """Average finite metric values and return NaN when none are finite.

    Args:
        values: Scalar metric values, possibly including NaN or infinity.

    Returns:
        Mean of the finite values, or NaN for an empty finite subset.
    """
    array = np.asarray(values, dtype=np.float32)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def average_metrics(metrics_list: list[Metrics]) -> Metrics:
    """Average each metric key across batches, ignoring non-finite values.

    Args:
        metrics_list: Non-empty sequence of dictionaries with identical keys.

    Returns:
        Per-key arithmetic means or NaN where no finite value exists.

    Raises:
        ValueError: If ``metrics_list`` is empty.
    """
    if not metrics_list:
        raise ValueError("Cannot average an empty metrics list.")

    return {
        key: finite_mean([metrics[key] for metrics in metrics_list])
        for key in metrics_list[0]
    }


def build_loaders(
    config: SyntheticPretrainingConfig,) -> tuple[DataLoader, DataLoader]:
    """Build training and validation loaders with identical label settings.

    Labels are loaded only when an augmentation or a loss needs them.

    Args:
        config: Validated synthetic pretraining configuration.

    Returns:
        Shuffled training loader and validation loader.
    """
    include_labels = (config.augmentation.synthmorph_like or
                      config.augmentation.pose_aware_probability > 0.0 or
                      config.loss.lambda_ct_label_dice > 0.0 or
                      config.loss.lambda_pet_label_dice > 0.0 or
                      config.loss.lambda_macro_ct_label_dice > 0.0 or
                      config.loss.pet_foreground_weight > 0.0)

    train_dataset = SingleScanRegistrationDataset(
        config.data.required("train_manifest"),
        exclusion_manifest=config.data.exclude_subjects_manifest,
        cache_dir=config.data.cache_dir,
        include_labels=include_labels,
    )
    val_dataset = SingleScanRegistrationDataset(
        config.data.required("val_manifest"),
        exclusion_manifest=config.data.exclude_subjects_manifest,
        cache_dir=config.data.cache_dir,
        include_labels=include_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=config.optimization.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.optimization.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.optimization.num_workers > 0,
    )
    return train_loader, val_loader


def autocast_context(device: torch.device,
                     enabled: bool) -> contextlib.AbstractContextManager:
    """Return a CUDA autocast context, or a no-op context elsewhere.

    Args:
        device: Device on which the forward pass runs.
        enabled: Whether mixed precision is requested.

    Returns:
        Context manager wrapping the forward pass.
    """
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda")


def _validation_config(
        config: SyntheticPretrainingConfig) -> SyntheticPretrainingConfig:
    """Disable the stochastic training-only terms for validation."""
    return dataclasses.replace(
        config,
        loss=dataclasses.replace(config.loss,
                                 lambda_symmetric_field=0.0,
                                 symmetric_probability=0.0),
        augmentation=dataclasses.replace(
            config.augmentation, pet_independent_intensity_probability=0.0),
    )


def _validation_rng(
        device: torch.device, config: SyntheticPretrainingConfig
) -> contextlib.AbstractContextManager:
    """Return a context isolating the validation random stream, if requested."""
    if not config.runtime.deterministic_validation:
        return contextlib.nullcontext()
    devices = []
    if device.type == "cuda":
        devices = [
            torch.cuda.current_device()
            if device.index is None else device.index
        ]
    return torch.random.fork_rng(devices=devices)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    config: SyntheticPretrainingConfig,
    epoch: int,
) -> Metrics:
    """Run one training epoch, or one validation epoch without optimizer.

    Validation uses full-severity deformations and, when
    ``deterministic_validation`` is set, a fixed random stream that leaves the
    training stream untouched.

    Args:
        model: Registration model.
        loader: Loader of single-scan batches.
        device: Device on which the model runs.
        optimizer: Optimizer for training; ``None`` selects validation.
        scaler: Optional gradient scaler for mixed precision.
        config: Validated synthetic pretraining configuration.
        epoch: Zero-based epoch index driving the curriculum.

    Returns:
        Metrics averaged over the finite values of all batches.
    """
    runtime = config.runtime
    model.train(mode=optimizer is not None)
    metrics_list: list[Metrics] = []

    if optimizer is None:
        validation_config = _validation_config(config)
        options = BatchExecutionOptions(
            deformation_scale=1.0,
            similarity_downsample_factor=(
                config.deformation.validation_similarity_downsample_factor),
            collect_expensive_metrics=True,
            metric_sampling_stride=runtime.validation_metric_sampling_stride,
        )
        with _validation_rng(device, config), torch.no_grad():
            if runtime.deterministic_validation:
                torch.manual_seed(runtime.seed + 10_000)
            for batch_idx, batch in enumerate(loader):
                if 0 < runtime.max_val_batches <= batch_idx:
                    break
                with autocast_context(device=device, enabled=runtime.amp):
                    _, metrics = compute_batch_loss_and_metrics(
                        model, batch, device, validation_config, options)
                metrics_list.append(metrics)
        return average_metrics(metrics_list)

    total_epochs = config.curriculum.curriculum_total_epochs
    if total_epochs is None:
        total_epochs = config.optimization.epochs
    for batch_idx, batch in enumerate(loader):
        if 0 < runtime.max_train_batches <= batch_idx:
            break
        options = BatchExecutionOptions(
            deformation_scale=curriculum_deformation_scale(
                epoch=epoch,
                total_epochs=total_epochs,
                device=device,
                settings=CurriculumSettings(
                    min_scale=config.curriculum.curriculum_min_scale,
                    warmup_fraction=config.curriculum.
                    curriculum_warmup_fraction,
                    ramp_fraction=config.curriculum.curriculum_ramp_fraction,
                    easy_probability=(
                        config.curriculum.curriculum_easy_probability),
                    extra_hard_probability=(
                        config.curriculum.curriculum_extra_hard_probability),
                    extra_hard_max_scale=(
                        config.curriculum.curriculum_extra_hard_max_scale)),
            ),
            similarity_downsample_factor=(
                config.deformation.similarity_downsample_factor),
            collect_expensive_metrics=runtime.train_expensive_metrics,
            metric_sampling_stride=runtime.metric_sampling_stride,
        )
        with autocast_context(device=device, enabled=runtime.amp):
            loss, metrics = compute_batch_loss_and_metrics(
                model, batch, device, config, options)
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics_list.append(metrics)
    return average_metrics(metrics_list)


def checkpoint_config(
        config: SyntheticPretrainingConfig
) -> dict[str, str | int | float | bool]:
    """Convert serializable parameters into checkpoint metadata.

    Args:
        config: Synthetic pretraining configuration.

    Returns:
        Flat mapping of scalar parameters, with paths converted to strings.
    """
    return checkpoint_metadata(config)


def write_metrics_row(
    path: Path,
    epoch: int,
    train_metrics: Metrics,
    val_metrics: Metrics,
    selection: LossSelectionState,
    epoch_seconds: float,
) -> None:
    """Append one epoch of synthetic training and validation metrics.

    Args:
        path: Metrics CSV path.
        epoch: Completed epoch index.
        train_metrics: Training metric values.
        val_metrics: Validation metric values.
        selection: Best validation loss and patience after this epoch.
        epoch_seconds: Elapsed time for this epoch, in seconds.
    """
    row: dict[str, int | float] = {
        "epoch": epoch,
        "best_val_loss": selection.best_val_loss,
        "epochs_without_improvement": selection.epochs_without_improvement,
        "epoch_seconds": epoch_seconds,
    }

    append_metrics_row(path, row, train_metrics, val_metrics)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    load_optimizer_state: bool = True,
    inherit_best_val_loss: bool = True,
) -> tuple[int, float]:
    """Restore model state and optionally resume the optimizer and best loss.

    Args:
        path: Checkpoint to load.
        model: Model receiving the stored weights.
        optimizer: Optimizer receiving stored state when enabled.
        device: Device used to load checkpoint tensors.
        load_optimizer_state: Whether to restore optimizer state.
        inherit_best_val_loss: Whether to reuse the stored best loss.

    Returns:
        Next epoch index and inherited or reset best validation loss.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer_state and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = (float(checkpoint.get("best_val_loss", float("inf")))
                     if inherit_best_val_loss else float("inf"))
    return start_epoch, best_val_loss


def initialize_matching_weights(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> int:
    """Load shape-compatible weights when adding a new model stage.

    Args:
        path: Source checkpoint path.
        model: Destination model.
        device: Device used when loading checkpoint tensors.

    Returns:
        Number of compatible tensors loaded into the model.
    """
    checkpoint = torch.load(path, map_location=device)
    source = checkpoint["model_state_dict"]
    destination = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


_LOG_COLUMNS = (
    ("loss", "loss", "{:.6f}"),
    ("ct", "loss_ct", "{:.6f}"),
    ("pet", "loss_pet", "{:.6f}"),
    ("field", "loss_field", "{:.6f}"),
    ("stage1_field", "loss_stage1_field", "{:.6f}"),
    ("affine_field", "loss_affine_field", "{:.6f}"),
    ("residual_field", "loss_residual_field", "{:.6f}"),
    ("smooth", "loss_smooth", "{:.6f}"),
    ("robust_epe", "loss_epe", "{:.6f}"),
    ("multiscale", "loss_multiscale_field", "{:.6f}"),
    ("field_gradient", "loss_field_gradient", "{:.6f}"),
    ("jacobian_loss", "loss_jacobian", "{:.6f}"),
    ("residual_bending", "loss_residual_bending", "{:.6f}"),
    ("global_residual", "loss_global_residual_penalty", "{:.6f}"),
    ("global_residual_weight", "global_residual_weight", "{:.3f}"),
    ("target_affine_abs", "target_affine_mean_abs", "{:.4f}"),
    ("residual_lowfreq_abs", "pred_residual_lowfreq_mean_abs", "{:.4f}"),
    ("ct_label_dice_loss", "loss_ct_label_dice", "{:.6f}"),
    ("pet_label_dice_loss", "loss_pet_label_dice", "{:.6f}"),
    ("pet_biomarker_loss", "loss_pet_biomarker", "{:.6f}"),
    ("epe", "endpoint_error", "{:.4f}"),
    ("target_epe", "target_endpoint", "{:.4f}"),
    ("ct_before", "lncc_ct_before", "{:.6f}"),
    ("ct_after", "lncc_ct_after", "{:.6f}"),
    ("pet_before", "mse_pet_before", "{:.6f}"),
    ("pet_after", "mse_pet_after", "{:.6f}"),
    ("pet_int_aug", "pet_intensity_augmented", "{:.3f}"),
    ("pose_aug", "pose_aware_applied", "{:.3f}"),
    ("pred_p95", "pred_p95", "{:.4f}"),
    ("target_p95", "target_p95", "{:.4f}"),
    ("severity", "deformation_scale", "{:.3f}"),
    ("pred_ndv", "pred_ndv_percent", "{:.4f}%"),
    ("pred_jac_p01", "pred_jac_p01", "{:.4f}"),
    ("target_ndv", "target_ndv_percent", "{:.4f}%"),
    ("target_jac_p01", "target_jac_p01", "{:.4f}"),
)


def format_metrics(prefix: str, metrics: Metrics) -> str:
    """Format synthetic pretraining metrics for a human-readable log line.

    Args:
        prefix: Phase label, such as ``train`` or ``val``.
        metrics: Metric values produced by an epoch.

    Returns:
        Stable, ordered metric summary.
    """
    return format_metric_line(prefix, metrics, _LOG_COLUMNS)
