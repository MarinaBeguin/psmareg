"""Fine-tuning steps, checkpointing, and metric aggregation for PSMAReg."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from psmareg.config.fine_tuning_config import FineTuningConfig
from psmareg.data.manifests import read_manifest
from psmareg.data.paired_dataset import PSMARegPairDataset
from psmareg.training.checkpoint_state import checkpoint_metadata
from psmareg.training.metrics_csv import format_metric_line
from psmareg.training.fine_tuning_batch import (
    EpochWeights,
    Metrics,
    TensorBatch,
    compute_batch_metrics,
)

LOGGER = logging.getLogger(__name__)
_OPTIONAL_LABEL_SCORES = frozenset({
    "ct_dsc",
    "challenge_ct_dsc",
    "challenge_pet_dsc",
    "challenge_mtv_percent_error",
    "challenge_tlg_percent_error",
})


def _average_metrics(metrics_list: list[Metrics]) -> Metrics:
    """Average metrics, ignoring missing observations for non-loss scores."""
    if not metrics_list:
        raise ValueError("Cannot average an empty metrics list.")
    averaged: Metrics = {}
    keys = metrics_list[0].keys()
    for key in keys:
        values = [metrics[key] for metrics in metrics_list]
        if key in _OPTIONAL_LABEL_SCORES:
            values = [value for value in values if math.isfinite(value)]
        averaged[key] = (sum(values) / len(values) if values else float("nan"))
    return averaged


def checkpoint_config(
        config: FineTuningConfig) -> dict[str, str | int | float | bool]:
    """Convert a typed configuration into historical checkpoint metadata.

    Args:
        config: Validated fine-tuning configuration.

    Returns:
        Flat serializable mapping, including the stable model identifier.
    """
    return checkpoint_metadata(config,
                               comma_joined=("multiscale_corr_scales",),
                               extra={"model": "affine_transmorph_v2_cascade"})


def build_loaders(config: FineTuningConfig) -> tuple[DataLoader, DataLoader]:
    """Create paired-data loaders with optional patient-balanced sampling.

    Args:
        config: Validated fine-tuning configuration.

    Returns:
        Training loader, shuffled or patient-balanced, and validation loader.
    """
    train_dataset = PSMARegPairDataset(
        config.data.train_manifest,
        cache_dir=config.data.paired_cache_dir,
        cache_float16=config.data.paired_cache_float16)
    val_dataset = PSMARegPairDataset(
        config.data.val_manifest,
        cache_dir=config.data.paired_cache_dir,
        cache_float16=config.data.paired_cache_float16)
    train_sampler = None
    if config.data.patient_balanced_sampling:
        subject_counts: dict[str, int] = {}
        for row in train_dataset.rows:
            subject = row["subject"]
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        sample_weights = [
            1.0 / subject_counts[row["subject"]] for row in train_dataset.rows
        ]
        train_sampler = WeightedRandomSampler(sample_weights,
                                              num_samples=len(train_dataset),
                                              replacement=True)
    loader_options: dict[str, Any] = {
        "batch_size":
            config.optimization.batch_size,
        "num_workers":
            config.runtime.num_workers,
        "pin_memory":
            torch.cuda.is_available(),
        "persistent_workers":
            config.runtime.num_workers > 0
            and not config.runtime.no_persistent_workers,
        "prefetch_factor":
            2 if config.runtime.num_workers > 0 else None,
    }
    train_loader = DataLoader(train_dataset,
                              shuffle=train_sampler is None,
                              sampler=train_sampler,
                              **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader


def validate_validation_labels(manifest_path: Path) -> None:
    """Require at least one paired CT label before training begins.

    Args:
        manifest_path: Validation manifest of the run.

    Raises:
        ValueError: If no pair provides both CT label maps, which model
            selection on the CT Dice needs.
    """
    if any(
            row.get("fixed_ct_label") and row.get("moving_ct_label")
            for row in read_manifest(manifest_path)):
        return
    raise ValueError(
        f"Validation manifest {manifest_path} has no pair with both CT "
        "labels; CT Dice model selection cannot run.")


def load_model_weights(checkpoint_path: Path, model: nn.Module,
                       device: torch.device) -> None:
    """Load a complete model state from a checkpoint.

    Args:
        checkpoint_path: Checkpoint containing ``model_state_dict``.
        model: Registration model receiving the weights.
        device: Device used when loading checkpoint tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def load_compatible_model_weights(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    prefixes: tuple[str, ...] = ()) -> tuple[int, int]:
    """Load shape-compatible checkpoint tensors, optionally by prefix.

    Args:
        checkpoint_path: Source checkpoint path.
        model: Destination registration model.
        device: Device used when loading checkpoint tensors.
        prefixes: Optional accepted state-dictionary key prefixes.

    Returns:
        Number of loaded tensors and number of candidate source tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint["model_state_dict"]
    destination = model.state_dict()
    if prefixes:
        source = {
            key: value
            for key, value in source.items()
            if any((key.startswith(prefix) for prefix in prefixes))
        }
    compatible = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    destination.update(compatible)
    model.load_state_dict(destination, strict=True)
    return (len(compatible), len(source))


def resume_checkpoint(checkpoint_path: Path, model: nn.Module,
                      optimizer: torch.optim.Optimizer,
                      scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau |
                      None, device: torch.device) -> tuple[int, float]:
    """Restore training state and return the next epoch and best score.

    Args:
        checkpoint_path: Checkpoint to resume.
        model: Model receiving stored weights.
        optimizer: Optimizer receiving stored state.
        scheduler: Optional scheduler receiving stored state.
        device: Device used when loading checkpoint tensors.

    Returns:
        Next epoch index and best checkpoint-selection score.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_score = float(checkpoint.get("best_val_score", -float("inf")))
    return (start_epoch, best_val_score)


def run_train_epoch(model: nn.Module, loader: DataLoader[TensorBatch],
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    config: FineTuningConfig, weights: EpochWeights) -> Metrics:
    """Run one optimization epoch with validated configuration.

    Args:
        model: Registration model to train.
        loader: Training batch loader.
        optimizer: Optimizer updated once per batch.
        device: Computation device.
        config: Validated fine-tuning configuration.
        weights: Loss weights after epoch-dependent ramps.

    Returns:
        Mean training metrics over processed batches.
    """
    model.train()
    metrics_list: list[Metrics] = []
    for batch in loader:
        loss, metrics = compute_batch_metrics(
            model=model,
            batch=batch,
            device=device,
            config=config,
            weights=weights,
            apply_training_augmentation=config.augmentation.train_augmentation)
        optimizer.zero_grad()
        loss.backward()
        if config.optimization.gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimization.gradient_clip_norm)
        optimizer.step()
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


@torch.no_grad()
def run_validation_epoch(model: nn.Module, loader: DataLoader[TensorBatch],
                         device: torch.device, config: FineTuningConfig,
                         weights: EpochWeights) -> Metrics:
    """Run one validation epoch without gradient tracking.

    Args:
        model: Registration model to evaluate.
        loader: Validation batch loader.
        device: Computation device.
        config: Validated fine-tuning configuration.
        weights: Loss weights used for this epoch.

    Returns:
        Mean validation metrics over processed batches.
    """
    model.eval()
    metrics_list: list[Metrics] = []
    for batch in loader:
        _, metrics = compute_batch_metrics(
            model=model,
            batch=batch,
            device=device,
            config=config,
            weights=weights,
            compute_challenge_metrics=config.runtime.log_challenge_metrics)
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


_LOG_COLUMNS = (
    ("loss", "loss", "{:.6f}"),
    ("ct", "loss_ct", "{:.6f}"),
    ("pet", "loss_pet", "{:.6f}"),
    ("pet_occ", "loss_pet_lesion_occupancy", "{:.6f}"),
    ("pet_trans", "loss_pet_lesion_translation", "{:.6f}"),
    ("pet_strain", "loss_pet_lesion_strain", "{:.6f}"),
    ("ct_dsc", "ct_dsc", "{:.6f}"),
    ("macro_ct", "loss_macro_ct_labels", "{:.6f}"),
    ("overlap", "valid_overlap", "{:.4f}"),
    ("mean_disp", "mean_disp", "{:.4f}"),
)


def format_metrics(prefix: str, metrics: Metrics) -> str:
    """Format the principal metrics as one compact log line.

    Args:
        prefix: Phase label, such as ``train`` or ``val``.
        metrics: Metric values produced by an epoch.

    Returns:
        Stable, ordered metric summary.
    """
    return format_metric_line(prefix, metrics, _LOG_COLUMNS)
