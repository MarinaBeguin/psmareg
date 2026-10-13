"""Train and validate the final PSMAReg lesion-aware registration model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from psmareg.config.sections import (
    add_set_argument,
    named_overrides,
    set_overrides,
)
from psmareg.config.fine_tuning_config import (
    FineTuningConfig,
    load_fine_tuning_config,
)

from psmareg.models.common.anatomy import macro_ct_label_groups
from psmareg.models.common.registration.cascades import (
    AffineCascadeSettings,
    AffineTransMorphV2CascadeRegistration,
)
from psmareg.training.fine_tuning import (
    build_loaders,
    validate_validation_labels,
    checkpoint_config,
    format_metrics,
    load_compatible_model_weights,
    load_model_weights,
    resume_checkpoint,
    run_train_epoch,
    run_validation_epoch,
    EpochWeights,
    TensorBatch,
)
from psmareg.training.checkpoint_state import (
    CheckpointState,
    save_checkpoint,
)
from psmareg.training.epoch_recovery import (
    LATEST_CHECKPOINT,
    PENDING_CHECKPOINT,
    recover_interrupted_epoch,
)
from psmareg.training.metrics_writer import FineTuningMetricsWriter
from psmareg.training.selection import SelectionState


def parse_fine_tuning_config(argv: list[str] | None = None) -> FineTuningConfig:
    """Parse the command line and load the selected training configuration.

    Args:
        argv: Optional argument vector. Uses ``sys.argv`` when omitted.

    Returns:
        Validated training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train PSMAReg from a structured Python configuration.")
    parser.add_argument("--config",
                        type=Path,
                        default=Path("psmareg/config/fine_tuning_final.py"))
    parser.add_argument("--train_manifest", type=Path)
    parser.add_argument("--val_manifest", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--resume_from", type=Path)
    parser.add_argument("--epochs", type=int)
    add_set_argument(
        parser,
        "Override a flattened training argument for a controlled diagnostic.")
    cli = parser.parse_args(argv)
    overrides = named_overrides(cli, ("train_manifest", "val_manifest",
                                      "output_dir", "resume_from", "epochs"))
    if cli.resume_from is not None:
        overrides["pretrained_checkpoint"] = None
        overrides["compatible_pretrained_checkpoint"] = None
    overrides.update(set_overrides(parser, cli.set))
    return load_fine_tuning_config(cli.config, overrides)


def _ramped_weight(weight: float, epoch: int, ramp_epochs: int) -> float:
    """Apply a linear epoch ramp without changing the configured weight."""
    progress = min((epoch + 1) / ramp_epochs, 1.0) if ramp_epochs > 0 else 1.0
    return weight * progress


def _build_optimizer(
    model: AffineTransMorphV2CascadeRegistration, config: FineTuningConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau |
           None]:
    """Create the AdamW optimizer and the optional plateau scheduler.

    Args:
        model: Registration model to optimize.
        config: Validated training configuration.

    Returns:
        Optimizer, and scheduler when ``lr_scheduler_patience`` is positive.
    """
    optimization = config.optimization
    affine_parameters = list(model.affine.parameters())
    affine_ids = {id(parameter) for parameter in affine_parameters}
    deformable_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in affine_ids
    ]
    # Both groups share one learning rate; they are kept separate so that
    # optimizer states of existing checkpoints can still be resumed.
    optimizer = torch.optim.AdamW([
        {
            "params": deformable_parameters,
            "lr": optimization.lr
        },
        {
            "params": affine_parameters,
            "lr": optimization.lr
        },
    ])
    scheduler = None
    if optimization.lr_scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=optimization.lr_scheduler_factor,
            patience=optimization.lr_scheduler_patience,
            min_lr=optimization.min_lr)
    return optimizer, scheduler


def _initialize_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: FineTuningConfig,
    device: torch.device,
) -> tuple[int, float]:
    """Load pretrained weights or resume an interrupted run.

    Args:
        model: Registration model receiving the weights.
        optimizer: Optimizer restored when resuming.
        scheduler: Optional scheduler restored when resuming.
        config: Validated training configuration.
        device: Device on which checkpoints are loaded.

    Returns:
        First epoch to run and the best validation score so far.
    """
    initialization = config.initialization
    if initialization.pretrained_checkpoint is not None:
        load_model_weights(checkpoint_path=initialization.pretrained_checkpoint,
                           model=model,
                           device=device)
        print(
            "Initialized model from pretrained checkpoint: "
            f"{initialization.pretrained_checkpoint}",
            flush=True)
    if initialization.compatible_pretrained_checkpoint is not None:
        prefixes = tuple(
            prefix.strip()
            for prefix in initialization.compatible_pretrained_prefixes.split(
                ",")
            if prefix.strip())
        loaded, considered = load_compatible_model_weights(
            checkpoint_path=initialization.compatible_pretrained_checkpoint,
            model=model,
            device=device,
            prefixes=prefixes)
        prefix_text = ", ".join(prefixes) if prefixes else "all prefixes"
        print(
            f"Initialized {loaded}/{considered} compatible tensors from "
            f"{initialization.compatible_pretrained_checkpoint} "
            f"({prefix_text}).",
            flush=True,
        )
    if initialization.resume_from is None:
        return 0, -float("inf")
    start_epoch, best_val_score = resume_checkpoint(
        checkpoint_path=initialization.resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device)
    print(
        f"Resumed from {initialization.resume_from} at epoch "
        f"{start_epoch:03d} with best_val_score={best_val_score:.6f}",
        flush=True,
    )
    return start_epoch, best_val_score


def _epoch_weights(config: FineTuningConfig, epoch: int) -> EpochWeights:
    """Return the ramped loss weights of one epoch."""
    anatomy, pet, lesion = (config.anatomical_loss, config.pet_loss,
                            config.lesion_loss)
    lesion_ramp = lesion.pet_lesion_occupancy_ramp_epochs
    return EpochWeights(
        lambda_macro_ct_labels=_ramped_weight(
            anatomy.lambda_macro_ct_labels, epoch,
            anatomy.macro_ct_label_ramp_epochs),
        lambda_pet_hot_region=_ramped_weight(pet.lambda_pet_hot_region, epoch,
                                             pet.pet_hot_ramp_epochs),
        lambda_pet_lesion_occupancy=_ramped_weight(
            lesion.lambda_pet_lesion_occupancy, epoch, lesion_ramp),
        lambda_pet_lesion_translation=_ramped_weight(
            lesion.lambda_pet_lesion_translation, epoch, lesion_ramp),
        lambda_pet_lesion_strain=_ramped_weight(lesion.lambda_pet_lesion_strain,
                                                epoch, lesion_ramp),
    )


def _print_macro_supervision(config: FineTuningConfig) -> None:
    """Log the macro CT-label groups when that supervision is enabled."""
    anatomy = config.anatomical_loss
    if anatomy.lambda_macro_ct_labels <= 0.0:
        return
    group_names = ", ".join(
        name
        for name, _ in macro_ct_label_groups(anatomy.macro_ct_label_preset))
    print(
        "Using weak macro CT-label supervision "
        f"(preset={anatomy.macro_ct_label_preset}, groups={group_names}, "
        f"downsample={anatomy.macro_ct_label_downsample}, "
        f"ramp_epochs={anatomy.macro_ct_label_ramp_epochs}).",
        flush=True)


@dataclass(frozen=True)
class _FineTuningRun:
    """Objects shared by every epoch of one fine-tuning run."""

    config: FineTuningConfig
    model: AffineTransMorphV2CascadeRegistration
    train_loader: DataLoader[TensorBatch]
    val_loader: DataLoader[TensorBatch]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None
    metadata: dict[str, str | int | float | bool]
    selection: SelectionState
    metrics_writer: FineTuningMetricsWriter


def _run_fine_tuning_epoch(run: _FineTuningRun, epoch: int) -> None:
    """Train, validate, select, and atomically record one completed epoch."""
    config = run.config
    output_dir = config.data.output_dir
    weights = _epoch_weights(config, epoch)
    train_metrics = run_train_epoch(
        model=run.model,
        loader=run.train_loader,
        optimizer=run.optimizer,
        device=next(run.model.parameters()).device,
        config=config,
        weights=weights,
    )
    val_metrics = run_validation_epoch(
        model=run.model,
        loader=run.val_loader,
        device=next(run.model.parameters()).device,
        config=config,
        weights=weights,
    )
    # The selection update rejects a non-finite score before it can reach
    # the scheduler.
    is_best, auxiliary_improvements = run.selection.update(
        val_metrics, config.runtime.log_challenge_metrics)
    if run.scheduler is not None:
        run.scheduler.step(val_metrics["ct_dsc"])

    checkpoint_state = CheckpointState(
        epoch=epoch,
        model=run.model,
        optimizer=run.optimizer,
        scheduler=run.scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        config=run.metadata,
    )
    pending_path = output_dir / PENDING_CHECKPOINT
    selected = {pending_path: run.selection.best_val_score}
    if is_best:
        selected[output_dir / "best_val.pt"] = run.selection.best_val_score
    for filename, score in auxiliary_improvements.items():
        selected[output_dir / filename] = score
    for path, score in selected.items():
        save_checkpoint(path,
                        checkpoint_state,
                        best_key="best_val_score",
                        best_value=score,
                        with_scheduler=True)
    run.metrics_writer.write(epoch, train_metrics, val_metrics,
                             run.selection.best_val_score,
                             run.selection.epochs_without_improvement)
    os.replace(pending_path, output_dir / LATEST_CHECKPOINT)
    print(
        f"Epoch {epoch:03d} | {format_metrics('train', train_metrics)} | "
        f"{format_metrics('val', val_metrics)} | "
        f"best_val_score={run.selection.best_val_score:.6f} | "
        f"lr={run.optimizer.param_groups[0]['lr']:.2e} | "
        f"patience={run.selection.epochs_without_improvement}/"
        f"{config.optimization.patience}",
        flush=True,
    )


def main() -> None:
    """Run final-model training and validation."""
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        stream=sys.stdout)
    config = parse_fine_tuning_config()
    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics_writer = FineTuningMetricsWriter(metrics_path)
    metrics_writer.validate()
    validate_validation_labels(config.data.val_manifest)
    recover_interrupted_epoch(output_dir, config.initialization.resume_from,
                              metrics_path)
    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config)
    model = AffineTransMorphV2CascadeRegistration(
        AffineCascadeSettings(
            max_velocity=config.architecture.max_velocity,
            max_residual_disp=config.architecture.max_residual_disp,
            scaling_steps=config.architecture.scaling_steps))
    model = model.to(device)
    _print_macro_supervision(config)
    optimizer, scheduler = _build_optimizer(model, config)
    metadata = checkpoint_config(config)
    start_epoch, best_val_score = _initialize_training_state(
        model, optimizer, scheduler, config, device)
    selection = SelectionState.restore(metrics_path, start_epoch,
                                       best_val_score)
    selection.drop_missing_checkpoints(output_dir)
    if start_epoch > 0 and selection.should_stop(config.optimization.patience):
        print("Run already reached early-stopping patience; no epoch resumed.",
              flush=True)
        return
    run = _FineTuningRun(config=config,
                         model=model,
                         train_loader=cast(DataLoader[TensorBatch],
                                           train_loader),
                         val_loader=cast(DataLoader[TensorBatch], val_loader),
                         optimizer=optimizer,
                         scheduler=scheduler,
                         metadata=metadata,
                         selection=selection,
                         metrics_writer=metrics_writer)
    for epoch in range(start_epoch, config.optimization.epochs):
        _run_fine_tuning_epoch(run, epoch)
        if selection.should_stop(config.optimization.patience):
            print(f"Early stopping at epoch {epoch:03d}.", flush=True)
            break


if __name__ == "__main__":
    main()
