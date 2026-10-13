"""Train PSMAReg registration models on synthetic CT/PET deformations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from psmareg.config.sections import (
    add_set_argument,
    named_overrides,
    set_overrides,
)
from psmareg.config.synthetic_pretraining_config import (
    SyntheticPretrainingConfig,
    load_synthetic_pretraining_config,
)
from psmareg.models.common.registration.cascades import (
    AffineTransMorphDiffCascadeRegistration,)
from psmareg.models.pretraining.factory import (
    build_model,
    set_effective_max_velocity,
)
from psmareg.training.pretraining import (
    build_loaders,
    checkpoint_config,
    format_metrics,
    initialize_matching_weights,
    load_checkpoint,
    run_epoch,
    write_metrics_row,
)
from psmareg.training.selection import LossSelectionState
from psmareg.training.checkpoint_state import (
    CheckpointState,
    save_checkpoint,
)


def parse_synthetic_pretraining_config(
    argv: list[str] | None = None,) -> SyntheticPretrainingConfig:
    """Parse operational CLI arguments and load a validated stage recipe.

    Args:
        argv: Optional argument vector, or the process arguments when omitted.

    Returns:
        Validated synthetic pretraining configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train PSMAReg from a synthetic pretraining configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("psmareg/config/synthetic_pretraining_stage_a.py"))
    parser.add_argument("--train_manifest", type=Path)
    parser.add_argument("--val_manifest", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--start_from", type=Path)
    parser.add_argument("--initialize_deformable_from", type=Path)
    parser.add_argument("--epochs", type=int)
    add_set_argument(
        parser, "Override a validated parameter for a controlled diagnostic.")
    cli = parser.parse_args(argv)
    overrides = named_overrides(
        cli, ("train_manifest", "val_manifest", "output_dir", "start_from",
              "initialize_deformable_from", "epochs"))
    if cli.start_from is not None:
        overrides["initialize_deformable_from"] = None
    if cli.initialize_deformable_from is not None:
        overrides["start_from"] = None
    overrides.update(set_overrides(parser, cli.set))
    return load_synthetic_pretraining_config(cli.config, overrides)


def _initialize_training_state(
    config: SyntheticPretrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Resume an experiment or initialize compatible model weights."""
    best_val_loss = float("inf")
    start_epoch = 0

    if config.initialization.start_from is not None:
        start_epoch, best_val_loss = load_checkpoint(
            path=config.initialization.start_from,
            model=model,
            optimizer=optimizer,
            device=device,
            load_optimizer_state=config.initialization.load_optimizer_state,
            inherit_best_val_loss=config.initialization.inherit_best_val_loss,
        )
        print(
            f"Resumed from {config.initialization.start_from} at epoch "
            f"{start_epoch:03d} "
            f"with best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
    elif config.initialization.initialize_deformable_from is not None:
        loaded_count = initialize_matching_weights(
            path=config.initialization.initialize_deformable_from,
            model=model,
            device=device,
        )
        print(
            f"Initialized {loaded_count} matching deformable tensors from "
            f"{config.initialization.initialize_deformable_from}; affine head "
            "starts at identity.",
            flush=True,
        )
        if config.architecture.reset_affine_head and isinstance(
                model, AffineTransMorphDiffCascadeRegistration):
            model.reset_affine_head()
            print("Reset learned affine residual head to identity.", flush=True)
    return start_epoch, best_val_loss


def _effective_max_velocity(config: SyntheticPretrainingConfig, epoch: int,
                            continuation_start_epoch: int) -> float:
    """Resolve the historical velocity ramp for one epoch."""
    if (config.architecture.max_velocity_start is not None and
            config.architecture.max_velocity_ramp_epochs > 0):
        velocity_progress = min(
            max((epoch - continuation_start_epoch) /
                config.architecture.max_velocity_ramp_epochs, 0.0),
            1.0,
        )
        return (config.architecture.max_velocity_start + velocity_progress *
                (config.architecture.max_velocity -
                 config.architecture.max_velocity_start))
    return config.architecture.max_velocity


@dataclass(frozen=True)
class _PretrainingRun:
    """Objects shared by every epoch of one synthetic pretraining run."""

    config: SyntheticPretrainingConfig
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    metadata: dict[str, str | int | float | bool]
    selection: LossSelectionState


def _run_epoch_metrics(
        run: _PretrainingRun,
        epoch: int) -> tuple[dict[str, float], dict[str, float], bool]:
    """Run training and, when scheduled, validation for one epoch."""
    config = run.config
    train_metrics = run_epoch(
        model=run.model,
        loader=run.train_loader,
        device=run.device,
        optimizer=run.optimizer,
        scaler=run.scaler,
        config=config,
        epoch=epoch,
    )
    should_validate = (config.runtime.validate_at_start and epoch == 0
                      ) or (epoch + 1) % config.runtime.validation_interval == 0
    if should_validate:
        val_metrics = run_epoch(
            model=run.model,
            loader=run.val_loader,
            device=run.device,
            optimizer=None,
            scaler=None,
            config=config,
            epoch=epoch,
        )
    else:
        val_metrics = {key: float("nan") for key in train_metrics}
    return train_metrics, val_metrics, should_validate


def _run_pretraining_epoch(run: _PretrainingRun, epoch: int,
                           continuation_start_epoch: int) -> None:
    """Train, validate, select, and record one completed epoch."""
    config = run.config
    output_dir = config.data.required("output_dir")
    effective_max_velocity = _effective_max_velocity(config, epoch,
                                                     continuation_start_epoch)
    set_effective_max_velocity(run.model, effective_max_velocity)
    run.metadata["max_velocity"] = effective_max_velocity
    run.metadata["max_velocity_target"] = config.architecture.max_velocity

    epoch_start_time = time.perf_counter()
    train_metrics, val_metrics, should_validate = _run_epoch_metrics(run, epoch)
    state = CheckpointState(
        epoch=epoch,
        model=run.model,
        optimizer=run.optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        config=run.metadata,
    )
    selection = run.selection
    is_best = selection.update(val_metrics["loss"], should_validate)
    for filename in (("best_val.pt",) if is_best else ()) + ("latest.pt",):
        save_checkpoint(output_dir / filename,
                        state,
                        best_key="best_val_loss",
                        best_value=selection.best_val_loss)
    epoch_seconds = time.perf_counter() - epoch_start_time
    write_metrics_row(
        path=output_dir / "metrics.csv",
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        selection=selection,
        epoch_seconds=epoch_seconds,
    )
    print(
        f"Epoch {epoch:03d} | "
        f"max_velocity={effective_max_velocity:.3f} | "
        f"{format_metrics('train', train_metrics)} | "
        f"{format_metrics('val', val_metrics)} | "
        f"best_val_loss={selection.best_val_loss:.6f} | "
        f"patience={selection.epochs_without_improvement}/"
        f"{config.optimization.patience} | "
        f"epoch_seconds={epoch_seconds:.1f}",
        flush=True,
    )


def main() -> None:
    """Run the configured synthetic pretraining experiment."""
    config = parse_synthetic_pretraining_config()
    config.data.required("output_dir").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config)

    model = build_model(config.architecture).to(device)
    print(
        "Built model "
        f"{type(model).__name__} | "
        f"requested_refinement_steps={config.architecture.refinement_steps} | "
        f"model_refinement_steps={getattr(model, 'refinement_steps', 'n/a')}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimization.lr)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.runtime.amp and device.type == "cuda",
    )
    start_epoch, best_val_loss = _initialize_training_state(
        config, model, optimizer, device)
    run = _PretrainingRun(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        metadata=checkpoint_config(config),
        selection=LossSelectionState(best_val_loss=best_val_loss))
    for epoch in range(start_epoch, config.optimization.epochs):
        _run_pretraining_epoch(run, epoch, start_epoch)
        if run.selection.should_stop(config.optimization.patience):
            print(f"Early stopping at epoch {epoch:03d}.", flush=True)
            break


if __name__ == "__main__":
    main()
