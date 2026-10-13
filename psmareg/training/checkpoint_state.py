"""Checkpoint contents and atomic writing shared by both training phases."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn

from psmareg.config.sections import flatten_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointState:
    """Completed epoch and mutable training objects to serialize."""

    epoch: int
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    config: dict[str, str | int | float | bool]
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None


def checkpoint_metadata(
    config: Any,
    *,
    comma_joined: tuple[str, ...] = (),
    extra: Mapping[str, str | int | float | bool] | None = None,
) -> dict[str, str | int | float | bool]:
    """Serialize sectioned configuration without changing field order.

    Args:
        config: Sectioned configuration of the run.
        comma_joined: Tuple parameters stored as comma-separated text.
        extra: Additional entries appended after the configuration.

    Returns:
        Flat mapping of scalar parameters; unset or unsupported values are
        omitted with a warning.
    """
    metadata: dict[str, str | int | float | bool] = {}
    omitted: list[str] = []
    for key, value in flatten_config(config).items():
        if isinstance(value, Path):
            metadata[key] = str(value)
        elif key in comma_joined:
            metadata[key] = ",".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            omitted.append(key)
    if omitted:
        LOGGER.warning(
            "Checkpoint metadata omits unset or unsupported fields: %s",
            ", ".join(omitted))
    if extra:
        metadata.update(extra)
    return metadata


def save_checkpoint(path: Path,
                    state: CheckpointState,
                    *,
                    best_key: str,
                    best_value: float,
                    with_scheduler: bool = False) -> None:
    """Serialize the complete resumable training state.

    Args:
        path: Destination checkpoint path.
        state: Completed epoch and training objects to save.
        best_key: Name under which the selection score is stored; existing
            checkpoints use ``best_val_score`` or ``best_val_loss``.
        best_value: Best selection score so far.
        with_scheduler: Whether to store the scheduler state, as fine-tuning
            checkpoints do even when no scheduler is configured.
    """
    payload: dict[str, object] = {
        "epoch": state.epoch,
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
    }
    if with_scheduler:
        payload["scheduler_state_dict"] = (state.scheduler.state_dict()
                                           if state.scheduler is not None else
                                           None)
    payload.update({
        "train_metrics": state.train_metrics,
        "val_metrics": state.val_metrics,
        best_key: best_value,
        "config": state.config,
    })
    save_checkpoint_payload(path, payload)


def save_checkpoint_payload(checkpoint_path: Path,
                            payload: dict[str, object]) -> None:
    """Atomically publish a checkpoint without damaging an earlier file.

    Args:
        checkpoint_path: Final checkpoint path.
        payload: Complete serializable checkpoint contents.
    """
    LOGGER.info("--> Saving checkpoint to %s...", checkpoint_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
