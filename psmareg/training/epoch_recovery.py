"""Roll an interrupted fine-tuning epoch forward before resuming.

An epoch is published in three steps: the checkpoint is staged as
``latest.pending.pt``, its row is appended to ``metrics.csv``, and the staged
file is renamed to ``latest.pt``. A run stopped between two steps is completed
here from the metrics stored inside the checkpoint.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from psmareg.training.checkpoint_state import save_checkpoint_payload
from psmareg.training.metrics_writer import FineTuningMetricsWriter
from psmareg.training.selection import (
    auxiliary_bests_from_rows,
    auxiliary_scores,
)

LOGGER = logging.getLogger(__name__)
PENDING_CHECKPOINT = "latest.pending.pt"
LATEST_CHECKPOINT = "latest.pt"


def _read_rows(metrics_path: Path) -> list[dict[str, str]]:
    """Read every recorded epoch, or nothing when the CSV is absent."""
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _check_recorded_row(row: dict[str, str], expected: dict[str, Any],
                        epoch: int) -> None:
    """Require an already written CSV row to match the checkpoint."""
    for key, value in expected.items():
        recorded = float(row[key])
        actual = float(value)
        if not (recorded == actual or
                (math.isnan(recorded) and math.isnan(actual))):
            raise ValueError(
                f"Pending checkpoint disagrees with metrics.csv at "
                f"epoch {epoch}, column {key}.")


def _complete_epoch(checkpoint: dict[str, Any], metrics_path: Path,
                    output_dir: Path) -> None:
    """Finish a checkpointed epoch whose CSV or selected files lag behind."""
    epoch = int(checkpoint["epoch"])
    best_score = float(checkpoint["best_val_score"])
    train_metrics = checkpoint["train_metrics"]
    val_metrics = checkpoint["val_metrics"]
    if not isinstance(train_metrics, dict) or not isinstance(val_metrics, dict):
        raise ValueError("Recovery checkpoint has no epoch metrics.")
    writer = FineTuningMetricsWriter(metrics_path)
    writer.check_schema(train_metrics, val_metrics)
    rows = _read_rows(metrics_path)
    if [int(row["epoch"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Cannot recover from a non-consecutive metrics CSV.")
    if len(rows) not in (epoch, epoch + 1):
        raise ValueError(
            f"Checkpoint epoch {epoch} and CSV with {len(rows)} rows are "
            "not one recoverable write apart.")

    previous = rows[epoch - 1] if epoch > 0 else None
    previous_best = (float(previous["best_val_score"])
                     if previous is not None else -float("inf"))
    current_score = float(val_metrics["ct_dsc"])
    improved = current_score > previous_best
    previous_patience = (int(previous["epochs_without_improvement"])
                         if previous is not None else 0)
    patience = 0 if improved else previous_patience + 1
    if not math.isclose(best_score,
                        max(previous_best, current_score),
                        rel_tol=1e-6,
                        abs_tol=1e-6):
        raise ValueError("Recovery checkpoint best score disagrees with CSV.")
    if len(rows) == epoch + 1:
        _check_recorded_row(
            rows[-1],
            writer.row(epoch, train_metrics, val_metrics, best_score, patience),
            epoch)

    if improved:
        save_checkpoint_payload(output_dir / "best_val.pt", checkpoint)
    earlier_scores = auxiliary_bests_from_rows(rows[:epoch])
    for filename, score in auxiliary_scores(val_metrics).items():
        if math.isfinite(score) and score > earlier_scores.get(
                filename, -float("inf")):
            save_checkpoint_payload(output_dir / filename,
                                    dict(checkpoint, best_val_score=score))
    if len(rows) == epoch:
        writer.write(epoch, train_metrics, val_metrics, best_score, patience)


def recover_interrupted_epoch(output_dir: Path, resume_from: Path | None,
                              metrics_path: Path) -> None:
    """Complete a staged epoch before model and optimizer restoration.

    Args:
        output_dir: Directory holding the run's checkpoints.
        resume_from: Checkpoint the run resumes from, or ``None`` when fresh.
        metrics_path: Metrics CSV of the run.

    Raises:
        ValueError: A staged epoch exists but the run does not resume from
            ``latest.pt``, or the checkpoint and CSV cannot be reconciled.
    """
    pending_path = output_dir / PENDING_CHECKPOINT
    latest_path = output_dir / LATEST_CHECKPOINT
    if resume_from is None or resume_from.resolve() != latest_path.resolve():
        if pending_path.exists():
            raise ValueError(
                f"Interrupted epoch found at {pending_path}; resume from "
                f"{LATEST_CHECKPOINT} to recover it.")
        return
    candidate = pending_path if pending_path.exists() else latest_path
    if not candidate.exists():
        return
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    if (not pending_path.exists() and
            len(_read_rows(metrics_path)) == epoch + 1):
        return
    _complete_epoch(checkpoint, metrics_path, output_dir)
    if pending_path.exists():
        os.replace(pending_path, latest_path)
    LOGGER.warning("Recovered interrupted epoch %s from %s.", epoch, candidate)
