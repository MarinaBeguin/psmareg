"""Stable CSV schema and writing for fine-tuning metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from psmareg.training.fine_tuning_batch_metrics import METRIC_KEYS
from psmareg.training.metrics_csv import (append_metrics_row,
                                          validate_metrics_header)


class FineTuningMetricsWriter:
    """Own the ordered fine-tuning CSV schema and its persistence."""

    def __init__(self, path: Path) -> None:
        """Remember the metrics path without creating a file.

        Args:
            path: CSV written by the fine-tuning run.
        """
        self.path = path
        # Column order that resumed runs rely on.
        self.fieldnames = [
            "epoch", "best_val_score", "epochs_without_improvement"
        ] + [
            f"{prefix}_{key}" for prefix in ("train", "val")
            for key in METRIC_KEYS
        ]

    def validate(self) -> None:
        """Reject an existing CSV with incompatible columns."""
        validate_metrics_header(self.path, self.fieldnames)

    def write(self, epoch: int, train_metrics: Mapping[str, float],
              val_metrics: Mapping[str, float], best_val_score: float,
              epochs_without_improvement: int) -> None:
        """Append one complete epoch after checking metric keys.

        Args:
            epoch: Completed epoch index.
            train_metrics: Ordered training metrics.
            val_metrics: Ordered validation metrics.
            best_val_score: Best CT Dice so far.
            epochs_without_improvement: Current early-stopping counter.
        """
        self.check_schema(train_metrics, val_metrics)
        append_metrics_row(
            self.path, {
                "epoch": epoch,
                "best_val_score": best_val_score,
                "epochs_without_improvement": epochs_without_improvement,
            }, train_metrics, val_metrics)

    @staticmethod
    def check_schema(train_metrics: Mapping[str, float],
                     val_metrics: Mapping[str, float]) -> None:
        """Reject metrics whose keys or order differ from the CSV schema.

        Raises:
            ValueError: If either mapping differs from ``METRIC_KEYS``.
        """
        if (tuple(train_metrics) != METRIC_KEYS or
                tuple(val_metrics) != METRIC_KEYS):
            raise ValueError("Fine-tuning metrics do not match the CSV schema.")

    def row(self, epoch: int, train_metrics: Mapping[str, float],
            val_metrics: Mapping[str, float], best_val_score: float,
            epochs_without_improvement: int) -> dict[str, int | float]:
        """Return the values ``write`` records for one epoch, by column.

        Args:
            epoch: Completed epoch index.
            train_metrics: Ordered training metrics.
            val_metrics: Ordered validation metrics.
            best_val_score: Best CT Dice so far.
            epochs_without_improvement: Current early-stopping counter.
        """
        values: dict[str, int | float] = {
            "epoch": epoch,
            "best_val_score": best_val_score,
            "epochs_without_improvement": epochs_without_improvement,
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            values.update(
                {f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
        return values
