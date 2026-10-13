"""Metrics CSV reading and writing, and log-line formatting."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


def append_metrics_row(
    path: Path,
    header: Mapping[str, int | float],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    """Append one epoch while retaining historical CSV column order.

    Args:
        path: Metrics CSV, created with its header when absent.
        header: Leading columns such as the epoch index.
        train_metrics: Training metrics, written with a ``train_`` prefix.
        val_metrics: Validation metrics, written with a ``val_`` prefix.

    Raises:
        ValueError: If an existing file has different columns.
    """
    row: dict[str, int | float] = dict(header)
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    fieldnames = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    validate_metrics_header(path, fieldnames)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.",
                                                  suffix=".tmp",
                                                  dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            if path.exists():
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            target.write(buffer.getvalue().encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_metrics_header(path: Path, fieldnames: list[str]) -> None:
    """Reject an existing metrics file with a different column order.

    Args:
        path: Existing or prospective metrics CSV path.
        fieldnames: Current training metric column names, in order.

    Raises:
        ValueError: The existing header differs from the current schema.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as file:
        existing_header = next(csv.reader(file), [])
    if existing_header != fieldnames:
        raise ValueError(f"Metrics CSV schema mismatch in {path}: expected "
                         f"{existing_header!r}, got {fieldnames!r}. "
                         "Use a new output directory or a matching recipe.")


def read_resume_metrics_rows(path: Path,
                             completed_epoch: int) -> list[dict[str, str]]:
    """Read the CSV history required to restore selection state.

    Args:
        path: Existing metrics CSV for the resumed run.
        completed_epoch: Last epoch recorded by the resumed checkpoint.

    Returns:
        Metrics rows through the checkpoint epoch.

    Raises:
        ValueError: The CSV is absent, empty, or out of sync with the
            checkpoint.
    """
    if not path.exists():
        raise ValueError(f"Cannot resume without metrics history: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Cannot resume from an empty metrics CSV: {path}")
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid epoch column in {path}") from error
    if epochs != list(range(completed_epoch + 1)):
        raise ValueError(
            f"Metrics CSV {path} has {len(epochs)} rows ending at epoch "
            f"{epochs[-1]}; the checkpoint ends at epoch {completed_epoch}. "
            "Expected consecutive epochs from zero through the checkpoint "
            "epoch. Use matching checkpoint and CSV files.")
    return rows


def format_metric_line(prefix: str, metrics: Mapping[str, float],
                       columns: Sequence[tuple[str, str, str]]) -> str:
    """Format selected metrics as one ``|``-separated log line.

    Args:
        prefix: Phase label, such as ``train`` or ``val``.
        metrics: Metric values produced by an epoch.
        columns: Logged label, metric key, and format template of each entry.

    Returns:
        Stable, ordered metric summary.
    """
    return " | ".join(f"{prefix}_{label}={template.format(metrics[key])}"
                      for label, key, template in columns)
