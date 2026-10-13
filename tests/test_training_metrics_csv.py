"""Tests for the fine-tuning metrics CSV schema, writing, and averaging."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from psmareg.training.metrics_csv import (append_metrics_row,
                                          read_resume_metrics_rows,
                                          validate_metrics_header)
from psmareg.training.fine_tuning import _average_metrics
from psmareg.training.fine_tuning_batch_metrics import METRIC_KEYS
from psmareg.training.metrics_writer import FineTuningMetricsWriter


class MetricsWriterTests(unittest.TestCase):
    """The writer owns the column order that resumed runs rely on."""

    def test_metrics_writer_uses_declared_order(self) -> None:
        """The schema owner writes exactly the columns it validates."""
        with TemporaryDirectory() as directory:
            writer = FineTuningMetricsWriter(Path(directory) / "metrics.csv")
            metrics = dict.fromkeys(METRIC_KEYS, 0.0)
            writer.validate()
            writer.write(0, metrics, metrics, 0.0, 0)
            with writer.path.open(newline="", encoding="utf-8") as file:
                header = next(csv.reader(file))
            self.assertEqual(header, writer.fieldnames)
            self.assertEqual(
                header[:3],
                ["epoch", "best_val_score", "epochs_without_improvement"])
            self.assertEqual(len(header), 3 + 2 * len(METRIC_KEYS))

    def test_writer_rejects_reordered_or_missing_metrics(self) -> None:
        """Metrics that do not follow the schema are never written."""
        with TemporaryDirectory() as directory:
            writer = FineTuningMetricsWriter(Path(directory) / "metrics.csv")
            metrics = dict.fromkeys(METRIC_KEYS, 0.0)
            reordered = dict(reversed(list(metrics.items())))
            incomplete = dict(list(metrics.items())[:-1])
            for invalid in (reordered, incomplete):
                with self.assertRaisesRegex(ValueError, "CSV schema"):
                    writer.write(0, metrics, invalid, 0.0, 0)
            self.assertFalse(writer.path.exists())


class MetricsCsvTests(unittest.TestCase):
    """Ensure resumed runs cannot silently corrupt metrics columns."""

    def test_matching_schema_appends(self) -> None:
        """A resumed run may append when the column order is unchanged."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics_row(path, {"epoch": 0}, {"loss": 1.0}, {"loss": 2.0})
            append_metrics_row(path, {"epoch": 1}, {"loss": 0.5}, {"loss": 1.5})
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["val_loss"], "1.5")

    def test_changed_schema_is_rejected_without_write(self) -> None:
        """A resumed run cannot append values beneath old column names."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics_row(path, {"epoch": 0}, {"loss": 1.0}, {"loss": 2.0})
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                append_metrics_row(path, {"epoch": 1}, {"ct_dsc": 0.5},
                                   {"ct_dsc": 0.6})
            self.assertEqual(path.read_bytes(), original)

    def test_preflight_rejects_changed_schema(self) -> None:
        """The header check works before an epoch or checkpoint is written."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics_row(path, {"epoch": 0}, {"loss": 1.0}, {"loss": 2.0})
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                validate_metrics_header(path,
                                        ["epoch", "train_ct_dsc", "val_ct_dsc"])

    def test_resume_requires_matching_last_epoch(self) -> None:
        """A checkpoint newer than its CSV history cannot be resumed."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics_row(path, {"epoch": 0}, {"loss": 1.0}, {"loss": 2.0})
            with self.assertRaisesRegex(ValueError, "checkpoint ends"):
                read_resume_metrics_rows(path, completed_epoch=1)

    def test_failed_csv_publish_keeps_previous_history(self) -> None:
        """An interrupted append leaves the existing CSV untouched."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics_row(path, {"epoch": 0}, {"loss": 1.0}, {"loss": 2.0})
            original = path.read_bytes()
            with patch("psmareg.training.metrics_csv.os.replace",
                       side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    append_metrics_row(path, {"epoch": 1}, {"loss": 0.5},
                                       {"loss": 1.5})
            self.assertEqual(path.read_bytes(), original)


class MetricAveragingTests(unittest.TestCase):
    """Check missing labels do not poison valid selection scores."""

    def test_unlabeled_pair_is_ignored_for_ct_dice(self) -> None:
        """Finite CT Dice is retained while unavailable scores stay NaN."""
        result = _average_metrics([
            {
                "loss": 1.0,
                "ct_dsc": 0.6,
                "challenge_pet_dsc": math.nan,
                "challenge_mtv_percent_error": 12.0,
                "challenge_tlg_percent_error": 8.0
            },
            {
                "loss": 2.0,
                "ct_dsc": math.nan,
                "challenge_pet_dsc": math.nan,
                "challenge_mtv_percent_error": math.nan,
                "challenge_tlg_percent_error": math.nan
            },
        ])
        self.assertEqual(result["loss"], 1.5)
        self.assertEqual(result["ct_dsc"], 0.6)
        self.assertTrue(math.isnan(result["challenge_pet_dsc"]))
        self.assertEqual(result["challenge_mtv_percent_error"], 12.0)
        self.assertEqual(result["challenge_tlg_percent_error"], 8.0)


if __name__ == "__main__":
    unittest.main()
