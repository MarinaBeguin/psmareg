"""Tests for the recovery of an interrupted fine-tuning epoch."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

from psmareg.training.metrics_csv import (append_metrics_row)
from psmareg.training.epoch_recovery import (recover_interrupted_epoch)
from psmareg.training.fine_tuning_batch_metrics import METRIC_KEYS


class InterruptedEpochTests(unittest.TestCase):
    """Recover either side of the staged-checkpoint/CSV publication."""

    @staticmethod
    def _checkpoint() -> dict[str, Any]:
        """Return a small serializable epoch-zero checkpoint."""
        metrics = dict.fromkeys(METRIC_KEYS, 0.0)
        metrics["ct_dsc"] = 0.6
        for key in ("challenge_ct_dsc", "challenge_pet_dsc",
                    "challenge_mtv_percent_error",
                    "challenge_tlg_percent_error"):
            metrics[key] = math.nan
        return {
            "epoch": 0,
            "best_val_score": 0.6,
            "train_metrics": metrics,
            "val_metrics": metrics,
        }

    def _recover(self, directory: str,
                 csv_already_written: bool) -> list[dict[str, str]]:
        """Run staged recovery and return the resulting metric rows."""
        output_dir = Path(directory)
        metrics_path = output_dir / "metrics.csv"
        checkpoint = self._checkpoint()
        if csv_already_written:
            append_metrics_row(metrics_path, {
                "epoch": 0,
                "best_val_score": 0.6,
                "epochs_without_improvement": 0,
            }, checkpoint["train_metrics"], checkpoint["val_metrics"])
        (output_dir / "latest.pending.pt").write_bytes(b"staged")
        with patch(
                "psmareg.training.epoch_recovery.torch.load",
                return_value=checkpoint), patch(
                    "psmareg.training.epoch_recovery.save_checkpoint_payload"):
            recover_interrupted_epoch(output_dir, output_dir / "latest.pt",
                                      metrics_path)
        self.assertFalse((output_dir / "latest.pending.pt").exists())
        self.assertTrue((output_dir / "latest.pt").exists())
        with metrics_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_checkpoint_ahead_of_csv_is_completed(self) -> None:
        """The missing CSV row is reconstructed from staged metrics."""
        with TemporaryDirectory() as directory:
            rows = self._recover(directory, csv_already_written=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["val_ct_dsc"], "0.6")

    def test_csv_ahead_of_latest_is_completed(self) -> None:
        """A written CSV row promotes the staged checkpoint once."""
        with TemporaryDirectory() as directory:
            rows = self._recover(directory, csv_already_written=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "0")

    def test_legacy_latest_ahead_without_pending_is_completed(self) -> None:
        """A checkpoint from the old write order can restore its CSV row."""
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            latest = output_dir / "latest.pt"
            latest.write_bytes(b"checkpoint")
            metrics_path = output_dir / "metrics.csv"
            with patch(
                    "psmareg.training.epoch_recovery.torch.load",
                    return_value=self._checkpoint()
            ), patch("psmareg.training.epoch_recovery.save_checkpoint_payload"):
                recover_interrupted_epoch(output_dir, output_dir / "latest.pt",
                                          metrics_path)
            with metrics_path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_unstaged_csv_ahead_is_left_untouched(self) -> None:
        """Without staged weights, a newer CSV must not be discarded."""
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            latest = output_dir / "latest.pt"
            latest.write_bytes(b"checkpoint")
            metrics_path = output_dir / "metrics.csv"
            checkpoint = self._checkpoint()
            for epoch in (0, 1):
                append_metrics_row(
                    metrics_path, {
                        "epoch": epoch,
                        "best_val_score": 0.6,
                        "epochs_without_improvement": epoch,
                    }, checkpoint["train_metrics"], checkpoint["val_metrics"])
            original = metrics_path.read_bytes()
            with patch("psmareg.training.epoch_recovery.torch.load",
                       return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "recoverable write"):
                    recover_interrupted_epoch(output_dir,
                                              output_dir / "latest.pt",
                                              metrics_path)
            self.assertEqual(metrics_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
