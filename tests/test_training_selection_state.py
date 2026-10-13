"""Tests for checkpoint selection, early stopping, and their restoration."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from psmareg.training.metrics_csv import (append_metrics_row)
from psmareg.training.selection import LossSelectionState, SelectionState
from psmareg.training.fine_tuning import validate_validation_labels


class ResumeSelectionTests(unittest.TestCase):
    """Restore early stopping and auxiliary maxima from existing history."""

    def test_auxiliary_scores_and_patience_are_restored(self) -> None:
        """A resumed epoch cannot reset the auxiliary selection threshold."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            metrics = {
                "challenge_ct_dsc": 0.7,
                "challenge_pet_dsc": 0.5,
                "challenge_mtv_percent_error": 10.0,
                "challenge_tlg_percent_error": 20.0,
            }
            append_metrics_row(path, {
                "epoch": 0,
                "best_val_score": 0.7,
                "epochs_without_improvement": 0,
            }, metrics, metrics)
            worse = dict(metrics,
                         challenge_ct_dsc=0.6,
                         challenge_pet_dsc=0.4,
                         challenge_mtv_percent_error=30.0)
            append_metrics_row(path, {
                "epoch": 1,
                "best_val_score": 0.7,
                "epochs_without_improvement": 1,
            }, worse, worse)
            state = SelectionState.restore(path, 2, 0.7)
            self.assertEqual(state.epochs_without_improvement, 1)
            scores = state.best_auxiliary_scores
            self.assertEqual(scores["best_challenge_ct_dsc.pt"], 0.7)
            self.assertEqual(scores["best_challenge_pet_dsc.pt"], 0.5)
            self.assertEqual(scores["best_challenge_pet_biomarker.pt"], -15.0)

    def test_selection_update_preserves_best_thresholds(self) -> None:
        """A worse epoch advances patience without replacing best scores."""
        state = SelectionState()
        first = {
            "ct_dsc": 0.7,
            "challenge_ct_dsc": 0.5,
            "challenge_pet_dsc": 0.4,
            "challenge_mtv_percent_error": 10.0,
            "challenge_tlg_percent_error": 20.0,
        }
        is_best, auxiliary = state.update(first, True)
        self.assertTrue(is_best)
        self.assertEqual(len(auxiliary), 3)
        worse = dict(first, ct_dsc=0.6, challenge_ct_dsc=0.3)
        is_best, auxiliary = state.update(worse, True)
        self.assertFalse(is_best)
        self.assertEqual(auxiliary, {})
        self.assertEqual(state.best_val_score, 0.7)
        self.assertEqual(state.epochs_without_improvement, 1)


class EarlyStoppingTests(unittest.TestCase):
    """Check the stopping rule and the auxiliary checkpoint bookkeeping."""

    def test_non_finite_selection_score_stops_the_run(self) -> None:
        """Model selection cannot proceed without a validation CT Dice."""
        metrics = {
            "ct_dsc": math.nan,
            "challenge_ct_dsc": 0.5,
            "challenge_pet_dsc": 0.4,
            "challenge_mtv_percent_error": 10.0,
            "challenge_tlg_percent_error": 20.0,
        }
        with self.assertRaisesRegex(ValueError, "not finite"):
            SelectionState().update(metrics, True)

    def test_non_finite_auxiliary_scores_are_skipped(self) -> None:
        """Missing challenge scores never select an auxiliary checkpoint."""
        metrics = {
            "ct_dsc": 0.6,
            "challenge_ct_dsc": 0.5,
            "challenge_pet_dsc": math.nan,
            "challenge_mtv_percent_error": math.nan,
            "challenge_tlg_percent_error": 20.0,
        }
        state = SelectionState()
        with self.assertLogs("psmareg.training.selection", level="WARNING"):
            _, improved = state.update(metrics, True)
        self.assertEqual(improved, {"best_challenge_ct_dsc.pt": 0.5})

    def test_patience_stops_after_enough_stale_epochs(self) -> None:
        """Training stops once patience is exhausted."""
        state = SelectionState(best_val_score=0.9)
        metrics = {
            "ct_dsc": 0.5,
            "challenge_ct_dsc": 0.5,
            "challenge_pet_dsc": 0.5,
            "challenge_mtv_percent_error": 1.0,
            "challenge_tlg_percent_error": 1.0,
        }
        for _ in range(2):
            self.assertFalse(state.should_stop(patience=3))
            state.update(metrics, False)
        state.update(metrics, False)
        self.assertTrue(state.should_stop(patience=3))

    def test_missing_auxiliary_files_can_be_selected_again(self) -> None:
        """A best score whose checkpoint is gone no longer blocks saving."""
        state = SelectionState(best_auxiliary_scores={
            "best_challenge_ct_dsc.pt": 0.7,
            "best_challenge_pet_dsc.pt": 0.5,
        })
        with TemporaryDirectory() as directory:
            (Path(directory) / "best_challenge_ct_dsc.pt").write_bytes(b"x")
            with self.assertLogs("psmareg.training.selection", level="WARNING"):
                state.drop_missing_checkpoints(Path(directory))
        self.assertEqual(state.best_auxiliary_scores,
                         {"best_challenge_ct_dsc.pt": 0.7})

    def test_fresh_run_refuses_an_existing_history(self) -> None:
        """A new run cannot append to the metrics of another run."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            self.assertEqual(SelectionState.restore(path, 0, -math.inf),
                             SelectionState())
            path.write_text("epoch\n0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                SelectionState.restore(path, 0, -math.inf)


class LossSelectionTests(unittest.TestCase):
    """Pretraining selects on the validation loss, when validation ran."""

    def test_lower_loss_is_better_and_resets_patience(self) -> None:
        """Only validated epochs move the best loss or the patience."""
        state = LossSelectionState()
        self.assertTrue(state.update(1.0, True))
        self.assertFalse(state.update(1.5, True))
        self.assertFalse(state.update(math.nan, False))
        self.assertEqual(
            (state.best_val_loss, state.epochs_without_improvement), (1.0, 1))
        self.assertTrue(state.should_stop(patience=1))
        self.assertTrue(state.update(0.5, True))
        self.assertEqual(state.epochs_without_improvement, 0)
        self.assertFalse(state.should_stop(patience=1))


class ValidationManifestTests(unittest.TestCase):
    """Check the validation manifest before training starts."""

    def test_manifest_without_ct_labels_fails_before_training(self) -> None:
        """Missing validation supervision is detected before epoch zero."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            path.write_text(
                "subject,fixed_ct_label,moving_ct_label\n"
                "subject_a,,moving_label.nii.gz\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no pair with both CT"):
                validate_validation_labels(path)


if __name__ == "__main__":
    unittest.main()
