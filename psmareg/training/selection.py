"""Checkpoint selection and early-stopping state for fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path

from psmareg.training.metrics_csv import read_resume_metrics_rows

LOGGER = logging.getLogger(__name__)
_AUXILIARY_KEYS = (
    "challenge_ct_dsc",
    "challenge_pet_dsc",
    "challenge_mtv_percent_error",
    "challenge_tlg_percent_error",
)


def auxiliary_scores(val_metrics: dict[str, float]) -> dict[str, float]:
    """Map challenge metrics to the three auxiliary checkpoint scores."""
    return {
        "best_challenge_ct_dsc.pt":
            val_metrics["challenge_ct_dsc"],
        "best_challenge_pet_dsc.pt":
            val_metrics["challenge_pet_dsc"],
        "best_challenge_pet_biomarker.pt":
            -(val_metrics["challenge_mtv_percent_error"] +
              val_metrics["challenge_tlg_percent_error"]) / 2.0,
    }


def auxiliary_bests_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    """Return the best finite auxiliary score recorded for each checkpoint.

    Args:
        rows: Metrics CSV rows of completed epochs.

    Returns:
        Best score per auxiliary checkpoint filename.
    """
    bests: dict[str, float] = {}
    for row in rows:
        scores = auxiliary_scores(
            {key: float(row[f"val_{key}"]) for key in _AUXILIARY_KEYS})
        for filename, score in scores.items():
            if math.isfinite(score):
                bests[filename] = max(score, bests.get(filename, -float("inf")))
    return bests


@dataclass
class _EarlyStopping:
    """Count the epochs elapsed since the selection score last improved."""

    epochs_without_improvement: int = 0

    def _record(self, improved: bool) -> None:
        """Reset the counter after an improvement, otherwise advance it."""
        self.epochs_without_improvement = (0 if improved else
                                           self.epochs_without_improvement + 1)

    def should_stop(self, patience: int) -> bool:
        """Return whether the configured early-stopping limit was reached."""
        return self.epochs_without_improvement >= patience


@dataclass
class SelectionState(_EarlyStopping):
    """Track primary and auxiliary best scores across training and resume."""

    best_val_score: float = -float("inf")
    best_auxiliary_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def restore(cls, metrics_path: Path, start_epoch: int,
                best_val_score: float) -> SelectionState:
        """Rebuild the selection state from a matching CSV history.

        Args:
            metrics_path: CSV from the run being resumed.
            start_epoch: First epoch after the restored checkpoint.
            best_val_score: Primary best score stored in that checkpoint.

        Returns:
            Selection state ready for the next epoch.

        Raises:
            ValueError: If a fresh run finds an existing history, or the history
                disagrees with the checkpoint.
        """
        if start_epoch == 0:
            if metrics_path.exists() and metrics_path.stat().st_size > 0:
                raise ValueError(
                    f"Metrics CSV already exists in a fresh run: "
                    f"{metrics_path}. Use a new output directory or resume "
                    "its checkpoint.")
            return cls(best_val_score=best_val_score)
        rows = read_resume_metrics_rows(metrics_path, start_epoch - 1)
        recorded_best = float(rows[-1]["best_val_score"])
        if not math.isclose(
                recorded_best, best_val_score, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Checkpoint best_val_score differs from metrics.csv; "
                "refusing an unsafe resume.")
        return cls(best_val_score=best_val_score,
                   epochs_without_improvement=int(
                       rows[-1]["epochs_without_improvement"]),
                   best_auxiliary_scores=auxiliary_bests_from_rows(rows))

    def drop_missing_checkpoints(self, output_dir: Path) -> None:
        """Let the next finite score recreate a missing auxiliary file."""
        for filename in tuple(self.best_auxiliary_scores):
            if not (output_dir / filename).exists():
                LOGGER.warning(
                    "No archived %s found; the next finite score will "
                    "create it.", filename)
                del self.best_auxiliary_scores[filename]

    def update(self, val_metrics: dict[str, float],
               log_challenge_metrics: bool) -> tuple[bool, dict[str, float]]:
        """Update best scores and return newly selected checkpoints.

        Args:
            val_metrics: Validation metrics of the completed epoch.
            log_challenge_metrics: Whether missing challenge scores are notable.

        Returns:
            Whether the primary score improved and the auxiliary improvements.

        Raises:
            ValueError: If the validation CT Dice is not finite.
        """
        current_score = val_metrics["ct_dsc"]
        if not math.isfinite(current_score):
            raise ValueError(
                "Validation CT Dice is not finite; model selection and the "
                "scheduler require at least one labeled validation pair.")
        is_best = current_score > self.best_val_score
        if is_best:
            self.best_val_score = current_score
        self._record(is_best)
        improvements: dict[str, float] = {}
        for filename, score in auxiliary_scores(val_metrics).items():
            if not math.isfinite(score):
                if log_challenge_metrics:
                    LOGGER.warning(
                        "Skipping %s: validation score is not finite.",
                        filename)
                continue
            if score <= self.best_auxiliary_scores.get(filename, -float("inf")):
                continue
            self.best_auxiliary_scores[filename] = score
            improvements[filename] = score
        return is_best, improvements


@dataclass
class LossSelectionState(_EarlyStopping):
    """Track the best validation loss and patience of synthetic pretraining."""

    best_val_loss: float = float("inf")

    def update(self, val_loss: float, validated: bool) -> bool:
        """Record one epoch and report whether it is the new best.

        Args:
            val_loss: Validation loss of the epoch, NaN when not validated.
            validated: Whether validation ran this epoch. Epochs without
                validation leave the state unchanged.

        Returns:
            Whether the validation loss improved.
        """
        if not validated:
            return False
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
        self._record(improved)
        return improved
