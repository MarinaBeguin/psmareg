"""Tests for fine-tuning epochs, checkpoints, and weight initialization."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from psmareg.config.fine_tuning_config import (
    FineTuningConfig,
    OptimizationConfig,
)
from psmareg.training.checkpoint_state import (
    CheckpointState,
    save_checkpoint,
    save_checkpoint_payload,
)
from psmareg.training.fine_tuning import (
    checkpoint_config,
    format_metrics,
    load_compatible_model_weights,
    load_model_weights,
    resume_checkpoint,
    run_train_epoch,
    run_validation_epoch,
)
from psmareg.training.fine_tuning_batch import EpochWeights
from psmareg.training.fine_tuning_batch_metrics import METRIC_KEYS

CPU = torch.device("cpu")
WEIGHTS = EpochWeights(0.0, 0.0, 0.0, 0.0, 0.0)


class _TwoStage(nn.Module):
    """Small model with the stage prefixes used by compatible loading."""

    def __init__(self, refiner_width: int = 3) -> None:
        super().__init__()
        self.coarse = nn.Linear(2, 2)
        self.refiner = nn.Linear(2, refiner_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both stages."""
        return self.refiner(self.coarse(x))


class EpochTests(unittest.TestCase):
    """Check the optimization step and the validation pass."""

    def test_global_gradient_is_clipped_before_the_step(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def batch_loss(**_: object) -> tuple[torch.Tensor, dict[str, float]]:
            return (model.weight * 10.0).sum(), {"loss": 10.0}

        for clip_norm, expected in ((1.0, 0.9), (0.0, 0.0)):
            with torch.no_grad():
                model.weight.fill_(1.0)
            config = replace(
                FineTuningConfig(),
                optimization=OptimizationConfig(gradient_clip_norm=clip_norm))
            with patch("psmareg.training.fine_tuning.compute_batch_metrics",
                       side_effect=batch_loss):
                metrics = run_train_epoch(model=model,
                                          loader=cast(Any, [{}]),
                                          optimizer=optimizer,
                                          device=CPU,
                                          config=config,
                                          weights=WEIGHTS)
            self.assertAlmostEqual(float(model.weight.detach()),
                                   expected,
                                   places=5)
            self.assertEqual(metrics, {"loss": 10.0})

    def test_gradients_do_not_accumulate_across_batches(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def batch_loss(**_: object) -> tuple[torch.Tensor, dict[str, float]]:
            return model.weight.sum(), {"loss": 1.0}

        with patch("psmareg.training.fine_tuning.compute_batch_metrics",
                   side_effect=batch_loss):
            run_train_epoch(model=model,
                            loader=cast(Any, [{}, {}, {}]),
                            optimizer=optimizer,
                            device=CPU,
                            config=FineTuningConfig(),
                            weights=WEIGHTS)
        self.assertAlmostEqual(float(model.weight.detach()), 0.7, places=5)

    def test_validation_averages_batches_in_eval_mode(self) -> None:
        model = nn.Linear(1, 1)
        model.train()
        modes: list[bool] = []
        values = iter([{
            "loss": 1.0,
            "ct_dsc": 0.4
        }, {
            "loss": 3.0,
            "ct_dsc": math.nan
        }])

        def batch_metrics(**_: object) -> tuple[torch.Tensor, dict[str, float]]:
            modes.append(model.training)
            return torch.zeros(()), next(values)

        with patch("psmareg.training.fine_tuning.compute_batch_metrics",
                   side_effect=batch_metrics):
            metrics = run_validation_epoch(model=model,
                                           loader=cast(Any, [{}, {}]),
                                           device=CPU,
                                           config=FineTuningConfig(),
                                           weights=WEIGHTS)
        self.assertEqual(modes, [False, False])
        self.assertEqual(metrics["loss"], 2.0)
        # A pair without labels does not dilute the selection score.
        self.assertEqual(metrics["ct_dsc"], 0.4)

    def test_log_line_reports_the_main_metrics(self) -> None:
        metrics = dict.fromkeys(METRIC_KEYS, 0.5)
        line = format_metrics("val", metrics)
        self.assertTrue(line.startswith("val_loss=0.500000 | "))
        self.assertIn("val_ct_dsc=0.500000", line)
        self.assertIn("val_pet_occ=0.500000", line)


class CheckpointTests(unittest.TestCase):
    """Check what a checkpoint stores and how it is restored."""

    def _state(self, model: nn.Module,
               optimizer: torch.optim.Optimizer) -> CheckpointState:
        """Return a checkpoint state of epoch four."""
        return CheckpointState(epoch=4,
                               model=model,
                               optimizer=optimizer,
                               train_metrics={"loss": 1.0},
                               val_metrics={"loss": 2.0},
                               config={"lr": 0.1})

    def test_resume_restores_model_optimizer_and_scheduler(self) -> None:
        torch.manual_seed(0)
        model = _TwoStage()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="max",
                                                               factor=0.5,
                                                               patience=0)
        model.coarse(torch.ones(2)).sum().backward()
        optimizer.step()
        scheduler.step(0.5)
        scheduler.step(0.4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            state = replace(self._state(model, optimizer), scheduler=scheduler)
            save_checkpoint(path,
                            state,
                            best_key="best_val_score",
                            best_value=0.75,
                            with_scheduler=True)
            payload = torch.load(path, weights_only=False)
            restored = _TwoStage()
            restored_optimizer = torch.optim.SGD(restored.parameters(),
                                                 lr=0.1,
                                                 momentum=0.9)
            restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                restored_optimizer, mode="max", factor=0.5, patience=0)
            start_epoch, best = resume_checkpoint(path, restored,
                                                  restored_optimizer,
                                                  restored_scheduler, CPU)
        self.assertEqual(list(payload), [
            "epoch", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "train_metrics", "val_metrics",
            "best_val_score", "config"
        ])
        self.assertEqual((start_epoch, best), (5, 0.75))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.05)
        self.assertEqual(restored_scheduler.best, scheduler.best)
        self.assertIn("momentum_buffer",
                      next(iter(restored_optimizer.state.values())))

    def test_pretraining_checkpoints_store_the_best_loss(self) -> None:
        model = _TwoStage()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_val.pt"
            save_checkpoint(path,
                            self._state(model, optimizer),
                            best_key="best_val_loss",
                            best_value=0.25)
            payload = torch.load(path, weights_only=False)
        self.assertNotIn("scheduler_state_dict", payload)
        self.assertEqual(payload["best_val_loss"], 0.25)
        self.assertEqual(payload["epoch"], 4)

    def test_failed_write_preserves_the_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.write_bytes(b"previous checkpoint")
            with patch("psmareg.training.checkpoint_state.torch.save",
                       side_effect=RuntimeError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    save_checkpoint_payload(path, {"epoch": 0})
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual([item.name for item in Path(directory).iterdir()],
                             ["latest.pt"])

    def test_metadata_is_flat_and_names_the_model(self) -> None:
        metadata = checkpoint_config(FineTuningConfig())
        self.assertEqual(metadata["model"], "affine_transmorph_v2_cascade")
        self.assertEqual(metadata["multiscale_corr_scales"], "2,4,8")
        self.assertTrue(
            all(
                isinstance(value, (str, int, float, bool))
                for value in metadata.values()))


class WeightInitializationTests(unittest.TestCase):
    """Check full and prefix-restricted weight loading."""

    def _checkpoint(self, directory: str, model: nn.Module) -> Path:
        """Save the weights of ``model`` and return the path."""
        path = Path(directory) / "weights.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def test_full_loading_copies_every_tensor(self) -> None:
        torch.manual_seed(0)
        source, target = _TwoStage(), _TwoStage()
        with tempfile.TemporaryDirectory() as directory:
            load_model_weights(self._checkpoint(directory, source), target, CPU)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_prefix_restricts_loading_to_one_stage(self) -> None:
        torch.manual_seed(0)
        source, target = _TwoStage(), _TwoStage()
        coarse_before = target.coarse.weight.clone()
        with tempfile.TemporaryDirectory() as directory:
            loaded, candidates = load_compatible_model_weights(
                self._checkpoint(directory, source),
                target,
                CPU,
                prefixes=("refiner.",))
        self.assertEqual((loaded, candidates), (2, 2))
        self.assertTrue(
            torch.equal(target.refiner.weight, source.refiner.weight))
        self.assertTrue(torch.equal(target.coarse.weight, coarse_before))

    def test_tensors_with_another_shape_are_skipped(self) -> None:
        torch.manual_seed(0)
        source, target = _TwoStage(refiner_width=5), _TwoStage()
        refiner_before = target.refiner.weight.clone()
        with tempfile.TemporaryDirectory() as directory:
            loaded, candidates = load_compatible_model_weights(
                self._checkpoint(directory, source), target, CPU)
        self.assertEqual((loaded, candidates), (2, 4))
        self.assertTrue(torch.equal(target.coarse.weight, source.coarse.weight))
        self.assertTrue(torch.equal(target.refiner.weight, refiner_before))


if __name__ == "__main__":
    unittest.main()
