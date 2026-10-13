"""Tests for synthetic pair generation and pretraining epochs."""

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from psmareg.models.common.registration.warping import warp_image_3d
from psmareg.models.pretraining.factory import build_model
from psmareg.training.pretraining import (
    average_metrics,
    finite_mean,
    initialize_matching_weights,
    load_checkpoint,
    run_epoch,
)
from psmareg.training.pretraining_batch import compute_batch_loss_and_metrics
from psmareg.training.pretraining_pair import (
    BatchExecutionOptions,
    synthesize_pair,
)
from tests.factories import SHAPE, scan_batch, small_pretraining_config

CPU = torch.device("cpu")
OPTIONS = BatchExecutionOptions(deformation_scale=0.5,
                                similarity_downsample_factor=1,
                                collect_expensive_metrics=True,
                                metric_sampling_stride=2)


class SynthesizePairTests(unittest.TestCase):
    """A synthetic pair comes with the deformation that relates its scans."""

    def test_fixed_scan_is_the_moving_scan_warped_by_the_target(self) -> None:
        config = small_pretraining_config("a")
        torch.manual_seed(0)
        pair = synthesize_pair(scan_batch(), CPU, config, OPTIONS)
        self.assertEqual(pair.target_displacement.shape, (1, 3, *SHAPE))
        self.assertGreater(float(pair.target_displacement.abs().max()), 0.1)
        expected = warp_image_3d(pair.moving_ct, pair.target_displacement)
        torch.testing.assert_close(pair.fixed_ct, expected)
        self.assertFalse(pair.pose_aware_applied)
        self.assertFalse(pair.synthmorph_fov_augmented)

    def test_deformation_scale_controls_the_target_magnitude(self) -> None:
        config = small_pretraining_config("a")
        magnitudes = []
        for scale in (0.25, 1.0):
            options = BatchExecutionOptions(scale, 1, False, 2)
            torch.manual_seed(0)
            pair = synthesize_pair(scan_batch(), CPU, config, options)
            magnitudes.append(float(pair.target_displacement.abs().mean()))
        self.assertLess(magnitudes[0], 0.6 * magnitudes[1])

    def test_label_driven_pair_requires_and_uses_labels(self) -> None:
        config = small_pretraining_config("a",
                                          synthmorph_like=True,
                                          pose_aware_probability=1.0)
        torch.manual_seed(0)
        pair = synthesize_pair(scan_batch(), CPU, config, OPTIONS)
        self.assertTrue(pair.pose_aware_applied)
        self.assertTrue(torch.isfinite(pair.fixed_ct).all())
        self.assertFalse(torch.equal(pair.moving_ct, scan_batch()["ct"]))
        without_labels = {
            key: value
            for key, value in scan_batch().items()
            if not key.endswith("label")
        }
        with self.assertRaisesRegex(ValueError, "requires CT labels"):
            synthesize_pair(without_labels, CPU, config, OPTIONS)


class PretrainingBatchTests(unittest.TestCase):
    """Run the supervised objective of each stage on one scan."""

    def test_each_stage_gives_a_finite_loss_and_gradients(self) -> None:
        for stage, has_affine in (("a", False), ("b", False), ("c", True)):
            config = small_pretraining_config(stage)
            torch.manual_seed(0)
            model = build_model(config.architecture)
            torch.manual_seed(1)
            loss, metrics = compute_batch_loss_and_metrics(
                model, scan_batch(), CPU, config, OPTIONS)
            loss.backward()
            self.assertTrue(math.isfinite(float(loss)), stage)
            self.assertAlmostEqual(metrics["loss"], float(loss), places=4)
            self.assertTrue(
                all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None), stage)
            self.assertEqual(metrics["loss_affine_field"] > 0.0, has_affine,
                             stage)
            self.assertGreater(metrics["target_endpoint"], 0.0)
            self.assertTrue(math.isfinite(metrics["pred_ndv_percent"]))

    def test_cheap_pass_skips_the_expensive_metrics(self) -> None:
        config = small_pretraining_config("a")
        torch.manual_seed(0)
        model = build_model(config.architecture)
        cheap = BatchExecutionOptions(0.5, 1, False, 2)
        _, metrics = compute_batch_loss_and_metrics(model, scan_batch(), CPU,
                                                    config, cheap)
        _, full = compute_batch_loss_and_metrics(model, scan_batch(), CPU,
                                                 config, OPTIONS)
        self.assertEqual(list(metrics), list(full))
        for key in ("lncc_ct_before", "mse_pet_before", "pred_ndv_percent",
                    "target_jac_p01"):
            self.assertTrue(math.isnan(metrics[key]), key)
            self.assertTrue(math.isfinite(full[key]), key)


class PretrainingEpochTests(unittest.TestCase):
    """Check the training and validation passes over a loader."""

    def setUp(self) -> None:
        self.config = small_pretraining_config("a")
        torch.manual_seed(0)
        self.model = build_model(self.config.architecture)
        self.loader: Any = [scan_batch(1), scan_batch(2)]

    def test_training_updates_the_weights(self) -> None:
        before = [
            parameter.detach().clone() for parameter in self.model.parameters()
        ]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        metrics = run_epoch(self.model,
                            self.loader,
                            CPU,
                            optimizer,
                            None,
                            self.config,
                            epoch=0)
        self.assertTrue(self.model.training)
        self.assertTrue(math.isfinite(metrics["loss"]))
        # Warm-up epochs use the minimum curriculum severity.
        self.assertEqual(metrics["deformation_scale"],
                         self.config.curriculum.curriculum_min_scale)
        changed = sum(not torch.equal(old, new)
                      for old, new in zip(before, self.model.parameters()))
        self.assertGreater(changed, 0)

    def test_validation_is_repeatable_and_leaves_the_rng_untouched(
            self) -> None:
        before = [
            parameter.detach().clone() for parameter in self.model.parameters()
        ]
        torch.manual_seed(3)
        state = torch.random.get_rng_state()
        first = run_epoch(self.model,
                          self.loader,
                          CPU,
                          None,
                          None,
                          self.config,
                          epoch=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        torch.manual_seed(99)
        second = run_epoch(self.model,
                           self.loader,
                           CPU,
                           None,
                           None,
                           self.config,
                           epoch=3)
        self.assertFalse(self.model.training)
        self.assertEqual(first, second)
        self.assertEqual(first["deformation_scale"], 1.0)
        self.assertEqual(first["symmetric_applied"], 0.0)
        for old, new in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_batch_limits_shorten_an_epoch(self) -> None:
        limited = small_pretraining_config("a", max_val_batches=1)
        one = run_epoch(self.model,
                        self.loader,
                        CPU,
                        None,
                        None,
                        limited,
                        epoch=0)
        first_only = run_epoch(self.model,
                               self.loader[:1],
                               CPU,
                               None,
                               None,
                               self.config,
                               epoch=0)
        self.assertEqual(one, first_only)


class MetricAveragingTests(unittest.TestCase):
    """Epoch averages skip metrics that were not computed for a batch."""

    def test_non_finite_values_are_ignored(self) -> None:
        averaged = average_metrics([{
            "loss": 1.0,
            "jacobian": math.nan
        }, {
            "loss": 3.0,
            "jacobian": 2.0
        }])
        self.assertEqual(averaged, {"loss": 2.0, "jacobian": 2.0})
        self.assertTrue(math.isnan(finite_mean([math.nan, math.inf])))


class PretrainingCheckpointTests(unittest.TestCase):
    """Check resuming a stage and initializing the next one."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.source = build_model(small_pretraining_config("a").architecture)
        self.optimizer = torch.optim.AdamW(self.source.parameters(), lr=1e-3)

    def _save(self, directory: str) -> Path:
        """Write a Stage A checkpoint of epoch seven."""
        path = Path(directory) / "best_val.pt"
        torch.save(
            {
                "epoch": 7,
                "model_state_dict": self.source.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": 0.5,
            }, path)
        return path

    def test_resume_can_inherit_or_reset_the_best_loss(self) -> None:
        torch.manual_seed(1)
        target = build_model(small_pretraining_config("a").architecture)
        optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory)
            resumed = load_checkpoint(path, target, optimizer, CPU)
            fresh = load_checkpoint(path,
                                    target,
                                    optimizer,
                                    CPU,
                                    load_optimizer_state=False,
                                    inherit_best_val_loss=False)
        self.assertEqual(resumed, (8, 0.5))
        self.assertEqual(fresh, (8, math.inf))
        for key, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_stage_c_reuses_the_deformable_weights_of_stage_b(self) -> None:
        torch.manual_seed(1)
        target = build_model(small_pretraining_config("c").architecture)
        affine_before = {
            key: value.clone()
            for key, value in target.state_dict().items()
            if key.startswith("affine.")
        }
        with tempfile.TemporaryDirectory() as directory:
            loaded = initialize_matching_weights(self._save(directory), target,
                                                 CPU)
        self.assertEqual(loaded, len(self.source.state_dict()))
        state = target.state_dict()
        for key, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)
        for key, value in affine_before.items():
            self.assertTrue(torch.equal(value, state[key]), key)


if __name__ == "__main__":
    unittest.main()
