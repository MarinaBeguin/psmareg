"""Tests for one complete fine-tuning batch with the final recipe."""

import math
import unittest

import torch

from psmareg.config.fine_tuning_config import FineTuningConfig
from psmareg.training.fine_tuning_batch import (
    EpochWeights,
    compute_batch_metrics,
)
from psmareg.training.fine_tuning_batch_inputs import prepare_batch
from psmareg.training.fine_tuning_batch_metrics import METRIC_KEYS
from tests.factories import (
    SHAPE,
    final_epoch_weights,
    paired_batch,
    small_final_model,
    small_fine_tuning_config,
)

CPU = torch.device("cpu")
_LABEL_SCORES = ("ct_dsc", "challenge_ct_dsc", "challenge_pet_dsc",
                 "challenge_mtv_percent_error", "challenge_tlg_percent_error")


def _tensor(batch: dict[str, torch.Tensor | list[str]],
            key: str) -> torch.Tensor:
    """Return one tensor entry of a batch."""
    value = batch[key]
    assert isinstance(value, torch.Tensor)
    return value


class FineTuningBatchTests(unittest.TestCase):
    """Run the objective of the published method on a small labeled pair."""

    config: FineTuningConfig
    weights: EpochWeights

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = small_fine_tuning_config()
        cls.weights = final_epoch_weights(cls.config)

    def _run(self, **options: bool) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one evaluation-mode batch with a fixed seed."""
        model = small_final_model(self.config)
        model.eval()
        torch.manual_seed(1)
        return compute_batch_metrics(model, paired_batch(), CPU, self.config,
                                     self.weights, **options)

    def test_training_batch_gives_a_loss_and_gradients_to_all_stages(
            self) -> None:
        model = small_final_model(self.config)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.abs().sum() == 0:
                    parameter.add_(0.02 * torch.randn_like(parameter))
        torch.manual_seed(1)
        loss, metrics = compute_batch_metrics(model,
                                              paired_batch(),
                                              CPU,
                                              self.config,
                                              self.weights,
                                              apply_training_augmentation=True)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss)))
        self.assertAlmostEqual(metrics["loss"], float(loss), places=5)
        for stage in ("affine", "coarse", "refiner"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(f"{stage}.") and parameter.grad is not None
            ]
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients), 0.0,
                stage)

    def test_metrics_follow_the_csv_schema(self) -> None:
        _, metrics = self._run()
        self.assertEqual(tuple(metrics), METRIC_KEYS)
        self.assertTrue(
            all(isinstance(value, float) for value in metrics.values()))
        self.assertEqual(metrics["pet_lesion_occupancy_components"], 2.0)
        self.assertEqual(metrics["pet_lesion_occupancy_weight"],
                         self.weights.lambda_pet_lesion_occupancy)
        self.assertGreater(metrics["valid_overlap"], 0.5)
        self.assertLessEqual(metrics["valid_overlap"], 1.0)

    def test_translation_prealignment_recovers_the_shift(self) -> None:
        _, metrics = self._run()
        # The moving scan is shifted by three voxels; the search step is two.
        self.assertAlmostEqual(metrics["det_translation_x"], 3.0, delta=1.0)
        self.assertAlmostEqual(metrics["det_translation_y"], 0.0, delta=0.5)
        self.assertAlmostEqual(metrics["det_translation_max"],
                               abs(metrics["det_translation_x"]),
                               places=5)

    def test_challenge_scores_are_computed_only_on_request(self) -> None:
        _, plain = self._run()
        _, scored = self._run(compute_challenge_metrics=True)
        self.assertTrue(0.0 < plain["ct_dsc"] <= 1.0)
        for key in _LABEL_SCORES[1:]:
            self.assertTrue(math.isnan(plain[key]), key)
            self.assertTrue(math.isfinite(scored[key]), key)
        self.assertEqual(plain["loss"], scored["loss"])

    def test_pair_without_labels_still_trains_on_images(self) -> None:
        batch = paired_batch()
        for key in [key for key in batch if key.endswith("label")]:
            del batch[key]
        model = small_final_model(self.config)
        model.eval()
        loss, metrics = compute_batch_metrics(model,
                                              batch,
                                              CPU,
                                              self.config,
                                              self.weights,
                                              compute_challenge_metrics=True)
        self.assertTrue(math.isfinite(float(loss)))
        for key in _LABEL_SCORES:
            self.assertTrue(math.isnan(metrics[key]), key)
        self.assertEqual(metrics["pet_lesion_occupancy_components"], 0.0)
        self.assertEqual(metrics["loss_pet_lesion_occupancy"], 0.0)

    def test_zero_weights_remove_the_lesion_terms_from_the_loss(self) -> None:
        loss, metrics = self._run()
        without = EpochWeights(0.0, 0.0, 0.0, 0.0, 0.0)
        model = small_final_model(self.config)
        model.eval()
        torch.manual_seed(1)
        reduced, reduced_metrics = compute_batch_metrics(
            model, paired_batch(), CPU, self.config, without)
        self.assertGreater(metrics["loss_pet_lesion_occupancy"], 0.0)
        self.assertEqual(reduced_metrics["loss_pet_lesion_occupancy"], 0.0)
        self.assertEqual(reduced_metrics["loss_macro_ct_labels"], 0.0)
        self.assertNotEqual(float(loss), float(reduced))

    def test_external_prealignment_is_composed_with_the_model(self) -> None:
        batch = paired_batch()
        external = torch.zeros(1, 3, 16, 16, 16)
        external[:, 1] = 1.0
        batch["prealign_displacement"] = external
        config = small_fine_tuning_config(
            deterministic_translation_prealign=False)
        model = small_final_model(config)
        model.eval()
        _, metrics = compute_batch_metrics(model, batch, CPU, config,
                                           self.weights)
        # The half-resolution field is upsampled and rescaled to two voxels.
        self.assertAlmostEqual(metrics["det_translation_y"], 2.0, places=4)
        self.assertAlmostEqual(metrics["det_translation_x"], 0.0, places=4)


class PrepareBatchTests(unittest.TestCase):
    """Check the augmentation applied before the model sees a pair."""

    def setUp(self) -> None:
        self.config = small_fine_tuning_config(
            train_spatial_aug_probability=1.0,
            train_intensity_aug_probability=1.0)

    def test_validation_batches_are_left_untouched(self) -> None:
        batch = paired_batch()
        generator_state = torch.random.get_rng_state()
        data = prepare_batch(batch,
                             CPU,
                             self.config,
                             apply_training_augmentation=False)
        self.assertTrue(torch.equal(data.fixed_ct, _tensor(batch, "fixed_ct")))
        self.assertTrue(
            torch.equal(data.moving_pet, _tensor(batch, "moving_pet")))
        self.assertIsNone(data.augmentation_validity)
        assert data.moving_ct_label is not None
        self.assertEqual(data.moving_ct_label.shape, (1, *SHAPE))
        self.assertTrue(
            torch.equal(torch.random.get_rng_state(), generator_state))

    def test_spatial_augmentation_moves_images_and_labels_together(
            self) -> None:
        batch = paired_batch()
        torch.manual_seed(0)
        data = prepare_batch(batch,
                             CPU,
                             self.config,
                             apply_training_augmentation=True)
        self.assertIsNotNone(data.augmentation_validity)
        self.assertFalse(
            torch.equal(data.moving_ct, _tensor(batch, "moving_ct")))
        # The fixed labels never move; moving labels follow their image.
        assert data.fixed_ct_label is not None
        assert data.moving_pet_label is not None
        self.assertTrue(
            torch.equal(data.fixed_ct_label,
                        _tensor(batch, "fixed_ct_label").squeeze(1)))
        lesion = data.moving_pet_label > 0
        self.assertGreater(float(data.moving_pet[0, 0][lesion[0]].mean()),
                           2.0 * float(data.moving_pet.mean()))

    def test_augmentation_is_reproducible_for_a_seed(self) -> None:
        results = []
        for _ in range(2):
            torch.manual_seed(5)
            data = prepare_batch(paired_batch(),
                                 CPU,
                                 self.config,
                                 apply_training_augmentation=True)
            results.append((data.moving_ct, torch.rand(1)))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))


if __name__ == "__main__":
    unittest.main()
