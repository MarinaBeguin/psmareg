"""Tests for the CT-label, PET, overlap, and smoothness fine-tuning losses."""

import unittest

import torch

from psmareg.config.fine_tuning_config import PetLossConfig
from psmareg.models.fine_tuning.losses import (
    adaptive_pet_correspondence_weight,
    capped_pet_mse_loss,
    displacement_smoothness_loss,
    downsample_labels_and_displacement,
    macro_label_dice_loss,
    multiclass_label_dice_loss,
    pet_hot_region_mse_loss,
    valid_overlap_hinge_loss,
)
from tests.factories import SHAPE, labeled_scan


class LabelDiceLossTests(unittest.TestCase):
    """Label Dice losses reward fields that realign the anatomy."""

    def setUp(self) -> None:
        self.fixed = labeled_scan(0)["ct_label"].squeeze(1)
        self.moving = labeled_scan(3)["ct_label"].squeeze(1)
        self.zero = torch.zeros(1, 3, *SHAPE)
        self.correct = self.zero.clone()
        self.correct[:, 0] = 3.0

    def test_multiclass_dice_prefers_the_aligning_field(self) -> None:
        aligned = multiclass_label_dice_loss(self.fixed,
                                             self.moving,
                                             self.correct,
                                             num_classes=6)
        misaligned = multiclass_label_dice_loss(self.fixed,
                                                self.moving,
                                                self.zero,
                                                num_classes=6)
        self.assertLess(float(aligned), 0.02)
        self.assertGreater(float(misaligned), float(aligned) + 0.1)

    def test_multiclass_dice_gradient_points_toward_alignment(self) -> None:
        field = self.zero.clone().requires_grad_(True)
        loss = multiclass_label_dice_loss(self.fixed,
                                          self.moving,
                                          field,
                                          num_classes=6)
        gradient, = torch.autograd.grad(loss, field)
        # Descending the loss must increase the x displacement toward +3.
        self.assertLess(float(gradient[:, 0].sum()), 0.0)

    def test_macro_dice_uses_the_given_groups(self) -> None:
        groups = [("body", (1, 2, 5)), ("organ", (5,))]
        aligned = macro_label_dice_loss(self.fixed, self.moving, self.correct,
                                        groups)
        misaligned = macro_label_dice_loss(self.fixed, self.moving, self.zero,
                                           groups)
        self.assertLess(float(aligned), 0.02)
        self.assertGreater(float(misaligned), float(aligned))

    def test_downsampling_rescales_the_displacement(self) -> None:
        fixed, moving, field = downsample_labels_and_displacement(
            self.fixed, self.moving, self.correct, 2)
        self.assertEqual(fixed.shape, (1, 16, 16, 16))
        self.assertEqual(moving.shape, (1, 16, 16, 16))
        torch.testing.assert_close(field[:, 0],
                                   torch.full_like(field[:, 0], 1.5))
        same = downsample_labels_and_displacement(self.fixed, self.moving,
                                                  self.correct, 1)
        self.assertIs(same[2], self.correct)


class PetLossTests(unittest.TestCase):
    """PET losses are robust to outliers and to missing correspondence."""

    def setUp(self) -> None:
        self.pet = labeled_scan(seed=2)["pet"]
        self.other = labeled_scan(seed=3)["pet"]

    def test_capped_mse_is_zero_for_identical_images(self) -> None:
        self.assertEqual(
            float(capped_pet_mse_loss(self.pet, self.pet, None, 0.995)), 0.0)
        self.assertGreater(
            float(capped_pet_mse_loss(self.other, self.pet, None, 0.995)), 0.0)

    def test_capped_mse_limits_the_effect_of_extreme_uptake(self) -> None:
        warped = self.pet.clone()
        warped[..., 2, 2, 2] = 50.0
        plain = capped_pet_mse_loss(warped, self.pet, None, 1.0)
        capped = capped_pet_mse_loss(warped, self.pet, None, 0.995)
        self.assertLess(float(capped), 0.01 * float(plain))

    def test_capped_mse_respects_the_valid_mask(self) -> None:
        warped = self.pet.clone()
        warped[..., :8] += 0.5
        mask = torch.ones_like(self.pet)
        mask[..., :8] = 0.0
        self.assertEqual(
            float(capped_pet_mse_loss(warped, self.pet, mask, 1.0)), 0.0)

    def test_hot_region_loss_detects_displaced_uptake(self) -> None:

        def hot_loss(warped: torch.Tensor) -> float:
            """Compare the hot regions of the fixed and warped PET."""
            return float(
                pet_hot_region_mse_loss(self.pet,
                                        warped,
                                        valid_mask=None,
                                        quantile=0.97,
                                        upper_quantile=0.995,
                                        warped_weight=0.04,
                                        smooth_kernel=7))

        self.assertEqual(hot_loss(self.pet), 0.0)
        self.assertGreater(hot_loss(torch.roll(self.pet, shifts=6, dims=4)),
                           0.01)

    def test_adaptive_weight_drops_without_hot_correspondence(self) -> None:
        settings = PetLossConfig(adaptive_pet_loss=True,
                                 adaptive_pet_min_weight=0.15)
        weight, dice = adaptive_pet_correspondence_weight(
            self.pet, self.pet, None, settings)
        self.assertEqual(float(weight), 1.0)
        self.assertAlmostEqual(dice, 1.0, places=5)
        far = torch.roll(self.pet, shifts=12, dims=4)
        weight, dice = adaptive_pet_correspondence_weight(
            self.pet, far, None, settings)
        self.assertAlmostEqual(float(weight), 0.15, places=6)
        self.assertLess(dice, 0.01)
        self.assertFalse(weight.requires_grad)


class RegularizationLossTests(unittest.TestCase):
    """Overlap and smoothness terms penalize only avoidable defects."""

    def test_overlap_loss_preserves_only_achievable_coverage(self) -> None:
        overlap = torch.full((2, 1, 2, 2, 2), 0.70)
        reference = torch.stack((
            torch.full((1, 2, 2, 2), 0.95),
            torch.full((1, 2, 2, 2), 0.72),
        ))
        loss = valid_overlap_hinge_loss(overlap,
                                        reference_overlap=reference,
                                        minimum_overlap=0.90)
        torch.testing.assert_close(loss, torch.tensor((0.20 + 0.02) / 2))

    def test_overlap_loss_is_zero_above_the_target(self) -> None:
        overlap = torch.full((1, 1, 2, 2, 2), 0.95)
        loss = valid_overlap_hinge_loss(overlap,
                                        reference_overlap=overlap,
                                        minimum_overlap=0.90)
        self.assertEqual(float(loss), 0.0)

    def test_smoothness_ignores_translation_at_every_resolution(self) -> None:
        translation = torch.ones(1, 3, 16, 16, 16)
        ramp = torch.zeros(1, 3, 16, 16, 16)
        ramp[:, 0] = torch.arange(16.0).view(1, 1, 16)
        for factor in (1, 2):
            self.assertEqual(
                float(displacement_smoothness_loss(translation, factor)), 0.0)
            self.assertGreater(
                float(displacement_smoothness_loss(ramp, factor)), 0.0)


if __name__ == "__main__":
    unittest.main()
