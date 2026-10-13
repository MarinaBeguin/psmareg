"""Tests for the real-pair and synthetic-pair augmentations."""

import dataclasses
import unittest

import torch

from psmareg.config.synthetic_pretraining_config import AugmentationConfig
from psmareg.data.augmentations import (
    independent_intensity_augmentation,
    random_affine_augmentation_field,
)
from psmareg.data.synthetic_augmentations import (
    apply_independent_fov_dropout,
    ct_label_to_random_synthetic_ct,
    independent_pet_intensity_augmentation,
    label_to_random_synthetic_pet,
    random_resolution_augmentation,
    synthmorph_like_pair_from_labels,
)
from tests.factories import SHAPE, labeled_scan


class RealPairAugmentationTests(unittest.TestCase):
    """Augmentations of real pairs stay mild and keep the background."""

    def test_affine_field_is_zero_without_any_range(self) -> None:
        field = random_affine_augmentation_field(torch.zeros(1, 1, 8, 10, 12),
                                                 max_translation=0.0,
                                                 max_rotation_degrees=0.0,
                                                 max_scaling=0.0)
        torch.testing.assert_close(field, torch.zeros_like(field))

    def test_affine_field_is_bounded_and_differs_per_sample(self) -> None:
        torch.manual_seed(0)
        field = random_affine_augmentation_field(torch.zeros(2, 1, 8, 10, 12),
                                                 max_translation=3.0,
                                                 max_rotation_degrees=0.0,
                                                 max_scaling=0.0)
        self.assertEqual(field.shape, (2, 3, 8, 10, 12))
        self.assertLessEqual(float(field.abs().max()), 3.0)
        self.assertFalse(torch.equal(field[0], field[1]))

    def test_intensity_augmentation_keeps_range_and_background(self) -> None:
        torch.manual_seed(17)
        image = torch.linspace(0, 1, 64).reshape(1, 1, 4, 4, 4)
        image[..., 0, :, :] = 0.0
        for is_pet in (False, True):
            augmented = independent_intensity_augmentation(image, is_pet=is_pet)
            self.assertTrue(torch.isfinite(augmented).all())
            self.assertGreaterEqual(float(augmented.min()), 0.0)
            self.assertLessEqual(float(augmented.max()), 1.0)
            self.assertFalse(torch.allclose(image, augmented))
            self.assertEqual(float(augmented[..., 0, :, :].abs().max()), 0.0)


class SyntheticImageTests(unittest.TestCase):
    """Images synthesized from labels follow the anatomy and the lesions."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        scan = labeled_scan()
        self.ct_label = scan["ct_label"]
        self.pet_label = scan["pet_label"]

    def test_pet_intensity_augmentation_changes_the_scale(self) -> None:
        pet = torch.linspace(0.0, 1.0, 16).view(1, 1, 2, 2, 4)
        augmented = independent_pet_intensity_augmentation(pet,
                                                           scale_log_std=0.25,
                                                           gamma_range=0.15,
                                                           bias_range=0.0)
        self.assertEqual(augmented.shape, pet.shape)
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertFalse(torch.allclose(augmented, pet))

    def test_synthetic_ct_keeps_air_darker_than_tissue(self) -> None:
        ct = ct_label_to_random_synthetic_ct(self.ct_label,
                                             noise_std=0.0,
                                             bias_amplitude=0.0)
        self.assertEqual(ct.shape, self.ct_label.shape)
        self.assertGreaterEqual(float(ct.min()), 0.0)
        self.assertLessEqual(float(ct.max()), 1.0)
        # Air stays dark while every tissue class gets a brighter intensity.
        self.assertLessEqual(float(ct[self.ct_label == 0].max()), 0.08 + 1e-6)
        self.assertGreater(float(ct[self.ct_label > 0].mean()), 0.1)

    def test_synthetic_pet_is_hot_in_lesions_unless_dropped(self) -> None:
        lesion = self.pet_label > 0
        body = (self.ct_label > 0) & ~lesion
        pet = label_to_random_synthetic_pet(self.ct_label,
                                            self.pet_label,
                                            noise_std=0.0,
                                            lesion_dropout_probability=0.0,
                                            false_hot_probability=0.0)
        self.assertGreater(float(pet[lesion].mean()),
                           3.0 * float(pet[body].mean()))
        self.assertEqual(float(pet[self.ct_label == 0].max()), 0.0)
        dropped = label_to_random_synthetic_pet(self.ct_label,
                                                self.pet_label,
                                                noise_std=0.0,
                                                lesion_dropout_probability=1.0,
                                                false_hot_probability=0.0)
        self.assertLess(float(dropped[lesion].mean()),
                        0.5 * float(pet[lesion].mean()))

    def test_fov_dropout_truncates_both_modalities_identically(self) -> None:
        image = torch.ones(2, 1, 16, 16, 16)
        ct, pet, applied = apply_independent_fov_dropout(image,
                                                         image,
                                                         probability=1.0,
                                                         max_fraction=0.3)
        self.assertTrue(applied)
        self.assertTrue(torch.equal(ct, pet))
        self.assertGreaterEqual(float(ct.mean()), 0.7 - 1e-6)
        self.assertLess(float(ct.mean()), 1.0)
        ct, pet, applied = apply_independent_fov_dropout(image,
                                                         image,
                                                         probability=0.0,
                                                         max_fraction=0.3)
        self.assertFalse(applied)
        self.assertTrue(torch.equal(ct, image))

    def test_resolution_augmentation_blurs_without_resizing(self) -> None:
        image = torch.rand(1, 1, 16, 16, 16)
        blurred = random_resolution_augmentation(image,
                                                 probability=1.0,
                                                 min_scale=0.4)
        self.assertEqual(blurred.shape, image.shape)
        self.assertLess(float(blurred.std()), float(image.std()))
        untouched = random_resolution_augmentation(image,
                                                   probability=0.0,
                                                   min_scale=0.4)
        self.assertTrue(torch.equal(untouched, image))


class SynthMorphPairTests(unittest.TestCase):
    """A label-driven pair shares geometry but not appearance."""

    def setUp(self) -> None:
        scan = labeled_scan()
        self.ct_label = scan["ct_label"]
        self.pet_label = scan["pet_label"]
        self.target = torch.zeros(1, 3, *SHAPE)
        self.target[:, 0] = 3.0
        self.augmentation = dataclasses.replace(
            AugmentationConfig(),
            synthmorph_like=True,
            synthmorph_fov_dropout_probability=0.0,
            synthmorph_pet_lesion_dropout_probability=0.0,
            synthmorph_pet_false_hot_probability=0.0)

    def test_fixed_image_follows_the_target_displacement(self) -> None:
        torch.manual_seed(0)
        moving_ct, moving_pet, fixed_ct, fixed_pet, fov = (
            synthmorph_like_pair_from_labels(self.ct_label, self.pet_label,
                                             self.target, self.augmentation,
                                             "zeros"))
        self.assertFalse(fov)
        for image in (moving_ct, moving_pet, fixed_ct, fixed_pet):
            self.assertEqual(image.shape, (1, 1, *SHAPE))
            self.assertTrue(torch.isfinite(image).all())
        # The fixed lesions sit where the target field pulls them from.
        fixed_lesion = torch.roll(self.pet_label, shifts=-3, dims=4) > 0
        outside = (torch.roll(self.ct_label, shifts=-3, dims=4)
                   > 0) & (~fixed_lesion)
        self.assertGreater(float(fixed_pet[fixed_lesion].mean()),
                           2.0 * float(fixed_pet[outside].mean()))
        # Both images are styled independently.
        self.assertFalse(torch.equal(moving_ct, fixed_ct))

    def test_pair_can_be_built_without_lesion_labels(self) -> None:
        torch.manual_seed(0)
        outputs = synthmorph_like_pair_from_labels(self.ct_label, None,
                                                   self.target,
                                                   self.augmentation, "zeros")
        self.assertEqual(outputs[1].shape, (1, 1, *SHAPE))

    def test_fov_dropout_is_reported(self) -> None:
        torch.manual_seed(0)
        augmentation = dataclasses.replace(
            self.augmentation, synthmorph_fov_dropout_probability=1.0)
        outputs = synthmorph_like_pair_from_labels(self.ct_label,
                                                   self.pet_label, self.target,
                                                   augmentation, "zeros")
        self.assertTrue(outputs[4])


if __name__ == "__main__":
    unittest.main()
