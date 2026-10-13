"""Tests for the deterministic translation and bone affine pre-alignment."""

import unittest

import torch

from psmareg.models.common.registration.prealignment.bone_affine import (
    BoneAffineAligner,
    BoneAffineSettings,
)
from psmareg.models.common.registration.prealignment.deterministic import (
    TranslationPrealignSettings,
    constant_translation_field,
    estimate_lowres_ct_translation_field,
)
from psmareg.models.common.registration.warping import warp_image_3d
from tests.factories import labeled_scan


class TranslationPrealignTests(unittest.TestCase):
    """Check that the grid search recovers a known rigid shift."""

    def setUp(self) -> None:
        self.fixed = labeled_scan(seed=1)["ct"]
        self.settings = TranslationPrealignSettings(downsample_factor=2,
                                                    search_radius=3,
                                                    search_step=1)

    def test_constant_field_broadcasts_the_translation(self) -> None:
        translation = torch.tensor([[1.0, -2.0, 3.0]])
        field = constant_translation_field((1, 1, 4, 5, 6), translation)
        self.assertEqual(field.shape, (1, 3, 4, 5, 6))
        torch.testing.assert_close(field[0, :, 2, 3, 4], translation[0])

    def test_recovers_shifts_along_each_axis(self) -> None:
        for dims, shifts, expected in (((4,), (4,), [4.0, 0.0, 0.0]),
                                       ((2, 3), (3, -4), [0.0, -4.0, 3.0])):
            moving = torch.roll(self.fixed, shifts=shifts, dims=dims)
            field, translation = estimate_lowres_ct_translation_field(
                self.fixed, moving, self.settings)
            torch.testing.assert_close(translation,
                                       torch.tensor([expected]),
                                       atol=0.5,
                                       rtol=0)
            self.assertEqual(field.shape, (1, 3, 32, 32, 32))

    def test_estimated_field_realigns_the_moving_image(self) -> None:
        moving = torch.roll(self.fixed, shifts=4, dims=4)
        field, _ = estimate_lowres_ct_translation_field(self.fixed, moving,
                                                        self.settings)
        before = (moving - self.fixed).abs().mean()
        after = (warp_image_3d(moving, field) - self.fixed).abs().mean()
        self.assertLess(float(after), 0.2 * float(before))

    def test_identical_images_need_no_translation(self) -> None:
        _, translation = estimate_lowres_ct_translation_field(
            self.fixed, self.fixed.clone(), self.settings)
        torch.testing.assert_close(translation,
                                   torch.zeros(1, 3),
                                   atol=1e-4,
                                   rtol=0)


class BoneAffineAlignerTests(unittest.TestCase):
    """Exercise fitting, evaluation, and conservative rejection."""

    def test_fit_and_evaluate_under_no_grad(self) -> None:
        torch.manual_seed(3)
        fixed = torch.rand(1, 1, 16, 16, 16)
        settings = BoneAffineSettings(downsample_factors=(2,),
                                      iterations_per_level=1,
                                      min_improvement=float("inf"))
        aligner = BoneAffineAligner(settings)
        with torch.no_grad():
            field, info = aligner.fit(fixed, fixed.clone()).evaluate()
        self.assertEqual(tuple(field.shape), (1, 3, 16, 16, 16))
        self.assertFalse(info.accepted)
        self.assertEqual(int(torch.count_nonzero(field)), 0)

    def test_accepted_alignment_reduces_the_bone_mismatch(self) -> None:
        fixed = torch.zeros(1, 1, 32, 32, 32)
        fixed[..., 6:26, 6:26, 6:26] = 0.4
        fixed[..., 10:14, 10:22, 10:22] = 0.9
        moving = torch.roll(fixed, shifts=3, dims=4)
        aligner = BoneAffineAligner(
            BoneAffineSettings(downsample_factors=(2,),
                               iterations_per_level=30))
        field, info = aligner.fit(fixed, moving).evaluate()
        self.assertTrue(info.accepted)
        self.assertGreater(info.affine_score, info.baseline_score)
        before = (moving - fixed).abs().mean()
        after = (warp_image_3d(moving, field) - fixed).abs().mean()
        self.assertLess(float(after), float(before))

    def test_evaluate_requires_fit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Fit the bone affine"):
            BoneAffineAligner().evaluate()

    def test_fit_rejects_batches_and_mismatched_shapes(self) -> None:
        aligner = BoneAffineAligner()
        with self.assertRaises(ValueError):
            aligner.fit(torch.zeros(2, 1, 8, 8, 8), torch.zeros(2, 1, 8, 8, 8))
        with self.assertRaises(ValueError):
            aligner.fit(torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 1, 8, 8, 6))


if __name__ == "__main__":
    unittest.main()
