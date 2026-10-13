"""Tests for the random deformations used by synthetic pretraining."""

import dataclasses
import unittest

import torch

from psmareg.models.common.registration.jacobian import jacobian_determinant
from psmareg.data.synthetic_deformations import (
    AffineSamplingSettings,
    CurriculumSettings,
    PoseAwareSettings,
    SyntheticTargetSettings,
    curriculum_deformation_scale,
    random_affine_displacement,
    random_elastic_velocity,
    random_pose_aware_displacement_from_labels,
    random_synthetic_displacement_components,
    safe_random_synthetic_displacement_components,
    smooth_coarse_target,
)
from tests.factories import labeled_scan

CPU = torch.device("cpu")


class CurriculumTests(unittest.TestCase):
    """The deformation severity grows with the epoch and stays bounded."""

    def _scales(self, epoch: int, settings: CurriculumSettings,
                count: int) -> list[float]:
        """Sample curriculum scales at one epoch."""
        return [
            curriculum_deformation_scale(epoch=epoch,
                                         total_epochs=100,
                                         device=CPU,
                                         settings=settings)
            for _ in range(count)
        ]

    def test_warmup_uses_the_minimum_scale(self) -> None:
        torch.manual_seed(7)
        self.assertEqual(self._scales(0, CurriculumSettings(), 5), [0.25] * 5)

    def test_ramp_widens_the_range_gradually(self) -> None:
        torch.manual_seed(7)
        middle = self._scales(45, CurriculumSettings(), 50)
        self.assertTrue(all(0.25 <= value <= 0.625 + 1e-6 for value in middle))
        self.assertGreater(max(middle), 0.45)

    def test_mature_curriculum_focuses_on_hard_examples(self) -> None:
        torch.manual_seed(11)
        settings = CurriculumSettings(easy_probability=0.2,
                                      extra_hard_probability=0.2,
                                      extra_hard_max_scale=1.15)
        values = self._scales(100, settings, 300)
        self.assertTrue(all(0.25 <= value <= 1.15 for value in values))
        easy = sum(value < 0.75 for value in values) / len(values)
        extra_hard = sum(value > 1.0 for value in values) / len(values)
        self.assertAlmostEqual(easy, 0.2, delta=0.08)
        self.assertAlmostEqual(extra_hard, 0.2, delta=0.08)

    def test_invalid_settings_are_rejected(self) -> None:
        for settings in (CurriculumSettings(min_scale=0.0),
                         CurriculumSettings(warmup_fraction=0.8,
                                            ramp_fraction=0.5),
                         CurriculumSettings(easy_probability=0.9,
                                            extra_hard_probability=0.3),
                         CurriculumSettings(extra_hard_max_scale=0.5)):
            with self.assertRaises(ValueError):
                curriculum_deformation_scale(0, 10, CPU, settings)


class RandomFieldTests(unittest.TestCase):
    """Sampled fields respect their bounds and remain invertible."""

    def test_affine_sampler_respects_the_translation_bound(self) -> None:
        torch.manual_seed(0)
        field = random_affine_displacement(
            (4, 16, 16, 16), AffineSamplingSettings(3.0, 0.0, 0.0), CPU)
        self.assertEqual(field.shape, (4, 3, 16, 16, 16))
        self.assertLessEqual(float(field.abs().max()), 3.0)
        # Without rotation or scaling each sample is a pure translation.
        self.assertEqual(float(field.std(dim=(2, 3, 4)).max()), 0.0)

    def test_elastic_velocity_respects_its_amplitude(self) -> None:
        torch.manual_seed(0)
        velocity = random_elastic_velocity(2, 16, 16, 16, 2.0, 4, CPU)
        self.assertEqual(velocity.shape, (2, 3, 16, 16, 16))
        self.assertLessEqual(float(velocity.abs().max()), 2.0)
        self.assertGreater(float(velocity.abs().max()), 0.0)

    def test_composed_field_returns_its_affine_component(self) -> None:
        torch.manual_seed(3)
        field, affine = random_synthetic_displacement_components(
            image=torch.zeros(2, 2, 12, 14, 16),
            max_elastic_displacement=0.0,
            elastic_control_point_spacing=6,
            max_translation=1.0,
            max_rotation_degrees=2.0,
            max_scaling=0.02,
        )
        self.assertEqual(field.shape, (2, 3, 12, 14, 16))
        # With no elastic part the composed field is the affine field.
        torch.testing.assert_close(field, affine, atol=1e-5, rtol=0)

    def test_safe_sampler_enforces_the_minimum_jacobian(self) -> None:
        torch.manual_seed(0)
        settings = SyntheticTargetSettings(max_elastic_displacement=1.5,
                                           elastic_control_point_spacing=6,
                                           max_translation=2.0,
                                           max_rotation_degrees=5.0,
                                           max_scaling=0.05,
                                           minimum_jacobian=0.3,
                                           max_attempts=8,
                                           jacobian_stride=1)
        field, affine = safe_random_synthetic_displacement_components(
            torch.zeros(2, 2, 16, 16, 16), settings)
        self.assertEqual(affine.shape, field.shape)
        self.assertGreaterEqual(
            float(jacobian_determinant(field, sampling_stride=1).min()), 0.3)

    def test_safe_sampler_fails_when_no_target_is_safe(self) -> None:
        torch.manual_seed(0)
        settings = SyntheticTargetSettings(max_elastic_displacement=6.0,
                                           elastic_control_point_spacing=4,
                                           max_translation=2.0,
                                           max_rotation_degrees=5.0,
                                           max_scaling=0.05,
                                           minimum_jacobian=0.3,
                                           max_attempts=3,
                                           jacobian_stride=1)
        with self.assertRaisesRegex(RuntimeError, "safe synthetic target"):
            safe_random_synthetic_displacement_components(
                torch.zeros(2, 2, 16, 16, 16), settings)

    def test_coarse_target_keeps_only_low_frequencies(self) -> None:
        torch.manual_seed(0)
        noise = torch.randn(1, 3, 16, 16, 16)
        smoothed = smooth_coarse_target(noise, downsample_factor=4)
        self.assertEqual(smoothed.shape, noise.shape)
        self.assertLess(float(smoothed.std()), 0.7 * float(noise.std()))
        # A linear field is fully representable at the coarse resolution.
        ramp = torch.zeros(1, 3, 16, 16, 16)
        ramp[:, 0] = torch.arange(16.0).view(1, 1, 16)
        torch.testing.assert_close(smooth_coarse_target(ramp,
                                                        downsample_factor=4),
                                   ramp,
                                   atol=1e-5,
                                   rtol=0)


class PoseAwareFieldTests(unittest.TestCase):
    """Articulated motion is bounded and needs a body to act on."""

    def setUp(self) -> None:
        self.label = labeled_scan()["ct_label"]
        self.settings = PoseAwareSettings(max_translation=3.0,
                                          max_rotation_degrees=5.0,
                                          max_scaling=0.02,
                                          region_blur_downsample=4,
                                          min_region_fraction=0.001)

    def test_field_moves_body_regions_within_bounds(self) -> None:
        torch.manual_seed(0)
        field = random_pose_aware_displacement_from_labels(
            self.label, self.settings)
        self.assertEqual(field.shape, (1, 3, 32, 32, 32))
        self.assertTrue(torch.isfinite(field).all())
        self.assertGreater(float(field.abs().max()), 0.1)
        # Translation plus small rotation and scaling of a 32-voxel volume.
        self.assertLess(float(field.abs().max()), 3.0 + 4.0)

    def test_empty_or_tiny_bodies_are_left_untouched(self) -> None:
        torch.manual_seed(0)
        empty = random_pose_aware_displacement_from_labels(
            torch.zeros_like(self.label), self.settings)
        self.assertEqual(float(empty.abs().max()), 0.0)
        strict = dataclasses.replace(self.settings, min_region_fraction=0.9)
        skipped = random_pose_aware_displacement_from_labels(self.label, strict)
        self.assertEqual(float(skipped.abs().max()), 0.0)

    def test_wrong_label_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            random_pose_aware_displacement_from_labels(self.label[0],
                                                       self.settings)


if __name__ == "__main__":
    unittest.main()
