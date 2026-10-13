"""Tests for the lesion-preserving losses of the fine-tuning phase."""

import dataclasses
import unittest

import torch

from psmareg.config.fine_tuning_config import LesionLossConfig
from psmareg.models.common.registration.geometry import voxel_grid_xyz
from psmareg.models.fine_tuning.lesion_losses import (
    mask_pet_similarity_around_moving_lesions,
    pet_lesion_occupancy_and_translation_losses,
    pet_lesion_strain_loss,
)
from tests.factories import SHAPE, labeled_scan


def _fields() -> dict[str, torch.Tensor]:
    """Return rigid and non-rigid pull-back fields on the test grid."""
    grid = voxel_grid_xyz(SHAPE,
                          device=torch.device("cpu"),
                          dtype=torch.float32)
    centered = (grid - 16.0).unsqueeze(0)
    zero = torch.zeros(1, 3, *SHAPE)
    translation = zero.clone()
    translation[:, 0] = 2.0
    subvoxel = zero.clone()
    subvoxel[:, 0] = 0.5
    return {
        "identity": zero,
        "translation": translation,
        "subvoxel_translation": subvoxel,
        # Sampling 1.5x farther apart shrinks lesions in the warped image.
        "compression": 0.5 * centered,
        # Sampling closer together enlarges them.
        "expansion": -0.4 * centered,
    }


class LesionOccupancyTests(unittest.TestCase):
    """Rigid lesion motion is free; changing lesion volume is penalized."""

    def setUp(self) -> None:
        scan = labeled_scan(seed=2)
        self.pet = scan["pet"]
        self.label = scan["pet_label"].squeeze(1)
        self.config = LesionLossConfig(pet_lesion_occupancy_downsample=1,
                                       pet_lesion_occupancy_min_voxels=3)
        self.fields = _fields()

    def _losses(self,
                name: str,
                config: LesionLossConfig | None = None) -> list[float]:
        """Return occupancy, translation, and component count."""
        occupancy, translation, components = (
            pet_lesion_occupancy_and_translation_losses(self.pet, self.label,
                                                        self.fields[name], None,
                                                        config or self.config))
        return [float(occupancy), float(translation), float(components)]

    def test_rigid_motion_is_not_penalized(self) -> None:
        for name in ("identity", "translation", "subvoxel_translation"):
            occupancy, translation, components = self._losses(name)
            self.assertAlmostEqual(occupancy, 0.0, places=5, msg=name)
            self.assertAlmostEqual(translation, 0.0, places=5, msg=name)
            self.assertEqual(components, 2.0)

    def test_volume_change_is_penalized(self) -> None:
        for name in ("compression", "expansion"):
            occupancy, translation, _ = self._losses(name)
            self.assertGreater(occupancy, 0.1, msg=name)
            self.assertGreater(translation, 0.01, msg=name)

    def test_small_or_absent_lesions_are_skipped(self) -> None:
        large_threshold = dataclasses.replace(
            self.config, pet_lesion_occupancy_min_voxels=1000)
        self.assertEqual(self._losses("compression", large_threshold),
                         [0.0, 0.0, 0.0])
        empty = pet_lesion_occupancy_and_translation_losses(
            self.pet, torch.zeros_like(self.label), self.fields["compression"],
            None, self.config)
        self.assertEqual([float(value) for value in empty], [0.0, 0.0, 0.0])

    def test_component_limit_keeps_the_largest_lesions(self) -> None:
        single = dataclasses.replace(self.config,
                                     pet_lesion_occupancy_max_components=1)
        self.assertEqual(self._losses("compression", single)[2], 1.0)

    def test_edge_guard_raises_the_weight_of_border_lesions(self) -> None:
        label = torch.zeros_like(self.label)
        label[:, 1:5, 10:14, 10:14] = 1
        guarded = dataclasses.replace(self.config,
                                      pet_lesion_edge_guard_margin_voxels=4,
                                      pet_lesion_edge_guard_weight=3.0)
        plain, _, _ = pet_lesion_occupancy_and_translation_losses(
            self.pet, label, self.fields["compression"], None, self.config)
        weighted, _, _ = pet_lesion_occupancy_and_translation_losses(
            self.pet, label, self.fields["compression"], None, guarded)
        self.assertGreater(float(plain), 0.0)
        torch.testing.assert_close(weighted, 3.0 * plain, rtol=1e-5, atol=0)

    def test_losses_are_differentiable(self) -> None:
        field = self.fields["compression"].clone().requires_grad_(True)
        occupancy, translation, _ = (
            pet_lesion_occupancy_and_translation_losses(self.pet, self.label,
                                                        field, None,
                                                        self.config))
        gradient, = torch.autograd.grad(occupancy + translation, field)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class LesionStrainTests(unittest.TestCase):
    """The strain loss measures non-rigid deformation inside lesions."""

    def setUp(self) -> None:
        self.label = labeled_scan(seed=2)["pet_label"].squeeze(1)
        self.fields = _fields()

    def _strain(self, name: str, label: torch.Tensor | None = None) -> float:
        """Return the strain loss of one named field."""
        return float(
            pet_lesion_strain_loss(self.label if label is None else label,
                                   self.fields[name],
                                   None,
                                   downsample_factor=1,
                                   min_voxels=3))

    def test_translations_have_no_strain(self) -> None:
        for name in ("identity", "translation", "subvoxel_translation"):
            self.assertAlmostEqual(self._strain(name), 0.0, places=6, msg=name)

    def test_scaling_has_strain_growing_with_its_magnitude(self) -> None:
        self.assertGreater(self._strain("expansion"), 0.1)
        self.assertGreater(self._strain("compression"),
                           self._strain("expansion"))

    def test_strain_is_only_measured_inside_lesions(self) -> None:
        self.assertEqual(
            self._strain("compression", torch.zeros_like(self.label)), 0.0)


class LesionSimilarityMaskTests(unittest.TestCase):
    """The PET similarity mask excludes where moving lesions land."""

    def setUp(self) -> None:
        self.label = labeled_scan(seed=2)["pet_label"].squeeze(1)
        self.fields = _fields()

    def test_mask_removes_lesions_and_their_margin(self) -> None:
        keep, removed = mask_pet_similarity_around_moving_lesions(
            None,
            self.label,
            self.fields["identity"],
            dilate=2,
            downsample=1,
            padding_mode="zeros")
        lesion = self.label[0] > 0
        self.assertEqual(keep.shape, (1, 1, *SHAPE))
        self.assertEqual(float(keep[0, 0][lesion].max()), 0.0)
        self.assertEqual(float(keep[0, 0, 0, 0, 0]), 1.0)
        self.assertGreater(float(removed), float(lesion.float().mean()))
        torch.testing.assert_close(removed, 1.0 - keep.mean())

    def test_mask_follows_the_displacement(self) -> None:
        keep, _ = mask_pet_similarity_around_moving_lesions(
            None,
            self.label,
            self.fields["translation"],
            dilate=0,
            downsample=1,
            padding_mode="zeros")
        landed = torch.roll(self.label[0], shifts=-2, dims=2) > 0
        self.assertLess(float(keep[0, 0][landed].max()), 1e-4)

    def test_mask_is_combined_with_the_base_mask(self) -> None:
        base = torch.zeros(1, 1, *SHAPE)
        keep, _ = mask_pet_similarity_around_moving_lesions(
            base,
            self.label,
            self.fields["identity"],
            dilate=1,
            downsample=1,
            padding_mode="zeros")
        self.assertEqual(float(keep.max()), 0.0)

    def test_downsampled_mask_matches_the_full_resolution_mask(self) -> None:
        full, _ = mask_pet_similarity_around_moving_lesions(
            None,
            self.label,
            self.fields["translation"],
            dilate=2,
            downsample=1,
            padding_mode="zeros")
        low, _ = mask_pet_similarity_around_moving_lesions(
            None,
            self.label,
            self.fields["translation"],
            dilate=2,
            downsample=2,
            padding_mode="zeros")
        self.assertLess(float((full - low).abs().mean()), 0.02)


if __name__ == "__main__":
    unittest.main()
