"""Tests for the supervised losses and model factory of pretraining."""

import dataclasses
import unittest

import torch

from psmareg.config.synthetic_pretraining_config import ArchitectureConfig
from psmareg.models.common.registration.cascades import (
    AffineTransMorphDiffCascadeRegistration,
    AffineTransMorphV2CascadeRegistration,
    TransMorphDiffCascadeRegistration,
    TransMorphV2CascadeRegistration,
)
from psmareg.models.common.registration.geometry import voxel_grid_xyz
from psmareg.models.pretraining.factory import build_model
from psmareg.models.pretraining.losses import (
    bending_energy,
    jacobian_hinge_loss,
    macro_ct_label_dice_loss,
    multiscale_field_loss,
    pet_biomarker_preservation_loss,
    robust_endpoint_loss,
    sampled_semantic_dice_loss,
)
from tests.factories import SHAPE, labeled_scan


def _centered_grid() -> torch.Tensor:
    """Return voxel coordinates relative to the volume center."""
    grid = voxel_grid_xyz(SHAPE,
                          device=torch.device("cpu"),
                          dtype=torch.float32)
    return (grid - 16.0).unsqueeze(0)


class FieldLossTests(unittest.TestCase):
    """Field losses vanish for an exact or perfectly regular prediction."""

    def test_supervised_losses_are_minimal_for_the_exact_field(self) -> None:
        torch.manual_seed(0)
        target = 0.1 * torch.randn(1, 3, 10, 12, 14)
        torch.testing.assert_close(robust_endpoint_loss(target, target),
                                   torch.tensor(1e-3))
        torch.testing.assert_close(multiscale_field_loss(target, target),
                                   torch.tensor(0.0))
        self.assertGreater(float(robust_endpoint_loss(target + 1.0, target)),
                           1.0)
        self.assertGreater(float(multiscale_field_loss(target + 1.0, target)),
                           0.5)

    def test_bending_energy_ignores_affine_fields(self) -> None:
        linear = _centered_grid()
        quadratic = 0.1 * linear.square()
        torch.testing.assert_close(bending_energy(linear),
                                   torch.tensor(0.0),
                                   atol=1e-8,
                                   rtol=0)
        self.assertGreater(float(bending_energy(quadratic)), 1e-3)

    def test_jacobian_hinge_penalizes_folding_only(self) -> None:
        identity = torch.zeros(1, 3, 16, 16, 16)
        folded = -2.0 * (_centered_grid()[..., :16, :16, :16] + 16.0)
        self.assertEqual(float(jacobian_hinge_loss(identity)), 0.0)
        self.assertGreater(
            float(jacobian_hinge_loss(folded, sampling_stride=1)), 0.5)


class LabelLossTests(unittest.TestCase):
    """Label losses compare the predicted warp with the target warp."""

    def setUp(self) -> None:
        self.scan = labeled_scan(seed=2)
        self.target = torch.zeros(1, 3, *SHAPE)
        self.target[:, 0] = 2.0

    def test_semantic_dice_prefers_the_target_field(self) -> None:
        label = self.scan["ct_label"]
        exact = sampled_semantic_dice_loss(label,
                                           self.target,
                                           self.target,
                                           downsample_factor=2,
                                           max_classes=3)
        wrong = sampled_semantic_dice_loss(label,
                                           self.target,
                                           -3.0 * self.target,
                                           downsample_factor=2,
                                           max_classes=3)
        self.assertLess(float(exact), 0.05)
        self.assertGreater(float(wrong), float(exact) + 0.1)

    def test_macro_dice_prefers_the_target_field(self) -> None:
        groups = [("body", (1, 2)), ("organ", (5,))]
        exact = macro_ct_label_dice_loss(self.scan["ct_label"], self.target,
                                         self.target, groups, 2, "border")
        wrong = macro_ct_label_dice_loss(self.scan["ct_label"], self.target,
                                         -3.0 * self.target, groups, 2,
                                         "border")
        self.assertLess(float(exact), 0.05)
        self.assertGreater(float(wrong), 0.3)

    def test_biomarker_loss_detects_changed_lesion_volume(self) -> None:
        exact = pet_biomarker_preservation_loss(self.scan["pet"],
                                                self.scan["pet_label"],
                                                self.target, self.target, 1,
                                                "border")
        compressed = pet_biomarker_preservation_loss(self.scan["pet"],
                                                     self.scan["pet_label"],
                                                     self.target,
                                                     0.5 * _centered_grid(), 1,
                                                     "border")
        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(compressed), 0.1)


class ModelFactoryTests(unittest.TestCase):
    """Each recipe model name builds its cascade with the given bounds."""

    def test_names_map_to_the_four_cascades(self) -> None:
        expected = {
            "transmorph_diff_cascade":
                TransMorphDiffCascadeRegistration,
            "transmorph_v2_cascade":
                TransMorphV2CascadeRegistration,
            "affine_transmorph_diff_cascade":
                AffineTransMorphDiffCascadeRegistration,
            "affine_transmorph_v2_cascade":
                AffineTransMorphV2CascadeRegistration,
        }
        for name, cascade in expected.items():
            architecture = dataclasses.replace(ArchitectureConfig(),
                                               model=name,
                                               max_velocity=5.0,
                                               scaling_steps=3)
            model = build_model(architecture)
            self.assertIs(type(model), cascade)
            self.assertEqual(getattr(model.coarse, "max_velocity"), 5.0)
            self.assertEqual(getattr(model, "scaling_steps"), 3)

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model(
                dataclasses.replace(ArchitectureConfig(), model="unknown"))


if __name__ == "__main__":
    unittest.main()
