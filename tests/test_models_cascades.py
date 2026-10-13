"""Tests for the affine heads and the registration cascades."""

import unittest

import torch
from torch import nn

from psmareg.models.common.registration.affine import (
    AffineRegistrationHead,
    StrongAffineRegistrationHead,
    affine_parameters_to_displacement,
    body_center_translation,
)
from psmareg.models.common.registration.blocks import (
    conv_norm_activation,
    pairwise_ct_pet_features,
)
from psmareg.models.common.registration.cascades import (
    AffineCascadeOutput,
    AffineCascadeSettings,
    AffineTransMorphDiffCascadeRegistration,
    AffineTransMorphV2CascadeRegistration,
    TransMorphDiffCascadeRegistration,
    TransMorphV2CascadeRegistration,
)
from psmareg.models.common.registration.warping import compose_displacements
from psmareg.models.pretraining.factory import (
    run_registration_model,
    set_effective_max_velocity,
)

SHAPE = (32, 32, 32)


class _ConstantCoarse(nn.Module):
    """Coarse stage stub predicting a constant x velocity."""

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Return a unit velocity along x."""
        velocity = torch.zeros(x.shape[0],
                               3,
                               *output_shape,
                               dtype=x.dtype,
                               device=x.device)
        velocity[:, 0] = 1.0
        return velocity


class _ConstantRefiner(nn.Module):
    """Residual stage stub predicting a constant y velocity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a half-voxel velocity along y."""
        velocity = torch.zeros(x.shape[0],
                               3,
                               *x.shape[-3:],
                               dtype=x.dtype,
                               device=x.device)
        velocity[:, 1] = 0.5
        return velocity


class _ConstantAffine(nn.Module):
    """Affine stage stub predicting a constant translation."""

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a constant affine displacement and its parameters."""
        field = torch.zeros(x.shape[0],
                            3,
                            *output_shape,
                            dtype=x.dtype,
                            device=x.device)
        field[:, 0] = 2.0
        return field, torch.zeros(x.shape[0], 9, device=x.device)


def _images() -> list[torch.Tensor]:
    """Return four random single-channel volumes."""
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(1, 1, *SHAPE, generator=generator) for _ in range(4)]


class AffineTests(unittest.TestCase):
    """Check the affine parameterization and its identity initialization."""

    def test_parameters_reproduce_a_pure_translation(self) -> None:
        translation = torch.tensor([[3.0, -2.0, 1.0]])
        field = affine_parameters_to_displacement(translation=translation,
                                                  rotation=torch.zeros(1, 3),
                                                  scale=torch.ones(1, 3),
                                                  shape=(7, 9, 11))
        torch.testing.assert_close(
            field, translation[:, :, None, None, None].expand_as(field))

    def test_scaling_is_centered_on_the_volume(self) -> None:
        field = affine_parameters_to_displacement(translation=torch.zeros(1, 3),
                                                  rotation=torch.zeros(1, 3),
                                                  scale=torch.full((1, 3), 1.1),
                                                  shape=(9, 9, 9))
        torch.testing.assert_close(field[0, :, 4, 4, 4],
                                   torch.zeros(3),
                                   atol=1e-6,
                                   rtol=0)
        torch.testing.assert_close(field[0, 0, 4, 4, 8],
                                   torch.tensor(0.4),
                                   atol=1e-5,
                                   rtol=0)

    def test_heads_start_at_identity(self) -> None:
        torch.manual_seed(0)
        image = torch.randn(1, 4, *SHAPE)
        for head in (AffineRegistrationHead(), StrongAffineRegistrationHead()):
            head.eval()
            field, parameters = head(image, SHAPE)
            torch.testing.assert_close(field, torch.zeros_like(field))
            torch.testing.assert_close(parameters[:, :6], torch.zeros(1, 6))
            torch.testing.assert_close(parameters[:, 6:], torch.ones(1, 3))

    def test_heads_respect_their_bounds_and_can_be_reset(self) -> None:
        torch.manual_seed(0)
        head = StrongAffineRegistrationHead(max_translation=5.0,
                                            max_rotation_degrees=3.0,
                                            max_scaling=0.05)
        head.eval()
        final_layer = head.parameters_head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.constant_(final_layer.bias, 50.0)
        _, parameters = head(torch.randn(1, 4, *SHAPE), SHAPE)
        self.assertLessEqual(float(parameters[:, :3].abs().max()), 5.0)
        self.assertLessEqual(float(parameters[:, 6:].max()), 1.05)
        head.reset_to_identity()
        field, _ = head(torch.randn(1, 4, *SHAPE), SHAPE)
        torch.testing.assert_close(field, torch.zeros_like(field))

    def test_body_center_translation_uses_pullback_convention(self) -> None:
        fixed = torch.zeros(1, 1, 8, 8, 12)
        moving = torch.zeros_like(fixed)
        fixed[:, :, 2:6, 2:6, 2:5] = 1.0
        moving[:, :, 2:6, 2:6, 6:9] = 1.0
        torch.testing.assert_close(body_center_translation(fixed, moving),
                                   torch.tensor([[4.0, 0.0, 0.0]]))
        clamped = body_center_translation(fixed, moving, max_translation=1.5)
        torch.testing.assert_close(clamped, torch.tensor([[1.5, 0.0, 0.0]]))


class BuildingBlockTests(unittest.TestCase):
    """Check the blocks whose layout existing checkpoints depend on."""

    def test_conv_factory_matches_an_explicit_sequential(self) -> None:
        torch.manual_seed(11)
        explicit = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        torch.manual_seed(11)
        factory = nn.Sequential(*conv_norm_activation(4, 8, stride=2))
        self.assertEqual(list(explicit.state_dict()),
                         list(factory.state_dict()))
        for key, value in explicit.state_dict().items():
            self.assertTrue(torch.equal(value, factory.state_dict()[key]))

    def test_pairwise_features_compare_fixed_and_moving_channels(self) -> None:
        x = torch.tensor([1.0, 4.0, 3.0, 1.0]).view(1, 4, 1, 1, 1)
        features = pairwise_ct_pet_features(x).flatten().tolist()
        self.assertEqual(features,
                         [1.0, 3.0, -2.0, 2.0, 3.0, 4.0, 1.0, 3.0, 3.0, 4.0])


class CascadeCompositionTests(unittest.TestCase):
    """Check how the cascades integrate and compose their stages."""

    def test_deformable_cascade_composes_coarse_then_residual(self) -> None:
        model = TransMorphDiffCascadeRegistration(scaling_steps=4)
        model.coarse = _ConstantCoarse()
        model.refiner = _ConstantRefiner()
        image = torch.zeros(1, 1, 8, 8, 8)

        total, coarse, residual = model(image, image, image, image)

        torch.testing.assert_close(total,
                                   compose_displacements(coarse, residual))
        torch.testing.assert_close(coarse[:, 0], torch.ones_like(coarse[:, 0]))
        torch.testing.assert_close(residual[:, 1],
                                   torch.full_like(residual[:, 1], 0.5))

    def test_affine_cascade_composes_affine_coarse_residual(self) -> None:
        model = AffineTransMorphDiffCascadeRegistration(
            AffineCascadeSettings(scaling_steps=4))
        model.affine = _ConstantAffine()
        model.coarse = _ConstantCoarse()
        model.refiner = _ConstantRefiner()
        image = torch.zeros(1, 1, 8, 8, 8)

        total, stage1, residual, affine = model(image, image, image, image)

        coarse = torch.zeros_like(affine)
        coarse[:, 0] = 1.0
        torch.testing.assert_close(stage1,
                                   compose_displacements(affine, coarse))
        torch.testing.assert_close(total,
                                   compose_displacements(stage1, residual))

    def test_body_center_initialization_feeds_the_affine_stage(self) -> None:
        model = AffineTransMorphDiffCascadeRegistration(
            AffineCascadeSettings(scaling_steps=4,
                                  body_center_initialization=True))
        model.eval()
        fixed = torch.zeros(1, 1, *SHAPE)
        fixed[..., 8:24, 8:24, 6:16] = 1.0
        moving = torch.roll(fixed, shifts=5, dims=4)
        output = model(fixed, fixed, moving, moving, return_diagnostics=True)
        torch.testing.assert_close(output.body_translation,
                                   torch.tensor([[5.0, 0.0, 0.0]]))
        # The untrained affine head is the identity, so the affine stage
        # equals the body-center translation.
        torch.testing.assert_close(output.affine[:, 0],
                                   torch.full_like(output.affine[:, 0], 5.0))


class FinalModelTests(unittest.TestCase):
    """Run the published cascades on a small volume."""

    def test_final_model_returns_bounded_fields(self) -> None:
        torch.manual_seed(0)
        model = AffineTransMorphV2CascadeRegistration(
            AffineCascadeSettings(max_velocity=6.0,
                                  max_residual_disp=3.0,
                                  scaling_steps=4))
        model.eval()
        with torch.no_grad():
            outputs = model(*_images())
        self.assertEqual(len(outputs), 4)
        for field in outputs:
            self.assertEqual(field.shape, (1, 3, *SHAPE))
            self.assertTrue(torch.isfinite(field).all())
        total, stage1, residual, _ = outputs
        torch.testing.assert_close(total,
                                   compose_displacements(stage1, residual))

    def test_diagnostics_extend_the_four_fields(self) -> None:
        torch.manual_seed(0)
        model = AffineTransMorphV2CascadeRegistration(
            AffineCascadeSettings(scaling_steps=4))
        model.eval()
        images = _images()
        with torch.no_grad():
            fields = model(*images)
            output = model(*images, return_diagnostics=True)
        self.assertIsInstance(output, AffineCascadeOutput)
        for field, named in zip(fields, output[:4]):
            self.assertTrue(torch.equal(field, named))
        self.assertEqual(output.body_translation.shape, (1, 3))
        self.assertEqual(output.affine_parameters.shape, (1, 9))

    def test_every_stage_receives_gradients(self) -> None:
        torch.manual_seed(0)
        model = AffineTransMorphV2CascadeRegistration(
            AffineCascadeSettings(scaling_steps=4))
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.abs().sum() == 0:
                    parameter.add_(0.02 * torch.randn_like(parameter))
        total, _, _, _ = model(*_images())
        total.square().mean().backward()
        for stage in ("affine", "coarse", "refiner"):
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(f"{stage}.") and parameter.grad is not None
            ]
            self.assertTrue(gradients, stage)
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients), 0.0,
                stage)

    def test_state_dict_keeps_the_three_stage_prefixes(self) -> None:
        model = AffineTransMorphV2CascadeRegistration()
        prefixes = {key.split(".")[0] for key in model.state_dict()}
        self.assertEqual(prefixes, {"affine", "coarse", "refiner"})
        other = AffineTransMorphV2CascadeRegistration()
        other.load_state_dict(model.state_dict())

    def test_velocity_cap_can_be_changed_without_new_weights(self) -> None:
        for model in (TransMorphV2CascadeRegistration(max_velocity=8.0),
                      AffineTransMorphV2CascadeRegistration()):
            before = [value.clone() for value in model.state_dict().values()]
            set_effective_max_velocity(model, 2.5)
            self.assertEqual(model.coarse.max_velocity, 2.5)
            for old, new in zip(before, model.state_dict().values()):
                self.assertTrue(torch.equal(old, new))

    def test_runner_normalizes_outputs_of_both_cascade_families(self) -> None:
        images = [torch.zeros(1, 1, 8, 8, 8)] * 4
        deformable = TransMorphDiffCascadeRegistration(scaling_steps=4)
        deformable.coarse = _ConstantCoarse()
        deformable.refiner = _ConstantRefiner()
        outputs = run_registration_model(deformable, *images)
        self.assertEqual(len(outputs), 4)
        self.assertIsNone(outputs[3])
        with self.assertRaises(TypeError):
            run_registration_model(nn.Identity(), *images)


if __name__ == "__main__":
    unittest.main()
