"""Tests for image warping, field composition, and velocity integration."""

import unittest

import torch

from psmareg.models.common.registration.warping import (
    compose_displacements,
    integrate_stationary_velocity,
    valid_sampling_mask,
    warp_image_3d,
)


class WarpingTests(unittest.TestCase):
    """Check the pull-back warping convention and its compositions."""

    def test_zero_displacement_is_identity(self) -> None:
        image = torch.randn(1, 1, 5, 6, 7)
        displacement = torch.zeros(1, 3, 5, 6, 7)
        warped = warp_image_3d(image, displacement)
        torch.testing.assert_close(warped, image, atol=1e-6, rtol=1e-6)

    def test_positive_width_displacement_samples_next_voxel(self) -> None:
        image = torch.arange(7.0).view(1, 1, 1, 1, 7).expand(1, 1, 3, 3, 7)
        displacement = torch.zeros(1, 3, 3, 3, 7)
        displacement[:, 0] = 1.0
        warped = warp_image_3d(image,
                               displacement,
                               interpolation_mode="nearest")
        torch.testing.assert_close(warped[..., :-1], image[..., 1:])

    def test_channels_follow_width_height_depth_order(self) -> None:
        image = torch.zeros(1, 1, 5, 6, 7)
        image[0, 0, 2, 3, 4] = 1.0
        for channel, source in ((0, (2, 3, 3)), (1, (2, 2, 4)), (2, (1, 3, 4))):
            displacement = torch.zeros(1, 3, 5, 6, 7)
            displacement[:, channel] = 1.0
            warped = warp_image_3d(image,
                                   displacement,
                                   interpolation_mode="nearest")
            self.assertEqual(float(warped[0, 0][source]), 1.0)

    def test_zero_padding_blanks_samples_outside_the_image(self) -> None:
        image = torch.ones(1, 1, 4, 4, 6)
        displacement = torch.zeros(1, 3, 4, 4, 6)
        displacement[:, 0] = 2.0
        zeros = warp_image_3d(image, displacement, padding_mode="zeros")
        border = warp_image_3d(image, displacement, padding_mode="border")
        torch.testing.assert_close(zeros[..., -2:],
                                   torch.zeros_like(zeros[..., -2:]))
        torch.testing.assert_close(border, image)

    def test_valid_mask_marks_samples_inside_the_image(self) -> None:
        displacement = torch.zeros(1, 3, 4, 4, 6)
        displacement[:, 0] = 2.0
        mask = valid_sampling_mask(displacement)
        self.assertEqual(mask.shape, (1, 1, 4, 4, 6))
        torch.testing.assert_close(mask[..., :4],
                                   torch.ones_like(mask[..., :4]))
        torch.testing.assert_close(mask[..., -1:],
                                   torch.zeros_like(mask[..., -1:]))

    def test_composition_matches_sequential_warp(self) -> None:
        image = torch.randn(1, 1, 9, 9, 9)
        first = torch.zeros(1, 3, 9, 9, 9)
        second = torch.zeros_like(first)
        first[:, 0] = 1.0
        second[:, 1] = -1.0

        sequential = warp_image_3d(warp_image_3d(image, first), second)
        composed = warp_image_3d(
            image,
            compose_displacements(first=first, second=second),
        )
        torch.testing.assert_close(composed, sequential, atol=1e-5, rtol=1e-5)

    def test_composition_spatially_warps_first_field(self) -> None:
        first = torch.zeros(1, 3, 5, 6, 7)
        second = torch.zeros_like(first)
        y = torch.arange(6.0).view(1, 1, 1, 6, 1)
        first[:, 0:1] = 0.1 * y
        second[:, 1] = 1.0

        composed = compose_displacements(first=first, second=second)

        expected_x = 0.1 * torch.arange(1.0, 6.0).view(1, 1, 1, 5, 1)
        torch.testing.assert_close(
            composed[:, 0:1, :, :-1, :],
            expected_x.expand(1, 1, 5, 5, 7),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_constant_velocity_integrates_to_same_translation(self) -> None:
        velocity = torch.zeros(1, 3, 7, 7, 7)
        velocity[:, 2] = 0.75
        displacement = integrate_stationary_velocity(velocity, scaling_steps=5)
        torch.testing.assert_close(displacement, velocity, atol=1e-6, rtol=1e-6)

    def test_integrated_velocity_inverts_with_negated_velocity(self) -> None:
        torch.manual_seed(0)
        coarse = 0.4 * torch.randn(1, 3, 4, 4, 4)
        velocity = torch.nn.functional.interpolate(coarse,
                                                   size=(16, 16, 16),
                                                   mode="trilinear",
                                                   align_corners=True)
        forward = integrate_stationary_velocity(velocity, scaling_steps=6)
        backward = integrate_stationary_velocity(-velocity, scaling_steps=6)
        round_trip = compose_displacements(first=forward, second=backward)
        # Trilinear resampling limits the accuracy of the discrete inverse.
        self.assertLess(float(round_trip.abs().max()),
                        0.15 * float(forward.abs().max()))


if __name__ == "__main__":
    unittest.main()
