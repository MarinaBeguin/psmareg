"""Tests for voxel grids, rotations, field resizing, and Jacobians."""

import math
import unittest

import torch

from psmareg.models.common.registration.displacement import (
    resize_displacement,
    resize_voxel_displacement,
)
from psmareg.models.common.registration.geometry import (
    euler_rotation_matrix_xyz,
    voxel_grid_xyz,
)
from psmareg.models.common.registration.jacobian import (
    jacobian_determinant,
    jacobian_summary,
)


class GeometryTests(unittest.TestCase):
    """Check coordinate conventions shared by every deformation."""

    def test_voxel_grid_channels_are_x_y_z(self) -> None:
        grid = voxel_grid_xyz((2, 3, 4),
                              device=torch.device("cpu"),
                              dtype=torch.float32)
        self.assertEqual(grid.shape, (3, 2, 3, 4))
        self.assertEqual(grid[:, 1, 2, 3].tolist(), [3.0, 2.0, 1.0])

    def test_euler_rotation_is_orthonormal(self) -> None:
        angles = torch.tensor([[0.3, -0.5, 1.1], [0.0, 0.0, 0.0]])
        rotation = euler_rotation_matrix_xyz(angles)
        identity = torch.eye(3).expand(2, 3, 3)
        torch.testing.assert_close(rotation @ rotation.transpose(1, 2),
                                   identity,
                                   atol=1e-6,
                                   rtol=0)
        torch.testing.assert_close(torch.linalg.det(rotation),
                                   torch.ones(2),
                                   atol=1e-6,
                                   rtol=0)
        torch.testing.assert_close(rotation[1], torch.eye(3))

    def test_euler_rotation_about_z_turns_x_into_y(self) -> None:
        rotation = euler_rotation_matrix_xyz(
            torch.tensor([0.0, 0.0, math.pi / 2]))
        torch.testing.assert_close(rotation @ torch.tensor([1.0, 0.0, 0.0]),
                                   torch.tensor([0.0, 1.0, 0.0]),
                                   atol=1e-6,
                                   rtol=0)


class DisplacementResizeTests(unittest.TestCase):
    """Check that resized fields keep describing the same physical motion."""

    def test_aligned_corners_scale_by_grid_spacing(self) -> None:
        field = torch.zeros(1, 3, 9, 11, 13)
        field[:, 0] = 2.0
        resized = resize_displacement(field, (5, 6, 7))
        self.assertEqual(resized.shape, (1, 3, 5, 6, 7))
        torch.testing.assert_close(resized[:, 0], torch.ones_like(resized[:,
                                                                          0]))

    def test_unaligned_corners_scale_by_size_ratio(self) -> None:
        field = torch.zeros(1, 3, 8, 8, 12)
        field[:, 0] = 6.0
        field[:, 2] = 4.0
        resized = resize_voxel_displacement(field, (4, 4, 4),
                                            align_corners=False)
        torch.testing.assert_close(resized[:, 0],
                                   torch.full_like(resized[:, 0], 2.0))
        torch.testing.assert_close(resized[:, 2],
                                   torch.full_like(resized[:, 2], 2.0))

    def test_resize_round_trip_preserves_a_smooth_field(self) -> None:
        coordinate = torch.linspace(0.0, 1.0, 16)
        field = torch.zeros(1, 3, 16, 16, 16)
        field[:, 0] = 3.0 * coordinate.view(1, 1, 16)
        low = resize_displacement(field, (8, 8, 8))
        restored = resize_displacement(low, (16, 16, 16))
        torch.testing.assert_close(restored, field, atol=1e-5, rtol=0)


class JacobianTests(unittest.TestCase):
    """Check local volume change of known deformations."""

    def test_translation_has_unit_jacobian(self) -> None:
        field = torch.zeros(1, 3, 12, 12, 12)
        field[:, 0] = 3.0
        field[:, 1] = -2.0
        determinants = jacobian_determinant(field, sampling_stride=2)
        torch.testing.assert_close(determinants, torch.ones_like(determinants))

    def test_uniform_scaling_has_cubed_jacobian(self) -> None:
        grid = voxel_grid_xyz((12, 12, 12),
                              device=torch.device("cpu"),
                              dtype=torch.float32)
        field = (0.1 * grid).unsqueeze(0)
        determinants = jacobian_determinant(field, sampling_stride=1)
        torch.testing.assert_close(determinants,
                                   torch.full_like(determinants, 1.1**3),
                                   atol=1e-5,
                                   rtol=0)

    def test_summary_reports_folding(self) -> None:
        grid = voxel_grid_xyz((16, 16, 16),
                              device=torch.device("cpu"),
                              dtype=torch.float32)
        folded = torch.zeros(1, 3, 16, 16, 16)
        folded[:, 0] = -2.0 * grid[0]
        self.assertEqual(
            jacobian_summary(torch.zeros(1, 3, 16, 16, 16))["ndv_percent"], 0.0)
        self.assertGreater(jacobian_summary(folded)["ndv_percent"], 99.0)


if __name__ == "__main__":
    unittest.main()
