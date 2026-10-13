"""Tests for the CT and PET intensity preprocessing."""

import unittest

import numpy as np
import torch

from psmareg.data.preprocessing import (
    CT_WINDOW,
    ct_window_normalize,
    preprocess_ct,
    preprocess_pet,
    remove_bed,
)


def _ct_with_bed() -> np.ndarray:
    """Return air, a central soft-tissue body, and a detached table slab."""
    volume = np.full((40, 40, 12), -1000.0, dtype=np.float32)
    volume[12:28, 10:26, :] = 40.0
    volume[8:32, 32:36, :] = 200.0
    return volume


class CtPreprocessingTests(unittest.TestCase):
    """CT is windowed to [0, 1] after the scanner table is removed."""

    def test_window_maps_to_unit_interval(self) -> None:
        lower, upper = CT_WINDOW
        volume = np.array([lower - 100.0, lower, 0.0, upper, upper + 100.0],
                          dtype=np.float32)
        np.testing.assert_allclose(
            ct_window_normalize(volume, CT_WINDOW),
            np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32))

    def test_bed_is_replaced_by_air_and_body_is_kept(self) -> None:
        volume = _ct_with_bed()
        cleaned = remove_bed(volume)
        np.testing.assert_allclose(cleaned[12:28, 10:26, :], 40.0)
        np.testing.assert_allclose(cleaned[8:32, 32:36, :], -1000.0)
        np.testing.assert_allclose(volume[8:32, 32:36, :], 200.0)

    def test_preprocessed_ct_is_normalized_without_the_bed(self) -> None:
        processed = preprocess_ct(_ct_with_bed())
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(float(processed.min()), 0.0)
        self.assertAlmostEqual(float(processed[20, 18, 6]),
                               340.0 / 600.0,
                               places=5)
        self.assertEqual(float(processed[20, 34, 6]), 0.0)


class PetPreprocessingTests(unittest.TestCase):
    """PET is robustly normalized whatever its absolute scale."""

    def test_pet_is_scale_invariant_and_finite(self) -> None:
        pet = torch.tensor(
            [-2.0, 0.0, 1.0, 2.0, 5.0,
             float("nan"), float("inf")])
        normalized = preprocess_pet(pet)
        torch.testing.assert_close(normalized, preprocess_pet(17.0 * pet))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_quantile_saturates_only_the_hottest_voxels(self) -> None:
        pet = torch.linspace(0.0, 10.0, 1001)
        normalized = preprocess_pet(pet)
        self.assertEqual(float(normalized[0]), 0.0)
        self.assertEqual(float(normalized[-1]), 1.0)
        self.assertLess(float((normalized == 1.0).float().mean()), 0.02)
        # Below the saturation level the mapping is linear.
        torch.testing.assert_close(normalized[500] / normalized[250],
                                   torch.tensor(2.0),
                                   rtol=1e-4,
                                   atol=0)


if __name__ == "__main__":
    unittest.main()
