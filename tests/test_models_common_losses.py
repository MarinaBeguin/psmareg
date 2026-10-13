"""Tests for the image-similarity and overlap losses shared by both phases."""

import unittest

import torch

from psmareg.models.common.anatomy import macro_ct_label_groups
from psmareg.models.common.losses import (
    label_group_mask,
    lncc_loss,
    masked_lncc_loss,
    mind_ssc_descriptor,
    mind_ssc_loss,
    multiscale_correlation_loss,
    smoothness_loss,
    soft_dice_loss,
)


class SimilarityLossTests(unittest.TestCase):
    """Check each similarity on identical, rescaled, and unrelated images."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.image = torch.rand(1, 1, 16, 16, 16)
        self.other = torch.rand(1, 1, 16, 16, 16)

    def test_lncc_is_minimal_for_identical_images(self) -> None:
        identical = lncc_loss(self.image, self.image)
        unrelated = lncc_loss(self.image, self.other)
        torch.testing.assert_close(identical,
                                   torch.tensor(-1.0),
                                   atol=1e-4,
                                   rtol=0)
        self.assertGreater(float(unrelated), -0.5)

    def test_lncc_ignores_a_linear_intensity_change(self) -> None:
        rescaled = lncc_loss(self.image, 2.0 * self.image + 0.3)
        self.assertLess(float(rescaled), -0.95)

    def test_masked_lncc_ignores_content_outside_the_mask(self) -> None:
        mask = torch.zeros_like(self.image)
        mask[..., :8] = 1.0
        corrupted = self.image.clone()
        corrupted[..., 12:] = self.other[..., 12:]
        clean = masked_lncc_loss(self.image, self.image, mask, window_size=3)
        with_corruption = masked_lncc_loss(corrupted,
                                           self.image,
                                           mask,
                                           window_size=3)
        torch.testing.assert_close(with_corruption, clean)
        self.assertGreater(
            float(lncc_loss(corrupted, self.image, window_size=3)),
            float(lncc_loss(self.image, self.image, window_size=3)))

    def test_mind_descriptor_ignores_affine_intensity_changes(self) -> None:
        descriptor = mind_ssc_descriptor(self.image)
        self.assertEqual(descriptor.shape, (1, 12, 16, 16, 16))
        self.assertLess(
            float(mind_ssc_loss(self.image, 3.0 * self.image + 0.2)), 1e-5)
        self.assertGreater(float(mind_ssc_loss(self.image, self.other)), 0.05)

    def test_multiscale_correlation_spans_minus_one_to_one(self) -> None:
        same = multiscale_correlation_loss(self.image, self.image)
        opposite = multiscale_correlation_loss(self.image, -self.image)
        unrelated = multiscale_correlation_loss(self.image, self.other)
        self.assertLess(float(same), -0.99)
        self.assertGreater(float(opposite), 0.99)
        self.assertLess(abs(float(unrelated)), 0.2)

    def test_similarities_give_gradients_to_the_moving_image(self) -> None:
        moving = self.other.clone().requires_grad_(True)
        mask = torch.ones_like(moving)
        for loss in (lncc_loss(moving, self.image),
                     masked_lncc_loss(moving, self.image, mask),
                     mind_ssc_loss(moving, self.image, mask=mask),
                     multiscale_correlation_loss(moving, self.image,
                                                 mask=mask)):
            gradient, = torch.autograd.grad(loss, moving)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


class OverlapAndSmoothnessTests(unittest.TestCase):
    """Check Dice, label grouping, and displacement smoothness."""

    def test_soft_dice_is_zero_for_a_match_and_one_when_disjoint(self) -> None:
        mask = torch.zeros(1, 1, 4, 4, 4)
        mask[..., :2] = 1.0
        torch.testing.assert_close(soft_dice_loss(mask, mask),
                                   torch.tensor(0.0),
                                   atol=1e-6,
                                   rtol=0)
        torch.testing.assert_close(soft_dice_loss(mask, 1.0 - mask),
                                   torch.tensor(1.0),
                                   atol=1e-6,
                                   rtol=0)

    def test_label_group_mask_merges_the_requested_classes(self) -> None:
        label = torch.tensor([0, 1, 2, 5, 7]).view(1, 1, 1, 1, 5)
        mask = label_group_mask(label, (2, 5))
        self.assertEqual(mask.flatten().tolist(), [0.0, 0.0, 1.0, 1.0, 0.0])

    def test_macro_groups_follow_the_preset(self) -> None:
        stable = dict(macro_ct_label_groups("stable"))
        without_bladder = dict(macro_ct_label_groups("stable_no_bladder"))
        self.assertIn("bladder", stable)
        self.assertNotIn("bladder", without_bladder)
        self.assertEqual(set(stable) - set(without_bladder), {"bladder"})
        with self.assertRaises(ValueError):
            macro_ct_label_groups("unknown")

    def test_smoothness_is_zero_for_a_translation(self) -> None:
        translation = torch.ones(1, 3, 6, 6, 6)
        noisy = translation + 0.5 * torch.randn(1, 3, 6, 6, 6)
        torch.testing.assert_close(smoothness_loss(translation),
                                   torch.tensor(0.0))
        self.assertGreater(float(smoothness_loss(noisy)), 0.0)


if __name__ == "__main__":
    unittest.main()
