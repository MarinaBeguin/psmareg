"""Check exported displacement axes and checkpoint inference composition."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from psmareg.data.nifti_io import load_nifti_image
from psmareg.data.manifests import PAIR_IMAGE_COLUMNS, read_manifest
from psmareg.evaluation.displacement_inference import (
    fullres_to_submitted_lowres,
    save_displacement,
    submitted_lowres_to_official_flow,
)


class DisplacementInferenceTests(unittest.TestCase):
    """Protect the NIfTI vector-component convention used by scoring."""

    def test_nifti_components_follow_volume_axes(self) -> None:
        flow = torch.zeros(1, 3, 6, 8, 10)
        flow[:, 0] = 1.0
        flow[:, 1] = 2.0
        flow[:, 2] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.nii.gz"
            save_displacement(flow, path, np.eye(4))
            array = np.asarray(load_nifti_image(path).dataobj)
        self.assertEqual(array.shape, (6, 8, 10, 3))
        np.testing.assert_array_equal(array[0, 0, 0], [3.0, 2.0, 1.0])

    def test_historical_lowres_roundtrip(self) -> None:
        flow = torch.zeros(1, 3, 6, 8, 10)
        flow[:, 0] = 1.0
        flow[:, 1] = 2.0
        flow[:, 2] = 3.0
        lowres = fullres_to_submitted_lowres(flow)
        official = submitted_lowres_to_official_flow(lowres)
        self.assertEqual(tuple(lowres.shape), (1, 3, 3, 4, 5))
        self.assertEqual(tuple(official.shape), tuple(flow.shape))
        torch.testing.assert_close(official[:, :, 0, 0, 0],
                                   torch.tensor([[3.0, 2.0, 1.0]]))

    def test_manifest_requires_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            path.write_text("subject,fixed_ct\ncase,a.nii.gz\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing manifest columns"):
                read_manifest(path, PAIR_IMAGE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
