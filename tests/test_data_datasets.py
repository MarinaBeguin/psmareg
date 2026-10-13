"""Tests for the paired and single-scan datasets and their caches."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from psmareg.data.manifests import exclude_manifest_patients
from psmareg.data.paired_dataset import PSMARegPairDataset
from psmareg.data.synthetic_registration_dataset import (
    SingleScanRegistrationDataset,)
from psmareg.data.tensor_cache import TensorCache
from tests.factories import save_nifti

SHAPE = (12, 12, 12)


def _write_scan(root: Path, name: str) -> dict[str, str]:
    """Write CT, PET, and both label maps of one synthetic scan."""
    ct = np.full(SHAPE, -500.0, dtype=np.float32)
    ct[3:9, 3:9, 3:9] = 50.0
    pet = np.zeros(SHAPE, dtype=np.float32)
    pet[4:8, 4:8, 4:8] = 5.0
    ct_label = np.zeros(SHAPE, dtype=np.float32)
    ct_label[3:9, 3:9, 3:9] = 17
    pet_label = np.zeros(SHAPE, dtype=np.float32)
    pet_label[4:8, 4:8, 4:8] = 1
    arrays = {
        "ct": ct,
        "pet": pet,
        "ct_label": ct_label,
        "pet_label": pet_label
    }
    return {
        key: str(save_nifti(array, root / f"{name}_{key}.nii.gz"))
        for key, array in arrays.items()
    }


def _write_pair_manifest(root: Path,
                         with_labels: bool = True,
                         prealign: str = "") -> Path:
    """Write a one-row paired manifest."""
    fixed, moving = _write_scan(root, "fixed"), _write_scan(root, "moving")
    row = {"subject": "PSMAReg_0001", "followup_index": "01"}
    for prefix, scan in (("fixed", fixed), ("moving", moving)):
        for key, path in scan.items():
            if with_labels or not key.endswith("label"):
                row[f"{prefix}_{key}"] = path
    if prealign:
        row["prealign_displacement"] = prealign
    manifest = root / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return manifest


class PairedDatasetTests(unittest.TestCase):
    """The paired dataset returns preprocessed images and optional labels."""

    def test_sample_contains_normalized_images_and_integer_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = PSMARegPairDataset(_write_pair_manifest(Path(directory)))
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
        self.assertEqual(sample["subject"], "PSMAReg_0001")
        for key in ("fixed_ct", "fixed_pet", "moving_ct", "moving_pet"):
            image = sample[key]
            assert isinstance(image, torch.Tensor)
            self.assertEqual(image.shape, (1, *SHAPE))
            self.assertEqual(image.dtype, torch.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
        label = sample["fixed_ct_label"]
        assert isinstance(label, torch.Tensor)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.unique().tolist(), [0, 17])
        self.assertNotIn("prealign_displacement", sample)

    def test_missing_labels_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_pair_manifest(Path(directory), with_labels=False)
            sample = PSMARegPairDataset(manifest)[0]
        self.assertFalse(any(key.endswith("label") for key in sample))

    def test_cache_is_written_once_and_returns_the_same_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            dataset = PSMARegPairDataset(_write_pair_manifest(root),
                                         cache_dir=cache_dir,
                                         cache_float16=True)
            uncached = PSMARegPairDataset(root / "pairs.csv")[0]
            first = dataset[0]
            files = {
                path: path.stat().st_mtime_ns for path in cache_dir.glob("*.pt")
            }
            second = dataset[0]
            unchanged = all(path.stat().st_mtime_ns == mtime
                            for path, mtime in files.items())
        # Four images and four label maps, one entry each.
        self.assertEqual(len(files), 8)
        self.assertTrue(unchanged)
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                reference = uncached[key]
                assert isinstance(reference, torch.Tensor)
                cached = second[key]
                assert isinstance(cached, torch.Tensor)
                self.assertEqual(cached.dtype, reference.dtype)
                torch.testing.assert_close(cached.float(),
                                           reference.float(),
                                           atol=5e-4,
                                           rtol=0)

    def test_displacement_components_are_reordered_to_x_y_z(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = np.zeros((*SHAPE, 3), dtype=np.float32)
            field[..., 0], field[..., 1], field[..., 2] = 1.0, 2.0, 3.0
            path = save_nifti(field, root / "prealign.nii.gz")
            manifest = _write_pair_manifest(root, prealign=str(path))
            sample = PSMARegPairDataset(manifest)[0]
            save_nifti(np.zeros(SHAPE, dtype=np.float32),
                       root / "scalar.nii.gz")
            invalid = _write_pair_manifest(root,
                                           prealign=str(root / "scalar.nii.gz"))
            invalid_dataset = PSMARegPairDataset(invalid)
            with self.assertRaisesRegex(ValueError, "displacement field"):
                _ = invalid_dataset[0]
        displacement = sample["prealign_displacement"]
        assert isinstance(displacement, torch.Tensor)
        self.assertEqual(displacement.shape, (3, *SHAPE))
        self.assertEqual(displacement[:, 0, 0, 0].tolist(), [3.0, 2.0, 1.0])


class SingleScanDatasetTests(unittest.TestCase):
    """The pretraining dataset caches scans and excludes held-out patients."""

    def _manifest(self, root: Path, labels: bool) -> Path:
        """Write a one-row single-scan manifest."""
        scan = _write_scan(root, "scan")
        columns = ["ct", "pet"] + (["ct_label", "pet_label"] if labels else [])
        manifest = root / "scans.csv"
        manifest.write_text(
            ",".join(columns + ["subject"]) + "\n" +
            ",".join([scan[key] for key in columns] + ["PSMARegPSMA_0001_00"]) +
            "\n",
            encoding="utf-8")
        return manifest

    def test_cache_is_reused_and_matches_the_uncached_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, labels=False)
            dataset = SingleScanRegistrationDataset(manifest,
                                                    cache_dir=root / "cache")
            reference = SingleScanRegistrationDataset(manifest)[0]
            first = dataset[0]
            cache_files = list((root / "cache").glob("*.pt"))
            first_mtime = cache_files[0].stat().st_mtime_ns
            second = dataset[0]
            second_mtime = cache_files[0].stat().st_mtime_ns
        self.assertEqual(len(cache_files), 1)
        self.assertEqual(first_mtime, second_mtime)
        self.assertEqual(set(first), {"ct", "pet"})
        for key in ("ct", "pet"):
            self.assertEqual(first[key].shape, (1, *SHAPE))
            torch.testing.assert_close(second[key],
                                       reference[key],
                                       atol=5e-4,
                                       rtol=0)

    def test_labels_keep_their_integer_values_through_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = SingleScanRegistrationDataset(self._manifest(root,
                                                                   labels=True),
                                                    cache_dir=root / "cache",
                                                    include_labels=True)
            first, second = dataset[0], dataset[0]
        torch.testing.assert_close(first["ct_label"], second["ct_label"])
        torch.testing.assert_close(first["pet_label"], second["pet_label"])
        self.assertEqual(second["ct_label"].dtype, torch.int64)
        self.assertEqual(int(second["ct_label"].max()), 17)

    def test_unreadable_cache_entry_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = SingleScanRegistrationDataset(self._manifest(
                root, labels=False),
                                                    cache_dir=root / "cache")
            first = dataset[0]
            cache_file = next((root / "cache").glob("*.pt"))
            cache_file.write_bytes(b"corrupted")
            with self.assertLogs("psmareg.data", level="WARNING"):
                rebuilt = dataset[0]
        torch.testing.assert_close(rebuilt["ct"], first["ct"])

    def test_missing_columns_and_emptied_manifests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, labels=False)
            with self.assertRaisesRegex(ValueError, "missing required"):
                SingleScanRegistrationDataset(manifest, include_labels=True)
            holdout = root / "holdout.csv"
            holdout.write_text("subject\nPSMAReg_0001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "removed every row"):
                SingleScanRegistrationDataset(manifest,
                                              exclusion_manifest=holdout)

    def test_holdout_patients_are_excluded(self) -> None:
        rows = [{
            "subject": "PSMARegPSMA_0001_00"
        }, {
            "subject": "PSMARegPSMA_0002_00"
        }]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "holdout.csv"
            manifest.write_text(
                "subject,fixed_ct\n"
                "PSMAReg_0002,/data/PSMARegPSMA_0002.nii.gz\n",
                encoding="utf-8")
            filtered = exclude_manifest_patients(rows, manifest)
            unfiltered = exclude_manifest_patients(rows, None)
        self.assertEqual([row["subject"] for row in filtered],
                         ["PSMARegPSMA_0001_00"])
        self.assertEqual(unfiltered, rows)


class TensorCacheTests(unittest.TestCase):
    """The cache publishes entries atomically."""

    def test_round_trip_preserves_tensors_and_dictionaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TensorCache(Path(directory) / "cache")
            tensor = torch.arange(6, dtype=torch.float16)
            TensorCache.save(cache.path("tensor"), tensor)
            TensorCache.save(cache.path("sample"), {"ct": tensor})
            self.assertTrue(
                torch.equal(TensorCache.load(cache.path("tensor")), tensor))
            self.assertTrue(
                torch.equal(
                    TensorCache.load(cache.path("sample"))["ct"], tensor))
            self.assertEqual(
                sorted(path.name for path in cache.directory.iterdir()),
                ["sample.pt", "tensor.pt"])

    def test_failed_write_keeps_the_previous_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TensorCache(Path(directory))
            path = cache.path("entry")
            TensorCache.save(path, torch.ones(2))
            with self.assertRaises(Exception):
                TensorCache.save(path, lambda: None)
            self.assertTrue(torch.equal(TensorCache.load(path), torch.ones(2)))
            self.assertEqual([item.name for item in cache.directory.iterdir()],
                             ["entry.pt"])


if __name__ == "__main__":
    unittest.main()
