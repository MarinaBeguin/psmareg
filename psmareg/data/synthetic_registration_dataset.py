"""Dataset used by synthetic registration pretraining."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from psmareg.data.manifests import (
    exclude_manifest_patients,
    read_manifest,
)
from psmareg.data.nifti_io import load_float_nifti
from psmareg.data.preprocessing import preprocess_ct, preprocess_pet
from psmareg.data.tensor_cache import TensorCache

Batch = dict[str, torch.Tensor]
LOGGER = logging.getLogger(__name__)


class SingleScanRegistrationDataset(Dataset[Batch]):
    """Dataset of single preprocessed CT/PET scans for synthetic pairing."""

    def __init__(
        self,
        manifest_path: Path,
        exclusion_manifest: Path | None = None,
        cache_dir: Path | None = None,
        include_labels: bool = False,
    ) -> None:
        self.rows = exclude_manifest_patients(read_manifest(manifest_path),
                                              exclusion_manifest)
        self.cache = TensorCache(cache_dir) if cache_dir is not None else None
        self.include_labels = include_labels

        if not self.rows:
            raise ValueError(f"No rows found in {manifest_path}.")

        required_columns = {"ct", "pet"}
        if self.include_labels:
            required_columns.update({"ct_label", "pet_label"})
        missing_columns = required_columns - set(self.rows[0])
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{manifest_path} is missing required columns: {columns}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Batch:
        row = self.rows[index]
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            try:
                cached = TensorCache.load(cache_path)
                sample = {
                    "ct": cached["ct"].float(),
                    "pet": cached["pet"].float(),
                }
                if self.include_labels:
                    sample["ct_label"] = cached["ct_label"].long()
                    sample["pet_label"] = cached["pet_label"].long()
                return sample
            except (OSError, RuntimeError, EOFError,
                    pickle.UnpicklingError) as error:
                LOGGER.warning("Removing unreadable cache file %s: %s",
                               cache_path, error)
                cache_path.unlink(missing_ok=True)

        ct = self._load_ct(Path(row["ct"]))
        pet = self._load_pet(Path(row["pet"]))
        sample = {"ct": ct, "pet": pet}
        if self.include_labels:
            sample["ct_label"] = self._load_label(Path(row["ct_label"]))
            sample["pet_label"] = self._load_label(Path(row["pet_label"]))
        if cache_path is not None:
            try:
                cached_sample = {
                    "ct": ct.half(),
                    "pet": pet.half(),
                }
                if self.include_labels:
                    cached_sample["ct_label"] = sample["ct_label"].short()
                    cached_sample["pet_label"] = sample["pet_label"].byte()
                TensorCache.save(cache_path, cached_sample)
            except (OSError, RuntimeError) as error:
                LOGGER.warning(
                    "Disabling preprocessing cache in worker %s after write "
                    "failure: %s", os.getpid(), error)
                # Each DataLoader worker owns its dataset copy. Disabling the
                # cache here prevents repeated failed writes in that worker.
                self.cache = None
        return sample

    def _cache_path(self, row: dict[str, str]) -> Path | None:
        """Return the cache entry of one manifest row, if caching is on."""
        if self.cache is None:
            return None
        # Existing caches depend on this versioned per-sample fingerprint.
        cache_version = ("synthetic_preprocessing_with_labels_v1" if
                         self.include_labels else "synthetic_preprocessing_v1")
        fingerprint_parts = [cache_version]
        keys = (("ct", "pet", "ct_label",
                 "pet_label") if self.include_labels else ("ct", "pet"))
        for key in keys:
            path = Path(row[key]).resolve()
            stat = path.stat()
            fingerprint_parts.extend(
                [str(path), str(stat.st_size),
                 str(stat.st_mtime_ns)])
        digest = hashlib.sha256(
            "\n".join(fingerprint_parts).encode("utf-8")).hexdigest()
        return self.cache.path(digest)

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        """Read a NIfTI volume with non-finite values set to zero."""
        return load_float_nifti(path, sanitize=True)

    @staticmethod
    def _load_ct(path: Path) -> torch.Tensor:
        """Load and preprocess a CT volume as ``[1, D, H, W]``."""
        image = preprocess_ct(SingleScanRegistrationDataset._load_nifti(path))
        return torch.from_numpy(image).float().unsqueeze(0)

    @staticmethod
    def _load_pet(path: Path) -> torch.Tensor:
        """Load and normalize a PET volume as ``[1, D, H, W]``."""
        image = torch.from_numpy(
            SingleScanRegistrationDataset._load_nifti(path))
        return preprocess_pet(image).unsqueeze(0)

    @staticmethod
    def _load_label(path: Path) -> torch.Tensor:
        """Load an integer label map as ``[1, D, H, W]``."""
        image = SingleScanRegistrationDataset._load_nifti(path)
        return torch.from_numpy(image).round().long().unsqueeze(0)
