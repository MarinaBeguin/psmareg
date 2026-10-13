"""Dataset for paired PSMAReg PET/CT registration."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from psmareg.data.manifests import read_manifest
from psmareg.data.nifti_io import load_float_nifti, load_nifti_image
from psmareg.data.preprocessing import preprocess_ct, preprocess_pet
from psmareg.data.tensor_cache import TensorCache

Sample = dict[str, torch.Tensor | str]


class PSMARegPairDataset(Dataset[Sample]):
    """Dataset returning fixed and moving CT/PET volumes."""

    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path | None = None,
        cache_float16: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.cache_float16 = cache_float16
        self.cache = TensorCache(cache_dir) if cache_dir is not None else None

    @staticmethod
    def _load_nifti(path: str) -> torch.Tensor:
        """Read a NIfTI volume as a ``[1, D, H, W]`` float tensor."""
        array = load_float_nifti(path, sanitize=False)
        return torch.from_numpy(array).float().unsqueeze(0)

    @staticmethod
    def _preprocess_ct(ct_volume: torch.Tensor) -> torch.Tensor:
        """Apply CT bed suppression and window normalization."""
        array = ct_volume.squeeze(0).detach().cpu().numpy()
        preprocessed = preprocess_ct(array)
        return torch.from_numpy(preprocessed).float().unsqueeze(0)

    @staticmethod
    def _load_label(path: str) -> torch.Tensor:
        """Load a label map as a single-channel tensor."""
        image = load_nifti_image(path)
        array = np.asarray(image.get_fdata(), dtype=np.int64)
        return torch.from_numpy(array).long().unsqueeze(0)

    @staticmethod
    def _load_displacement(path: str) -> torch.Tensor:
        """Load a voxel displacement field saved as [X,Y,Z,3] NIfTI.

        NIfTI vector components follow array-axis order [X,Y,Z]. Reorder them
        to the [W,H,D] component order expected by ``grid_sample``.
        """
        image = load_nifti_image(path)
        array = np.asarray(image.get_fdata(dtype=np.float32))
        if array.ndim == 5 and array.shape[-2] == 1 and array.shape[-1] == 3:
            array = array[..., 0, :]
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(
                "Expected displacement field with shape [X,Y,Z,3], got "
                f"{array.shape}")
        array = np.moveaxis(array, -1, 0)
        array = array[[2, 1, 0]]
        return torch.from_numpy(array).float()

    @staticmethod
    def _cache_key(path: str, kind: str) -> str:
        """Fingerprint one source file; existing caches depend on it."""
        resolved = str(Path(path).resolve())
        stat = os.stat(resolved)
        payload = (f"{kind}|{resolved}|{stat.st_size}|"
                   f"{stat.st_mtime_ns}").encode()
        return hashlib.sha1(payload).hexdigest()

    def _load_or_cache_image(self, path: str, kind: str) -> torch.Tensor:
        """Return a preprocessed CT or PET, cached when enabled."""
        if self.cache is None:
            if kind == "ct":
                return self._preprocess_ct(self._load_nifti(path))
            if kind == "pet":
                return preprocess_pet(self._load_nifti(path))
            raise ValueError(f"Unsupported image kind: {kind}")

        cache_path = self.cache.path(self._cache_key(path, kind))
        if cache_path.exists():
            tensor = TensorCache.load(cache_path)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Invalid cache entry in {cache_path}")
            return tensor.float()

        if kind == "ct":
            tensor = self._preprocess_ct(self._load_nifti(path))
        elif kind == "pet":
            tensor = preprocess_pet(self._load_nifti(path))
        else:
            raise ValueError(f"Unsupported image kind: {kind}")

        tensor_to_store = tensor.half() if self.cache_float16 else tensor.float(
        )
        TensorCache.save(cache_path, tensor_to_store.cpu())
        return tensor.float()

    def _load_or_cache_label(self, path: str, kind: str) -> torch.Tensor:
        """Return a label map, cached when enabled."""
        if self.cache is None:
            return self._load_label(path)

        cache_path = self.cache.path(self._cache_key(path, kind))
        if cache_path.exists():
            tensor = TensorCache.load(cache_path)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Invalid cache entry in {cache_path}")
            return tensor.long()

        tensor = self._load_label(path)
        TensorCache.save(cache_path, tensor.short().cpu())
        return tensor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]

        fixed_ct = self._load_or_cache_image(row["fixed_ct"], "ct")
        fixed_pet = self._load_or_cache_image(row["fixed_pet"], "pet")
        moving_ct = self._load_or_cache_image(row["moving_ct"], "ct")
        moving_pet = self._load_or_cache_image(row["moving_pet"], "pet")
        sample: Sample = {
            "subject": row["subject"],
            "followup_index": row["followup_index"],
            "fixed_ct": fixed_ct,
            "fixed_pet": fixed_pet,
            "moving_ct": moving_ct,
            "moving_pet": moving_pet,
        }

        if "fixed_ct_label" in row and row["fixed_ct_label"]:
            sample["fixed_ct_label"] = self._load_or_cache_label(
                row["fixed_ct_label"], "ct_label")

        if "fixed_pet_label" in row and row["fixed_pet_label"]:
            sample["fixed_pet_label"] = self._load_or_cache_label(
                row["fixed_pet_label"], "pet_label")

        if "moving_ct_label" in row and row["moving_ct_label"]:
            sample["moving_ct_label"] = self._load_or_cache_label(
                row["moving_ct_label"], "ct_label")

        if "moving_pet_label" in row and row["moving_pet_label"]:
            sample["moving_pet_label"] = self._load_or_cache_label(
                row["moving_pet_label"], "pet_label")

        if "prealign_displacement" in row and row["prealign_displacement"]:
            sample["prealign_displacement"] = self._load_displacement(
                row["prealign_displacement"])

        return sample
