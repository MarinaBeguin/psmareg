"""Dataset used for validation and challenge submission inference."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing import preprocess_ct

Sample = dict[str, torch.Tensor | str]


class SubmissionDataset(Dataset[Sample]):
    """Dataset without labels for validation/submission inference."""

    def __init__(self, manifest_path: Path) -> None:
        self.rows = self._load_manifest(manifest_path)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    @staticmethod
    def _load_nifti(path: str) -> torch.Tensor:
        image = nib.load(path)
        array = np.asarray(image.get_fdata(dtype=np.float32))
        return torch.from_numpy(array).float().unsqueeze(0)

    @staticmethod
    def _preprocess_ct(ct_volume: torch.Tensor) -> torch.Tensor:
        """Apply CT bed suppression and window normalization."""
        array = ct_volume.squeeze(0).detach().cpu().numpy()
        preprocessed = preprocess_ct(array)
        return torch.from_numpy(preprocessed).float().unsqueeze(0)

    @staticmethod
    def _normalize_pet(pet_volume: torch.Tensor) -> torch.Tensor:
        """Normalize PET using the 99th percentile."""
        positive_values = pet_volume[pet_volume > 0.0]

        if positive_values.numel() == 0:
            return torch.zeros_like(pet_volume)

        percentile_99 = torch.quantile(positive_values, 0.99)
        scale = torch.clamp(percentile_99, min=1e-6)

        return torch.clamp(
            pet_volume / scale,
            min=0.0,
            max=1.0,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]

        fixed_ct = self._preprocess_ct(
            self._load_nifti(row["fixed_ct"])
        )
        fixed_pet = self._normalize_pet(
            self._load_nifti(row["fixed_pet"])
        )
        moving_ct = self._preprocess_ct(
            self._load_nifti(row["moving_ct"])
        )
        moving_pet = self._normalize_pet(
            self._load_nifti(row["moving_pet"])
        )

        fixed = torch.cat(
            [fixed_ct, fixed_pet],
            dim=0,
        )

        moving = torch.cat(
            [moving_ct, moving_pet],
            dim=0,
        )

        return {
            "subject": row["subject"],
            "followup_index": row["followup_index"],
            "split": row["split"],
            "fixed": fixed,
            "moving": moving,
            "fixed_ct": fixed_ct,
            "fixed_pet": fixed_pet,
            "moving_ct": moving_ct,
            "moving_pet": moving_pet,
        }
