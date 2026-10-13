"""NIfTI reading shared by the datasets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np


def load_nifti_image(path: str | Path) -> nib.Nifti1Image:
    """Open a NIfTI file without reading its voxel data.

    Args:
        path: Source NIfTI file.

    Returns:
        Lazily loaded image.
    """
    return cast(nib.Nifti1Image, nib.load(str(path)))


def load_float_nifti_with_header(
        path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Read float32 NIfTI values together with the image they come from.

    Args:
        path: Source NIfTI file.

    Returns:
        Voxel array in the original axis order, and the image carrying the
        affine and header.
    """
    image = load_nifti_image(path)
    return np.asarray(image.get_fdata(dtype=np.float32)), image


def load_float_nifti(path: str | Path, *, sanitize: bool) -> np.ndarray:
    """Read float32 NIfTI values with optional non-finite replacement.

    Args:
        path: Source NIfTI file.
        sanitize: Replace NaN and infinite values with zero when true.

    Returns:
        Image array in the original NIfTI axis order.
    """
    image, _ = load_float_nifti_with_header(path)
    if sanitize:
        return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image
