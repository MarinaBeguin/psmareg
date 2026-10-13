"""Local PSMAReg scoring on training pairs with available labels."""

from __future__ import annotations

import csv
import functools
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from psmareg.data.nifti_io import load_nifti_image
from psmareg.evaluation.displacement_inference import (
    OfficialSpatialTransformer,)

_SCORING_PACKAGE = "Learn2Reg26_PSMA.psmareg_scoring_program"

MetricRow = dict[str, float | str]


@functools.lru_cache(maxsize=None)
def _scoring_module(name: str) -> Any:
    """Import one module of the official challenge scoring program.

    The scoring program is an external checkout that is not distributed with
    this package; it must be importable as ``Learn2Reg26_PSMA``.

    Args:
        name: Module path inside the scoring program.

    Raises:
        ImportError: If the scoring program is not on the import path.
    """
    try:
        return importlib.import_module(f"{_SCORING_PACKAGE}.{name}")
    except ImportError as error:
        raise ImportError(
            "Local scoring needs the official Learn2Reg26_PSMA scoring "
            "program. Clone it and add its parent directory to PYTHONPATH."
        ) from error


@dataclass(frozen=True)
class _ReferencePair:
    """Reference images and labels needed for one pair's challenge metrics."""

    fixed_ct_labels: np.ndarray
    moving_ct_labels: np.ndarray
    moving_pet_labels: np.ndarray
    fixed_ct_image: np.ndarray
    moving_pet_image: np.ndarray
    spacing_mm: tuple[float, float, float]


def _surface_metrics() -> Any:
    """Return the official surface-distance metrics module."""
    return _scoring_module("surface_distance.metrics")


def _digital_diffeomorphism() -> Any:
    """Return the official non-diffeomorphic-volume module."""
    return _scoring_module("digital_diffeomorphism")


def get_entry_value(entry: dict[str, Any], *keys: str) -> str:
    """Return the first available value among several possible JSON keys.

    Raises:
        KeyError: If none of the keys holds a string.
    """
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str):
            return value
    raise KeyError(f"Missing keys {keys}. Available keys: {list(entry.keys())}")


def get_compute_device(device_name: str) -> torch.device:
    """Return CUDA device if available and requested, otherwise CPU."""
    if device_name == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to float tensor on device."""
    return torch.from_numpy(array).float().to(
        device=device,
        non_blocking=device.type == "cuda",
    )


def load_displacement(path: Path, device: torch.device) -> torch.Tensor:
    """Load low-resolution displacement and upsample it to full resolution."""
    displacement_array = load_nifti_image(str(path)).get_fdata()
    displacement = to_device_tensor(displacement_array, device)

    if displacement.shape[0] != 3:
        displacement = displacement.permute(3, 0, 1, 2).contiguous()
    else:
        displacement = displacement.clone()

    return F.interpolate(
        displacement.unsqueeze(0),
        scale_factor=2.0,
        mode="trilinear",
        align_corners=False,
    )


def compute_average_label_dice(
    fixed_labels: np.ndarray,
    warped_labels: np.ndarray,
    label_ids: range,
) -> float:
    """Compute average Dice over selected labels.

    Args:
        fixed_labels: Fixed label map.
        warped_labels: Moving label map warped to the fixed grid.
        label_ids: Labels over which the Dice is averaged.

    Returns:
        Mean Dice over the labels present, or NaN when there is none.
    """
    dice_scores: list[float] = []

    for label_id in label_ids:
        dice_score = _surface_metrics().compute_dice_coefficient(
            fixed_labels == label_id,
            warped_labels == label_id,
        )
        if not np.isnan(dice_score):
            dice_scores.append(float(dice_score))

    if not dice_scores:
        return float("nan")

    return float(np.mean(dice_scores))


def compute_average_label_hd95(
    fixed_labels: np.ndarray,
    moving_labels: np.ndarray,
    warped_labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    label_ids: range,
) -> float:
    """Compute average HD95 over selected labels.

    Args:
        fixed_labels: Fixed label map.
        moving_labels: Moving label map before warping.
        warped_labels: Moving label map warped to the fixed grid.
        spacing_mm: Voxel spacing of the fixed image.
        label_ids: Labels over which the distance is averaged.

    Returns:
        Mean 95th-percentile Hausdorff distance in millimeters, or NaN when no
        label is usable.
    """
    hd95_scores: list[float] = []

    for label_id in label_ids:
        fixed_mask = fixed_labels == label_id
        moving_mask = moving_labels == label_id

        if not fixed_mask.any() or not moving_mask.any():
            continue

        warped_mask = warped_labels == label_id
        surface_distances = _surface_metrics().compute_surface_distances(
            fixed_mask,
            warped_mask,
            spacing_mm,
        )
        hd95_score = _surface_metrics().compute_robust_hausdorff(
            surface_distances,
            95.0,
        )

        if np.isfinite(hd95_score):
            hd95_scores.append(float(hd95_score))

    if not hd95_scores:
        return float("nan")

    return float(np.mean(hd95_scores))


def compute_percent_ndv(
    displacement: torch.Tensor,
    fixed_ct_image: np.ndarray,
) -> float:
    """Compute percentage of non-diffeomorphic volume."""
    displacement_np = displacement.squeeze(0).detach().cpu().numpy()

    mask = fixed_ct_image[1:-1, 1:-1, 1:-1] > 0
    transformation = displacement_np + _digital_diffeomorphism(
    ).get_identity_grid(displacement_np)

    jacobian_determinants = _digital_diffeomorphism().calc_jac_dets(
        transformation)
    _, _, non_diff_volume, _ = _digital_diffeomorphism().calc_measurements(
        jacobian_determinants, mask)

    total_voxels = float(np.sum(mask))
    if total_voxels <= 0.0:
        return float("nan")

    return float(non_diff_volume / total_voxels * 100.0)


def compute_pet_burden_metrics(
    pet_label_moving: torch.Tensor,
    pet_label_warped: torch.Tensor,
    pet_image_moving: torch.Tensor,
) -> tuple[float, float]:
    """Compute MTV and TLG percentage errors.

    Args:
        pet_label_moving: Moving lesion labels.
        pet_label_warped: Lesion labels warped to the fixed grid.
        pet_image_moving: Moving PET image.

    Returns:
        Percent errors of the metabolic tumor volume and of the total lesion
        glycolysis.
    """
    mtv_before = torch.count_nonzero(pet_label_moving > 0).item()
    mtv_after = torch.count_nonzero(pet_label_warped > 0).item()

    tlg_before = torch.sum(
        (pet_label_moving > 0).float() * pet_image_moving).item()
    tlg_after = torch.sum(
        (pet_label_warped > 0).float() * pet_image_moving).item()

    mtv_percent_error = 100.0 * abs(mtv_after - mtv_before) / max(
        float(mtv_before),
        1e-6,
    )
    tlg_percent_error = 100.0 * abs(tlg_after - tlg_before) / max(
        float(tlg_before),
        1e-6,
    )

    return mtv_percent_error, tlg_percent_error


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize metric values."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def write_csv(rows: list[MetricRow], output_path: Path) -> None:
    """Write per-pair metrics to CSV.

    Raises:
        ValueError: If there is no row to write.
    """
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write JSON summary."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=True)


def get_followup_indices(entry: dict[str, Any], requested: str) -> list[str]:
    """Return follow-up indices to score."""
    if requested != "all":
        return [requested]

    indices: list[str] = []
    for key in entry:
        if key.startswith("Follow-up ") and key.endswith(" CT Label"):
            indices.append(key.split()[1])

    return sorted(indices)


def _load_reference_pair(entry: dict[str, Any], followup_index: str,
                         reference_dir: Path) -> _ReferencePair:
    """Resolve and load the baseline/follow-up images used by scoring."""
    fixed_ct_label_path = reference_dir / get_entry_value(
        entry, "Baseline CT Label", "CT Label")
    moving_ct_label_path = reference_dir / get_entry_value(
        entry, f"Follow-up {followup_index} CT Label")
    moving_pet_label_path = reference_dir / get_entry_value(
        entry, f"Follow-up {followup_index} PET Label")
    fixed_ct_image_path = reference_dir / get_entry_value(
        entry, "Baseline CT", "CT")
    moving_pet_image_path = reference_dir / get_entry_value(
        entry, f"Follow-up {followup_index} PET")

    fixed_ct_label_nii = load_nifti_image(str(fixed_ct_label_path))
    fixed_ct_labels = fixed_ct_label_nii.get_fdata()
    moving_ct_labels = load_nifti_image(str(moving_ct_label_path)).get_fdata()
    moving_pet_labels = load_nifti_image(str(moving_pet_label_path)).get_fdata()
    fixed_ct_image = load_nifti_image(str(fixed_ct_image_path)).get_fdata()
    moving_pet_image = load_nifti_image(str(moving_pet_image_path)).get_fdata()
    zooms = fixed_ct_label_nii.header.get_zooms()
    spacing_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return _ReferencePair(fixed_ct_labels, moving_ct_labels, moving_pet_labels,
                          fixed_ct_image, moving_pet_image, spacing_mm)


def _warp_reference_labels(
    spatial_transformer: OfficialSpatialTransformer,
    displacement: torch.Tensor,
    moving_ct_labels: torch.Tensor,
    moving_pet_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp CT and PET labels in the same inference context and order."""
    with torch.inference_mode():
        warped_ct_labels = spatial_transformer(
            moving_ct_labels.unsqueeze(0).unsqueeze(0),
            displacement,
        ).squeeze(0).squeeze(0)
        warped_pet_labels = spatial_transformer(
            moving_pet_labels.unsqueeze(0).unsqueeze(0),
            displacement,
        ).squeeze(0).squeeze(0)
    return warped_ct_labels, warped_pet_labels


def score_pair(
    entry: dict[str, Any],
    followup_index: str,
    reference_dir: Path,
    prediction_dir: Path,
    spatial_transformer: OfficialSpatialTransformer,
    device: torch.device,
) -> MetricRow:
    """Score one baseline-to-follow-up pair.

    Args:
        entry: Dataset JSON entry of one subject.
        followup_index: Follow-up scan scored against the baseline.
        reference_dir: Root of the reference images and labels.
        prediction_dir: Directory of the exported displacement fields.
        spatial_transformer: Official warping module for the label size.
        device: Device on which the warps run.

    Returns:
        Subject, follow-up, and the five challenge metrics.

    Raises:
        FileNotFoundError: If the displacement field of the pair is missing.
    """
    subject = str(entry["subject"])
    subject_id = subject.rsplit("_", maxsplit=1)[-1]

    displacement_path = (
        prediction_dir /
        f"disp_{subject_id}_00_{subject_id}_{followup_index}.nii.gz")
    if not displacement_path.exists():
        raise FileNotFoundError(f"Missing displacement: {displacement_path}")

    displacement = load_displacement(displacement_path, device)

    reference = _load_reference_pair(entry, followup_index, reference_dir)
    moving_ct_labels_torch = to_device_tensor(reference.moving_ct_labels,
                                              device)
    moving_pet_labels_torch = to_device_tensor(reference.moving_pet_labels,
                                               device)
    moving_pet_image_torch = to_device_tensor(reference.moving_pet_image,
                                              device)

    warped_ct_labels_torch, warped_pet_labels_torch = _warp_reference_labels(
        spatial_transformer, displacement, moving_ct_labels_torch,
        moving_pet_labels_torch)

    warped_ct_labels = torch.round(
        warped_ct_labels_torch).detach().cpu().numpy()

    ct_dsc = compute_average_label_dice(
        fixed_labels=reference.fixed_ct_labels,
        warped_labels=warped_ct_labels,
        label_ids=range(1, 118),
    )
    ct_hd95 = compute_average_label_hd95(
        fixed_labels=reference.fixed_ct_labels,
        moving_labels=reference.moving_ct_labels,
        warped_labels=warped_ct_labels,
        spacing_mm=reference.spacing_mm,
        label_ids=range(1, 118),
    )

    mtv_percent_error, tlg_percent_error = compute_pet_burden_metrics(
        pet_label_moving=moving_pet_labels_torch,
        pet_label_warped=warped_pet_labels_torch,
        pet_image_moving=moving_pet_image_torch,
    )

    percent_ndv = compute_percent_ndv(
        displacement=displacement,
        fixed_ct_image=reference.fixed_ct_image,
    )

    return {
        "subject": subject,
        "subject_id": subject_id,
        "followup_index": followup_index,
        "DSC": ct_dsc,
        "HD95": ct_hd95,
        "MTV Percent Error": mtv_percent_error,
        "TLG Percent Error": tlg_percent_error,
        "Percent NDV": percent_ndv,
    }
