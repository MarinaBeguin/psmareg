"""Case selection, registration, and caching for the qualitative figure.

The displacement prediction follows the low-resolution challenge submission
and the official scorer's upsampling convention.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import nibabel as nib
import torch

from psmareg.data.nifti_io import load_float_nifti_with_header
from psmareg.data.preprocessing import preprocess_ct, preprocess_pet
from psmareg.evaluation.displacement_inference import (
    OfficialSpatialTransformer,
    fullres_to_submitted_lowres,
    submitted_lowres_to_official_flow,
    predict_displacement,
)
from psmareg.evaluation.visualization_inputs import (
    VisualizationPair,
    VisualizationSettings,
)

LOGGER = logging.getLogger(__name__)


def load_prediction_cache(path: Path) -> dict[str, np.ndarray] | None:
    """Load a prediction cache, ignoring incomplete or invalid files.

    Args:
        path: Cache entry written by ``save_prediction_cache``.

    Returns:
        Cached arrays, or ``None`` when the file is absent or invalid.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            data = {name: cached[name] for name in cached.files}
        required_volumes = {
            f"{phase}_{modality}" for phase in ("fixed", "moving", "warped")
            for modality in ("ct", "pet", "pet_label")
        }
        if not required_volumes.union({"display_spacing", "ras_affine"
                                      }).issubset(data):
            raise ValueError("Missing cached volumes or orientation metadata.")
        shape = data["fixed_ct"].shape
        if len(shape) != 3 or any(
                data[name].shape != shape for name in required_volumes):
            raise ValueError("Invalid cached volume shapes.")
        if data["display_spacing"].shape != (
                3,) or data["ras_affine"].shape != (4, 4):
            raise ValueError("Invalid cached orientation metadata.")
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile,
            zlib.error) as exc:
        LOGGER.info("Ignoring invalid prediction cache %s: %s. Recomputing.",
                    path, exc)
        return None
    LOGGER.info("Loaded cached prediction: %s", path)
    return data


def save_prediction_cache(path: Path, data: dict[str, np.ndarray]) -> None:
    """Publish a cache atomically only after every array has been written."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb",
                                         dir=path.parent,
                                         prefix=path.name + ".",
                                         suffix=".tmp",
                                         delete=False) as handle:
            temporary_path = Path(handle.name)
            np.savez(handle, **cast(dict[str, Any], data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def axial_radiological_orientation(affine: np.ndarray) -> np.ndarray:
    """Reorder source voxel axes to RAS using NIfTI metadata (no interpolation).

    The released scans must have cardinal axes. Oblique images require a
    physical reslice, not merely an array transpose, and are rejected here.

    Raises:
        ValueError: If the image axes are oblique.
    """
    spacing = nib.affines.voxel_sizes(affine)
    directions = affine[:3, :3] / spacing[np.newaxis, :]
    cardinal = np.zeros((3, 3))
    for axis in range(3):
        world_axis = int(np.argmax(np.abs(directions[:, axis])))
        cardinal[world_axis, axis] = np.sign(directions[world_axis, axis])
    if not np.allclose(directions, cardinal, atol=1e-4):
        raise ValueError(
            "Oblique NIfTI: a physical axial reslice is required for display.")
    return nib.orientations.ornt_transform(
        nib.orientations.io_orientation(affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )


def axial_radiological_view(array: np.ndarray,
                            orientation: np.ndarray) -> np.ndarray:
    """Reorient to [superior slice, anterior-posterior, right-left]."""
    ras = nib.orientations.apply_orientation(array, orientation)
    return ras.transpose(2, 1, 0)[:, ::-1, ::-1]


@dataclass(frozen=True)
class CaseSpec:
    """One labeled subject and optional follow-up selected for display."""
    label: str
    subject: str
    followup: str | None


@dataclass(frozen=True)
class MetricSpec:
    """Four optional challenge metrics supplied for one displayed case."""
    subject: str
    followup: str
    dsc: float
    hd95: float
    mtv: float
    tlg: float


def parse_case(text: str) -> CaseSpec:
    """Parse labels such as good=0336_fu01, medium=0200_fu02, hard=0359."""
    if "=" in text:
        label, token = text.split("=", 1)
        label = label.strip()
    else:
        label, token = "", text
    token = token.strip()
    if "_fu" in token:
        subject, followup = token.split("_fu", 1)
    else:
        subject, followup = token, None
    subject = subject.strip()
    if not subject.startswith("PSMARegPSMA_"):
        subject = f"PSMARegPSMA_{subject}"
    return CaseSpec(label=label or subject.split("_")[-1],
                    subject=subject,
                    followup=followup)


def case_token(row: Mapping[str, str]) -> str:
    """Return the short subject and follow-up identifier."""
    return f"{row['subject'].split('_')[-1]}_fu{row['followup_index']}"


def select_rows(
    rows: list[dict[str, str]],
    cases: list[CaseSpec],
) -> list[tuple[CaseSpec, dict[str, str]]]:
    """Select the requested cases from the labeled manifest.

    Raises:
        ValueError: If a requested case is not in the manifest.
    """
    selected: list[tuple[CaseSpec, dict[str, str]]] = []
    for case in cases:
        matches = [
            row for row in rows if row["subject"] == case.subject and
            (case.followup is None or row["followup_index"] == case.followup)
        ]
        if not matches:
            raise ValueError(
                f"Could not find case {case.subject} followup={case.followup}.")
        selected.append((case, matches[0]))
    return selected


def choose_slice(
    moving_pet_label: np.ndarray,
    warped_pet_label: np.ndarray,
    fixed_pet: np.ndarray,
    moving_pet: np.ndarray,
    row_index: int,
    slice_args: list[str] | None,
) -> int:
    """Choose a representative axial slice, prioritizing visible lesions.

    Args:
        moving_pet_label: Moving lesion labels, slice axis first.
        warped_pet_label: Warped lesion labels with the same shape.
        fixed_pet: Fixed PET, used when no lesion is annotated.
        moving_pet: Moving PET, used when no lesion is annotated.
        row_index: Position of the case in the figure.
        slice_args: Optional slice indices given on the command line.

    Returns:
        Index of the displayed axial slice.
    """
    if slice_args and row_index < len(slice_args):
        return int(slice_args[row_index])
    moving_count = (moving_pet_label > 0).reshape(moving_pet_label.shape[0],
                                                  -1).sum(axis=1)
    warped_count = (warped_pet_label > 0).reshape(warped_pet_label.shape[0],
                                                  -1).sum(axis=1)
    # For the paper figure, prefer slices where the annotated PET lesions are
    # actually visible.  Warped lesion is only a weak tie-breaker.  Do not use
    # PET intensity here: physiological uptake can otherwise dominate small
    # annotated lesions and select a visually unhelpful slice.
    # Only moving/warped lesions are outlined in the paper figure. Prefer slices
    # where both are visible, rather than selecting a fixed-only lesion slice.
    lesion_score = moving_count + warped_count + np.minimum(
        moving_count, warped_count)
    if float(np.max(lesion_score)) > 0:
        return int(np.argmax(lesion_score))
    pet_score = (
        np.nan_to_num(fixed_pet).reshape(fixed_pet.shape[0], -1).sum(axis=1) +
        np.nan_to_num(moving_pet).reshape(moving_pet.shape[0], -1).sum(axis=1))
    return int(np.argmax(pet_score)) if np.isfinite(
        pet_score).any() else fixed_pet.shape[0] // 2


def report_top_lesion_slices(
    fixed_pet_label: np.ndarray,
    moving_pet_label: np.ndarray,
    warped_pet_label: np.ndarray,
    token: str,
    top_n: int,
) -> None:
    """Print the slices containing the most annotated PET lesions.

    Args:
        fixed_pet_label: Fixed lesion labels, slice axis first.
        moving_pet_label: Moving lesion labels with the same shape.
        warped_pet_label: Warped lesion labels with the same shape.
        token: Case identifier used in the log line.
        top_n: Number of slices to report; zero disables the report.
    """
    fixed_count = (fixed_pet_label > 0).reshape(fixed_pet_label.shape[0],
                                                -1).sum(axis=1)
    moving_count = (moving_pet_label > 0).reshape(moving_pet_label.shape[0],
                                                  -1).sum(axis=1)
    warped_count = (warped_pet_label > 0).reshape(warped_pet_label.shape[0],
                                                  -1).sum(axis=1)
    score = moving_count + warped_count + np.minimum(moving_count, warped_count)
    order = np.argsort(score)[::-1][:max(0, top_n)]
    entries = [(f"z={int(z)} fixed={int(fixed_count[z])} "
                f"moving={int(moving_count[z])} warped={int(warped_count[z])}")
               for z in order
               if score[z] > 0]
    if entries:
        LOGGER.info("Top lesion slices for %s: %s", token, " | ".join(entries))


def load_metrics(
    path: Path | None,
    overrides: list[MetricSpec] | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Read challenge scores and apply optional manual overrides."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = (row["subject"], row["followup_index"])
            out[key] = {
                "DSC": float(row["DSC"]),
                "HD95": float(row["HD95"]),
                "MTV": float(row["MTV Percent Error"]),
                "TLG": float(row["TLG Percent Error"]),
            }
    for metric in overrides or []:
        out[(metric.subject, metric.followup)] = {
            "DSC": metric.dsc,
            "HD95": metric.hd95,
            "MTV": metric.mtv,
            "TLG": metric.tlg,
        }
    return out


def _load_case_volumes(
        row: Mapping[str, str]) -> tuple[dict[str, np.ndarray], Any]:
    """Load the raw volumes of one case and check their common voxel grid."""
    fixed_ct, fixed_header = load_float_nifti_with_header(row["fixed_ct"])
    volumes = {"fixed_ct": fixed_ct}
    for name in ("moving_ct", "fixed_pet", "moving_pet", "fixed_pet_label",
                 "moving_pet_label"):
        array, header = load_float_nifti_with_header(row[name])
        if (array.shape != fixed_ct.shape or
                not np.allclose(header.affine, fixed_header.affine, atol=1e-3)):
            raise ValueError(f"{name} is not on the common fixed voxel grid; "
                             "resampling is required.")
        volumes[name] = array
    return volumes, fixed_header


def _normalized_pet(pet: np.ndarray) -> np.ndarray:
    """Apply the training PET normalization to a raw volume."""
    return preprocess_pet(
        torch.from_numpy(pet).float().unsqueeze(0)).squeeze(0).numpy()


def _warp_moving_volumes(
    volumes: Mapping[str, np.ndarray],
    flow: torch.Tensor,
    image_warp_mode: str,
) -> dict[str, np.ndarray]:
    """Warp the raw moving CT, PET, and lesion label with the official flow."""
    device = flow.device
    size = (int(flow.shape[-3]), int(flow.shape[-2]), int(flow.shape[-1]))
    warpers = {
        "moving_ct": OfficialSpatialTransformer(size, mode=image_warp_mode),
        "moving_pet": OfficialSpatialTransformer(size, mode=image_warp_mode),
        "moving_pet_label": OfficialSpatialTransformer(size, mode="nearest"),
    }
    warped = {}
    for name, warper in warpers.items():
        tensor = torch.from_numpy(
            volumes[name]).float().to(device).unsqueeze(0).unsqueeze(0)
        warped[name.replace("moving", "warped")] = warper.to(device)(
            tensor, flow).squeeze().detach().cpu().numpy()
    return warped


def predict_case(
    image_warp_mode: str,
    model: torch.nn.Module,
    settings: VisualizationSettings,
    checkpoint_config: Mapping[str, Any],
    row: Mapping[str, str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Register one case, warp the raw images, and orient display arrays.

    Args:
        image_warp_mode: Interpolation used to warp the raw images.
        model: Loaded registration model.
        settings: Visualization and pre-alignment settings.
        checkpoint_config: Configuration stored in the checkpoint.
        row: Manifest row of the case.
        device: Device on which the model runs.

    Returns:
        Preprocessed fixed, moving, and warped volumes with their lesion
        labels in axial radiological orientation, the display voxel spacing,
        and the reoriented affine.
    """
    token = case_token(row)
    LOGGER.info("Loading volumes for %s", token)
    volumes, fixed_header = _load_case_volumes(row)
    LOGGER.info("Preprocessing %s", token)
    data = {
        "fixed_ct": preprocess_ct(volumes["fixed_ct"]),
        "moving_ct": preprocess_ct(volumes["moving_ct"]),
        "fixed_pet": _normalized_pet(volumes["fixed_pet"]),
        "moving_pet": _normalized_pet(volumes["moving_pet"]),
    }
    fixed = torch.from_numpy(np.stack(
        (data["fixed_ct"], data["fixed_pet"]))).float().unsqueeze(0).to(device)
    moving = torch.from_numpy(np.stack(
        (data["moving_ct"],
         data["moving_pet"]))).float().unsqueeze(0).to(device)
    LOGGER.info("Running registration for %s", token)

    with torch.inference_mode():
        pair = VisualizationPair(fixed[:, :1], fixed[:, 1:2], moving[:, :1],
                                 moving[:, 1:2], fixed, moving, None)
        flow = submitted_lowres_to_official_flow(
            fullres_to_submitted_lowres(
                predict_displacement(model, pair, settings, checkpoint_config)))
        LOGGER.info("Warping images for %s", token)
        warped = _warp_moving_volumes(volumes, flow, image_warp_mode)

    del fixed, moving, pair, flow
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    data.update({
        "warped_ct": preprocess_ct(warped["warped_ct"]),
        "warped_pet": _normalized_pet(warped["warped_pet"]),
        "fixed_pet_label": volumes["fixed_pet_label"],
        "moving_pet_label": volumes["moving_pet_label"],
        "warped_pet_label": warped["warped_pet_label"],
    })
    orientation = axial_radiological_orientation(fixed_header.affine)
    ras_affine = fixed_header.affine @ nib.orientations.inv_ornt_aff(
        orientation, volumes["fixed_ct"].shape)
    LOGGER.info(
        "NIfTI axes for %s: %s; display: axial radiological (R left, A top)",
        token, nib.aff2axcodes(fixed_header.affine))
    data = {
        name: axial_radiological_view(array, orientation)
        for name, array in data.items()
    }
    data["display_spacing"] = nib.affines.voxel_sizes(ras_affine)[::-1].copy()
    data["ras_affine"] = ras_affine
    return data
