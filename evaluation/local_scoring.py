"""Local PSMAReg scoring on training pairs with available labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from Learn2Reg26_PSMA.psmareg_scoring_program.surface_distance import metrics as surface_metrics
import Learn2Reg26_PSMA.psmareg_scoring_program.digital_diffeomorphism as dd


MetricRow = dict[str, float | str]


class SpatialTransformer(torch.nn.Module):
    """N-D spatial transformer adapted from VoxelMorph-style warping."""

    def __init__(self, size: tuple[int, int, int], mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, size_item) for size_item in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)

        self.register_buffer("grid", grid)

    def forward(self, source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp source with displacement field."""
        new_locations = self.grid + flow
        shape = flow.shape[2:]

        for axis_index, axis_size in enumerate(shape):
            new_locations[:, axis_index, ...] = (
                2.0 * (new_locations[:, axis_index, ...] / (axis_size - 1) - 0.5)
            )

        new_locations = new_locations.permute(0, 2, 3, 4, 1)
        new_locations = new_locations[..., [2, 1, 0]]

        return functional.grid_sample(
            source,
            new_locations,
            align_corners=False,
            mode=self.mode,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--json_name", type=str, default="PSMAReg_dataset.json")
    parser.add_argument("--followups", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def get_entry_value(entry: dict[str, Any], *keys: str) -> str:
    """Return the first available value among several possible JSON keys."""
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
    displacement_array = nib.load(str(path)).get_fdata()
    displacement = to_device_tensor(displacement_array, device)

    if displacement.shape[0] != 3:
        displacement = displacement.permute(3, 0, 1, 2).contiguous()
    else:
        displacement = displacement.clone()

    return functional.interpolate(
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
    """Compute average Dice over selected labels."""
    dice_scores: list[float] = []

    for label_id in label_ids:
        dice_score = surface_metrics.compute_dice_coefficient(
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
    """Compute average HD95 over selected labels."""
    hd95_scores: list[float] = []

    for label_id in label_ids:
        fixed_mask = fixed_labels == label_id
        moving_mask = moving_labels == label_id

        if not fixed_mask.any() or not moving_mask.any():
            continue

        warped_mask = warped_labels == label_id
        surface_distances = surface_metrics.compute_surface_distances(
            fixed_mask,
            warped_mask,
            spacing_mm,
        )
        hd95_score = surface_metrics.compute_robust_hausdorff(
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
    transformation = displacement_np + dd.get_identity_grid(displacement_np)

    jacobian_determinants = dd.calc_jac_dets(transformation)
    _, _, non_diff_volume, _ = dd.calc_measurements(jacobian_determinants, mask)

    total_voxels = float(np.sum(mask))
    if total_voxels <= 0.0:
        return float("nan")

    return float(non_diff_volume / total_voxels * 100.0)


def compute_pet_burden_metrics(
    pet_label_moving: torch.Tensor,
    pet_label_warped: torch.Tensor,
    pet_image_moving: torch.Tensor,
) -> tuple[float, float]:
    """Compute MTV and TLG percentage errors."""
    mtv_before = torch.count_nonzero(pet_label_moving > 0).item()
    mtv_after = torch.count_nonzero(pet_label_warped > 0).item()

    tlg_before = torch.sum((pet_label_moving > 0).float() * pet_image_moving).item()
    tlg_after = torch.sum((pet_label_warped > 0).float() * pet_image_moving).item()

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
    """Write per-pair metrics to CSV."""
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


def score_pair(
    entry: dict[str, Any],
    followup_index: str,
    reference_dir: Path,
    prediction_dir: Path,
    spatial_transformer: SpatialTransformer,
    device: torch.device,
) -> MetricRow:
    """Score one baseline-to-follow-up pair."""
    subject = str(entry["subject"])
    subject_id = subject.split("_")[-1]

    displacement_path = (
        prediction_dir / f"disp_{subject_id}_00_{subject_id}_{followup_index}.nii.gz"
    )
    if not displacement_path.exists():
        raise FileNotFoundError(f"Missing displacement: {displacement_path}")

    displacement = load_displacement(displacement_path, device)

    fixed_ct_label_path = reference_dir / get_entry_value(
        entry,
        "Baseline CT Label",
        "CT Label",
    )
    moving_ct_label_path = reference_dir / get_entry_value(
        entry,
        f"Follow-up {followup_index} CT Label",
    )
    moving_pet_label_path = reference_dir / get_entry_value(
        entry,
        f"Follow-up {followup_index} PET Label",
    )
    fixed_ct_image_path = reference_dir / get_entry_value(
        entry,
        "Baseline CT",
        "CT",
    )
    moving_pet_image_path = reference_dir / get_entry_value(
        entry,
        f"Follow-up {followup_index} PET",
    )

    fixed_ct_label_nii = nib.load(str(fixed_ct_label_path))
    fixed_ct_labels = fixed_ct_label_nii.get_fdata()
    moving_ct_labels = nib.load(str(moving_ct_label_path)).get_fdata()
    moving_pet_labels = nib.load(str(moving_pet_label_path)).get_fdata()
    fixed_ct_image = nib.load(str(fixed_ct_image_path)).get_fdata()
    moving_pet_image = nib.load(str(moving_pet_image_path)).get_fdata()

    moving_ct_labels_torch = to_device_tensor(moving_ct_labels, device)
    moving_pet_labels_torch = to_device_tensor(moving_pet_labels, device)
    moving_pet_image_torch = to_device_tensor(moving_pet_image, device)

    with torch.inference_mode():
        warped_ct_labels_torch = spatial_transformer(
            moving_ct_labels_torch.unsqueeze(0).unsqueeze(0),
            displacement,
        ).squeeze(0).squeeze(0)

        warped_pet_labels_torch = spatial_transformer(
            moving_pet_labels_torch.unsqueeze(0).unsqueeze(0),
            displacement,
        ).squeeze(0).squeeze(0)

    warped_ct_labels = torch.round(warped_ct_labels_torch).detach().cpu().numpy()

    spacing_mm = tuple(float(value) for value in fixed_ct_label_nii.header.get_zooms()[:3])

    ct_dsc = compute_average_label_dice(
        fixed_labels=fixed_ct_labels,
        warped_labels=warped_ct_labels,
        label_ids=range(1, 118),
    )
    ct_hd95 = compute_average_label_hd95(
        fixed_labels=fixed_ct_labels,
        moving_labels=moving_ct_labels,
        warped_labels=warped_ct_labels,
        spacing_mm=spacing_mm,
        label_ids=range(1, 118),
    )

    mtv_percent_error, tlg_percent_error = compute_pet_burden_metrics(
        pet_label_moving=moving_pet_labels_torch,
        pet_label_warped=warped_pet_labels_torch,
        pet_image_moving=moving_pet_image_torch,
    )

    percent_ndv = compute_percent_ndv(
        displacement=displacement,
        fixed_ct_image=fixed_ct_image,
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


def main() -> None:
    """Run local scoring on training pairs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json_path = args.reference_dir / args.json_name
    with dataset_json_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    entries = dataset.get("training_paired", [])
    if not entries:
        raise RuntimeError(f"No training_paired entries found in {dataset_json_path}")

    device = get_compute_device(args.device)
    print(f"Using device: {device}")

    spatial_transformer = SpatialTransformer(
        size=(192, 192, 288),
        mode="nearest",
    ).to(device)
    spatial_transformer.eval()

    rows: list[MetricRow] = []

    for entry in entries:
        subject = str(entry["subject"])
        followup_indices = get_followup_indices(entry, args.followups)

        for followup_index in followup_indices:
            try:
                row = score_pair(
                    entry=entry,
                    followup_index=followup_index,
                    reference_dir=args.reference_dir,
                    prediction_dir=args.prediction_dir,
                    spatial_transformer=spatial_transformer,
                    device=device,
                )
            except FileNotFoundError as error:
                print(f"Skipping {subject} FU{followup_index}: {error}")
                continue

            rows.append(row)
            print(
                f"{subject} FU{followup_index} | "
                f"DSC={row['DSC']:.4f} | "
                f"HD95={row['HD95']:.4f} | "
                f"MTV={row['MTV Percent Error']:.4f}% | "
                f"TLG={row['TLG Percent Error']:.4f}% | "
                f"NDV={row['Percent NDV']:.4f}%"
            )

    if not rows:
        raise RuntimeError("No pairs were scored. Check prediction filenames.")

    metric_names = [
        "DSC",
        "HD95",
        "MTV Percent Error",
        "TLG Percent Error",
        "Percent NDV",
    ]
    summary = {
        metric_name: summarize([float(row[metric_name]) for row in rows])
        for metric_name in metric_names
    }

    write_csv(rows, args.output_dir / "local_scoring_per_pair.csv")
    write_json(summary, args.output_dir / "local_scoring_summary.json")

    print("")
    print("Summary:")
    for metric_name, metric_summary in summary.items():
        print(
            f"{metric_name}: "
            f"{metric_summary['mean']:.4f} ± {metric_summary['std']:.4f}"
        )


if __name__ == "__main__":
    main()
