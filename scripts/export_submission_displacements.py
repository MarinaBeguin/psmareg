"""Export PSMAReg displacement fields in the expected submission format.

python -m scripts.export_submission_displacements \
  --manifest ./manifests/val_pairs.csv \
  --checkpoint ./results/baseline/best_val.pt \
  --output_dir ./results/baseline/predictions \
  --reference_json /path/to/reference_data/PSMAReg_dataset.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional

from models.network import SmallRegistrationUNet
from scripts.submission_dataset import SubmissionDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> SmallRegistrationUNet:
    model = SmallRegistrationUNet().to(device)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _subject_id(subject: str) -> str:
    return subject.split("_")[-1]


def _make_output_name(subject: str) -> str:
    subject_id = _subject_id(subject)
    return f"disp_{subject_id}_00_{subject_id}_01.nii.gz"


def _load_affine_from_reference(reference_json: Path, dataset_root: Path, subject: str) -> np.ndarray:
    with reference_json.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    for entry in metadata["validation_paired"]:
        if entry["subject"] == subject:
            image_path = dataset_root / entry["Baseline CT"]
            return np.asarray(nib.load(str(image_path)).affine)

    return np.eye(4, dtype=np.float32)


def _save_displacement(
    displacement: torch.Tensor,
    output_path: Path,
    affine: np.ndarray,
) -> None:
    displacement_np = displacement.detach().cpu().numpy()[0]
    displacement_np = np.moveaxis(displacement_np, 0, -1).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(displacement_np, affine), str(output_path))


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    dataset = SubmissionDataset(args.manifest)
    model = _load_model(args.checkpoint, device)

    dataset_root = args.reference_json.parent

    for index in range(len(dataset)):
        sample = dataset[index]
        subject = str(sample["subject"])
        followup_index = str(sample["followup_index"])

        if followup_index != "01":
            continue

        fixed = sample["fixed"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        moving = sample["moving"].unsqueeze(0).to(device)  # type: ignore[union-attr]

        full_res_disp = model(fixed, moving)

        low_res_disp = functional.interpolate(
            full_res_disp,
            scale_factor=0.5,
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )

        output_name = _make_output_name(subject)
        affine = _load_affine_from_reference(
            reference_json=args.reference_json,
            dataset_root=dataset_root,
            subject=subject,
        )

        _save_displacement(
            displacement=low_res_disp,
            output_path=args.output_dir / output_name,
            affine=affine,
        )

        print(f"Saved {output_name} | low-res shape={tuple(low_res_disp.shape)}")


if __name__ == "__main__":
    main()
