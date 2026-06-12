"""Visualize fixed, moving, warped images and displacement fields.

python -m visualization.visualize_registration \
  --manifest ./manifests/train_pairs.csv \
  --checkpoint ./results/ct_pet_registration/latest.pt \
  --output_dir ./results/ct_pet_registration/visualizations \
  --sample_index 0 \
  --slice_index -1

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.paired_dataset import PSMARegPairDataset
from models.attention_unet3d import AttentionUNet3DRegistration
from utils.warping import warp_image_3d


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--slice_index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _to_numpy_slice(volume: torch.Tensor, slice_index: int) -> np.ndarray:
    """Extract one axial slice from a tensor shaped [1, 1, D, H, W]."""
    array = volume.detach().cpu().numpy()

    if array.ndim != 5:
        raise ValueError(f"Expected [B, C, D, H, W], got shape {array.shape}")

    return cast(np.ndarray, array[0, 0, slice_index, :, :])


def _to_numpy_disp_slice(displacement: torch.Tensor, channel: int, slice_index: int) -> np.ndarray:
    """Extract one displacement component slice from [1, 3, D, H, W]."""
    array = displacement.detach().cpu().numpy()

    if array.ndim != 5:
        raise ValueError(f"Expected [B, 3, D, H, W], got shape {array.shape}")

    return cast(np.ndarray, array[0, channel, slice_index, :, :])


def _compute_mse(image_a: torch.Tensor, image_b: torch.Tensor) -> float:
    """Compute MSE between two tensors."""
    return float(torch.mean((image_a - image_b) ** 2).detach().cpu())


def _load_model(checkpoint_path: Path, device: torch.device) -> AttentionUNet3DRegistration:
    """Load model from checkpoint."""
    model = AttentionUNet3DRegistration(
        in_channels=4,
        out_channels=3,
        features=(16, 32, 64, 128),
    ).to(device)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _plot_registration(
    output_path: Path,
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    warped_ct: torch.Tensor,
    fixed_pet: torch.Tensor,
    moving_pet: torch.Tensor,
    warped_pet: torch.Tensor,
    slice_index: int,
    title: str,
) -> None:
    """Save fixed, moving, warped and residual comparison figure."""
    ct_residual_before = torch.abs(fixed_ct - moving_ct)
    ct_residual_after = torch.abs(fixed_ct - warped_ct)

    pet_residual_before = torch.abs(fixed_pet - moving_pet)
    pet_residual_after = torch.abs(fixed_pet - warped_pet)

    ct_vmin = -1.0
    ct_vmax = 1.0
    pet_vmin = 0.0
    pet_vmax = 1.0

    figure, axes = plt.subplots(4, 4, figsize=(18, 16))

    ct_images = [
        fixed_ct,
        moving_ct,
        warped_ct,
        ct_residual_after,
    ]
    ct_titles = [
        "Fixed CT",
        "Moving CT",
        "Warped CT",
        "|Fixed CT - Warped CT|",
    ]

    pet_images = [
        fixed_pet,
        moving_pet,
        warped_pet,
        pet_residual_after,
    ]
    pet_titles = [
        "Fixed PET",
        "Moving PET",
        "Warped PET",
        "|Fixed PET - Warped PET|",
    ]

    residual_images = [
        ct_residual_before,
        ct_residual_after,
        pet_residual_before,
        pet_residual_after,
    ]
    residual_titles = [
        "|Fixed CT - Moving CT|",
        "|Fixed CT - Warped CT|",
        "|Fixed PET - Moving PET|",
        "|Fixed PET - Warped PET|",
    ]

    for column, image in enumerate(ct_images):
        axes[0, column].imshow(
            _to_numpy_slice(image, slice_index),
            cmap="gray",
            vmin=ct_vmin,
            vmax=ct_vmax,
        )
        axes[0, column].set_title(ct_titles[column])
        axes[0, column].axis("off")

    for column, image in enumerate(pet_images):
        axes[1, column].imshow(
            _to_numpy_slice(image, slice_index),
            cmap="magma",
            vmin=pet_vmin,
            vmax=pet_vmax,
        )
        axes[1, column].set_title(pet_titles[column])
        axes[1, column].axis("off")

    for column, image in enumerate(residual_images):
        axes[2, column].imshow(
            _to_numpy_slice(image, slice_index),
            cmap="hot",
        )
        axes[2, column].set_title(residual_titles[column])
        axes[2, column].axis("off")

    axes[3, 0].imshow(
        _to_numpy_slice(fixed_ct, slice_index),
        cmap="gray",
        vmin=ct_vmin,
        vmax=ct_vmax,
    )
    axes[3, 0].imshow(
        _to_numpy_slice(moving_pet, slice_index),
        cmap="magma",
        alpha=0.45,
        vmin=pet_vmin,
        vmax=pet_vmax,
    )
    axes[3, 0].set_title("Fixed CT + Moving PET")
    axes[3, 0].axis("off")

    axes[3, 1].imshow(
        _to_numpy_slice(fixed_ct, slice_index),
        cmap="gray",
        vmin=ct_vmin,
        vmax=ct_vmax,
    )
    axes[3, 1].imshow(
        _to_numpy_slice(warped_pet, slice_index),
        cmap="magma",
        alpha=0.45,
        vmin=pet_vmin,
        vmax=pet_vmax,
    )
    axes[3, 1].set_title("Fixed CT + Warped PET")
    axes[3, 1].axis("off")

    axes[3, 2].axis("off")
    axes[3, 3].axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_displacement(
    output_path: Path,
    displacement: torch.Tensor,
    slice_index: int,
) -> None:
    """Save displacement component and norm figure."""
    displacement_norm = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    component_titles = [
        "Displacement X",
        "Displacement Y",
        "Displacement Z",
        "Displacement norm",
    ]

    component_images = [
        _to_numpy_disp_slice(displacement, 0, slice_index),
        _to_numpy_disp_slice(displacement, 1, slice_index),
        _to_numpy_disp_slice(displacement, 2, slice_index),
        _to_numpy_slice(displacement_norm, slice_index),
    ]

    for index, image in enumerate(component_images):
        image_handle = axes[index].imshow(image, cmap="coolwarm")
        axes[index].set_title(component_titles[index])
        axes[index].axis("off")
        figure.colorbar(image_handle, ax=axes[index], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    dataset = PSMARegPairDataset(args.manifest)
    sample = dataset[args.sample_index]

    fixed = cast(torch.Tensor, sample["fixed"]).unsqueeze(0).to(device)
    moving = cast(torch.Tensor, sample["moving"]).unsqueeze(0).to(device)

    fixed_ct = cast(torch.Tensor, sample["fixed_ct"]).unsqueeze(0).to(device)
    fixed_pet = cast(torch.Tensor, sample["fixed_pet"]).unsqueeze(0).to(device)
    moving_ct = cast(torch.Tensor, sample["moving_ct"]).unsqueeze(0).to(device)
    moving_pet = cast(torch.Tensor, sample["moving_pet"]).unsqueeze(0).to(device)

    _, _, depth, _, _ = fixed_ct.shape
    slice_index = args.slice_index
    if slice_index < 0:
        slice_index = depth // 2

    model = _load_model(args.checkpoint, device)

    with torch.no_grad():
        displacement = model(fixed, moving)
        warped_ct = warp_image_3d(moving_ct, displacement)
        warped_pet = warp_image_3d(moving_pet, displacement)

    subject = str(sample["subject"])
    followup_index = str(sample["followup_index"])

    mse_before_ct = _compute_mse(moving_ct, fixed_ct)
    mse_after_ct = _compute_mse(warped_ct, fixed_ct)
    mse_before_pet = _compute_mse(moving_pet, fixed_pet)
    mse_after_pet = _compute_mse(warped_pet, fixed_pet)

    mean_displacement = float(displacement.abs().mean().detach().cpu())
    max_displacement = float(displacement.abs().max().detach().cpu())
    std_displacement = float(displacement.std().detach().cpu())

    print(f"Subject: {subject}")
    print(f"Follow-up: {followup_index}")
    print(f"Slice: {slice_index}")
    print(f"CT MSE before:  {mse_before_ct:.6f}")
    print(f"CT MSE after:   {mse_after_ct:.6f}")
    print(f"PET MSE before: {mse_before_pet:.6f}")
    print(f"PET MSE after:  {mse_after_pet:.6f}")
    print(f"Mean |disp|:    {mean_displacement:.4f}")
    print(f"Max |disp|:     {max_displacement:.4f}")
    print(f"Std disp:       {std_displacement:.4f}")

    base_name = f"{subject}_followup_{followup_index}_slice_{slice_index:03d}"

    _plot_registration(
        output_path=args.output_dir / f"{base_name}_registration.png",
        fixed_ct=fixed_ct,
        moving_ct=moving_ct,
        warped_ct=warped_ct,
        fixed_pet=fixed_pet,
        moving_pet=moving_pet,
        warped_pet=warped_pet,
        slice_index=slice_index,
        title=f"{subject} follow-up {followup_index} slice {slice_index}",
    )

    _plot_displacement(
        output_path=args.output_dir / f"{base_name}_displacement.png",
        displacement=displacement,
        slice_index=slice_index,
    )

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
