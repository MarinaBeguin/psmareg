"""Registration panels and displacement-field plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def _to_numpy_slice(volume: Tensor, slice_index: int) -> np.ndarray:
    """Extract one axial slice of a ``[B, C, D, H, W]`` volume."""
    array = volume.detach().cpu().numpy()

    if array.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {array.shape}")

    return cast(np.ndarray, array[0, 0, slice_index, :, :])


def _to_numpy_vector_slice(field: Tensor, channel: int,
                           slice_index: int) -> np.ndarray:
    """Extract one axial slice of one displacement component."""
    array = field.detach().cpu().numpy()

    if array.ndim != 5:
        raise ValueError(f"Expected [B,3,D,H,W], got {array.shape}")

    return cast(np.ndarray, array[0, channel, slice_index, :, :])


def compute_mse(image_a: Tensor, image_b: Tensor) -> float:
    """Return mean squared error between two image tensors.

    Args:
        image_a: First image tensor.
        image_b: Second image tensor with the same shape.

    Returns:
        Scalar mean squared error on the CPU.
    """
    return float(torch.mean((image_a - image_b)**2).detach().cpu())


@dataclass(frozen=True)
class RegistrationImages:
    """Fixed, moving, and registered CT/PET views for one subject."""

    fixed_ct: Tensor
    moving_ct: Tensor
    warped_ct: Tensor
    fixed_pet: Tensor
    moving_pet: Tensor
    warped_pet: Tensor


_CT_DISPLAY = {"cmap": "gray", "vmin": -1.0, "vmax": 1.0}
_PET_DISPLAY = {"cmap": "magma", "vmin": 0.0, "vmax": 1.0}


def _show_row(axes: Any, slice_index: int, panels: Sequence[tuple[str, Tensor]],
              display: Mapping[str, Any]) -> None:
    """Draw one row of titled image panels."""
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(_to_numpy_slice(image, slice_index), **display)
        axis.set_title(title)
        axis.axis("off")


def _show_overlay(axis: Any, slice_index: int, title: str, ct: Tensor,
                  pet: Tensor) -> None:
    """Draw a PET image blended over a CT image."""
    axis.imshow(_to_numpy_slice(ct, slice_index), **_CT_DISPLAY)
    axis.imshow(_to_numpy_slice(pet, slice_index), alpha=0.45, **_PET_DISPLAY)
    axis.set_title(title)
    axis.axis("off")


def plot_registration(
    output_path: Path,
    images: RegistrationImages,
    slice_index: int,
    title: str,
) -> None:
    """Save the CT/PET registration and residual panels.

    Args:
        output_path: Destination PNG path.
        images: Fixed, moving, and warped image tensors.
        slice_index: Axial slice to display.
        title: Figure title.
    """
    ct_before = torch.abs(images.fixed_ct - images.moving_ct)
    ct_after = torch.abs(images.fixed_ct - images.warped_ct)
    pet_before = torch.abs(images.fixed_pet - images.moving_pet)
    pet_after = torch.abs(images.fixed_pet - images.warped_pet)

    figure, axes = plt.subplots(4, 4, figsize=(18, 16))
    _show_row(axes[0], slice_index, (
        ("Fixed CT", images.fixed_ct),
        ("Moving CT", images.moving_ct),
        ("Warped CT", images.warped_ct),
        ("|Fixed CT - Warped CT|", ct_after),
    ), _CT_DISPLAY)
    _show_row(axes[1], slice_index, (
        ("Fixed PET", images.fixed_pet),
        ("Moving PET", images.moving_pet),
        ("Warped PET", images.warped_pet),
        ("|Fixed PET - Warped PET|", pet_after),
    ), _PET_DISPLAY)
    _show_row(axes[2], slice_index, (
        ("|Fixed CT - Moving CT|", ct_before),
        ("|Fixed CT - Warped CT|", ct_after),
        ("|Fixed PET - Moving PET|", pet_before),
        ("|Fixed PET - Warped PET|", pet_after),
    ), {"cmap": "hot"})
    _show_overlay(axes[3, 0], slice_index, "Fixed CT + Moving PET",
                  images.fixed_ct, images.moving_pet)
    _show_overlay(axes[3, 1], slice_index, "Fixed CT + Warped PET",
                  images.fixed_ct, images.warped_pet)
    axes[3, 2].axis("off")
    axes[3, 3].axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_vector_field(
    output_path: Path,
    field: Tensor,
    slice_index: int,
    title_prefix: str,
) -> None:
    """Save axial displacement components and magnitude as a PNG.

    Args:
        output_path: Destination PNG path.
        field: Displacement tensor in x/y/z voxel order.
        slice_index: Axial slice to display.
        title_prefix: Label used for each panel.
    """
    field_norm = torch.linalg.vector_norm(field, dim=1, keepdim=True)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    titles = [
        f"{title_prefix} X",
        f"{title_prefix} Y",
        f"{title_prefix} Z",
        f"{title_prefix} norm",
    ]

    images = [
        _to_numpy_vector_slice(field, 0, slice_index),
        _to_numpy_vector_slice(field, 1, slice_index),
        _to_numpy_vector_slice(field, 2, slice_index),
        _to_numpy_slice(field_norm, slice_index),
    ]

    for index, image in enumerate(images):
        image_handle = axes[index].imshow(image, cmap="coolwarm")
        axes[index].set_title(titles[index])
        axes[index].axis("off")
        figure.colorbar(image_handle, ax=axes[index], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
