"""Checkpoint inference and NIfTI export for paired PSMAReg volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from psmareg.data.nifti_io import load_float_nifti_with_header
from psmareg.data.preprocessing import preprocess_ct, preprocess_pet
from psmareg.evaluation.visualization_inputs import (
    VisualizationPair,
    VisualizationSettings,
    prepare_visualization_inputs,
)
from psmareg.models.common.registration.warping import compose_displacements


def pair_from_row(
    row: Mapping[str, str],
    device: torch.device,
) -> tuple[VisualizationPair, nib.Nifti1Image]:
    """Load a manifest pair with the CT/PET normalization used in training.

    Raises:
        ValueError: If an image is not on the fixed CT voxel grid or the
            external displacement has an unexpected shape.
    """
    arrays = {}
    reference: nib.Nifti1Image | None = None
    for name in ("fixed_ct", "fixed_pet", "moving_ct", "moving_pet"):
        array, image = load_float_nifti_with_header(row[name])
        if reference is None:
            reference = image
        elif (array.shape != reference.shape or
              not np.allclose(image.affine, reference.affine, atol=1e-3)):
            raise ValueError(f"{name} is not on the fixed CT voxel grid")
        arrays[name] = array
    if reference is None:
        raise ValueError("Manifest pair contains no fixed CT image")

    def image_tensor(name: str) -> Tensor:
        """Preprocess one loaded volume as a ``[1, 1, D, H, W]`` tensor."""
        array = arrays[name]
        if name.endswith("ct"):
            array = preprocess_ct(array)
            return (torch.from_numpy(array).float().unsqueeze(0).unsqueeze(
                0).to(device))
        pet = preprocess_pet(torch.from_numpy(array).float().unsqueeze(0))
        return pet.unsqueeze(0).to(device)

    fixed_ct, fixed_pet, moving_ct, moving_pet = (image_tensor(name)
                                                  for name in ("fixed_ct",
                                                               "fixed_pet",
                                                               "moving_ct",
                                                               "moving_pet"))
    external = None
    if row.get("prealign_displacement"):
        field, _ = load_float_nifti_with_header(row["prealign_displacement"])
        if field.ndim == 5 and field.shape[-2:] == (1, 3):
            field = field[..., 0, :]
        if field.ndim != 4 or field.shape[-1] != 3:
            raise ValueError("External displacement must have shape [X,Y,Z,3]")
        external = (torch.from_numpy(
            np.moveaxis(field, -1, 0)[[2, 1,
                                       0]].copy()).unsqueeze(0).to(device))
    pair = VisualizationPair(fixed_ct, fixed_pet, moving_ct, moving_pet,
                             torch.cat((fixed_ct, fixed_pet), dim=1),
                             torch.cat((moving_ct, moving_pet), dim=1),
                             external)
    return pair, reference


@torch.no_grad()
def predict_displacement(
    model: nn.Module,
    pair: VisualizationPair,
    settings: VisualizationSettings,
    checkpoint_config: Mapping[str, object],
) -> Tensor:
    """Return full-resolution pull-back flow in grid_sample channel order.

    Args:
        model: Loaded registration model.
        pair: Preprocessed images of one manifest row.
        settings: Visualization and pre-alignment settings.
        checkpoint_config: Configuration stored in the checkpoint.

    Returns:
        Field of shape ``[1, 3, D, H, W]`` mapping the fixed grid to the
        original moving image, pre-alignment included.
    """
    prepared = prepare_visualization_inputs(settings, pair, model,
                                            checkpoint_config)
    moving = prepared.moving
    output = model(fixed_ct=prepared.fixed_ct,
                   fixed_pet=prepared.fixed_pet,
                   moving_ct=moving.moving_ct,
                   moving_pet=moving.moving_pet)
    model_flow = output[0].float()
    if (prepared.use_prealign or pair.external_prealign is not None or
            settings.bone_affine is not None):
        return compose_displacements(moving.displacement, model_flow).float()
    return model_flow


def fullres_to_submitted_lowres(flow: Tensor) -> Tensor:
    """Downsample a field to the half-resolution challenge submission."""
    return F.interpolate(flow,
                         scale_factor=0.5,
                         mode="trilinear",
                         align_corners=False,
                         recompute_scale_factor=False)


def submitted_lowres_to_official_flow(flow: Tensor) -> Tensor:
    """Emulate NIfTI readback and the official scorer's twofold upsampling."""
    return F.interpolate(flow[:, [2, 1, 0]],
                         scale_factor=2,
                         mode="trilinear",
                         align_corners=False,
                         recompute_scale_factor=False)


def save_displacement(flow: Tensor, path: Path, affine: np.ndarray) -> None:
    """Save one flow as [X,Y,Z,3] NIfTI with axis-order vector components.

    Args:
        flow: Field of shape ``[1, 3, X, Y, Z]`` in x/y/z channel order.
        path: Destination NIfTI file.
        affine: Affine of the fixed image.

    Raises:
        ValueError: If ``flow`` is not a single three-channel field.
    """
    if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError("Expected one displacement field [1,3,X,Y,Z]")
    array = np.moveaxis(flow.detach().cpu().numpy()[0, [2, 1, 0]], 0,
                        -1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))


class OfficialSpatialTransformer(nn.Module):
    """Warp with the grid convention of the official challenge scorer.

    Displacement channels follow the array axes of the volume, unlike the
    x/y/z order used by the training code.
    """

    def __init__(self,
                 size: tuple[int, int, int],
                 mode: str = "bilinear",
                 padding_mode: str = "zeros") -> None:
        """Precompute the identity grid of one volume size.

        Args:
            size: Spatial shape of the warped volumes.
            mode: Interpolation mode passed to ``grid_sample``.
            padding_mode: Padding mode passed to ``grid_sample``.
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        grid = torch.stack(
            torch.meshgrid(*(torch.arange(s) for s in size), indexing="ij"))
        self.register_buffer("grid",
                             grid.unsqueeze(0).float(),
                             persistent=False)

    def forward(self, src: Tensor, flow: Tensor) -> Tensor:
        """Warp one image with an axis-ordered voxel displacement.

        Args:
            src: Image of shape ``[B, C, X, Y, Z]``.
            flow: Displacement of shape ``[B, 3, X, Y, Z]``.

        Returns:
            Warped image with the shape of ``src``.
        """
        grid = cast(Tensor, self.grid)
        locations = grid + flow
        shape = flow.shape[2:]
        for axis, length in enumerate(shape):
            locations[:, axis] = 2 * (locations[:, axis] / (length - 1) - 0.5)
        locations = locations.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src,
                             locations,
                             align_corners=True,
                             mode=self.mode,
                             padding_mode=self.padding_mode)
