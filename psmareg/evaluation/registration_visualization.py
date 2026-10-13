"""Register one PET/CT pair from a checkpoint and render diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from psmareg.data.paired_dataset import PSMARegPairDataset
from psmareg.evaluation.registration_figures import (
    RegistrationImages,
    compute_mse,
    plot_registration,
    plot_vector_field,
)
from psmareg.evaluation.visualization_inputs import (
    VisualizationPair,
    VisualizationSettings,
    load_checkpoint_config,
    load_model,
    prepare_visualization_inputs,
)
from psmareg.models.common.registration.cascades import (
    AffineTransMorphDiffCascadeRegistration,)
from psmareg.models.common.registration.warping import (
    compose_displacements,
    warp_image_3d,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _VisualizationArtifacts:
    """Images and fields produced for one registration visualization."""

    fixed_ct: Tensor
    moving_ct: Tensor
    warped_ct: Tensor
    fixed_pet: Tensor
    moving_pet: Tensor
    warped_pet: Tensor
    body_translation: Tensor | None
    affine_parameters: Tensor | None
    use_prealign: bool
    external_prealign: Tensor | None
    prealign_translation: Tensor
    prealign_displacement: Tensor
    stage1: Tensor
    affine: Tensor | None
    residual: Tensor
    displacement: Tensor


def run_visualization(settings: VisualizationSettings) -> None:
    """Render registration images and displacement diagnostics for one pair.

    Args:
        settings: Paths, model bounds, and pre-alignment choices.
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(settings.device if settings.device == "cuda" and
                          torch.cuda.is_available() else "cpu")

    dataset = PSMARegPairDataset(settings.manifest)
    sample = dataset[settings.sample_index]

    fixed_ct = cast(Tensor, sample["fixed_ct"]).unsqueeze(0).to(device)
    fixed_pet = cast(Tensor, sample["fixed_pet"]).unsqueeze(0).to(device)
    moving_ct = cast(Tensor, sample["moving_ct"]).unsqueeze(0).to(device)
    moving_pet = cast(Tensor, sample["moving_pet"]).unsqueeze(0).to(device)
    fixed = torch.cat((fixed_ct, fixed_pet), dim=1)
    moving = torch.cat((moving_ct, moving_pet), dim=1)
    external_prealign = (cast(
        Tensor, sample["prealign_displacement"]).unsqueeze(0).to(device)
                         if "prealign_displacement" in sample else None)

    _, _, depth, _, _ = fixed_ct.shape
    slice_index = (settings.slice_index
                   if settings.slice_index >= 0 else depth // 2)

    model = load_model(settings.checkpoint, device, settings.model)
    checkpoint_config = load_checkpoint_config(settings.checkpoint, device)
    pair = VisualizationPair(
        fixed_ct=fixed_ct,
        fixed_pet=fixed_pet,
        moving_ct=moving_ct,
        moving_pet=moving_pet,
        fixed=fixed,
        moving=moving,
        external_prealign=external_prealign,
    )
    artifacts = _infer_registration(settings, pair, model, checkpoint_config)
    subject = str(sample["subject"])
    followup_index = str(sample["followup_index"])
    _log_diagnostics(settings, subject, followup_index, slice_index, artifacts)
    _save_figures(settings, subject, followup_index, slice_index, artifacts)


def _infer_registration(
    settings: VisualizationSettings,
    pair: VisualizationPair,
    model: nn.Module,
    checkpoint_config: Mapping[str, Any],
) -> _VisualizationArtifacts:
    """Apply optional pre-alignment, run the network, and warp the pair."""
    with torch.no_grad():
        prepared = prepare_visualization_inputs(settings, pair, model,
                                                checkpoint_config)
        moving = prepared.moving
        prealigned = (prepared.use_prealign or
                      pair.external_prealign is not None or
                      settings.bone_affine is not None)
        model_inputs = {
            "fixed_ct": prepared.fixed_ct,
            "fixed_pet": prepared.fixed_pet,
            "moving_ct": moving.moving_ct,
            "moving_pet": moving.moving_pet,
        }
        body_translation = affine_parameters = None
        if isinstance(model, AffineTransMorphDiffCascadeRegistration):
            output = model(**model_inputs, return_diagnostics=True)
            body_translation = output.body_translation
            affine_parameters = output.affine_parameters
        else:
            output = model(**model_inputs)

        model_displacement = cast(Tensor, output[0]).float()
        displacement = (compose_displacements(moving.displacement,
                                              model_displacement).float()
                        if prealigned else model_displacement)
        LOGGER.info("Mean |disp| %s", displacement.abs().mean().item())
        LOGGER.info("Max |disp| %s", displacement.abs().max().item())
        affine = None
        if len(output) > 3:
            affine = cast(Tensor, output[3])
            if prealigned:
                affine = compose_displacements(moving.displacement, affine)
            affine = affine.float()
        warped_ct = warp_image_3d(pair.moving_ct,
                                  displacement,
                                  padding_mode=prepared.output_padding_mode)
        warped_pet = warp_image_3d(pair.moving_pet,
                                   displacement,
                                   padding_mode=prepared.output_padding_mode)

    return _VisualizationArtifacts(
        fixed_ct=pair.fixed_ct,
        moving_ct=pair.moving_ct,
        warped_ct=warped_ct,
        fixed_pet=pair.fixed_pet,
        moving_pet=pair.moving_pet,
        warped_pet=warped_pet,
        body_translation=body_translation,
        affine_parameters=affine_parameters,
        use_prealign=prepared.use_prealign,
        external_prealign=pair.external_prealign,
        prealign_translation=moving.translation,
        prealign_displacement=moving.displacement,
        stage1=cast(Tensor, output[1]).float(),
        affine=affine,
        residual=cast(Tensor, output[2]).float(),
        displacement=displacement,
    )


def _magnitude_lines(label: str, field: Tensor) -> list[str]:
    """Format the mean and maximum absolute value of one field."""
    magnitude = field.abs()
    return [
        f"{f'Mean |{label}|:':<17}{float(magnitude.mean().detach().cpu()):.4f}",
        f"{f'Max |{label}|:':<17}{float(magnitude.max().detach().cpu()):.4f}",
    ]


def _xyz(values: Tensor) -> str:
    """Format three components with millivoxel precision."""
    return ", ".join(f"{value:.3f}" for value in values.detach().cpu().tolist())


def _log_diagnostics(
    settings: VisualizationSettings,
    subject: str,
    followup_index: str,
    slice_index: int,
    artifacts: _VisualizationArtifacts,
) -> None:
    """Log image similarity and displacement diagnostics."""
    result = artifacts
    lines = [
        f"Subject: {subject}",
        f"Follow-up: {followup_index}",
        f"Slice: {slice_index}",
        "CT MSE before:  "
        f"{compute_mse(result.moving_ct, result.fixed_ct):.6f}",
        "CT MSE after:   "
        f"{compute_mse(result.warped_ct, result.fixed_ct):.6f}",
        "PET MSE before: "
        f"{compute_mse(result.moving_pet, result.fixed_pet):.6f}",
        "PET MSE after:  "
        f"{compute_mse(result.warped_pet, result.fixed_pet):.6f}",
    ]
    if (result.use_prealign or result.external_prealign is not None or
            settings.bone_affine is not None):
        lines.append("Prealign mean displacement (x,y,z): " +
                     _xyz(result.prealign_translation[0]))
    lines.extend(_magnitude_lines("stage1", result.stage1))
    if result.affine is not None:
        lines.extend(_magnitude_lines("affine", result.affine))
        if result.body_translation is not None:
            lines.append("Body translation (x,y,z): " +
                         _xyz(result.body_translation[0]))
        if result.affine_parameters is not None:
            lines.append("Learned residual translation (x,y,z): " +
                         _xyz(result.affine_parameters[0][:3]))
    lines.extend(_magnitude_lines("residual", result.residual))
    lines.extend(_magnitude_lines("disp", result.displacement))
    lines.append("Std disp:        "
                 f"{float(result.displacement.std().detach().cpu()):.4f}")
    for line in lines:
        LOGGER.info("%s", line)


def _save_figures(
    settings: VisualizationSettings,
    subject: str,
    followup_index: str,
    slice_index: int,
    artifacts: _VisualizationArtifacts,
) -> None:
    """Write registration and displacement figures in their original order."""
    result = artifacts
    base_name = f"{subject}_followup_{followup_index}_slice_{slice_index:03d}"

    plot_registration(
        output_path=settings.output_dir / f"{base_name}_registration.png",
        images=RegistrationImages(result.fixed_ct, result.moving_ct,
                                  result.warped_ct, result.fixed_pet,
                                  result.moving_pet, result.warped_pet),
        slice_index=slice_index,
        title=f"{subject} follow-up {followup_index} slice {slice_index}",
    )
    if settings.registration_only:
        plt.close("all")
        LOGGER.info("Saved registration figure to: %s", settings.output_dir)
        return

    fields: list[tuple[str, str, Tensor | None]] = [
        ("stage1", "Stage 1 displacement", result.stage1),
        ("affine", "Affine displacement", result.affine),
        ("prealign", "Prealign displacement", result.prealign_displacement if
         result.use_prealign or result.external_prealign is not None else None),
        ("residual", "Residual displacement", result.residual),
        ("displacement", "Displacement", result.displacement),
    ]
    for suffix, title_prefix, field in fields:
        if field is None:
            continue
        plot_vector_field(
            output_path=settings.output_dir / f"{base_name}_{suffix}.png",
            field=field,
            slice_index=slice_index,
            title_prefix=title_prefix,
        )
    LOGGER.info("Saved figures to: %s", settings.output_dir)
