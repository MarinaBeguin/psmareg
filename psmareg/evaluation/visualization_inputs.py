"""Load the model and prepare pre-aligned inputs for visualization."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from psmareg.models.common.registration.prealignment.bone_affine import (
    BoneAffineAligner,
    BoneAffineSettings,
)
from psmareg.models.common.registration.prealignment.deterministic import (
    TranslationPrealignSettings,
    estimate_lowres_ct_translation_field,
)
from psmareg.models.common.registration.warping import (
    compose_displacements,
    warp_image_3d,
)
from psmareg.models.common.registration.cascades import (
    AffineCascadeSettings,
    AffineTransMorphDiffCascadeRegistration,
    AffineTransMorphV2CascadeRegistration,
    TransMorphV2CascadeRegistration,
    TransMorphDiffCascadeRegistration,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOverrides:
    """Model bounds replacing the values stored in the checkpoint."""

    max_velocity: float | None = None
    max_residual_disp: float | None = None
    scaling_steps: int | None = None


@dataclass(frozen=True)
class TranslationPrealignOverrides:
    """Translation pre-alignment; ``None`` keeps the checkpoint value."""

    enabled: bool = False
    downsample: int | None = None
    search_radius: int | None = None
    search_step: int | None = None
    max_translation: float | None = None
    feature_mode: str | None = None
    padding_mode: str | None = None
    input_padding_mode: str | None = None


@dataclass(frozen=True)
class VisualizationSettings:
    """Inputs, outputs, and pre-alignment choices of one visualization.

    Attributes:
        manifest: Paired manifest holding the pair to render.
        checkpoint: Trained checkpoint to load.
        output_dir: Directory receiving the figures.
        sample_index: Row of the manifest to render.
        slice_index: Axial slice, or a negative value for the middle slice.
        device: Requested device name; CUDA falls back to the CPU.
        zero_padding: Whether to force zero padding when warping.
        registration_only: Whether to skip the displacement figures.
        model: Model bounds overriding the checkpoint.
        translation: Translation pre-alignment overrides.
        bone_affine: Bone affine pre-alignment settings, or ``None`` to skip.
    """

    manifest: Path
    checkpoint: Path
    output_dir: Path
    sample_index: int = 0
    slice_index: int = -1
    device: str = "cuda"
    zero_padding: bool = False
    registration_only: bool = False
    model: ModelOverrides = ModelOverrides()
    translation: TranslationPrealignOverrides = TranslationPrealignOverrides()
    bone_affine: BoneAffineSettings | None = None


@dataclass(frozen=True)
class _TranslationPrealign:
    """Translation pre-alignment resolved from overrides and checkpoint."""

    enabled: bool
    downsample: int
    search_radius: int
    search_step: int
    max_translation: float
    padding_mode: str
    input_padding_mode: str
    feature_mode: str


@dataclass(frozen=True)
class VisualizationPair:
    """Images and optional external field loaded from one manifest row."""

    fixed_ct: Tensor
    fixed_pet: Tensor
    moving_ct: Tensor
    moving_pet: Tensor
    fixed: Tensor
    moving: Tensor
    external_prealign: Tensor | None


@dataclass(frozen=True)
class PrealignState:
    """Moving model inputs and accumulated pull-back pre-alignment."""

    moving_ct: Tensor
    moving_pet: Tensor
    displacement: Tensor
    translation: Tensor


@dataclass(frozen=True)
class PreparedVisualization:
    """Network inputs and display padding after optional pre-alignment."""

    fixed_ct: Tensor
    fixed_pet: Tensor
    moving: PrealignState
    use_prealign: bool
    output_padding_mode: str


def load_checkpoint_config(checkpoint_path: Path,
                           device: torch.device) -> Mapping[str, Any]:
    """Read the training configuration stored in a checkpoint.

    Args:
        checkpoint_path: Checkpoint written by a training run.
        device: Device on which the checkpoint is mapped.

    Returns:
        Flat configuration saved with the checkpoint.
    """
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    return config if isinstance(config, Mapping) else {}


def _resolve_prealign_settings(
    overrides: TranslationPrealignOverrides,
    config: Mapping[str, Any],
) -> _TranslationPrealign:
    """Merge command-line and checkpoint pre-alignment settings."""

    def pick(override: Any, name: str, default: Any) -> Any:
        """Prefer the override, then the checkpoint, then the default.

        Args:
            override: Command-line value, or ``None`` when not given.
            name: Suffix of the checkpoint key.
            default: Value used when neither source defines the setting.

        Returns:
            Selected setting.
        """
        if override is not None:
            return override
        return config.get(f"deterministic_translation_{name}", default)

    return _TranslationPrealign(
        enabled=overrides.enabled or
        bool(config.get("deterministic_translation_prealign", False)),
        downsample=int(pick(overrides.downsample, "downsample", 4)),
        search_radius=int(pick(overrides.search_radius, "search_radius", 8)),
        search_step=int(pick(overrides.search_step, "search_step", 2)),
        max_translation=float(pick(overrides.max_translation, "max", 80.0)),
        padding_mode=str(pick(overrides.padding_mode, "padding_mode", "auto")),
        input_padding_mode=str(
            pick(overrides.input_padding_mode, "input_padding_mode", "auto")),
        feature_mode=str(pick(overrides.feature_mode, "feature_mode", "ct")),
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    overrides: ModelOverrides,
) -> nn.Module:
    """Rebuild the checkpoint's cascade and load its weights.

    Args:
        checkpoint_path: Checkpoint written by a training run.
        device: Device receiving the model.
        overrides: Model bounds replacing the checkpoint values when set.

    Returns:
        Model in evaluation mode.
    """
    max_velocity = overrides.max_velocity
    max_residual_disp = overrides.max_residual_disp
    scaling_steps = overrides.scaling_steps
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}

    resolved_max_velocity = (max_velocity if max_velocity is not None else
                             float(config.get("max_velocity", 8.0)))
    resolved_max_residual_disp = (max_residual_disp
                                  if max_residual_disp is not None else float(
                                      config.get("max_residual_disp", 2.0)))
    resolved_scaling_steps = (scaling_steps if scaling_steps is not None else
                              int(config.get("scaling_steps", 7)))

    model_name = str(config.get("model", "transmorph_diff_cascade"))
    model: nn.Module
    if model_name in ("affine_transmorph_diff_cascade",
                      "affine_transmorph_v2_cascade"):
        affine_cascade = (AffineTransMorphDiffCascadeRegistration
                          if model_name == "affine_transmorph_diff_cascade" else
                          AffineTransMorphV2CascadeRegistration)
        model = affine_cascade(
            AffineCascadeSettings(
                max_velocity=resolved_max_velocity,
                max_residual_disp=resolved_max_residual_disp,
                scaling_steps=resolved_scaling_steps,
                max_affine_translation=float(
                    config.get("max_affine_translation", 30.0)),
                max_affine_rotation_degrees=float(
                    config.get("max_affine_rotation_degrees", 15.0)),
                max_affine_scaling=float(config.get("max_affine_scaling",
                                                    0.10)),
                body_center_initialization=bool(
                    config.get("body_center_initialization", False)),
                body_mask_threshold=float(
                    config.get("body_mask_threshold", 0.02)),
                max_body_translation=float(
                    config.get("max_body_translation", 64.0)),
                zero_padding=bool(config.get("zero_padding", False)),
            ))
    else:
        # Unknown names fall back to the first retained cascade.
        cascade = (TransMorphV2CascadeRegistration
                   if model_name == "transmorph_v2_cascade" else
                   TransMorphDiffCascadeRegistration)
        model = cascade(max_velocity=resolved_max_velocity,
                        max_residual_disp=resolved_max_residual_disp,
                        scaling_steps=resolved_scaling_steps)
    model = model.to(device)

    LOGGER.info(
        "Loaded model config: max_velocity=%s, max_residual_disp=%s, "
        "scaling_steps=%s", resolved_max_velocity, resolved_max_residual_disp,
        resolved_scaling_steps)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _apply_external_prealignment(
    state: PrealignState,
    fixed_ct: Tensor,
    external_prealign: Tensor,
    padding_mode: str,
) -> PrealignState:
    """Resize an external field and warp the original moving inputs."""
    displacement = external_prealign.to(dtype=fixed_ct.dtype)
    source_shape = displacement.shape[-3:]
    if displacement.shape[-3:] != fixed_ct.shape[-3:]:
        displacement = torch.nn.functional.interpolate(
            displacement,
            size=fixed_ct.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        displacement = displacement.clone()
        displacement[:, 0] *= fixed_ct.shape[-1] / source_shape[-1]
        displacement[:, 1] *= fixed_ct.shape[-2] / source_shape[-2]
        displacement[:, 2] *= fixed_ct.shape[-3] / source_shape[-3]
    translation = displacement.mean(dim=(2, 3, 4))
    moving_ct = warp_image_3d(state.moving_ct,
                              displacement,
                              padding_mode=padding_mode)
    moving_pet = warp_image_3d(state.moving_pet,
                               displacement,
                               padding_mode=padding_mode)
    return PrealignState(moving_ct, moving_pet, displacement, translation)


def _apply_bone_prealignment(
    settings: BoneAffineSettings,
    state: PrealignState,
    fixed_ct: Tensor,
    original_moving: tuple[Tensor, Tensor],
    padding_mode: str,
) -> PrealignState:
    """Fit a bone affine field and rewarp the original moving images."""
    aligner = BoneAffineAligner(
        dataclasses.replace(settings, padding_mode=padding_mode))
    bone_displacement, bone_info = aligner.fit(fixed_ct,
                                               state.moving_ct).evaluate()
    displacement = compose_displacements(state.displacement, bone_displacement)
    translation = displacement.mean(dim=(2, 3, 4))
    moving_ct = warp_image_3d(original_moving[0],
                              displacement,
                              padding_mode=padding_mode)
    moving_pet = warp_image_3d(original_moving[1],
                               displacement,
                               padding_mode=padding_mode)
    LOGGER.info(
        "Bone affine prealign accepted=%s score=%.4f->%.4f "
        "improvement=%.4f translation=%s rotation_deg=%s scale=%s",
        bone_info.accepted, bone_info.baseline_score, bone_info.affine_score,
        bone_info.improvement, bone_info.translation_xyz,
        bone_info.rotation_xyz_degrees, bone_info.scale_xyz)
    return PrealignState(moving_ct, moving_pet, displacement, translation)


def _apply_ct_translation_prealignment(
    state: PrealignState,
    fixed_ct: Tensor,
    external_prealign: Tensor | None,
    settings: TranslationPrealignSettings,
    input_padding_mode: str,
) -> PrealignState:
    """Search CT translation and warp the current moving model inputs."""
    deterministic_displacement, _ = estimate_lowres_ct_translation_field(
        fixed_ct, state.moving_ct, settings)
    displacement = (compose_displacements(
        state.displacement, deterministic_displacement) if external_prealign
                    is not None else deterministic_displacement)
    translation = displacement.mean(dim=(2, 3, 4))
    moving_ct = warp_image_3d(state.moving_ct,
                              displacement,
                              padding_mode=input_padding_mode)
    moving_pet = warp_image_3d(state.moving_pet,
                               displacement,
                               padding_mode=input_padding_mode)
    return PrealignState(moving_ct, moving_pet, displacement, translation)


def prepare_visualization_inputs(
    settings: VisualizationSettings,
    pair: VisualizationPair,
    model: nn.Module,
    checkpoint_config: Mapping[str, Any],
) -> PreparedVisualization:
    """Resolve padding and run the selected pre-alignment stages.

    Args:
        settings: Visualization and pre-alignment settings.
        pair: Images and optional external field of one manifest row.
        model: Loaded registration model.
        checkpoint_config: Configuration stored in the checkpoint.

    Returns:
        Network inputs and the accumulated pre-alignment.
    """
    prealign = _resolve_prealign_settings(settings.translation,
                                          checkpoint_config)
    fixed_ct = pair.fixed[:, 0:1]
    body_threshold = 0.02
    model_padding = "border"
    if isinstance(model, AffineTransMorphDiffCascadeRegistration):
        body_threshold = model.body_mask_threshold
        model_padding = "zeros" if model.padding_mode == "zeros" else "border"
    output_padding = ("zeros" if settings.zero_padding or
                      model_padding == "zeros" else "border")
    state = PrealignState(
        pair.moving[:, 0:1],
        pair.moving[:, 1:2],
        torch.zeros(fixed_ct.shape[0],
                    3,
                    *fixed_ct.shape[-3:],
                    device=fixed_ct.device,
                    dtype=fixed_ct.dtype),
        torch.zeros(fixed_ct.shape[0],
                    3,
                    device=fixed_ct.device,
                    dtype=fixed_ct.dtype),
    )
    if pair.external_prealign is not None:
        state = _apply_external_prealignment(state, fixed_ct,
                                             pair.external_prealign,
                                             model_padding)
    if settings.bone_affine is not None:
        state = _apply_bone_prealignment(settings.bone_affine, state, fixed_ct,
                                         (pair.moving_ct, pair.moving_pet),
                                         model_padding)
    if prealign.enabled:
        state = _apply_ct_translation_prealignment(
            state,
            fixed_ct,
            pair.external_prealign,
            TranslationPrealignSettings(
                downsample_factor=prealign.downsample,
                search_radius=prealign.search_radius,
                search_step=prealign.search_step,
                body_threshold=body_threshold,
                max_translation=prealign.max_translation,
                padding_mode=(model_padding if prealign.padding_mode == "auto"
                              else prealign.padding_mode),
                feature_mode=prealign.feature_mode,
            ),
            (model_padding if prealign.input_padding_mode == "auto" else
             prealign.input_padding_mode),
        )
    return PreparedVisualization(fixed_ct, pair.fixed[:, 1:2], state,
                                 prealign.enabled, output_padding)
