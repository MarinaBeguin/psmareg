"""Structured configuration and validation for synthetic pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from psmareg.config.sections import (
    apply_flat_overrides,
    build_config,
    flatten_config,
    load_python_recipe,
)


@dataclass(frozen=True)
class DataConfig:
    """Input manifests, output directory, cache, and hold-out exclusion."""

    train_manifest: Path | None = None
    val_manifest: Path | None = None
    output_dir: Path | None = None
    cache_dir: Path | None = None
    exclude_subjects_manifest: Path | None = None

    def required(self, name: str) -> Path:
        """Return a path that a validated configuration always defines.

        Args:
            name: ``train_manifest``, ``val_manifest``, or ``output_dir``.

        Raises:
            ValueError: If the path is unset.
        """
        value = getattr(self, name)
        if not isinstance(value, Path):
            raise ValueError(f"{name} is required.")
        return value


@dataclass(frozen=True)
class InitializationConfig:
    """Checkpoint continuation and compatible-weight initialization."""

    start_from: Path | None = None
    initialize_deformable_from: Path | None = None
    load_optimizer_state: bool = True
    inherit_best_val_loss: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    """Epoch count, batch size, optimizer, and early stopping."""

    epochs: int = 100
    batch_size: int = 1
    lr: float = 0.0001
    num_workers: int = 4
    patience: int = 20


@dataclass(frozen=True)
class RuntimeConfig:
    """Mixed precision, validation cadence, diagnostics, and reproducibility."""

    amp: bool = True
    validation_interval: int = 1
    validate_at_start: bool = False
    train_expensive_metrics: bool = False
    metric_sampling_stride: int = 4
    validation_metric_sampling_stride: int = 1
    max_train_batches: int = 0
    max_val_batches: int = 0
    seed: int = 42
    deterministic_validation: bool = True


@dataclass(frozen=True)
class ArchitectureConfig:
    """Registration model and affine/deformable bounds."""

    model: str = "affine_transmorph_diff_cascade"
    max_velocity: float = 8.0
    max_velocity_start: float | None = None
    max_velocity_ramp_epochs: int = 0
    max_residual_disp: float = 4.0
    refinement_steps: int = 2
    scaling_steps: int = 7
    max_affine_translation: float = 30.0
    max_affine_rotation_degrees: float = 15.0
    max_affine_scaling: float = 0.1
    body_center_initialization: bool = False
    body_mask_threshold: float = 0.02
    max_body_translation: float = 64.0
    zero_padding: bool = False
    reset_affine_head: bool = False


@dataclass(frozen=True)
class LossConfig:  # pylint: disable=too-many-instance-attributes
    """Synthetic supervision, similarity, and regularization."""

    lambda_pet: float = 0.2
    image_similarity_weight: float = 1.0
    lambda_field: float = 0.1
    lambda_stage1_field: float = 0.01
    lambda_affine_field: float = 0.0
    lambda_residual_field: float = 0.03
    lambda_smooth: float = 0.01
    lambda_epe: float = 0.0
    lambda_multiscale_field: float = 0.0
    lambda_field_gradient: float = 0.0
    lambda_jacobian: float = 0.0
    lambda_residual_bending: float = 0.0
    lambda_global_residual_penalty: float = 0.0
    global_residual_target_threshold: float = 6.0
    global_residual_target_scale: float = 12.0
    global_residual_downsample_factor: int = 8
    lambda_ct_label_dice: float = 0.0
    lambda_pet_label_dice: float = 0.0
    lambda_pet_biomarker: float = 0.0
    pet_biomarker_downsample_factor: int = 4
    lambda_macro_ct_label_dice: float = 0.0
    macro_ct_label_preset: str = "stable_no_bladder"
    macro_ct_label_downsample_factor: int = 2
    lambda_symmetric_field: float = 0.0
    symmetric_probability: float = 0.0
    pet_foreground_weight: float = 0.0
    label_loss_downsample_factor: int = 2
    max_ct_label_classes_per_batch: int = 4
    jacobian_margin: float = 0.05
    jacobian_loss_stride: int = 4


@dataclass(frozen=True)
class AugmentationConfig:  # pylint: disable=too-many-instance-attributes
    """PET intensity, SynthMorph-like, and pose-aware augmentation."""

    pet_independent_intensity_probability: float = 0.0
    pet_independent_scale_log_std: float = 0.0
    pet_independent_gamma_range: float = 0.0
    pet_independent_bias_range: float = 0.0
    synthmorph_like: bool = False
    synthmorph_ct_noise_std: float = 0.03
    synthmorph_ct_bias_amplitude: float = 0.12
    synthmorph_pet_noise_std: float = 0.02
    synthmorph_pet_lesion_dropout_probability: float = 0.35
    synthmorph_pet_false_hot_probability: float = 0.15
    synthmorph_fov_dropout_probability: float = 0.25
    synthmorph_fov_max_fraction: float = 0.18
    synthmorph_resolution_augmentation_probability: float = 0.0
    synthmorph_min_resolution_scale: float = 0.55
    pose_aware_probability: float = 0.0
    pose_aware_attempts: int = 3
    pose_max_translation: float = 14.0
    pose_max_rotation_degrees: float = 8.0
    pose_max_scaling: float = 0.03
    pose_region_blur_downsample: int = 8
    pose_min_region_fraction: float = 0.01


@dataclass(frozen=True)
class DeformationConfig:
    """Synthetic target generation and metric resolution."""

    target_min_jacobian: float = 0.0
    target_generation_attempts: int = 4
    max_elastic_displacement: float = 8.0
    elastic_control_point_spacing: int = 24
    max_translation: float = 8.0
    max_rotation_degrees: float = 5.0
    max_scaling: float = 0.05
    similarity_downsample_factor: int = 2
    validation_similarity_downsample_factor: int = 1
    coarse_target_downsample_factor: int = 4


@dataclass(frozen=True)
class CurriculumConfig:
    """Synthetic deformation difficulty schedule."""

    curriculum_min_scale: float = 0.25
    curriculum_warmup_fraction: float = 0.2
    curriculum_ramp_fraction: float = 0.5
    curriculum_easy_probability: float = 0.2
    curriculum_extra_hard_probability: float = 0.0
    curriculum_extra_hard_max_scale: float = 1.0
    curriculum_total_epochs: int | None = None


@dataclass(frozen=True)
class SyntheticPretrainingConfig:
    """Immutable collection of all grouped synthetic pretraining options."""

    data: DataConfig = field(default_factory=DataConfig)
    initialization: InitializationConfig = field(
        default_factory=InitializationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


PATH_FIELDS = {
    "train_manifest",
    "val_manifest",
    "output_dir",
    "cache_dir",
    "exclude_subjects_manifest",
    "start_from",
    "initialize_deformable_from",
}
OPTIONAL_NUMERIC_FIELDS = {
    "max_velocity_start": float,
    "curriculum_total_epochs": int,
}
NULLABLE_FIELDS = (PATH_FIELDS -
                   {"train_manifest", "val_manifest", "output_dir"
                   }) | OPTIONAL_NUMERIC_FIELDS.keys()
MODEL_NAMES = {
    "transmorph_diff_cascade",
    "affine_transmorph_diff_cascade",
    "transmorph_v2_cascade",
    "affine_transmorph_v2_cascade",
}


def _load_module(path: Path) -> ModuleType:
    """Load a Python configuration without changing import search paths."""
    return load_python_recipe(
        path,
        module_name=f"psmareg_synthetic_config_{path.stem}",
        error_label="synthetic configuration")


def validate_synthetic_config(values: dict[str, Any]) -> None:
    """Reject invalid combinations before data loading or CUDA setup.

    Args:
        values: Flattened synthetic pretraining configuration.

    Raises:
        ValueError: If a parameter or combination is invalid.
    """
    for name in ("train_manifest", "val_manifest", "output_dir"):
        if values[name] is None:
            raise ValueError(f"{name} must be defined in the config or CLI.")
    if values["model"] not in MODEL_NAMES:
        raise ValueError(
            f"Unknown synthetic registration model: {values['model']}")
    if values["start_from"] is not None and values[
            "initialize_deformable_from"] is not None:
        raise ValueError(
            "Use either --start_from or --initialize_deformable_from, not both."
        )
    for name in (
            "validation_interval",
            "coarse_target_downsample_factor",
            "similarity_downsample_factor",
            "validation_similarity_downsample_factor",
            "metric_sampling_stride",
            "validation_metric_sampling_stride",
            "label_loss_downsample_factor",
            "max_ct_label_classes_per_batch",
            "macro_ct_label_downsample_factor",
            "jacobian_loss_stride",
            "global_residual_downsample_factor",
            "target_generation_attempts",
            "pose_aware_attempts",
            "pose_region_blur_downsample",
            "epochs",
            "batch_size",
            "patience",
    ):
        if values[name] < 1:
            raise ValueError(f"{name} must be at least 1.")
    for name in (
            "lambda_macro_ct_label_dice",
            "lambda_symmetric_field",
            "lambda_global_residual_penalty",
            "global_residual_target_threshold",
            "max_velocity_ramp_epochs",
            "image_similarity_weight",
            "pet_independent_scale_log_std",
            "pet_independent_gamma_range",
            "pet_independent_bias_range",
            "pose_max_translation",
            "pose_max_rotation_degrees",
            "pose_max_scaling",
            "num_workers",
    ):
        if values[name] < 0:
            raise ValueError(f"{name} cannot be negative.")
    for name in (
            "symmetric_probability",
            "pet_independent_intensity_probability",
            "synthmorph_pet_lesion_dropout_probability",
            "synthmorph_pet_false_hot_probability",
            "synthmorph_fov_dropout_probability",
            "synthmorph_resolution_augmentation_probability",
            "pose_aware_probability",
            "pose_min_region_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if values["global_residual_target_scale"] <= 0.0:
        raise ValueError("global_residual_target_scale must be positive.")
    if values["lr"] <= 0.0:
        raise ValueError("lr must be positive.")
    if not 0.0 <= values["synthmorph_fov_max_fraction"] < 1.0:
        raise ValueError("synthmorph_fov_max_fraction must be in [0, 1).")
    if not 0.0 < values["synthmorph_min_resolution_scale"] <= 1.0:
        raise ValueError("synthmorph_min_resolution_scale must be in (0, 1].")
    if (values["curriculum_easy_probability"] < 0.0 or
            values["curriculum_extra_hard_probability"] < 0.0 or
            values["curriculum_easy_probability"] +
            values["curriculum_extra_hard_probability"] > 1.0):
        raise ValueError("Invalid mature curriculum probabilities.")


def load_synthetic_pretraining_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> SyntheticPretrainingConfig:
    """Load a stage recipe, apply operational overrides, and validate it.

    Args:
        path: Python file containing uppercase configuration constants.
        overrides: Optional flattened values supplied by the command line.

    Returns:
        Validated, immutable synthetic pretraining configuration.

    Raises:
        ValueError: If a name, value, or parameter combination is invalid.
    """
    module = _load_module(path)
    values = flatten_config(SyntheticPretrainingConfig())
    for name, value in vars(module).items():
        if name.isupper() and not name.startswith("_"):
            key = name.lower()
            if key not in values:
                raise ValueError(
                    f"Unknown synthetic configuration parameter: {name}")
            values[key] = value
    apply_flat_overrides(values,
                         overrides,
                         unknown_label="synthetic configuration parameter",
                         path_fields=PATH_FIELDS,
                         nullable_fields=NULLABLE_FIELDS,
                         optional_numeric_fields=OPTIONAL_NUMERIC_FIELDS)
    validate_synthetic_config(values)
    return build_config(SyntheticPretrainingConfig, values)
