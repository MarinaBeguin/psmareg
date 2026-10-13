"""Structured training configuration, loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any

from psmareg.config.sections import (
    apply_flat_overrides,
    build_config,
    check_fields,
    flatten_config,
    load_python_recipe,
)


@dataclass(frozen=True)
class DataConfig:
    """Data sources, output locations, caching, and sampling options."""

    train_manifest: Path = Path(
        "manifests/paired/train_pairs.csv")
    val_manifest: Path = Path(
        "manifests/paired/val_pairs.csv")
    output_dir: Path = Path("results/fine_tuning")
    paired_cache_dir: Path | None = None
    paired_cache_float16: bool = False
    patient_balanced_sampling: bool = False


@dataclass(frozen=True)
class InitializationConfig:
    """Checkpoint initialization and training-resume options."""

    pretrained_checkpoint: Path | None = None
    compatible_pretrained_checkpoint: Path | None = None
    compatible_pretrained_prefixes: str = ""
    resume_from: Path | None = None


@dataclass(frozen=True)
class ArchitectureConfig:
    """Registration-network architecture and displacement bounds."""

    num_ct_classes: int = 118
    max_velocity: float = 8.0
    max_residual_disp: float = 4.0
    scaling_steps: int = 7


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer, scheduler, and early-stopping options."""

    epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-4
    gradient_clip_norm: float = 0.0
    patience: int = 20
    lr_scheduler_patience: int = 0
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-7


@dataclass(frozen=True)
class AugmentationConfig:
    """Online spatial and intensity augmentation options."""

    train_augmentation: bool = False
    train_spatial_aug_probability: float = 0.0
    train_aug_max_translation: float = 3.0
    train_aug_max_rotation_degrees: float = 2.0
    train_aug_max_scaling: float = 0.02
    train_intensity_aug_probability: float = 0.0


@dataclass(frozen=True)
class SimilarityLossConfig:
    """CT and PET image-similarity loss options."""

    lambda_pet: float = 0.2
    lambda_mind_ct: float = 0.0
    lambda_mind_pet: float = 0.0
    mind_loss_downsample: int = 2
    mind_radius: int = 1
    mind_dilation: int = 2
    lambda_multiscale_corr_ct: float = 0.0
    lambda_multiscale_corr_pet: float = 0.0
    multiscale_corr_scales: tuple[int, ...] = (2, 4, 8)


@dataclass(frozen=True)
class AnatomicalLossConfig:
    """Detailed and grouped CT-label supervision options."""

    lambda_ct_dice: float = 0.5
    ct_dice_loss_downsample: int = 1
    lambda_macro_ct_labels: float = 0.0
    macro_ct_label_downsample: int = 2
    macro_ct_label_preset: str = "stable_no_bladder"
    macro_ct_label_ramp_epochs: int = 20


@dataclass(frozen=True)
class PetLossConfig:  # pylint: disable=too-many-instance-attributes
    """PET hot-region masking and adaptive weighting options."""

    lambda_pet_hot_region: float = 0.0
    pet_hot_quantile: float = 0.98
    pet_hot_upper_quantile: float = 1.0
    pet_hot_warped_weight: float = 0.25
    pet_hot_smooth_kernel: int = 5
    pet_hot_ramp_epochs: int = 0
    pet_loss_upper_quantile: float = 1.0
    pet_loss_quantile_downsample: int = 1
    pet_mask_similarity_near_lesions: bool = False
    pet_lesion_similarity_mask_dilate: int = 5
    pet_lesion_similarity_mask_downsample: int = 1
    adaptive_pet_loss: bool = False
    adaptive_pet_hot_quantile: float = 0.97
    adaptive_pet_smooth_kernel: int = 7
    adaptive_pet_downsample: int = 1
    adaptive_pet_min_weight: float = 0.15
    adaptive_pet_low_dice: float = 0.05
    adaptive_pet_high_dice: float = 0.35


@dataclass(frozen=True)
class LesionLossConfig:
    """Lesion occupancy, translation, strain, and edge protection options."""

    lambda_pet_lesion_occupancy: float = 0.0
    lambda_pet_lesion_translation: float = 0.0
    lambda_pet_lesion_strain: float = 0.0
    pet_lesion_occupancy_downsample: int = 4
    pet_lesion_occupancy_min_voxels: int = 3
    pet_lesion_occupancy_max_components: int = 12
    pet_lesion_occupancy_ramp_epochs: int = 10
    pet_lesion_adaptive_size_weighting: bool = False
    pet_lesion_small_voxels: float = 80.0
    pet_lesion_large_voxels: float = 1500.0
    pet_lesion_small_weight: float = 1.8
    pet_lesion_large_weight: float = 0.45
    pet_lesion_halo_voxels: int = 0
    pet_lesion_halo_weight: float = 0.25
    pet_lesion_edge_guard_margin_voxels: int = 0
    pet_lesion_edge_guard_weight: float = 1.0


@dataclass(frozen=True)
class RegularizationConfig:
    """Displacement smoothness and valid-overlap regularization options."""

    lambda_smooth: float = 0.01
    smoothness_downsample: int = 1
    lambda_overlap: float = 0.0
    minimum_valid_overlap: float = 0.9


@dataclass(frozen=True)
class PrealignmentConfig:
    """Deterministic coarse translation pre-alignment options."""

    deterministic_translation_prealign: bool = False
    deterministic_translation_downsample: int = 4
    deterministic_translation_search_radius: int = 8
    deterministic_translation_search_step: int = 2
    deterministic_translation_max: float = 80.0
    deterministic_translation_feature_mode: str = "ct"
    deterministic_translation_padding_mode: str = "auto"
    deterministic_translation_input_padding_mode: str = "auto"


@dataclass(frozen=True)
class RuntimeConfig:
    """Data-loading, logging, and reproducibility options."""

    num_workers: int = 4
    no_persistent_workers: bool = False
    log_challenge_metrics: bool = False
    seed: int = 42


@dataclass(frozen=True)
class FineTuningConfig:
    """Immutable collection of all grouped training options."""

    data: DataConfig = field(default_factory=DataConfig)
    initialization: InitializationConfig = field(
        default_factory=InitializationConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    similarity_loss: SimilarityLossConfig = field(
        default_factory=SimilarityLossConfig)
    anatomical_loss: AnatomicalLossConfig = field(
        default_factory=AnatomicalLossConfig)
    pet_loss: PetLossConfig = field(default_factory=PetLossConfig)
    lesion_loss: LesionLossConfig = field(default_factory=LesionLossConfig)
    regularization: RegularizationConfig = field(
        default_factory=RegularizationConfig)
    prealignment: PrealignmentConfig = field(default_factory=PrealignmentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


PATH_ARGUMENTS = {
    "train_manifest", "val_manifest", "output_dir", "paired_cache_dir",
    "pretrained_checkpoint", "compatible_pretrained_checkpoint", "resume_from"
}
OPTIONAL_PATH_ARGUMENTS = PATH_ARGUMENTS - {
    "train_manifest", "val_manifest", "output_dir"
}


def _load_module(path: Path) -> ModuleType:
    """Load a Python configuration file as an isolated module.

    Args:
        path: Path to the Python configuration file.

    Returns:
        Loaded configuration module.

    Raises:
        ValueError: If Python cannot create an import specification.
    """
    return load_python_recipe(path,
                              module_name="_psmareg_user_config",
                              error_label="configuration file")


def _config_from_module(module: ModuleType) -> FineTuningConfig:
    """Build an immutable configuration from flat uppercase constants.

    Args:
        module: Loaded user-configuration module.

    Returns:
        Structured training configuration with defaults applied.

    Raises:
        ValueError: If the module contains an unknown uppercase constant.
    """
    defaults = FineTuningConfig()
    sections: dict[str, Any] = {}
    known_constants: set[str] = set()
    for section_field in fields(defaults):
        default_section = getattr(defaults, section_field.name)
        section_values = {}
        for item in fields(default_section):
            constant_name = item.name.upper()
            known_constants.add(constant_name)
            if hasattr(module, constant_name):
                section_values[item.name] = getattr(module, constant_name)
        sections[section_field.name] = type(default_section)(**section_values)
    helper_constants = {"MANIFEST_ROOT", "PRETRAINING_ROOT"}
    unknown = sorted(
        name for name in vars(module)
        if name.isupper() and name not in known_constants | helper_constants)
    if unknown:
        raise ValueError(
            f"Unknown configuration constants: {', '.join(unknown)}")
    return FineTuningConfig(**sections)


def validate_fine_tuning_config(values: dict[str, Any]) -> None:
    """Validate parameter ranges and cross-parameter constraints.

    Args:
        values: Flattened training configuration.

    Raises:
        ValueError: If a parameter or combination is invalid.
    """
    check_fields(values, ("train_manifest", "val_manifest", "output_dir"),
                 lambda value: value is not None, "is required")
    check_fields(values, [key for key in values if key.startswith("lambda_")],
                 lambda value: value >= 0, "cannot be negative")
    check_fields(
        values,
        ("epochs", "batch_size", "num_ct_classes", "scaling_steps",
         "mind_loss_downsample", "mind_dilation", "ct_dice_loss_downsample",
         "smoothness_downsample", "macro_ct_label_downsample",
         "pet_loss_quantile_downsample",
         "pet_lesion_similarity_mask_downsample", "adaptive_pet_downsample",
         "pet_lesion_occupancy_downsample", "pet_lesion_occupancy_min_voxels",
         "pet_lesion_occupancy_max_components",
         "deterministic_translation_downsample",
         "deterministic_translation_search_step"), lambda value: value >= 1,
        "must be >= 1")
    check_fields(
        values,
        ("num_workers", "patience", "lr_scheduler_patience", "mind_radius",
         "pet_hot_ramp_epochs", "macro_ct_label_ramp_epochs",
         "pet_lesion_occupancy_ramp_epochs", "pet_lesion_halo_voxels",
         "pet_lesion_edge_guard_margin_voxels",
         "pet_lesion_similarity_mask_dilate",
         "deterministic_translation_search_radius"), lambda value: value >= 0,
        "cannot be negative")
    check_fields(values,
                 ("pet_hot_smooth_kernel", "adaptive_pet_smooth_kernel"),
                 lambda value: value >= 1 and value % 2 == 1,
                 "must be a positive odd integer")
    check_fields(values,
                 ("pet_hot_quantile", "pet_hot_upper_quantile",
                  "pet_loss_upper_quantile", "adaptive_pet_hot_quantile"),
                 lambda value: 0 < value <= 1, "must be in (0, 1]")
    check_fields(
        values,
        ("train_spatial_aug_probability", "train_intensity_aug_probability",
         "adaptive_pet_min_weight", "adaptive_pet_low_dice",
         "adaptive_pet_high_dice", "minimum_valid_overlap"),
        lambda value: 0 <= value <= 1, "must be in [0, 1]")
    if values["adaptive_pet_high_dice"] <= values["adaptive_pet_low_dice"]:
        raise ValueError(
            "adaptive_pet_high_dice must exceed adaptive_pet_low_dice.")
    if values["pet_lesion_large_voxels"] <= values["pet_lesion_small_voxels"]:
        raise ValueError(
            "pet_lesion_large_voxels must exceed pet_lesion_small_voxels.")
    check_fields(values,
                 ("pet_lesion_small_voxels", "pet_lesion_small_weight",
                  "pet_lesion_large_weight", "pet_lesion_edge_guard_weight",
                  "lr", "min_lr", "max_velocity", "max_residual_disp"),
                 lambda value: value > 0, "must be positive")
    check_fields(values,
                 ("gradient_clip_norm", "train_aug_max_translation",
                  "train_aug_max_rotation_degrees", "train_aug_max_scaling",
                  "pet_hot_warped_weight", "pet_lesion_halo_weight",
                  "deterministic_translation_max"), lambda value: value >= 0,
                 "cannot be negative")
    if values["pet_hot_upper_quantile"] < values["pet_hot_quantile"]:
        raise ValueError("pet_hot_upper_quantile must be >= pet_hot_quantile.")
    if not 0 < values["lr_scheduler_factor"] < 1:
        raise ValueError("lr_scheduler_factor must be in (0, 1).")
    if values["macro_ct_label_preset"] not in {
            "stable", "stable_no_bladder", "skeleton_only", "organs_only"
    }:
        raise ValueError("Invalid macro_ct_label_preset.")
    if values["deterministic_translation_feature_mode"] not in {"ct", "mind"}:
        raise ValueError("Invalid deterministic_translation_feature_mode.")
    for name in ("deterministic_translation_padding_mode",
                 "deterministic_translation_input_padding_mode"):
        if values[name] not in {"auto", "border", "zeros"}:
            raise ValueError(f"Invalid {name}.")
    if not values["multiscale_corr_scales"] or any(
            scale < 1 for scale in values["multiscale_corr_scales"]):
        raise ValueError(
            "multiscale_corr_scales must contain positive integers.")
    if values["resume_from"] is not None and (
            values["pretrained_checkpoint"] is not None or
            values["compatible_pretrained_checkpoint"] is not None):
        raise ValueError(
            "resume_from cannot be combined with pretrained checkpoints.")


def load_fine_tuning_config(
        path: Path,
        overrides: dict[str, Any] | None = None) -> FineTuningConfig:
    """Load, override, convert, and validate a training configuration.

    Args:
        path: Python file containing uppercase configuration constants.
        overrides: Optional flattened values supplied by the command line.

    Returns:
        Validated, immutable training configuration.

    Raises:
        TypeError: If an override cannot be converted to the expected type.
        ValueError: If a name, value, or parameter combination is invalid.
    """
    module = _load_module(path)
    config = _config_from_module(module)
    values = flatten_config(config)
    apply_flat_overrides(values,
                         overrides,
                         unknown_label="configuration parameter",
                         path_fields=PATH_ARGUMENTS,
                         nullable_fields=OPTIONAL_PATH_ARGUMENTS)
    validate_fine_tuning_config(values)
    return build_config(FineTuningConfig, values)
