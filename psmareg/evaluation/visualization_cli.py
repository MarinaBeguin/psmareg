"""Command-line options shared by the visualization scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from psmareg.evaluation.visualization_inputs import (
    ModelOverrides,
    TranslationPrealignOverrides,
    VisualizationSettings,
)
from psmareg.models.common.registration.prealignment.bone_affine import (
    BoneAffineSettings,)

# Value options of the bone affine pre-alignment and their defaults.
BONE_AFFINE_OPTIONS: tuple[tuple[str, type, object], ...] = (
    ("bone_affine_downsamples", str, "8,4"),
    ("bone_affine_iterations", int, 80),
    ("bone_affine_lr", float, 0.05),
    ("bone_affine_threshold", float, 0.66),
    ("bone_affine_softness", float, 0.04),
    ("bone_affine_max_translation", float, 120.0),
    ("bone_affine_max_rotation_degrees", float, 8.0),
    ("bone_affine_max_scaling", float, 0.04),
    ("bone_affine_min_improvement", float, 0.005),
)
# Value options of the translation pre-alignment forwarded between scripts.
TRANSLATION_OPTIONS = (
    "deterministic_translation_downsample",
    "deterministic_translation_search_radius",
    "deterministic_translation_search_step",
    "deterministic_translation_max",
)


def add_bone_affine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the bone affine pre-alignment switch and its settings."""
    parser.add_argument(
        "--bone_affine_prealign",
        action="store_true",
        help=("Estimate a conservative CT bone-aware affine pre-alignment "
              "before inference."),
    )
    for name, value_type, default in BONE_AFFINE_OPTIONS:
        parser.add_argument(f"--{name}", type=value_type, default=default)


def add_visualization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every option of the single-pair visualization script."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--slice_index", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_velocity", type=float, default=None)
    parser.add_argument("--max_residual_disp", type=float, default=None)
    parser.add_argument("--scaling_steps", type=int, default=None)
    parser.add_argument(
        "--deterministic_translation_prealign",
        action="store_true",
        help=
        "Force deterministic CT translation pre-alignment before inference.",
    )
    for name, value_type in zip(TRANSLATION_OPTIONS, (int, int, int, float)):
        parser.add_argument(f"--{name}", type=value_type, default=None)
    parser.add_argument("--deterministic_translation_feature_mode",
                        choices=["ct", "mind"],
                        default=None)
    parser.add_argument(
        "--deterministic_translation_padding_mode",
        choices=["auto", "border", "zeros"],
        default=None,
        help="Padding used only during deterministic translation search.",
    )
    parser.add_argument(
        "--deterministic_translation_input_padding_mode",
        choices=["auto", "border", "zeros"],
        default=None,
        help=
        "Padding used when pre-warping moving images before feeding the model.",
    )
    parser.add_argument(
        "--zero_padding",
        action="store_true",
        help=("Force zero padding for warping, overriding the checkpoint "
              "padding mode."),
    )
    parser.add_argument(
        "--registration_only",
        action="store_true",
        help="Only save the main CT/PET registration panel, not vector-field "
        "panels.",
    )
    add_bone_affine_arguments(parser)


def bone_affine_settings(args: argparse.Namespace) -> BoneAffineSettings | None:
    """Build the bone affine settings, or ``None`` when it is disabled."""
    if not args.bone_affine_prealign:
        return None
    return BoneAffineSettings(
        downsample_factors=tuple(
            int(value.strip())
            for value in args.bone_affine_downsamples.split(",")
            if value.strip()),
        iterations_per_level=args.bone_affine_iterations,
        lr=args.bone_affine_lr,
        bone_threshold=args.bone_affine_threshold,
        bone_softness=args.bone_affine_softness,
        max_translation=args.bone_affine_max_translation,
        max_rotation_degrees=args.bone_affine_max_rotation_degrees,
        max_scaling=args.bone_affine_max_scaling,
        min_improvement=args.bone_affine_min_improvement,
    )


def translation_overrides(
        args: argparse.Namespace) -> TranslationPrealignOverrides:
    """Build optional translation settings shared by inference front ends."""
    return TranslationPrealignOverrides(
        enabled=args.deterministic_translation_prealign,
        downsample=args.deterministic_translation_downsample,
        search_radius=args.deterministic_translation_search_radius,
        search_step=args.deterministic_translation_search_step,
        max_translation=args.deterministic_translation_max,
        feature_mode=args.deterministic_translation_feature_mode,
        padding_mode=args.deterministic_translation_padding_mode,
        input_padding_mode=args.deterministic_translation_input_padding_mode,
    )


def visualization_settings(args: argparse.Namespace) -> VisualizationSettings:
    """Convert parsed single-pair options into typed settings."""
    return VisualizationSettings(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        slice_index=args.slice_index,
        device=args.device,
        zero_padding=args.zero_padding,
        registration_only=args.registration_only,
        model=ModelOverrides(args.max_velocity, args.max_residual_disp,
                             args.scaling_steps),
        translation=translation_overrides(args),
        bone_affine=bone_affine_settings(args),
    )
