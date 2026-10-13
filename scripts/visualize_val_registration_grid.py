"""Generate a small visual registration panel for each subject in a manifest.

This is a lightweight wrapper around ``scripts/visualize_registration.py``.
It picks a few axial slices per subject and stores the outputs in one folder
per subject, which makes challenge-val failure analysis much less painful.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from psmareg.data.manifests import read_manifest
from psmareg.data.nifti_io import load_nifti_image
from psmareg.evaluation.visualization_cli import (
    BONE_AFFINE_OPTIONS,
    TRANSLATION_OPTIONS,
    add_bone_affine_arguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Parse the grid visualization options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--num_slices",
        type=int,
        default=3,
        help="Number of slices per subject. Recommended: 2 or 3.",
    )
    parser.add_argument(
        "--slice_fractions",
        type=float,
        nargs="*",
        default=None,
        help=("Optional relative slice positions in [0, 1]. "
              "Default uses 0.35, 0.50, 0.65 for 3 slices."),
    )
    parser.add_argument(
        "--slice_source",
        choices=["fractions", "hot_pet"],
        default="fractions",
        help=(
            "How to select slices. 'fractions' uses fixed relative positions. "
            "'hot_pet' selects axial slices with the strongest fixed/moving "
            "PET "
            "hot-region signal, useful for MTV/TLG failure analysis."),
    )
    parser.add_argument(
        "--hot_pet_quantile",
        type=float,
        default=0.985,
        help="Per-volume PET quantile used by --slice_source hot_pet.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--retry_cpu_on_sigkill",
        action="store_true",
        help=("If a CUDA visualization subprocess is killed by SIGKILL, retry "
              "the "
              "same slice on CPU instead of aborting the whole grid."),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject filters, e.g. 0047 PSMARegPSMA_0035.",
    )
    parser.add_argument(
        "--start_sample_index",
        type=int,
        default=0,
        help=
        "Resume helper: skip manifest rows with index smaller than this value.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a slice when its main *_registration.png already exists.",
    )
    parser.add_argument(
        "--deterministic_translation_prealign",
        action="store_true",
        help="Forward deterministic pre-alignment args to "
        "visualize_registration.py.",
    )
    parser.add_argument("--deterministic_translation_downsample",
                        type=int,
                        default=4)
    parser.add_argument("--deterministic_translation_search_radius",
                        type=int,
                        default=12)
    parser.add_argument("--deterministic_translation_search_step",
                        type=int,
                        default=2)
    parser.add_argument("--deterministic_translation_max",
                        type=float,
                        default=120.0)
    add_bone_affine_arguments(parser)
    parser.add_argument(
        "--zero_padding",
        action="store_true",
        help="Force zero padding in visualize_registration.py.",
    )
    parser.add_argument(
        "--registration_only",
        action="store_true",
        help=
        "Only generate *_registration.png panels, not vector-field figures.",
    )
    return parser.parse_args()


def _subject_matches(subject: str, filters: list[str] | None) -> bool:
    """Return whether a subject passes the optional name filters."""
    if not filters:
        return True
    return any(item in subject for item in filters)


def _slice_indices(depth: int, count: int,
                   fractions: list[float] | None) -> list[int]:
    """Choose distinct axial slices at the requested depth fractions."""
    if depth <= 0:
        raise ValueError("Volume depth must be positive.")
    if count <= 0:
        raise ValueError("--num_slices must be positive.")

    if fractions:
        selected = fractions[:]
    elif count == 1:
        selected = [0.50]
    elif count == 2:
        selected = [0.40, 0.60]
    elif count == 3:
        selected = [0.35, 0.50, 0.65]
    else:
        step = 0.70 / max(count - 1, 1)
        selected = [0.15 + index * step for index in range(count)]

    indices: list[int] = []
    for fraction in selected:
        clipped = min(max(float(fraction), 0.0), 1.0)
        indices.append(int(round(clipped * (depth - 1))))

    # Preserve order while avoiding duplicate slices for small volumes.
    deduplicated: list[int] = []
    for index in indices:
        if index not in deduplicated:
            deduplicated.append(index)
    return deduplicated


def _hot_pet_slice_indices(
    fixed_pet_path: str,
    moving_pet_path: str,
    count: int,
    quantile: float,
) -> list[int]:
    """Choose the axial slices with the most PET uptake."""
    if count <= 0:
        raise ValueError("--num_slices must be positive.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("--hot_pet_quantile must be in (0, 1).")

    fixed = load_nifti_image(fixed_pet_path).get_fdata(dtype="float32")
    moving = load_nifti_image(moving_pet_path).get_fdata(dtype="float32")
    if fixed.shape != moving.shape:
        raise ValueError(
            f"Fixed/moving PET shape mismatch: {fixed.shape} vs {moving.shape}")
    if fixed.ndim != 3:
        raise ValueError(f"Expected 3D PET volumes, got shape={fixed.shape}.")

    scores = []
    for image in (fixed, moving):
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            continue
        threshold = float(np.quantile(finite, quantile))
        hot = np.clip(image - threshold, a_min=0.0, a_max=None)
        scores.append(hot.sum(axis=(1, 2)))
    if not scores:
        depth = int(fixed.shape[0])
        return _slice_indices(depth, count, None)

    score = np.sum(np.stack(scores, axis=0), axis=0)
    if not np.any(score > 0):
        depth = int(fixed.shape[0])
        return _slice_indices(depth, count, None)

    candidate_order = np.argsort(score)[::-1]
    selected: list[int] = []
    min_gap = max(1, int(round(fixed.shape[0] / max(count * 8, 1))))
    for raw_index in candidate_order:
        index = int(raw_index)
        if all(abs(index - previous) >= min_gap for previous in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for raw_index in candidate_order:
            index = int(raw_index)
            if index not in selected:
                selected.append(index)
            if len(selected) >= count:
                break
    return sorted(selected)


def _volume_depth(path: str) -> int:
    """Return the number of axial slices of a NIfTI volume."""
    image = load_nifti_image(path)
    shape = image.shape
    if len(shape) < 3:
        raise ValueError(f"Expected a 3D image for {path}, got shape={shape}.")
    # visualize_registration.py uses fixed_ct.shape[-3] as the slice axis.
    return int(shape[0])


def _select_slices(args: argparse.Namespace, row: dict[str, str],
                   depth: int) -> list[int]:
    """Return the axial slices to render for one manifest row."""
    if args.slice_source == "hot_pet":
        return _hot_pet_slice_indices(
            row["fixed_pet"],
            row["moving_pet"],
            count=args.num_slices,
            quantile=args.hot_pet_quantile,
        )
    return _slice_indices(depth, args.num_slices, args.slice_fractions)


def _forwarded_options(args: argparse.Namespace) -> list[str]:
    """Build the pre-alignment and display options passed to each slice."""
    forwarded: list[str] = []
    if args.deterministic_translation_prealign:
        forwarded.append("--deterministic_translation_prealign")
        for name in TRANSLATION_OPTIONS:
            forwarded.extend([f"--{name}", str(getattr(args, name))])
    if args.bone_affine_prealign:
        forwarded.append("--bone_affine_prealign")
        for name, _, _ in BONE_AFFINE_OPTIONS:
            forwarded.extend([f"--{name}", str(getattr(args, name))])
    if args.zero_padding:
        forwarded.append("--zero_padding")
    if args.registration_only:
        forwarded.append("--registration_only")
    return forwarded


def _run_slice(args: argparse.Namespace, command: list[str],
               slice_index: int) -> None:
    """Render one slice, optionally retrying on the CPU after a SIGKILL."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if not (args.retry_cpu_on_sigkill and args.device == "cuda" and
                error.returncode == -9):
            raise
        print(
            "  CUDA subprocess was killed for slice "
            f"{slice_index}; "
            "retrying this slice on CPU.",
            flush=True,
        )
        cpu_command = command[:]
        cpu_command[cpu_command.index("--device") + 1] = "cpu"
        subprocess.run(cpu_command, check=True)


def _render_subject(args: argparse.Namespace, sample_index: int,
                    row: dict[str, str], subject: str) -> None:
    """Render the selected slices of one manifest row into its folder."""
    depth = _volume_depth(row["fixed_ct"])
    slices = _select_slices(args, row, depth)
    subject_dir = args.output_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{sample_index:03d}] {subject}: depth={depth}, slices={slices}",
        flush=True,
    )
    followup_index = row.get("followup_index", "01")
    for slice_index in slices:
        expected_registration = (
            subject_dir /
            f"{subject}_followup_{followup_index}_slice_{slice_index:03d}"
            "_registration.png")
        if args.skip_existing and expected_registration.exists():
            print(
                f"  skip existing slice {slice_index}: "
                f"{expected_registration.name}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_registration.py"),
            "--manifest",
            str(args.manifest),
            "--checkpoint",
            str(args.checkpoint),
            "--output_dir",
            str(subject_dir),
            "--sample_index",
            str(sample_index),
            "--slice_index",
            str(slice_index),
            "--device",
            args.device,
        ] + _forwarded_options(args)
        _run_slice(args, command, slice_index)


def main() -> None:
    """Render a grid of selected validation subjects for visual review."""
    args = _parse_args()
    rows = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    for sample_index, row in enumerate(rows):
        if sample_index < args.start_sample_index:
            continue
        subject = row.get("subject", f"sample_{sample_index:03d}")
        if not _subject_matches(subject, args.subjects):
            continue
        _render_subject(args, sample_index, row, subject)
        selected_count += 1

    print(f"Generated visual checks for {selected_count} subject(s): "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
