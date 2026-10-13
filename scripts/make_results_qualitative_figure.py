#!/usr/bin/env python3
"""Generate the article-style qualitative PET/CT registration figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from psmareg.data.manifests import PAIR_IMAGE_COLUMNS, read_manifest
from psmareg.evaluation.qualitative_cases import (
    CaseSpec,
    MetricSpec,
    case_token,
    choose_slice,
    load_metrics,
    load_prediction_cache,
    parse_case,
    predict_case,
    report_top_lesion_slices,
    save_prediction_cache,
    select_rows,
)
from psmareg.evaluation.qualitative_figure import (
    MOVING_LESION_COLOR,
    SliceView,
    add_orientation_markers,
    blank_axis,
    body_difference_mask,
    draw_difference,
    draw_scan,
    manually_compact_case_blocks,
    manually_set_ct_pet_gap,
    metric_caption,
    plt,
    pull_image_axes_together,
    slice_crop,
)
from psmareg.evaluation.visualization_cli import (
    add_bone_affine_arguments,
    bone_affine_settings,
    translation_overrides,
)
from psmareg.evaluation.visualization_inputs import (
    ModelOverrides,
    VisualizationSettings,
    load_checkpoint_config,
    load_model,
)

COLUMN_TITLES = ("Fixed", "Moving", "|Fixed - moving|", "|Fixed - warped|")
CaseData = tuple[CaseSpec, dict[str, str], dict[str, np.ndarray]]


def _parse_metric(text: str) -> MetricSpec:
    """Parse metric overrides: 0081_fu02=0.8405,3.8055,0.0000,0.0000."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "--metric must be formatted as case=DSC,HD95,MTV,TLG, "
            "e.g. 0081_fu02=0.84,3.8,0,0")
    token, values = text.split("=", 1)
    case = parse_case(token)
    if case.followup is None:
        raise argparse.ArgumentTypeError("--metric case must include _fuXX.")
    parts = [
        float(value.strip()) for value in values.split(",") if value.strip()
    ]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--metric must contain four comma-separated values: "
            "DSC,HD95,MTV,TLG.")
    return MetricSpec(
        subject=case.subject,
        followup=case.followup,
        dsc=parts[0],
        hd95=parts[1],
        mtv=parts[2],
        tlg=parts[3],
    )


def _parse_args() -> argparse.Namespace:
    """Parse figure, case-selection, and checkpoint inference options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Optional dataset root containing imagesTr/labelsTr, "
        "to relocate manifest paths.")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional prediction cache for regenerating figures "
        "without repeating inference.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/paired/val_pairs.csv"),
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=None,
        help=
        ("Case to include, e.g. good=0336_fu01. Repeat three times. "
         "Default: good=0336_fu01, intermediate=0200_fu02, difficult=0359_fu01."
        ),
    )
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--metrics_csv", type=Path, default=None)
    parser.add_argument(
        "--metric",
        action="append",
        type=_parse_metric,
        default=None,
        help="Manual metric override, e.g. 0081_fu02=0.8405,3.8055,0.0,0.0.",
    )
    parser.add_argument(
        "--show_metrics",
        action="store_true",
        help="Write per-case metrics into row labels when --metrics_csv is "
        "provided.",
    )
    parser.add_argument(
        "--image_warp_mode",
        default="bilinear",
        choices=("bilinear", "nearest"),
        help="Image interpolation for display. Labels are always nearest.",
    )
    parser.add_argument("--slice", action="append", default=None)
    parser.add_argument("--pet_quantile", type=float, default=0.995)
    parser.add_argument("--pet_alpha", type=float, default=0.55)
    parser.add_argument(
        "--top_slice_report",
        type=int,
        default=10,
        help="Print the top-N lesion-rich slices for each selected case.",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Do not crop panels around the body on the selected slice.",
    )
    parser.add_argument(
        "--crop_to_body",
        action="store_true",
        help="Crop each slice around the body. Default is the full scan, "
        "as in visualize_registration.py.",
    )
    parser.add_argument("--crop_margin", type=int, default=12)
    parser.add_argument(
        "--layout",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Figure aspect. Portrait is more compact for LNCS columns/pages.",
    )
    parser.add_argument("--max_velocity", type=float, default=None)
    parser.add_argument("--max_residual_disp", type=float, default=None)
    parser.add_argument("--scaling_steps", type=int, default=None)

    # Optional pre-alignment overrides; omitted values come from the checkpoint.
    parser.add_argument("--deterministic_translation_prealign",
                        action="store_true")
    parser.add_argument("--deterministic_translation_downsample",
                        type=int,
                        default=None)
    parser.add_argument("--deterministic_translation_search_radius",
                        type=int,
                        default=None)
    parser.add_argument("--deterministic_translation_search_step",
                        type=int,
                        default=None)
    parser.add_argument("--deterministic_translation_max",
                        type=float,
                        default=None)
    parser.add_argument("--deterministic_translation_feature_mode",
                        default=None)
    parser.add_argument("--deterministic_translation_padding_mode",
                        default=None)
    parser.add_argument("--deterministic_translation_input_padding_mode",
                        default=None)
    add_bone_affine_arguments(parser)
    return parser.parse_args()


def _relocate_rows(rows: list[dict[str, str]], data_root: Path) -> None:
    """Point every image path of the manifest to another dataset root."""
    for row in rows:
        for key, value in list(row.items()):
            if key.startswith(("fixed_", "moving_")) and value:
                original = Path(value)
                row[key] = str(data_root / original.parent.name / original.name)


def _cache_path(args: argparse.Namespace, row: Mapping[str, str],
                settings: VisualizationSettings) -> Path | None:
    """Return the prediction cache entry of one case, if caching is on."""
    if args.cache_dir is None:
        return None
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    stat = args.checkpoint.stat()
    signature = json.dumps(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "mtime": stat.st_mtime_ns,
            "size": stat.st_size,
            "row": dict(row),
            "prealign": str(settings.translation),
            "inference_settings": {
                key: value
                for key, value in vars(args).items()
                if key.startswith("bone_affine") or key in (
                    "device", "max_velocity", "max_residual_disp",
                    "scaling_steps", "image_warp_mode")
            },
            "display_version": 1
        },
        sort_keys=True,
        default=str)
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return args.cache_dir / f"{case_token(row)}_{digest}.npz"


def _predict_cases(args: argparse.Namespace,
                   selected: list[tuple[CaseSpec, dict[str, str]]],
                   device: torch.device) -> list[CaseData]:
    """Register every selected case, reusing cached predictions."""
    model_overrides = ModelOverrides(args.max_velocity, args.max_residual_disp,
                                     args.scaling_steps)
    model = load_model(args.checkpoint, device, model_overrides)
    checkpoint_config = load_checkpoint_config(args.checkpoint,
                                               torch.device("cpu"))
    settings = VisualizationSettings(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_prefix.parent,
        device=args.device,
        model=model_overrides,
        translation=translation_overrides(args),
        bone_affine=bone_affine_settings(args),
    )
    case_data: list[CaseData] = []
    for case, row in selected:
        print(f"Predicting {case.label}: {case_token(row)}", flush=True)
        cache_path = _cache_path(args, row, settings)
        data = (load_prediction_cache(cache_path)
                if cache_path is not None else None)
        if data is None:
            data = predict_case(args.image_warp_mode, model, settings,
                                checkpoint_config, row, device)
            if cache_path is not None:
                save_prediction_cache(cache_path, data)
                print(f"Cached prediction: {cache_path}", flush=True)
        case_data.append((case, row, data))
    return case_data


def _difference_vmax(case_data: list[CaseData]) -> float:
    """Return one robust color-scale maximum shared by all difference maps."""

    def percentile(modality: str, level: float) -> float:
        """Pool the absolute differences of one modality over all cases."""
        values = np.concatenate([
            np.abs(
                np.asarray(data[f"fixed_{modality}"]) -
                np.asarray(data[name])).ravel()
            for _, _, data in case_data
            for name in (f"moving_{modality}", f"warped_{modality}")
        ])
        return float(np.percentile(values[np.isfinite(values)], level))

    return max(percentile("ct", 98.5), percentile("pet", 99.2), 1e-6)


def _build_axes(layout: str, case_count: int) -> tuple[Any, np.ndarray]:
    """Create the figure and its grid of three rows by five columns per case."""
    figsize = ((7.85, 4.1 * case_count) if layout == "portrait" else
               (12.8, 6.1 * case_count))
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        nrows=case_count,
        ncols=1,
        left=0.018,
        right=0.965,
        top=0.975,
        bottom=0.010,
        hspace=0.020,
    )
    axes_rows: list[list[Any]] = []
    for patient_index in range(case_count):
        inner = outer[patient_index].subgridspec(
            nrows=3,
            ncols=5,
            height_ratios=[1.0, 1.0, 0.085],
            width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055],
            wspace=0.026,
            hspace=0.000,
        )
        for local_row in range(3):
            axes_rows.append([
                fig.add_subplot(inner[local_row, column]) for column in range(5)
            ])
    return fig, np.asarray(axes_rows, dtype=object)


def _select_view(args: argparse.Namespace, row_index: int, token: str,
                 data: Mapping[str, np.ndarray]) -> SliceView:
    """Choose the displayed slice and crop of one case."""
    z = choose_slice(
        np.asarray(data["moving_pet_label"]),
        np.asarray(data["warped_pet_label"]),
        np.asarray(data["fixed_pet"]),
        np.asarray(data["moving_pet"]),
        row_index,
        args.slice,
    )
    report_top_lesion_slices(
        np.asarray(data["fixed_pet_label"]),
        np.asarray(data["moving_pet_label"]),
        np.asarray(data["warped_pet_label"]),
        token,
        args.top_slice_report,
    )
    crop = (slice_crop(
        np.asarray(data["fixed_ct"]),
        np.asarray(data["moving_ct"]),
        np.asarray(data["warped_ct"]),
        z,
        args.crop_margin,
    ) if args.crop_to_body and not args.no_crop else (slice(None), slice(None)))
    return SliceView(z, crop)


def _slice_record(case: CaseSpec, token: str, z: int,
                  data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Describe the displayed slice of one case for the figure metadata."""
    ras_affine = np.asarray(data["ras_affine"])
    return {
        "case":
            token,
        "label":
            case.label,
        "axial_slice":
            z,
        "superior_coordinate_mm":
            float(ras_affine[2, 2] * z + ras_affine[2, 3]),
        "moving_lesion_voxels":
            int((np.asarray(data["moving_pet_label"])[z] > 0).sum()),
        "warped_lesion_voxels":
            int((np.asarray(data["warped_pet_label"])[z] > 0).sum()),
    }


def _draw_case(axes: np.ndarray, row_index: int, data: Mapping[str, np.ndarray],
               view: SliceView, diff_vmax: float) -> Any:
    """Draw the CT and PET rows of one case.

    Returns:
        Difference image of the last PET panel, used for the colorbar.
    """
    ct_row, pet_row = 3 * row_index, 3 * row_index + 1
    volumes = {name: np.asarray(array) for name, array in data.items()}
    masks = {
        name:
            body_difference_mask(volumes["fixed_ct"],
                                 volumes[f"{name}_ct"],
                                 view,
                                 mode="fixed") for name in ("moving", "warped")
    }
    for column, name in enumerate(("fixed_ct", "moving_ct")):
        draw_scan(axes[ct_row, column], volumes[name], None, view, (0.0, 1.0),
                  "gray")
    for column, name in ((2, "moving"), (3, "warped")):
        draw_difference(axes[ct_row, column],
                        volumes["fixed_ct"],
                        volumes[f"{name}_ct"],
                        view,
                        diff_vmax,
                        display_mask=masks[name])
    pull_image_axes_together(axes, ct_row, pet_row)
    draw_scan(axes[pet_row, 0],
              volumes["fixed_pet"],
              None,
              view, (0.0, 1.0),
              "magma",
              contour_color=MOVING_LESION_COLOR)
    draw_scan(axes[pet_row, 1],
              volumes["moving_pet"],
              volumes["moving_pet_label"],
              view, (0.0, 1.0),
              "magma",
              contour_color=MOVING_LESION_COLOR)
    mappable = None
    for column, name in ((2, "moving"), (3, "warped")):
        mappable = draw_difference(axes[pet_row, column],
                                   volumes["fixed_pet"],
                                   volumes[f"{name}_pet"],
                                   view,
                                   diff_vmax,
                                   lesion_contours=[
                                       (volumes[f"{name}_pet_label"],
                                        MOVING_LESION_COLOR)
                                   ],
                                   display_mask=masks[name])
    return mappable


def _finish_case(axes: np.ndarray, row_index: int,
                 data: Mapping[str, np.ndarray], caption: str) -> None:
    """Set the aspect, orientation markers, and caption of one case."""
    ct_row, pet_row, metrics_row = (3 * row_index, 3 * row_index + 1,
                                    3 * row_index + 2)
    spacing = np.asarray(data["display_spacing"])
    for image_row in (ct_row, pet_row):
        for column in range(4):
            axes[image_row, column].set_aspect(float(spacing[1] / spacing[2]))
            add_orientation_markers(axes[image_row, column])
    for column in range(5):
        blank_axis(axes[metrics_row, column])
    blank_axis(axes[ct_row, 4])
    blank_axis(axes[pet_row, 4])
    if caption:
        axes[metrics_row, 0].text(
            0.0,
            1.04,
            caption,
            transform=axes[metrics_row, 0].transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
            color="black",
            clip_on=False,
        )


def _add_titles_and_colorbars(fig: Any, axes: np.ndarray,
                              mappables: list[Any]) -> None:
    """Compact the layout, then add the column titles and the colorbars."""
    manually_set_ct_pet_gap(fig, axes, len(mappables), gap=0.006)
    manually_compact_case_blocks(fig, axes, len(mappables), gap=0.018)
    for column, title in enumerate(COLUMN_TITLES):
        pos = axes[0, column].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2.0,
            pos.y1 + 0.007,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
    for row_index, mappable in enumerate(mappables):
        ct_row = 3 * row_index
        top = axes[ct_row, 4].get_position().y1
        bottom = axes[ct_row + 1, 4].get_position().y0
        pad = 0.080 * (top - bottom)
        top -= pad
        bottom += pad
        pos = axes[ct_row, 4].get_position()
        cax = fig.add_axes(
            (pos.x0 + pos.width * 0.32, bottom, pos.width * 0.36, top - bottom))
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=6, length=2, pad=1)
        cbar.set_label("Abs. diff.", fontsize=6, labelpad=1)


def _save_figure(fig: Any, args: argparse.Namespace,
                 slice_records: list[dict[str, Any]]) -> None:
    """Write the figure in three formats and its slice metadata."""
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        args.output_prefix.with_suffix(suffix)
        for suffix in (".png", ".pdf", ".svg")
    ]
    fig.savefig(paths[0], dpi=220)
    fig.savefig(paths[1])
    fig.savefig(paths[2])
    metadata_path = args.output_prefix.with_suffix(".json")
    metadata_path.write_text(json.dumps(
        {
            "checkpoint": str(args.checkpoint),
            "orientation": "axial radiological: R left, A top",
            "cases": slice_records
        },
        indent=2),
                             encoding="utf-8")
    plt.close(fig)
    for path in paths + [metadata_path]:
        print(f"Saved {path}", flush=True)


def main() -> None:
    """Generate and save the qualitative figure and slice metadata.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        stream=sys.stdout)
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but CUDA is not available.")
    cases = args.case or [
        parse_case("Good=0336_fu01"),
        parse_case("Intermediate=0200_fu02"),
        parse_case("Difficult=0359_fu01"),
    ]
    rows = read_manifest(
        args.manifest,
        PAIR_IMAGE_COLUMNS + ("fixed_pet_label", "moving_pet_label"))
    if args.data_root is not None:
        _relocate_rows(rows, args.data_root)
    selected = select_rows(rows, cases)
    metrics = load_metrics(args.metrics_csv, args.metric)
    case_data = _predict_cases(args, selected, torch.device(args.device))
    diff_vmax = _difference_vmax(case_data)

    fig, axes = _build_axes(args.layout, len(case_data))
    mappables: list[Any] = []
    slice_records = []
    for row_index, (case, row, data) in enumerate(case_data):
        token = case_token(row)
        view = _select_view(args, row_index, token, data)
        print(f"Selected slice for {case.label} {token}: z={view.z}",
              flush=True)
        slice_records.append(_slice_record(case, token, view.z, data))
        mappables.append(_draw_case(axes, row_index, data, view, diff_vmax))
        _finish_case(
            axes, row_index, data,
            metric_caption(metrics, row, chr(ord("a") + row_index), case.label))
    _add_titles_and_colorbars(fig, axes, mappables)
    _save_figure(fig, args, slice_records)


if __name__ == "__main__":
    main()
