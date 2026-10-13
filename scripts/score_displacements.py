#!/usr/bin/env python3
"""Score exported displacement fields with the official challenge metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psmareg.evaluation.displacement_inference import (
    OfficialSpatialTransformer,)
from psmareg.evaluation.local_scoring import (
    MetricRow,
    get_compute_device,
    get_followup_indices,
    score_pair,
    summarize,
    write_csv,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--json_name", type=str, default="PSMAReg_dataset.json")
    parser.add_argument("--followups", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run local scoring on training pairs.

    Raises:
        RuntimeError: If the dataset lists no pair or no pair could be scored.
    """
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json_path = args.reference_dir / args.json_name
    with dataset_json_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    entries = dataset.get("training_paired", [])
    if not entries:
        raise RuntimeError(
            f"No training_paired entries found in {dataset_json_path}")

    device = get_compute_device(args.device)
    print(f"Using device: {device}")

    spatial_transformer = OfficialSpatialTransformer(
        size=(192, 192, 288),
        mode="nearest",
        padding_mode="border",
    ).to(device)
    spatial_transformer.eval()

    rows: list[MetricRow] = []

    for entry in entries:
        subject = str(entry["subject"])
        followup_indices = get_followup_indices(entry, args.followups)

        for followup_index in followup_indices:
            try:
                row = score_pair(
                    entry=entry,
                    followup_index=followup_index,
                    reference_dir=args.reference_dir,
                    prediction_dir=args.prediction_dir,
                    spatial_transformer=spatial_transformer,
                    device=device,
                )
            except FileNotFoundError as error:
                print(f"Skipping {subject} FU{followup_index}: {error}")
                continue

            rows.append(row)
            print(f"{subject} FU{followup_index} | "
                  f"DSC={row['DSC']:.4f} | "
                  f"HD95={row['HD95']:.4f} | "
                  f"MTV={row['MTV Percent Error']:.4f}% | "
                  f"TLG={row['TLG Percent Error']:.4f}% | "
                  f"NDV={row['Percent NDV']:.4f}%")

    if not rows:
        raise RuntimeError("No pairs were scored. Check prediction filenames.")

    metric_names = [
        "DSC",
        "HD95",
        "MTV Percent Error",
        "TLG Percent Error",
        "Percent NDV",
    ]
    summary = {
        metric_name: summarize([float(row[metric_name]) for row in rows])
        for metric_name in metric_names
    }

    write_csv(rows, args.output_dir / "local_scoring_per_pair.csv")
    write_json(summary, args.output_dir / "local_scoring_summary.json")

    print("")
    print("Summary:")
    for metric_name, metric_summary in summary.items():
        print(f"{metric_name}: "
              f"{metric_summary['mean']:.4f} ± {metric_summary['std']:.4f}")


if __name__ == "__main__":
    main()
