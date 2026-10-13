"""Build paired registration manifests from the PSMAReg dataset JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PairRow = dict[str, str]


def _load_json(json_path: Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""
    with json_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    return data


def _resolve_path(dataset_root: Path, relative_path: str) -> str:
    """Resolve paths stored in the JSON file relative to the dataset root."""
    return str((dataset_root / relative_path).resolve())


def _extract_followup_indices(subject_entry: dict[str, Any]) -> list[str]:
    """Return all follow-up indices available for one subject."""
    indices: list[str] = []

    for key in subject_entry:
        if key.startswith("Follow-up ") and key.endswith(" CT"):
            parts = key.split()
            if len(parts) >= 2:
                indices.append(parts[1])

    return sorted(indices)


def _build_rows_for_split(
    entries: list[dict[str, Any]],
    dataset_root: Path,
    split_name: str,
) -> list[PairRow]:
    """Build one row per baseline-to-follow-up registration pair."""
    rows: list[PairRow] = []

    for subject_entry in entries:
        subject = str(subject_entry["subject"])
        followup_indices = _extract_followup_indices(subject_entry)

        if not followup_indices:
            continue

        for followup_index in followup_indices:
            row: PairRow = {
                "split":
                    split_name,
                "subject":
                    subject,
                "followup_index":
                    followup_index,
                "fixed_ct":
                    _resolve_path(
                        dataset_root,
                        str(subject_entry["Baseline CT"]),
                    ),
                "fixed_pet":
                    _resolve_path(
                        dataset_root,
                        str(subject_entry["Baseline PET"]),
                    ),
                "moving_ct":
                    _resolve_path(
                        dataset_root,
                        str(subject_entry[f"Follow-up {followup_index} CT"]),
                    ),
                "moving_pet":
                    _resolve_path(
                        dataset_root,
                        str(subject_entry[f"Follow-up {followup_index} PET"]),
                    ),
            }

            if "Baseline CT Label" in subject_entry:
                row["fixed_ct_label"] = _resolve_path(
                    dataset_root,
                    str(subject_entry["Baseline CT Label"]),
                )
            if "Baseline PET Label" in subject_entry:
                row["fixed_pet_label"] = _resolve_path(
                    dataset_root,
                    str(subject_entry["Baseline PET Label"]),
                )
            if f"Follow-up {followup_index} CT Label" in subject_entry:
                row["moving_ct_label"] = _resolve_path(
                    dataset_root,
                    str(subject_entry[f"Follow-up {followup_index} CT Label"]),
                )
            if f"Follow-up {followup_index} PET Label" in subject_entry:
                row["moving_pet_label"] = _resolve_path(
                    dataset_root,
                    str(subject_entry[f"Follow-up {followup_index} PET Label"]),
                )

            rows.append(row)

    return rows


def _write_csv(rows: list[PairRow], output_path: Path) -> None:
    """Write manifest rows to CSV."""
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    json_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> None:
    """Build train and validation manifests from the dataset JSON.

    Args:
        json_path: Dataset description of the challenge.
        dataset_root: Root against which the JSON paths are resolved.
        output_dir: Directory receiving the training and validation manifests.
    """
    metadata = _load_json(json_path)

    train_entries = list(metadata.get("training_paired", []))
    val_entries = list(metadata.get("validation_paired", []))

    train_rows = _build_rows_for_split(
        entries=train_entries,
        dataset_root=dataset_root,
        split_name="train",
    )
    val_rows = _build_rows_for_split(
        entries=val_entries,
        dataset_root=dataset_root,
        split_name="val",
    )

    _write_csv(train_rows, output_dir / "train_pairs.csv")
    _write_csv(val_rows, output_dir / "val_pairs.csv")

    print(f"Saved {len(train_rows)} train pairs")
    print(f"Saved {len(val_rows)} validation pairs")


def _parse_args() -> argparse.Namespace:
    """Parse the manifest-building options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build paired training and validation manifests from metadata."""
    args = _parse_args()
    build_manifests(
        json_path=args.json_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
