"""Split a labeled single-scan manifest into train and validation CSVs."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

FIELDNAMES = ["ct", "ct_label", "pet", "pet_label", "subject"]


def parse_args() -> argparse.Namespace:
    """Parse the manifest-splitting command line.

    Returns:
        Validated command-line values from ``argparse``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("manifests_pretraining/all_scans.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("manifests_pretraining"),
    )
    parser.add_argument("--train_output_name", default="training_data.csv")
    parser.add_argument("--val_output_name", default="validation_data.csv")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_by",
        choices=["patient", "scan"],
        default="patient",
        help=
        "Use patient to keep all timepoints from a subject in the same split.",
    )
    parser.add_argument(
        "--skip_missing_files",
        action="store_true",
        help="Drop rows whose CT/PET images or labels do not exist.",
    )
    return parser.parse_args()


def patient_id_from_subject(subject: str) -> str:
    """Collapse a scan id like PSMARegPSMA_0006_01 to PSMARegPSMA_0006."""
    return re.sub(r"_\d{2}$", "", subject)


def read_manifest(path: Path, skip_missing_files: bool) -> list[dict[str, str]]:
    """Read the required columns from a single-scan manifest.

    Args:
        path: Input CSV path.
        skip_missing_files: Whether to omit rows whose image or label files
            are missing.

    Returns:
        Nonempty list of manifest rows.

    Raises:
        ValueError: If columns are missing or no usable row remains.
    """
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = set(FIELDNAMES) - set(reader.fieldnames or [])
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {columns}")

        rows: list[dict[str, str]] = []
        skipped = 0
        for row in reader:
            item = {field: row[field] for field in FIELDNAMES}
            if skip_missing_files and not all(
                    Path(item[field]).exists()
                    for field in ("ct", "ct_label", "pet", "pet_label")):
                skipped += 1
                continue
            rows.append(item)

    if skipped:
        print(f"Skipped {skipped} rows with missing files.")
    if not rows:
        raise ValueError(f"No usable rows found in {path}.")
    return rows


def split_rows(
    rows: list[dict[str, str]],
    val_fraction: float,
    seed: int,
    split_by: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows reproducibly by patient or by scan.

    Args:
        rows: Single-scan manifest rows.
        val_fraction: Target fraction of validation groups.
        seed: Random seed for group shuffling.
        split_by: ``patient`` to keep timepoints together, otherwise ``scan``.

    Returns:
        Training and validation rows, each sorted by subject.

    Raises:
        ValueError: If the validation fraction is outside ``(0, 1)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1.")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if split_by == "patient":
            group_key = patient_id_from_subject(row["subject"])
        else:
            group_key = f"{row['subject']}::{index}"
        grouped_rows[group_key].append(row)

    groups = list(grouped_rows.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    num_val_groups = max(1, round(len(groups) * val_fraction))
    if num_val_groups >= len(groups):
        num_val_groups = len(groups) - 1

    val_group_keys = {group_key for group_key, _ in groups[:num_val_groups]}

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for group_key, group_rows in groups:
        if group_key in val_group_keys:
            val_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    train_rows.sort(key=lambda row: row["subject"])
    val_rows.sort(key=lambda row: row["subject"])

    return train_rows, val_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write single-scan rows to a CSV manifest.

    Args:
        path: Output CSV path.
        rows: Rows to serialize in the standard column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the requested split and reject patient leakage.

    Raises:
        RuntimeError: If a patient group appears in both splits.
    """
    args = parse_args()

    rows = read_manifest(
        path=args.input_manifest,
        skip_missing_files=args.skip_missing_files,
    )
    train_rows, val_rows = split_rows(
        rows=rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split_by=args.split_by,
    )

    train_path = args.output_dir / args.train_output_name
    val_path = args.output_dir / args.val_output_name
    write_manifest(train_path, train_rows)
    write_manifest(val_path, val_rows)

    train_groups = {
        patient_id_from_subject(row["subject"])
        if args.split_by == "patient" else row["subject"] for row in train_rows
    }
    val_groups = {
        patient_id_from_subject(row["subject"])
        if args.split_by == "patient" else row["subject"] for row in val_rows
    }
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(
            f"Found split leakage for groups: {sorted(overlap)[:5]}")

    print(f"Input rows: {len(rows)}")
    print(f"Split by: {args.split_by}")
    print(f"Validation fraction: {args.val_fraction:.3f}")
    print(f"Seed: {args.seed}")
    print(f"Training rows: {len(train_rows)} -> {train_path}")
    print(f"Validation rows: {len(val_rows)} -> {val_path}")


if __name__ == "__main__":
    main()
