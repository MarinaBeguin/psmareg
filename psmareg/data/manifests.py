"""CSV manifest reading and patient-level split hygiene."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Collection

_PATIENT_PATTERN = re.compile(r"(?:PSMARegPSMA_|PSMAReg_)?(\d{4})")

# Image columns that every paired manifest row must provide.
PAIR_IMAGE_COLUMNS = ("subject", "followup_index", "fixed_ct", "fixed_pet",
                      "moving_ct", "moving_pet")


def read_manifest(
    path: Path, required_columns: Collection[str] = ()) -> list[dict[str, str]]:
    """Read every row of a CSV manifest.

    Args:
        path: Manifest with one header row.
        required_columns: Column names that must be present.

    Returns:
        One mapping from column name to value per row.

    Raises:
        ValueError: If a required column is missing.
    """
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(required_columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Missing manifest columns in {path}: {sorted(missing)}")
        return list(reader)


def patient_id_from_row(row: dict[str, str]) -> str:
    """Extract the four-digit patient identifier from a manifest row.

    Raises:
        ValueError: If no four-digit patient identifier is found.
    """
    for key in ("subject", "ct", "fixed_ct", "moving_ct"):
        match = _PATIENT_PATTERN.search(row.get(key, ""))
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract a patient id from manifest row: {row}")


def patient_ids_from_manifest(path: Path) -> set[str]:
    """Read all unique patient identifiers from a CSV manifest."""
    return {patient_id_from_row(row) for row in read_manifest(path)}


def exclude_manifest_patients(
    rows: list[dict[str, str]],
    exclusion_manifest: Path | None,
) -> list[dict[str, str]]:
    """Remove patients listed in another manifest.

    Raises:
        ValueError: If the exclusion removes every row.
    """
    if exclusion_manifest is None:
        return rows

    excluded_ids = patient_ids_from_manifest(exclusion_manifest)
    filtered = [
        row for row in rows if patient_id_from_row(row) not in excluded_ids
    ]
    if not filtered:
        raise ValueError(
            f"Excluding patients from {exclusion_manifest} removed every row.")
    return filtered
