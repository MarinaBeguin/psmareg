"""Create single-scan manifest for CT/PET + labels pretraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse image, label, and output paths from the command line.

    Returns:
        Parsed command-line values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write a single-scan manifest for complete CT/PET label pairs."""
    args = parse_args()
    rows: list[dict[str, str]] = []

    for ct_path in sorted(args.images_dir.glob("*_0000_*.nii.gz")):
        pet_path = Path(str(ct_path).replace("_0000_", "_0001_"))

        ct_label_path = args.labels_dir / ct_path.name
        pet_label_path = args.labels_dir / pet_path.name

        if not pet_path.exists():
            print(f"Missing PET image: {pet_path}")
            continue
        if not ct_label_path.exists():
            print(f"Missing CT label: {ct_label_path}")
            continue
        if not pet_label_path.exists():
            print(f"Missing PET label: {pet_label_path}")
            continue

        subject = ct_path.name.replace("_0000_", "_").replace(".nii.gz", "")

        rows.append({
            "ct": str(ct_path),
            "ct_label": str(ct_label_path),
            "pet": str(pet_path),
            "pet_label": str(pet_label_path),
            "subject": subject,
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ct", "ct_label", "pet", "pet_label", "subject"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created manifest with {len(rows)} scans.")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
