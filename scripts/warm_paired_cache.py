"""Populate the paired PET/CT preprocessing cache before fine-tuning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from psmareg.data.paired_dataset import PSMARegPairDataset


def main() -> None:
    """Read every pair once so fine-tuning can reuse its disk cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Paired CSV manifest; repeat for training and validation.")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(
            os.environ.get("PSMAREG_PAIRED_CACHE_DIR",
                           "./psmareg_paired_cache")),
    )
    args = parser.parse_args()
    for manifest in args.manifest:
        dataset = PSMARegPairDataset(manifest,
                                    cache_dir=args.cache_dir,
                                    cache_float16=True)
        for index in range(len(dataset)):
            _ = dataset[index]
        print(f"Cached {len(dataset)} pairs from {manifest}", flush=True)


if __name__ == "__main__":
    main()
