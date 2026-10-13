#!/usr/bin/env python3
"""Export PSMAReg pull-back displacement fields from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from psmareg.data.manifests import PAIR_IMAGE_COLUMNS, read_manifest
from psmareg.evaluation.displacement_inference import (
    fullres_to_submitted_lowres,
    pair_from_row,
    predict_displacement,
    save_displacement,
)
from psmareg.evaluation.visualization_inputs import (
    ModelOverrides,
    VisualizationSettings,
    load_checkpoint_config,
    load_model,
)


def main() -> None:
    """Save a full- or half-resolution NIfTI field for each manifest pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--lowres",
        action="store_true",
        help="Write the half-resolution challenge submission field")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, ModelOverrides())
    checkpoint_config = load_checkpoint_config(args.checkpoint, device)
    settings = VisualizationSettings(args.manifest,
                                     args.checkpoint,
                                     args.output_dir,
                                     device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in read_manifest(args.manifest, PAIR_IMAGE_COLUMNS):
        pair, reference = pair_from_row(row, device)
        flow = predict_displacement(model, pair, settings, checkpoint_config)
        if args.lowres:
            flow = fullres_to_submitted_lowres(flow)
        subject = row["subject"].split("_")[-1]
        followup = row["followup_index"]
        output = (args.output_dir /
                  f"disp_{subject}_00_{subject}_{followup}.nii.gz")
        save_displacement(flow, output, reference.affine)
        print(output, flush=True)


if __name__ == "__main__":
    main()
