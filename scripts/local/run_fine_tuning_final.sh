#!/bin/bash
# Final lesion-aware fine-tuning on a local GPU.
#
# Every training parameter comes from psmareg/config/fine_tuning_final.py.
# Optional overrides:
#   TRAIN_CONFIG    Another recipe file.
#   TRAIN_MANIFEST  Training pairs.
#   VAL_MANIFEST    Validation pairs.
#   OUTPUT_DIR      Run directory.
#   RESUME_FROM     Checkpoint of an interrupted run, usually latest.pt.
#   EPOCHS          Number of epochs.
#   NUM_WORKERS     DataLoader workers.
# Extra arguments are passed to the trainer, for example:
#   scripts/local/run_fine_tuning_final.sh --set mind_loss_downsample=2

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

args=(--config "${TRAIN_CONFIG:-psmareg/config/fine_tuning_final.py}")
[[ -n "${TRAIN_MANIFEST:-}" ]] && args+=(--train_manifest "${TRAIN_MANIFEST}")
[[ -n "${VAL_MANIFEST:-}" ]] && args+=(--val_manifest "${VAL_MANIFEST}")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${RESUME_FROM:-}" ]] && args+=(--resume_from "${RESUME_FROM}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")

run_logged fine_tuning_final "${PYTHON}" train_fine_tuning.py "${args[@]}" "$@"
