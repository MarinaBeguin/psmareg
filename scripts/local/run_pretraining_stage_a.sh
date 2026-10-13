#!/bin/bash
# Stage A synthetic pretraining on a local GPU.
#
# Every training parameter comes from
# psmareg/config/synthetic_pretraining_stage_a.py. Optional overrides:
#   OUTPUT_DIR   Run directory.
#   EPOCHS       Number of epochs.
#   START_FROM   Checkpoint to continue from.
#   NUM_WORKERS  DataLoader workers.
# Extra arguments are passed to the trainer, for example:
#   GPU=1 scripts/local/run_pretraining_stage_a.sh --set lr=5e-5

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

args=(--config psmareg/config/synthetic_pretraining_stage_a.py)
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${START_FROM:-}" ]] && args+=(--start_from "${START_FROM}")
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")

run_logged pretraining_stage_a \
  "${PYTHON}" train_synthetic_registration_pretraining.py "${args[@]}" "$@"
