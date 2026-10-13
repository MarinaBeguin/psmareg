#!/bin/bash
# Stage C (global-motion expert) synthetic pretraining on a local GPU.
#
# Every training parameter comes from
# psmareg/config/synthetic_pretraining_stage_c.py, which trains the affine
# cascade from scratch. Optional overrides:
#   OUTPUT_DIR       Run directory.
#   EPOCHS           Number of epochs.
#   INITIALIZE_FROM  Checkpoint providing shape-compatible deformable weights.
#   START_FROM       Stage C checkpoint to resume, with its optimizer state
#                    and best validation loss.
#   NUM_WORKERS      DataLoader workers.
# Extra arguments are passed to the trainer, for example:
#   scripts/local/run_pretraining_stage_c.sh --set lambda_affine_field=0.5

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

args=(--config psmareg/config/synthetic_pretraining_stage_c.py)
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
if [[ -n "${START_FROM:-}" ]]; then
  args+=(--start_from "${START_FROM}"
         --set load_optimizer_state=true --set inherit_best_val_loss=true)
elif [[ -n "${INITIALIZE_FROM:-}" ]]; then
  args+=(--initialize_deformable_from "${INITIALIZE_FROM}")
fi
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")

run_logged pretraining_stage_c \
  "${PYTHON}" train_synthetic_registration_pretraining.py "${args[@]}" "$@"
