#!/bin/bash
# Stage B synthetic pretraining on a local GPU.
#
# Every training parameter comes from
# psmareg/config/synthetic_pretraining_stage_b.py, which starts from the
# Stage A weights with a fresh optimizer. Optional overrides:
#   OUTPUT_DIR       Run directory.
#   EPOCHS           Number of epochs.
#   INITIALIZE_FROM  Stage A checkpoint to start from (fresh optimizer).
#   START_FROM       Stage B checkpoint to resume, with its optimizer state
#                    and best validation loss.
#   NUM_WORKERS      DataLoader workers.
# Extra arguments are passed to the trainer, for example:
#   scripts/local/run_pretraining_stage_b.sh --set lambda_field=0.2

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

args=(--config psmareg/config/synthetic_pretraining_stage_b.py)
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${INITIALIZE_FROM:-}" ]] && args+=(--start_from "${INITIALIZE_FROM}")
if [[ -n "${START_FROM:-}" ]]; then
  args+=(--start_from "${START_FROM}"
         --set load_optimizer_state=true --set inherit_best_val_loss=true)
fi
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")

run_logged pretraining_stage_b \
  "${PYTHON}" train_synthetic_registration_pretraining.py "${args[@]}" "$@"
