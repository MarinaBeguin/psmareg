#!/bin/bash
#SBATCH --job-name=psmareg_synth_v2_A
#SBATCH --output=logs/pretraining_synth_v2_stageA_%j.out
#SBATCH --error=logs/pretraining_synth_v2_stageA_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export \
  MODEL=transmorph_v2_cascade \
  EPOCHS="${EPOCHS:-80}" \
  OUTPUT_DIR="${OUTPUT_DIR:-results/transmorph_v2_synthetic_stageA}"

set -e
source ~/.bashrc
set -u -o pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/mbeguin/code/psmareg-lesion-aware-registration}}"

args=(--config psmareg/config/synthetic_pretraining_stage_a.py)
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${START_FROM:-}" ]] && args+=(--start_from "${START_FROM}")
[[ -n "${MODEL:-}" ]] && args+=(--set "model=${MODEL}")
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")

python3 train_synthetic_registration_pretraining.py "${args[@]}" "$@"
