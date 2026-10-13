#!/bin/bash
#SBATCH --job-name=psmareg_synth_v2_B
#SBATCH --output=logs/pretraining_synth_v2_stageB_%j.out
#SBATCH --error=logs/pretraining_synth_v2_stageB_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export \
  MODEL=transmorph_v2_cascade \
  EPOCHS="${EPOCHS:-120}" \
  INITIALIZE_FROM="${INITIALIZE_FROM:-results/transmorph_v2_synthetic_stageA/best_val.pt}" \
  OUTPUT_DIR="${OUTPUT_DIR:-results/transmorph_v2_synthetic_stageB}"

set -e
source ~/.bashrc
set -u -o pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/mbeguin/code/psmareg-lesion-aware-registration}}"

args=(--config psmareg/config/synthetic_pretraining_stage_b.py)
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${MODEL:-}" ]] && args+=(--set "model=${MODEL}")
[[ -n "${INITIALIZE_FROM:-}" ]] && args+=(--start_from "${INITIALIZE_FROM}")
if [[ -n "${START_FROM:-}" ]]; then
  args+=(--start_from "${START_FROM}" --set load_optimizer_state=true --set inherit_best_val_loss=true)
fi
[[ -n "${PET_INT_AUG_PROB:-}" ]] && args+=(--set "pet_independent_intensity_probability=${PET_INT_AUG_PROB}")
[[ -n "${PET_INT_SCALE_LOG_STD:-}" ]] && args+=(--set "pet_independent_scale_log_std=${PET_INT_SCALE_LOG_STD}")
[[ -n "${PET_INT_GAMMA_RANGE:-}" ]] && args+=(--set "pet_independent_gamma_range=${PET_INT_GAMMA_RANGE}")
[[ -n "${PET_INT_BIAS_RANGE:-}" ]] && args+=(--set "pet_independent_bias_range=${PET_INT_BIAS_RANGE}")

python3 train_synthetic_registration_pretraining.py "${args[@]}" "$@"
