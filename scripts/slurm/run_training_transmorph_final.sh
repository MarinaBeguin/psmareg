#!/bin/bash
#SBATCH --job-name=psmareg_final
#SBATCH --output=logs/psmareg_final_%j.out
#SBATCH --error=logs/psmareg_final_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/mbeguin/code/psmareg-lesion-aware-registration}}"
cd "${PROJECT_DIR}"

args=(--config "${TRAIN_CONFIG:-psmareg/config/fine_tuning_final.py}")
[[ -n "${TRAIN_MANIFEST:-}" ]] && args+=(--train_manifest "${TRAIN_MANIFEST}")
[[ -n "${VAL_MANIFEST:-}" ]] && args+=(--val_manifest "${VAL_MANIFEST}")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${RESUME_FROM:-}" ]] && args+=(--resume_from "${RESUME_FROM}")
[[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")

# Diagnostic-only overrides. Normal fine-tuning reads these values from the recipe.
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--set "num_workers=${NUM_WORKERS}")
[[ -n "${MIND_LOSS_DOWNSAMPLE:-}" ]] && args+=(--set "mind_loss_downsample=${MIND_LOSS_DOWNSAMPLE}")
[[ -n "${MACRO_CT_LABEL_DOWNSAMPLE:-}" ]] && args+=(--set "macro_ct_label_downsample=${MACRO_CT_LABEL_DOWNSAMPLE}")
[[ -n "${SMOOTHNESS_DOWNSAMPLE:-}" ]] && args+=(--set "smoothness_downsample=${SMOOTHNESS_DOWNSAMPLE}")
[[ -n "${CT_DICE_LOSS_DOWNSAMPLE:-}" ]] && args+=(--set "ct_dice_loss_downsample=${CT_DICE_LOSS_DOWNSAMPLE}")

python3 train_fine_tuning.py "${args[@]}"
