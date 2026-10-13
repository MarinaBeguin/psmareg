#!/bin/bash
#SBATCH --job-name=psmareg_val_viz
#SBATCH --output=logs/val_viz_%A_%a.out
#SBATCH --error=logs/val_viz_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-6

set -eo pipefail

source ~/.bashrc

set -u

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -d /home/mbeguin/code/psmareg-lesion-aware-registration ]]; then
    PROJECT_DIR=/home/mbeguin/code/psmareg-lesion-aware-registration
  elif [[ -d /home/mbeguin/cluster_AI/home/mbeguin/code/psmareg-lesion-aware-registration ]]; then
    PROJECT_DIR=/home/mbeguin/cluster_AI/home/mbeguin/code/psmareg-lesion-aware-registration
  else
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi
cd "${PROJECT_DIR}"

MANIFEST="${MANIFEST:-manifests/val_pairs_cluster.csv}"
CHECKPOINT="${CHECKPOINT:-results/transmorph_v2_hybrid_stageCglobal_refinerB_mcbo_mind_adaptive_pet_anatomy_v7_regularized_v2/best_val.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/transmorph_v2_hybrid_stageCglobal_refinerB_mcbo_mind_adaptive_pet_anatomy_v7_regularized_v2/visual_checks_key_patients_best}"
NUM_SLICES="${NUM_SLICES:-4}"
SLICE_FRACTIONS="${SLICE_FRACTIONS:-0.30 0.45 0.60 0.75}"
SLICE_SOURCE="${SLICE_SOURCE:-fractions}"
HOT_PET_QUANTILE="${HOT_PET_QUANTILE:-0.985}"
DEVICE="${DEVICE:-cuda}"
REGISTRATION_ONLY="${REGISTRATION_ONLY:-1}"
RETRY_CPU_ON_SIGKILL="${RETRY_CPU_ON_SIGKILL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BONE_AFFINE_PREALIGN="${BONE_AFFINE_PREALIGN:-0}"
BONE_AFFINE_DOWNSAMPLES="${BONE_AFFINE_DOWNSAMPLES:-8,4}"
BONE_AFFINE_ITERATIONS="${BONE_AFFINE_ITERATIONS:-80}"
BONE_AFFINE_LR="${BONE_AFFINE_LR:-0.05}"
BONE_AFFINE_THRESHOLD="${BONE_AFFINE_THRESHOLD:-0.66}"
BONE_AFFINE_SOFTNESS="${BONE_AFFINE_SOFTNESS:-0.04}"
BONE_AFFINE_MAX_TRANSLATION="${BONE_AFFINE_MAX_TRANSLATION:-120}"
BONE_AFFINE_MAX_ROTATION_DEGREES="${BONE_AFFINE_MAX_ROTATION_DEGREES:-8}"
BONE_AFFINE_MAX_SCALING="${BONE_AFFINE_MAX_SCALING:-0.04}"
BONE_AFFINE_MIN_IMPROVEMENT="${BONE_AFFINE_MIN_IMPROVEMENT:-0.005}"

# Comma- or space-separated subject list. The default is the set we selected for
# failure-pattern analysis. Override with e.g. SUBJECTS="0001,0003,0005".
SUBJECTS="${SUBJECTS:-0007 0008 0013 0021 0035 0036 0047}"
read -r -a SUBJECT_ARRAY <<< "${SUBJECTS//,/ }"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SUBJECT_ARRAY[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside subject list of length ${#SUBJECT_ARRAY[@]}"
  exit 0
fi
SUBJECT="${SUBJECT_ARRAY[$TASK_ID]}"

args=(
  scripts/visualize_val_registration_grid.py
  --manifest "${MANIFEST}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --num_slices "${NUM_SLICES}"
  --slice_source "${SLICE_SOURCE}"
  --hot_pet_quantile "${HOT_PET_QUANTILE}"
  --subjects "${SUBJECT}"
  --device "${DEVICE}"
  --deterministic_translation_prealign
  --deterministic_translation_downsample 4
  --deterministic_translation_search_radius 12
  --deterministic_translation_search_step 2
  --deterministic_translation_max 120
)

if [[ "${SLICE_SOURCE}" == "fractions" ]]; then
  args+=(--slice_fractions ${SLICE_FRACTIONS})
fi

if [[ "${REGISTRATION_ONLY}" == "1" ]]; then
  args+=(--registration_only)
fi
if [[ "${RETRY_CPU_ON_SIGKILL}" == "1" ]]; then
  args+=(--retry_cpu_on_sigkill)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip_existing)
fi
if [[ "${BONE_AFFINE_PREALIGN}" == "1" ]]; then
  args+=(
    --bone_affine_prealign
    --bone_affine_downsamples "${BONE_AFFINE_DOWNSAMPLES}"
    --bone_affine_iterations "${BONE_AFFINE_ITERATIONS}"
    --bone_affine_lr "${BONE_AFFINE_LR}"
    --bone_affine_threshold "${BONE_AFFINE_THRESHOLD}"
    --bone_affine_softness "${BONE_AFFINE_SOFTNESS}"
    --bone_affine_max_translation "${BONE_AFFINE_MAX_TRANSLATION}"
    --bone_affine_max_rotation_degrees "${BONE_AFFINE_MAX_ROTATION_DEGREES}"
    --bone_affine_max_scaling "${BONE_AFFINE_MAX_SCALING}"
    --bone_affine_min_improvement "${BONE_AFFINE_MIN_IMPROVEMENT}"
  )
fi

echo "Visualizing subject ${SUBJECT} into ${OUTPUT_DIR}"
python3 "${args[@]}"
