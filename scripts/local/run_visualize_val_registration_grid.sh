#!/bin/bash
# Registration panels for selected validation subjects on a local GPU.
#
# Required:
#   CHECKPOINT  Trained checkpoint, for example
#               results/fine_tuning/best_val.pt.
# Optional overrides:
#   MANIFEST          Paired validation manifest.
#   OUTPUT_DIR        Figure directory (default: visual_checks next to the
#                     checkpoint).
#   SUBJECTS          Comma- or space-separated subject filters; empty renders
#                     every subject of the manifest.
#   NUM_SLICES        Slices per subject (default: 4).
#   SLICE_SOURCE      fractions (default) or hot_pet.
#   SLICE_FRACTIONS   Relative slice positions used with fractions.
#   HOT_PET_QUANTILE  PET quantile used with hot_pet.
#   DEVICE            cuda (default) or cpu.
#   REGISTRATION_ONLY, RETRY_CPU_ON_SIGKILL, SKIP_EXISTING
#                     Set to 0 to disable (default: 1).
#   BONE_AFFINE_PREALIGN
#                     Set to 1 to add the bone affine pre-alignment.
# Extra arguments are passed to scripts/visualize_val_registration_grid.py,
# for example: --bone_affine_iterations 40

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "CHECKPOINT is required, e.g. CHECKPOINT=path/to/best_val.pt $0" >&2
  exit 2
fi
MANIFEST="${MANIFEST:-manifests/paired/val_pairs.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CHECKPOINT}")/visual_checks}"
SLICE_SOURCE="${SLICE_SOURCE:-fractions}"

args=(
  scripts/visualize_val_registration_grid.py
  --manifest "${MANIFEST}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --num_slices "${NUM_SLICES:-4}"
  --slice_source "${SLICE_SOURCE}"
  --hot_pet_quantile "${HOT_PET_QUANTILE:-0.985}"
  --device "${DEVICE:-cuda}"
  --deterministic_translation_prealign
  --deterministic_translation_downsample 4
  --deterministic_translation_search_radius 12
  --deterministic_translation_search_step 2
  --deterministic_translation_max 120
)
if [[ -n "${SUBJECTS:-}" ]]; then
  read -r -a subject_filters <<< "${SUBJECTS//,/ }"
  args+=(--subjects "${subject_filters[@]}")
fi
if [[ "${SLICE_SOURCE}" == "fractions" ]]; then
  read -r -a slice_fractions <<< "${SLICE_FRACTIONS:-0.30 0.45 0.60 0.75}"
  args+=(--slice_fractions "${slice_fractions[@]}")
fi
[[ "${REGISTRATION_ONLY:-1}" == "1" ]] && args+=(--registration_only)
[[ "${RETRY_CPU_ON_SIGKILL:-1}" == "1" ]] && args+=(--retry_cpu_on_sigkill)
[[ "${SKIP_EXISTING:-1}" == "1" ]] && args+=(--skip_existing)
[[ "${BONE_AFFINE_PREALIGN:-0}" == "1" ]] && args+=(--bone_affine_prealign)

run_logged visualize_val_registration_grid "${PYTHON}" "${args[@]}" "$@"
