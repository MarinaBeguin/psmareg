#!/bin/bash
# Shared setup for the local GPU launchers. Source this file; do not run it.
#
# Environment variables understood by every launcher:
#   GPU          Index of the GPU to use (default: 0).
#   PYTHON       Python interpreter (default: python3).
#   PROJECT_DIR  Repository root (default: deduced from this file's location).
#   DRY_RUN      Set to 1 to print the command instead of running it.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Usage: run_logged <job_name> <command> [arguments...]
# Runs the command from the repository root and copies its output to
# logs/<job_name>_<timestamp>.log.
run_logged() {
  local job_name="$1"
  shift
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  mkdir -p logs
  local log_file
  log_file="logs/${job_name}_$(date +%Y%m%d_%H%M%S).log"
  echo "GPU ${CUDA_VISIBLE_DEVICES} | log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}
