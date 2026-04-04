#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}"
DEFAULT_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
JOBS="${JOBS:-$DEFAULT_JOBS}"
TRIALS="${TRIALS:-}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
ALLOW_SMOKE_SUBMISSION="${ALLOW_SMOKE_SUBMISSION:-0}"
SUBMISSION_DEFAULT_TRIALS=64
SCENE_CLASS="${SCENE_CLASS:-all}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$TRIALS" && "$TRIALS" -lt "$SUBMISSION_DEFAULT_TRIALS" && "$ALLOW_SMOKE_SUBMISSION" != "1" ]]; then
  echo "Submission bundle requires at least $SUBMISSION_DEFAULT_TRIALS trials unless ALLOW_SMOKE_SUBMISSION=1 is set." >&2
  exit 1
fi

export PYTHON_BIN
export PROFILE=submission
export SUITE=headline
export ANCHOR=fr1
export SCENE_CLASS
export JOBS
export TRIALS
export CLEAN_OUTPUTS

"$ROOT_DIR/scripts/build_results_bundle.sh"

manifest_path="$ROOT_DIR/results/submission/build_manifest.txt"
submission_mode="final"
if [[ -n "$TRIALS" && "$TRIALS" -lt "$SUBMISSION_DEFAULT_TRIALS" ]]; then
  submission_mode="smoke_override"
elif [[ -n "$TRIALS" && "$TRIALS" -ge "$SUBMISSION_DEFAULT_TRIALS" ]]; then
  submission_mode="custom_final"
fi

{
  echo "submission_bundle=1"
  echo "submission_trial_floor=$SUBMISSION_DEFAULT_TRIALS"
  echo "submission_mode=$submission_mode"
  echo "allow_smoke_submission=$ALLOW_SMOKE_SUBMISSION"
  echo "command=bash scripts/build_submission_bundle.sh"
} >> "$manifest_path"

echo "Wrote manifest: $manifest_path"
