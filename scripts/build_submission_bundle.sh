#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}"
JOBS="${JOBS:-4}"
TRIALS="${TRIALS:-}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

export ROOT_DIR PYTHON_BIN JOBS TRIALS CLEAN_OUTPUTS

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/run_study.py"
  --profile submission
  --suite headline
  --anchor fr1
  --scene-class open_aisle
  --jobs "$JOBS"
)

if [[ -n "$TRIALS" ]]; then
  cmd+=(--trials "$TRIALS")
fi

if [[ "$CLEAN_OUTPUTS" == "1" ]]; then
  cmd+=(--clean-outputs)
fi

"${cmd[@]}"

OUTPUT_ROOT="$ROOT_DIR/results/submission"
DATA_DIR="$OUTPUT_ROOT/data"
FIGURES_DIR="$OUTPUT_ROOT/figures"

required_data=(
  allocation_family.csv
  occupied_fraction.csv
  pilot_fraction.csv
  fragmentation.csv
  range_separation.csv
  velocity_separation.csv
  angle_separation.csv
  nominal_summary.csv
  runtime_summary.csv
  failure_modes.csv
)

required_figures=(
  allocation_family.png
  occupied_fraction.png
  pilot_fraction.png
  fragmentation.png
  range_separation.png
  velocity_separation.png
  angle_separation.png
  runtime_summary.png
  representative_resource_mask.png
  representative_spectrum.png
)

for filename in "${required_data[@]}"; do
  [[ -f "$DATA_DIR/$filename" ]] || { echo "Missing data artifact: $filename" >&2; exit 1; }
done

for filename in "${required_figures[@]}"; do
  [[ -f "$FIGURES_DIR/$filename" ]] || { echo "Missing figure artifact: $filename" >&2; exit 1; }
done

actual_data="$(find "$DATA_DIR" -maxdepth 1 -type f -exec basename {} \; | sort)"
actual_figures="$(find "$FIGURES_DIR" -maxdepth 1 -type f -exec basename {} \; | sort)"
expected_data="$(printf '%s\n' "${required_data[@]}" | sort)"
expected_figures="$(printf '%s\n' "${required_figures[@]}" | sort)"

[[ "$actual_data" == "$expected_data" ]] || {
  echo "Unexpected data artifact set: $actual_data" >&2
  exit 1
}
[[ "$actual_figures" == "$expected_figures" ]] || {
  echo "Unexpected figure artifact set: $actual_figures" >&2
  exit 1
}

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
manifest_path="$OUTPUT_ROOT/build_manifest.txt"
{
  echo "timestamp_utc=$timestamp_utc"
  echo "profile=submission"
  echo "suite=headline"
  echo "anchor=fr1"
  echo "scenes=open_aisle"
  echo "sweeps=allocation_family,occupied_fraction,pilot_fraction,fragmentation,range_separation,velocity_separation,angle_separation"
  echo "jobs=$JOBS"
  echo "trials=${TRIALS:-default}"
  echo "python_bin=$PYTHON_BIN"
  echo "command=bash scripts/build_submission_bundle.sh"
} > "$manifest_path"

echo "Wrote manifest: $manifest_path"
