#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}"
PROFILE="${PROFILE:-submission}"
SUITE="${SUITE:-headline}"
ANCHOR="${ANCHOR:-all}"
SCENE_CLASS="${SCENE_CLASS:-open_aisle}"
JOBS="${JOBS:-4}"
TRIALS="${TRIALS:-}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/run_study.py"
  --profile "$PROFILE"
  --suite "$SUITE"
  --anchor "$ANCHOR"
  --scene-class "$SCENE_CLASS"
  --jobs "$JOBS"
)

if [[ -n "$TRIALS" ]]; then
  cmd+=(--trials "$TRIALS")
fi

if [[ "$CLEAN_OUTPUTS" == "1" ]]; then
  cmd+=(--clean-outputs)
fi

"${cmd[@]}"

OUTPUT_ROOT="$ROOT_DIR/results/$PROFILE"
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
  anchor_comparison.csv
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

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
manifest_path="$OUTPUT_ROOT/build_manifest.txt"
{
  echo "timestamp_utc=$timestamp_utc"
  echo "profile=$PROFILE"
  echo "suite=$SUITE"
  echo "anchor=$ANCHOR"
  echo "scene_class=$SCENE_CLASS"
  echo "jobs=$JOBS"
  echo "trials=${TRIALS:-default}"
  echo "python_bin=$PYTHON_BIN"
  echo "command=${cmd[*]}"
} > "$manifest_path"

echo "Wrote manifest: $manifest_path"
