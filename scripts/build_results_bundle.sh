#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}"
PROFILE="${PROFILE:-submission}"
SUITE="${SUITE:-headline}"
ANCHOR="${ANCHOR:-all}"
SCENE_CLASS="${SCENE_CLASS:-all}"
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
  range_separation.csv
  velocity_separation.csv
  angle_separation.csv
  absolute_range.csv
  burst_profile.csv
  aperture.csv
  scene_comparison.csv
  fr1_vs_fr2.csv
  crb_gap.csv
)

required_figures=(
  range_separation.png
  velocity_separation.png
  angle_separation.png
  absolute_range.png
  burst_profile.png
  aperture.png
  scene_comparison.png
  fr1_vs_fr2.png
  crb_gap.png
  representative_cube_slices.png
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
