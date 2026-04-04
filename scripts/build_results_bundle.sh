#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}"
DEFAULT_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
PROFILE="${PROFILE:-submission}"
SUITE="${SUITE:-headline}"
ANCHOR="${ANCHOR:-all}"
SCENE_CLASS="${SCENE_CLASS:-open_aisle}"
JOBS="${JOBS:-$DEFAULT_JOBS}"
TRIALS="${TRIALS:-}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
SCHEMA_VERSION="2.0"
ESTIMATOR_SET="fft_masked,music_masked"
FBSS_ABLATION_SET="fbss_spatial_only,fbss_spatial_range,fbss_spatial_doppler,fbss_spatial_range_doppler"
EVIDENCE_PROFILE="music_waveform_limited_v2"

ACTIVE_SWEEPS=(
  allocation_family
  occupied_fraction
  fragmentation
  bandwidth_span
  slow_time_span
  range_separation
  velocity_separation
  angle_separation
)

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

required_data=()
for sweep_name in "${ACTIVE_SWEEPS[@]}"; do
  required_data+=("${sweep_name}.csv")
done
required_data+=(
  all_sweep_results.csv
  trial_level_results.csv
  nominal_summary.csv
  pilot_only_nominal_summary.csv
  runtime_summary.csv
  failure_modes.csv
  usefulness_windows.csv
  fbss_ablation_results.csv
  representative_resource_mask.csv
  representative_scene_geometry.csv
  representative_range_doppler.csv
  representative_music_spectra.csv
  representative_fbss_ablation_spectra.csv
)

if [[ "$SCENE_CLASS" == "all" ]]; then
  required_data+=(scene_comparison.csv)
fi
if [[ "$ANCHOR" == "all" ]]; then
  required_data+=(anchor_comparison.csv)
fi

required_figures=()
for sweep_name in "${ACTIVE_SWEEPS[@]}"; do
  required_figures+=("${sweep_name}.png")
done
required_figures+=(
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
submission_mode="non_submission"
if [[ "$PROFILE" == "submission" ]]; then
  if [[ -n "$TRIALS" ]]; then
    submission_mode="custom_trial_count"
  else
    submission_mode="default_submission"
  fi
fi
{
  echo "timestamp_utc=$timestamp_utc"
  echo "schema_version=$SCHEMA_VERSION"
  echo "evidence_profile=$EVIDENCE_PROFILE"
  echo "estimator_set=$ESTIMATOR_SET"
  echo "fbss_ablation_set=$FBSS_ABLATION_SET"
  echo "knowledge_modes=known_symbols,pilot_only"
  echo "profile=$PROFILE"
  echo "suite=$SUITE"
  echo "anchor=$ANCHOR"
  echo "scenes=$SCENE_CLASS"
  echo "sweeps=$(IFS=,; echo "${ACTIVE_SWEEPS[*]}")"
  echo "jobs=$JOBS"
  echo "trials=${TRIALS:-default}"
  echo "submission_mode=$submission_mode"
  echo "python_bin=$PYTHON_BIN"
  echo "command=${cmd[*]}"
} > "$manifest_path"

echo "Wrote manifest: $manifest_path"
