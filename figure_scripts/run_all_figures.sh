#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_RUNNER=("$PYTHON_BIN")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run --active python)
else
  PYTHON_RUNNER=(python3)
fi

scripts=(
  01_motivation_1d_range.py
  02_nominal_scene_verdict.py
  03_nominal_trial_delta.py
  04_scene_coherence_overlap.py
  05_regime_map.py
  06_rack_aisle_failure_diagnostic.py
  07_nominal_resource_mask.py
  08_representative_intersection_case.py
  09_model_order_nominal_comparison.py
  10_expected_order_nuisance_sweep.py
  11_nominal_runtime_comparison.py
  12_masked_observation_equation.py
  13_music_pseudospectrum_equation.py
)

for script_name in "${scripts[@]}"; do
  echo "Running ${script_name}"
  "${PYTHON_RUNNER[@]}" "$SCRIPT_DIR/$script_name"
done
