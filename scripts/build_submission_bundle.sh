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

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"])
sys.path.insert(0, str(root_dir / "src"))

from aisle_isac.reporting import write_all_outputs
from aisle_isac.scenarios import build_study_config
from aisle_isac.study import SUBMISSION_SWEEP_NAMES, run_study

trial_override = os.environ["TRIALS"].strip()
jobs = int(os.environ["JOBS"])
clean_outputs = os.environ["CLEAN_OUTPUTS"] == "1"

studies = []
for scene_name in ("open_aisle", "rack_aisle"):
    cfg = build_study_config(
        "fr1",
        scene_name,
        "submission",
        suite="headline",
        trial_count_override=int(trial_override) if trial_override else None,
    )
    studies.append(
        run_study(
            cfg,
            show_progress=True,
            max_workers=jobs,
            suite="headline",
            sweep_names=SUBMISSION_SWEEP_NAMES,
        )
    )

write_all_outputs(
    studies,
    root_dir / "results" / "submission",
    clean_outputs=clean_outputs,
    sweep_names=SUBMISSION_SWEEP_NAMES,
    include_scene_comparison=True,
    include_fr1_vs_fr2=False,
    include_crb_gap=True,
    include_representative_cube_slices=True,
)
PY

OUTPUT_ROOT="$ROOT_DIR/results/submission"
DATA_DIR="$OUTPUT_ROOT/data"
FIGURES_DIR="$OUTPUT_ROOT/figures"

required_data=(
  range_separation.csv
  velocity_separation.csv
  angle_separation.csv
  burst_profile.csv
  aperture.csv
  scene_comparison.csv
  crb_gap.csv
)

required_figures=(
  range_separation.png
  velocity_separation.png
  angle_separation.png
  burst_profile.png
  aperture.png
  scene_comparison.png
  crb_gap.png
  representative_cube_slices.png
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
  echo "scenes=open_aisle,rack_aisle"
  echo "sweeps=range_separation,velocity_separation,angle_separation,burst_profile,aperture"
  echo "jobs=$JOBS"
  echo "trials=${TRIALS:-default}"
  echo "python_bin=$PYTHON_BIN"
  echo "command=bash scripts/build_submission_bundle.sh"
} > "$manifest_path"

echo "Wrote manifest: $manifest_path"
