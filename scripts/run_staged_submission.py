#!/usr/bin/env python3
"""Run the full 64-trial submission experiments one scene at a time.

Each stage saves its results independently so the process can be
interrupted between scenes without losing completed work.

Stages (in order):
  1. Model-order comparison (MDL vs expected-order) — 3 scenes × 2 modes
  2. Nuisance sweep (MDL)                          — 3 scenes
  3. Nuisance sweep (expected-order)                — 3 scenes

Total: 12 stage steps.  Each can be skipped via --skip-* flags or by
passing --start-from <step_number> to resume after an interruption.
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import os

from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import run_communications_study
from aisle_isac.scheduled_reporting import write_all_outputs

SCENES = ("open_aisle", "intersection", "rack_aisle")
TRIALS = 64
PROFILE = "submission"
ANCHOR = "fr1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step: int, total: int, label: str) -> None:
    ts = time.strftime("%H:%M:%S")
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  [{ts}]  Step {step}/{total}: {label}")
    print(f"{bar}\n", flush=True)


def _elapsed(t0: float) -> str:
    dt = time.perf_counter() - t0
    m, s = divmod(int(dt), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Stage 1: Model-order comparison (MDL vs expected-order nominal)
# ---------------------------------------------------------------------------

def run_model_order_comparison(step_offset: int, total: int) -> int:
    """Run nominal MDL + expected-order for each scene. Returns steps used."""
    output = REPO / "results" / "analysis" / "model_order_nominal_64trials.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check for partial results from a previous run.
    existing_rows: list[dict] = []
    done_keys: set[tuple[str, str]] = set()
    if output.exists():
        with open(output) as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                done_keys.add((row["scene"], row["music_model_order_mode"]))

    modes = [("mdl", None), ("expected", None)]
    step = 0
    rows = list(existing_rows)

    for scene in SCENES:
        for mode, fixed_order in modes:
            step += 1
            if (scene, mode) in done_keys:
                print(f"  [{time.strftime('%H:%M:%S')}]  Skipping {scene}/{mode} — already in {output.name}")
                continue
            _banner(step_offset + step, total, f"Model-order comparison: {scene} / {mode}")
            t0 = time.perf_counter()
            cfg = build_study_config(
                ANCHOR, scene, PROFILE,
                trial_count_override=TRIALS,
                music_model_order_mode=mode,
                music_fixed_model_order=fixed_order,
            )
            study = run_communications_study(
                cfg, show_progress=True,
                suite="headline",
                sweep_names=(),
                include_pilot_only=False,
                include_representative=False,
            )
            nom = study.nominal_point
            fft = nom.method_summaries["fft_masked"]
            mus = nom.method_summaries["music_masked"]
            rows.append({
                "scene": scene,
                "music_model_order_mode": mode,
                "fft_joint_pres": f"{fft.joint_resolution_probability:.6f}",
                "music_joint_pres": f"{mus.joint_resolution_probability:.6f}",
                "fft_joint_pdet": f"{fft.joint_detection_probability:.6f}",
                "music_joint_pdet": f"{mus.joint_detection_probability:.6f}",
                "music_mean_order": f"{mus.mean_estimated_model_order:.1f}" if mus.mean_estimated_model_order is not None else "",
            })
            # Write after every scene/mode so progress is never lost.
            with open(output, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  [{time.strftime('%H:%M:%S')}]  {scene}/{mode} done in {_elapsed(t0)}")
            print(f"    FFT  Pres={fft.joint_resolution_probability:.3f}  Pdet={fft.joint_detection_probability:.3f}")
            print(f"    MUSIC Pres={mus.joint_resolution_probability:.3f}  Pdet={mus.joint_detection_probability:.3f}  mean_K={mus.mean_estimated_model_order or 0:.1f}")
            del study, nom, fft, mus, cfg
            gc.collect()

    return step


# ---------------------------------------------------------------------------
# Stage 2 & 3: Nuisance sweeps (MDL / expected-order)
# ---------------------------------------------------------------------------

def run_nuisance_sweep(step_offset: int, total: int, mode: str, output_dir: str) -> int:
    """Run nuisance_gain_offset sweep one scene at a time. Returns steps used."""
    out_root = REPO / output_dir
    step = 0

    for scene in SCENES:
        step += 1
        scene_dir = out_root / scene
        marker = scene_dir / ".done"
        if marker.exists():
            print(f"  [{time.strftime('%H:%M:%S')}]  Skipping nuisance/{mode}/{scene} — already complete")
            continue

        _banner(step_offset + step, total, f"Nuisance sweep ({mode}): {scene}")
        t0 = time.perf_counter()

        cfg = build_study_config(
            ANCHOR, scene, PROFILE,
            trial_count_override=TRIALS,
            music_model_order_mode=mode,
            enable_fbss_ablation=False,
        )
        study = run_communications_study(
            cfg, show_progress=True,
            sweep_names=("nuisance_gain_offset",),
            include_pilot_only=False,
            include_representative=False,
        )
        # Write per-scene outputs.
        scene_dir.mkdir(parents=True, exist_ok=True)
        write_all_outputs(
            [study],
            scene_dir,
            clean_outputs=True,
            sweep_names=("nuisance_gain_offset",),
            include_scene_comparison=False,
            include_anchor_comparison=False,
        )
        marker.touch()
        print(f"  [{time.strftime('%H:%M:%S')}]  nuisance/{mode}/{scene} done in {_elapsed(t0)}")
        print(f"    Results in {scene_dir}")
        del study, cfg
        gc.collect()

    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Staged 64-trial submission runner")
    parser.add_argument("--start-from", type=int, default=1, help="Resume from this step number (1-indexed)")
    parser.add_argument("--skip-model-order", action="store_true", help="Skip model-order comparison stage")
    parser.add_argument("--skip-nuisance-mdl", action="store_true", help="Skip nuisance sweep (MDL)")
    parser.add_argument("--skip-nuisance-expected", action="store_true", help="Skip nuisance sweep (expected-order)")
    args = parser.parse_args()

    stages = []
    if not args.skip_model_order:
        stages.append(("Model-order comparison (3 scenes × 2 modes)", 6, "model_order"))
    if not args.skip_nuisance_mdl:
        stages.append(("Nuisance sweep — MDL (3 scenes)", 3, "nuisance_mdl"))
    if not args.skip_nuisance_expected:
        stages.append(("Nuisance sweep — expected-order (3 scenes)", 3, "nuisance_expected"))

    total_steps = sum(s[1] for s in stages)

    print("=" * 60)
    print("  Staged Submission Runner")
    print(f"  Profile: {PROFILE}  Anchor: {ANCHOR}  Trials: {TRIALS}")
    print(f"  Total steps: {total_steps}")
    print(f"  Start from: step {args.start_from}")
    print()
    for i, (label, n, _) in enumerate(stages):
        offset = sum(s[1] for s in stages[:i])
        print(f"    Steps {offset+1}-{offset+n}: {label}")
    print("=" * 60)
    print(f"  [{time.strftime('%H:%M:%S')}]  Starting...\n", flush=True)

    global_t0 = time.perf_counter()
    step_offset = 0

    for label, n, tag in stages:
        # Allow skipping ahead to a specific step.
        if step_offset + n < args.start_from:
            step_offset += n
            continue

        if tag == "model_order":
            run_model_order_comparison(step_offset, total_steps)
        elif tag == "nuisance_mdl":
            run_nuisance_sweep(step_offset, total_steps, "mdl", "results/submission_nuisance")
        elif tag == "nuisance_expected":
            run_nuisance_sweep(step_offset, total_steps, "expected", "results/submission_expected_order")
        step_offset += n

    print(f"\n{'=' * 60}")
    print(f"  [{time.strftime('%H:%M:%S')}]  All stages complete in {_elapsed(global_t0)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
