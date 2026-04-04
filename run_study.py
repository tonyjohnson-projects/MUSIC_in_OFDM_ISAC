"""CLI entrypoint for the communications-limited MUSIC study."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.scheduled_reporting import write_all_outputs
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import (
    METHOD_ORDER,
    PUBLIC_SWEEP_NAMES,
    SUBMISSION_SWEEP_NAMES,
    run_communications_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the communications-limited OFDM ISAC MUSIC study.")
    parser.add_argument(
        "--anchor",
        default="all",
        choices=("fr1", "fr2", "all"),
        help="Waveform anchor selection.",
    )
    parser.add_argument(
        "--scene-class",
        default="open_aisle",
        choices=("open_aisle", "rack_aisle", "intersection", "all"),
        help="Scene class selection.",
    )
    parser.add_argument(
        "--profile",
        default="quick",
        choices=("quick", "submission"),
        help="Runtime profile to use.",
    )
    parser.add_argument(
        "--suite",
        default="headline",
        choices=("headline", "full"),
        help="Study sweep suite to run.",
    )
    parser.add_argument(
        "--clean-outputs",
        action="store_true",
        help="Remove stale generated outputs before writing the new result tree.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker-process count for sweep-point parallelism.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override the Monte Carlo trial count for every sweep point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    anchors = ("fr1", "fr2") if args.anchor == "all" else (args.anchor,)
    scenes = ("open_aisle", "rack_aisle", "intersection") if args.scene_class == "all" else (args.scene_class,)
    max_workers = args.jobs if args.jobs is not None else max(1, os.cpu_count() or 1)

    studies = []
    total_start_time = time.perf_counter()
    active_sweeps = SUBMISSION_SWEEP_NAMES if args.profile == "submission" else PUBLIC_SWEEP_NAMES
    for anchor_name in anchors:
        for scene_name in scenes:
            cfg = build_study_config(
                anchor_name,
                scene_name,
                args.profile,
                suite=args.suite,
                trial_count_override=args.trials,
            )
            studies.append(
                run_communications_study(
                    cfg,
                    show_progress=True,
                    max_workers=max_workers,
                    suite=args.suite,
                    sweep_names=active_sweeps,
                )
            )

    output_root = REPO_ROOT / "results" / args.profile
    include_anchor_comparison = len(anchors) > 1
    include_scene_comparison = len(scenes) > 1
    write_all_outputs(
        studies,
        output_root,
        clean_outputs=args.clean_outputs,
        sweep_names=active_sweeps,
        include_scene_comparison=include_scene_comparison,
        include_anchor_comparison=include_anchor_comparison,
    )
    total_runtime_s = time.perf_counter() - total_start_time

    print("Communications-Limited OFDM ISAC MUSIC Study")
    print("-------------------------------------------")
    print(f"Profile: {args.profile}")
    print(f"Suite: {args.suite}")
    print(f"Anchors: {', '.join(anchors)}")
    print(f"Scene classes: {', '.join(scenes)}")
    print(f"Trials per sweep point: {studies[0].nominal_point.method_summaries['fft_masked'].trial_count}")
    print(f"Worker processes: {max_workers}")
    print(f"Total runtime: {total_runtime_s:.1f} s")
    print(f"Output root: {output_root}")
    print("Generated CSVs:")
    csv_filenames = [f"{sweep_name}.csv" for sweep_name in active_sweeps]
    csv_filenames.extend(
        [
            "all_sweep_results.csv",
            "trial_level_results.csv",
            "nominal_summary.csv",
            "pilot_only_nominal_summary.csv",
            "runtime_summary.csv",
            "failure_modes.csv",
            "usefulness_windows.csv",
            "fbss_ablation_results.csv",
            "representative_resource_mask.csv",
            "representative_scene_geometry.csv",
            "representative_range_doppler.csv",
            "representative_music_spectra.csv",
            "representative_fbss_ablation_spectra.csv",
        ]
    )
    if include_scene_comparison:
        csv_filenames.append("scene_comparison.csv")
    if include_anchor_comparison:
        csv_filenames.append("anchor_comparison.csv")
    for filename in csv_filenames:
        print(f"- {output_root / 'data' / filename}")
    print("Generated figures:")
    figure_filenames = [f"{sweep_name}.png" for sweep_name in active_sweeps]
    figure_filenames.extend(["runtime_summary.png", "representative_resource_mask.png", "representative_spectrum.png"])
    for filename in figure_filenames:
        print(f"- {output_root / 'figures' / filename}")
    print(f"CSV-driven plotting script: {REPO_ROOT / 'scripts' / 'plot_results_from_csv.py'}")

    first_study = studies[0]
    print()
    print(f"Nominal summary: {first_study.anchor_label} / {first_study.scene_label}")
    for method_name in METHOD_ORDER:
        summary = first_study.nominal_point.method_summaries[method_name]
        print(
            f"- {method_name}: Pdet={summary.joint_detection_probability:.2f}, "
            f"Pres={summary.joint_resolution_probability:.2f}, "
            f"Prange={summary.range_resolution_probability:.2f}, "
            f"Pvel={summary.velocity_resolution_probability:.2f}, "
            f"Pangle={summary.angle_resolution_probability:.2f}, "
            f"uncond_joint_rmse={summary.unconditional_joint_assignment_rmse:.3f}, "
            f"total_runtime={summary.total_runtime_s:.4f}s"
        )


if __name__ == "__main__":
    main()
