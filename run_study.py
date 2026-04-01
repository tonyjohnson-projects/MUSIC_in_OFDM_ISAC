"""CLI entrypoint for the private-5G angle-range-Doppler study."""

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

from aisle_isac.reporting import write_all_outputs
from aisle_isac.scenarios import build_study_config
from aisle_isac.study import METHOD_ORDER, PUBLIC_SWEEP_NAMES, run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private-5G angle-range-Doppler study.")
    parser.add_argument(
        "--anchor",
        default="all",
        choices=("fr1", "fr2", "all"),
        help="Waveform anchor selection.",
    )
    parser.add_argument(
        "--scene-class",
        default="all",
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
    max_workers = args.jobs if args.jobs is not None else max(1, min(4, os.cpu_count() or 1))

    studies = []
    total_start_time = time.perf_counter()
    for anchor_name in anchors:
        for scene_name in scenes:
            cfg = build_study_config(
                anchor_name,
                scene_name,
                args.profile,
                suite=args.suite,
                trial_count_override=args.trials,
            )
            studies.append(run_study(cfg, show_progress=True, max_workers=max_workers, suite=args.suite))

    output_root = REPO_ROOT / "results" / args.profile
    include_fr1_vs_fr2 = len(anchors) > 1
    write_all_outputs(
        studies,
        output_root,
        clean_outputs=args.clean_outputs,
        sweep_names=PUBLIC_SWEEP_NAMES,
        include_scene_comparison=True,
        include_fr1_vs_fr2=include_fr1_vs_fr2,
        include_crb_gap=True,
        include_representative_cube_slices=True,
    )
    total_runtime_s = time.perf_counter() - total_start_time

    print("Private-5G Angle-Range-Doppler Study")
    print("------------------------------------")
    print(f"Profile: {args.profile}")
    print(f"Suite: {args.suite}")
    print(f"Anchors: {', '.join(anchors)}")
    print(f"Scene classes: {', '.join(scenes)}")
    print(f"Trials per sweep point: {studies[0].nominal_point.method_summaries['fft'].trial_count}")
    print(f"Worker processes: {max_workers}")
    print(f"Total runtime: {total_runtime_s:.1f} s")
    print(f"Output root: {output_root}")
    print("Generated CSVs:")
    csv_filenames = [f"{sweep_name}.csv" for sweep_name in PUBLIC_SWEEP_NAMES]
    csv_filenames.extend(["scene_comparison.csv", "crb_gap.csv"])
    if include_fr1_vs_fr2:
        csv_filenames.append("fr1_vs_fr2.csv")
    for filename in csv_filenames:
        print(f"- {output_root / 'data' / filename}")
    print("Generated figures:")
    figure_filenames = [f"{sweep_name}.png" for sweep_name in PUBLIC_SWEEP_NAMES]
    figure_filenames.extend(["scene_comparison.png", "crb_gap.png", "representative_cube_slices.png"])
    if include_fr1_vs_fr2:
        figure_filenames.append("fr1_vs_fr2.png")
    for filename in figure_filenames:
        print(f"- {output_root / 'figures' / filename}")

    first_study = studies[0]
    print()
    print(f"Nominal summary: {first_study.anchor_label} / {first_study.scene_label}")
    for method_name in METHOD_ORDER:
        summary = first_study.nominal_point.method_summaries[method_name]
        print(
            f"- {method_name}: Pdet={summary.joint_detection_probability:.2f}, "
            f"Pres={summary.joint_resolution_probability:.2f}, "
            f"scene_cost={summary.scene_cost:.3f}, "
            f"uncond_joint_rmse={summary.unconditional_joint_assignment_rmse:.3f}, "
            f"uncond_rmse_over_crb={summary.unconditional_rmse_over_crb:.3f}, "
            f"total_runtime={summary.total_runtime_s:.4f}s"
        )


if __name__ == "__main__":
    main()
