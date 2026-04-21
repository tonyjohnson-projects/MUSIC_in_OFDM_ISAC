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


def _parse_sweep_names(profile: str, sweeps_arg: str | None) -> tuple[str, ...]:
    default_sweeps = SUBMISSION_SWEEP_NAMES if profile == "submission" else PUBLIC_SWEEP_NAMES
    if sweeps_arg is None:
        return default_sweeps

    requested = []
    seen = set()
    for raw_name in sweeps_arg.split(","):
        sweep_name = raw_name.strip()
        if not sweep_name or sweep_name in seen:
            continue
        requested.append(sweep_name)
        seen.add(sweep_name)

    if not requested:
        raise SystemExit("--sweeps must include at least one sweep name")

    invalid = sorted(set(requested) - set(PUBLIC_SWEEP_NAMES) - set(SUBMISSION_SWEEP_NAMES))
    if invalid:
        supported = ", ".join(PUBLIC_SWEEP_NAMES)
        raise SystemExit(f"Unsupported sweep name(s): {', '.join(invalid)}. Supported sweeps: {supported}")
    return tuple(requested)


def _resolve_output_root(output_dir: str | None, profile: str) -> Path:
    if output_dir is None:
        return REPO_ROOT / "results" / profile
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


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
        help="Worker-process count for sweep-point evaluation.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override the Monte Carlo trial count for every sweep point.",
    )
    parser.add_argument(
        "--sweeps",
        default=None,
        help="Comma-separated subset of public sweeps to run. Nominal is always evaluated.",
    )
    parser.add_argument(
        "--skip-pilot-only",
        action="store_true",
        help="Skip the pilot-only nominal diagnostic for faster iteration runs.",
    )
    parser.add_argument(
        "--skip-representative",
        action="store_true",
        help="Skip representative-trial artifacts for faster iteration runs.",
    )
    parser.add_argument(
        "--disable-fbss-ablation",
        action="store_true",
        help="Disable FBSS ablation runs on nominal, bandwidth-span, and slow-time-span points.",
    )
    parser.add_argument(
        "--music-model-order",
        default="mdl",
        choices=("mdl", "eigengap", "fixed", "expected"),
        help="MUSIC model-order mode. 'expected' uses the known target count.",
    )
    parser.add_argument(
        "--music-fixed-order",
        type=int,
        default=None,
        help="Fixed MUSIC model order to use when --music-model-order=fixed.",
    )
    parser.add_argument(
        "--skip-local-refinement",
        action="store_true",
        help="Skip local matched-filter refinement for both FFT and MUSIC to isolate coarse-candidate quality.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.music_fixed_order is not None and args.music_fixed_order < 1:
        raise SystemExit("--music-fixed-order must be at least 1")
    if args.music_model_order == "fixed" and args.music_fixed_order is None:
        raise SystemExit("--music-fixed-order is required when --music-model-order=fixed")
    if args.music_model_order == "expected" and args.music_fixed_order is not None:
        raise SystemExit("--music-fixed-order is not compatible with --music-model-order=expected")
    music_model_order_mode = "fixed" if args.music_fixed_order is not None else args.music_model_order

    anchors = ("fr1", "fr2") if args.anchor == "all" else (args.anchor,)
    scenes = ("open_aisle", "rack_aisle", "intersection") if args.scene_class == "all" else (args.scene_class,)
    max_workers = args.jobs if args.jobs is not None else max(1, os.cpu_count() or 1)
    active_sweeps = _parse_sweep_names(args.profile, args.sweeps)
    output_root = _resolve_output_root(args.output_dir, args.profile)

    studies = []
    total_start_time = time.perf_counter()
    total_scene_runs = len(anchors) * len(scenes)
    scene_run_index = 0
    print("Starting study run", flush=True)
    print(f"- profile: {args.profile}", flush=True)
    print(f"- suite: {args.suite}", flush=True)
    print(f"- anchors: {', '.join(anchors)}", flush=True)
    print(f"- scene classes: {', '.join(scenes)}", flush=True)
    print(f"- sweeps: {', '.join(active_sweeps)}", flush=True)
    print(f"- worker processes: {max_workers}", flush=True)
    print(f"- output root: {output_root}", flush=True)
    for anchor_name in anchors:
        for scene_name in scenes:
            scene_run_index += 1
            scene_start_time = time.perf_counter()
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Scene {scene_run_index}/{total_scene_runs} starting: {anchor_name} / {scene_name}",
                flush=True,
            )
            cfg = build_study_config(
                anchor_name,
                scene_name,
                args.profile,
                suite=args.suite,
                trial_count_override=args.trials,
                music_model_order_mode=music_model_order_mode,
                music_fixed_model_order=args.music_fixed_order,
                enable_fbss_ablation=not args.disable_fbss_ablation,
                skip_local_refinement=args.skip_local_refinement,
            )
            studies.append(
                run_communications_study(
                    cfg,
                    show_progress=True,
                    max_workers=max_workers,
                    suite=args.suite,
                    sweep_names=active_sweeps,
                    include_pilot_only=not args.skip_pilot_only,
                    include_representative=not args.skip_representative,
                )
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Scene {scene_run_index}/{total_scene_runs} finished: {anchor_name} / {scene_name} "
                f"in {time.perf_counter() - scene_start_time:.1f} s",
                flush=True,
            )

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

    print("Communications-Limited OFDM ISAC MUSIC Study", flush=True)
    print("-------------------------------------------", flush=True)
    print(f"Profile: {args.profile}", flush=True)
    print(f"Suite: {args.suite}", flush=True)
    print(f"Sweeps: {', '.join(active_sweeps)}", flush=True)
    print(f"Anchors: {', '.join(anchors)}", flush=True)
    print(f"Scene classes: {', '.join(scenes)}", flush=True)
    print(f"Trials per sweep point: {studies[0].nominal_point.method_summaries['fft_masked'].trial_count}", flush=True)
    print(f"Worker processes: {max_workers}", flush=True)
    print(f"MUSIC model order mode: {music_model_order_mode}", flush=True)
    if args.music_fixed_order is not None:
        print(f"MUSIC fixed order: {args.music_fixed_order}", flush=True)
    print(f"Total runtime: {total_runtime_s:.1f} s", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print("Generated CSVs:", flush=True)
    csv_filenames = [f"{sweep_name}.csv" for sweep_name in active_sweeps]
    csv_filenames.extend(
        [
            "all_sweep_results.csv",
            "trial_level_results.csv",
            "nominal_summary.csv",
            "runtime_summary.csv",
            "failure_modes.csv",
            "usefulness_windows.csv",
            "stage_diagnostics.csv",
        ]
    )
    if not args.skip_pilot_only:
        csv_filenames.append("pilot_only_nominal_summary.csv")
    if not args.disable_fbss_ablation:
        csv_filenames.append("fbss_ablation_results.csv")
    if not args.skip_representative:
        csv_filenames.extend(
            [
                "representative_resource_mask.csv",
                "representative_scene_geometry.csv",
                "representative_range_doppler.csv",
                "representative_music_spectra.csv",
            ]
        )
        if not args.disable_fbss_ablation:
            csv_filenames.append("representative_fbss_ablation_spectra.csv")
    if include_scene_comparison:
        csv_filenames.append("scene_comparison.csv")
    if include_anchor_comparison:
        csv_filenames.append("anchor_comparison.csv")
    for filename in csv_filenames:
        print(f"- {output_root / 'data' / filename}", flush=True)
    print("Generated figures:", flush=True)
    figure_filenames = [f"{sweep_name}.png" for sweep_name in active_sweeps]
    figure_filenames.append("runtime_summary.png")
    if not args.skip_representative:
        figure_filenames.extend(["representative_resource_mask.png", "representative_spectrum.png"])
    for filename in figure_filenames:
        print(f"- {output_root / 'figures' / filename}", flush=True)
    print(f"CSV-driven plotting script: {REPO_ROOT / 'scripts' / 'plot_results_from_csv.py'}", flush=True)

    first_study = studies[0]
    print(flush=True)
    print(f"Nominal summary: {first_study.anchor_label} / {first_study.scene_label}", flush=True)
    for method_name in METHOD_ORDER:
        summary = first_study.nominal_point.method_summaries[method_name]
        print(
            f"- {method_name}: Pdet={summary.joint_detection_probability:.2f}, "
            f"Pres={summary.joint_resolution_probability:.2f}, "
            f"Prange={summary.range_resolution_probability:.2f}, "
            f"Pvel={summary.velocity_resolution_probability:.2f}, "
            f"Pangle={summary.angle_resolution_probability:.2f}, "
            f"uncond_joint_rmse={summary.unconditional_joint_assignment_rmse:.3f}, "
            f"total_runtime={summary.total_runtime_s:.4f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
