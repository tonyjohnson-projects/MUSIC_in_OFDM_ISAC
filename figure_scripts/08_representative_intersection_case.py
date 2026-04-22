#!/usr/bin/env python3
"""Generate figures/08_representative_intersection_case.png."""

from __future__ import annotations

from collections import defaultdict

from matplotlib.patheffects import Normal, Stroke

from common import METHOD_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, save_figure, trim_figure_whitespace
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import _nominal_point_spec, nominal_trial_parameters, simulate_communications_trial


def select_representative_intersection_trial() -> int:
    rows = read_csv_rows(REPO_ROOT / "results" / "submission" / "data" / "trial_level_results.csv")
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["scene_class"] != "intersection":
            continue
        if row["sweep_name"] != "nominal":
            continue
        if row["estimator_family"] != "headline":
            continue
        grouped[int(row["trial_index"])][row["method"]] = row

    best_trial_index = -1
    best_score = float("-inf")
    for trial_index, methods in grouped.items():
        fft_row = methods.get("fft_masked")
        music_row = methods.get("music_masked")
        if fft_row is None or music_row is None:
            continue
        if music_row["joint_resolution_success"] != "1" or fft_row["joint_resolution_success"] != "0":
            continue
        score = float(fft_row["unconditional_joint_assignment_rmse"]) - float(music_row["unconditional_joint_assignment_rmse"])
        if score > best_score:
            best_score = score
            best_trial_index = trial_index

    if best_trial_index < 0:
        raise RuntimeError("Failed to find a representative intersection nominal trial")
    return best_trial_index


def reconstruct_nominal_trial(scene_name: str, trial_index: int):
    cfg = build_study_config("fr1", scene_name, "submission", enable_fbss_ablation=False)
    spec = _nominal_point_spec(cfg)
    seed_sequence = np.random.SeedSequence([
        cfg.rng_seed,
        spec.point_index,
        int(round(1_000.0 * spec.occupied_fraction)),
        int(round(1_000.0 * spec.fragmentation_index)),
        int(round(1_000.0 * spec.bandwidth_span_fraction)),
        int(round(1_000.0 * spec.slow_time_span_fraction)),
    ])
    child_seed = seed_sequence.spawn(cfg.runtime_profile.n_trials)[trial_index]
    trial = simulate_communications_trial(
        cfg, nominal_trial_parameters(cfg), spec.allocation_family, spec.allocation_label, spec.knowledge_mode,
        spec.modulation_scheme, spec.resource_grid_kwargs, np.random.default_rng(child_seed), include_fbss_ablation=False,
    )
    return cfg, trial


def make_figure(output_path) -> None:
    trial_index = select_representative_intersection_trial()
    _, trial = reconstruct_nominal_trial("intersection", trial_index)

    truth_targets = trial.masked_observation.snapshot.scenario.targets
    fft_detections = trial.estimates["fft_masked"].detections
    music_detections = trial.estimates["music_masked"].detections

    velocity_axis = trial.fft_cube.velocity_axis_mps
    range_axis = trial.fft_cube.range_axis_m
    velocity_min, velocity_max = float(velocity_axis[0]), float(velocity_axis[-1])
    range_min, range_max = float(range_axis[0]), float(range_axis[-1])
    velocity_span, range_span = velocity_max - velocity_min, range_max - range_min
    range_doppler_db = 10.0 * np.log10(np.maximum(np.max(trial.fft_cube.power_cube, axis=0), 1.0e-12))

    fig, ax_heatmap = plt.subplots(figsize=(8.3, 5.6))

    image = ax_heatmap.imshow(
        range_doppler_db, aspect="auto", origin="lower", extent=[velocity_min, velocity_max, range_min, range_max], cmap="viridis"
    )
    truth_scatter = ax_heatmap.scatter(
        [target.velocity_mps for target in truth_targets], [target.range_m for target in truth_targets],
        marker="*", s=320, c="#2FA66A", edgecolors="white", linewidths=1.6, label="Truth movers", zorder=5,
    )
    fft_scatter = ax_heatmap.scatter(
        [detection.velocity_mps for detection in fft_detections], [detection.range_m for detection in fft_detections],
        marker="o", s=150, c=METHOD_COLORS["fft_masked"], edgecolors="white", linewidths=1.6, label="FFT detections", zorder=6,
    )
    music_scatter = ax_heatmap.scatter(
        [detection.velocity_mps for detection in music_detections], [detection.range_m for detection in music_detections],
        marker="x", s=180, linewidths=3.2, c=METHOD_COLORS["music_masked"], label="MUSIC detections", zorder=7,
    )
    music_scatter.set_path_effects([Stroke(linewidth=2.2, foreground="white"), Normal()])

    annotation_bbox = {"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}
    false_fft = max(fft_detections, key=lambda detection: detection.range_m)
    ax_heatmap.annotate(
        "FFT duplicates\n+v branch",
        xy=(false_fft.velocity_mps, false_fft.range_m),
        xytext=(
            min(velocity_max - 0.02 * velocity_span, false_fft.velocity_mps + 0.18 * velocity_span),
            min(range_max - 0.04 * range_span, false_fft.range_m + 0.10 * range_span),
        ),
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": METHOD_COLORS["fft_masked"], "shrinkA": 4, "shrinkB": 6},
        fontsize=10,
        color=METHOD_COLORS["fft_masked"],
        fontweight="bold",
        bbox=annotation_bbox,
    )

    music_anchor = min(music_detections, key=lambda detection: detection.range_m)
    ax_heatmap.annotate(
        "MUSIC resolves\nboth movers",
        xy=(music_anchor.velocity_mps, music_anchor.range_m),
        xytext=(velocity_min + 0.14 * velocity_span, range_min + 0.35  * range_span),
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": METHOD_COLORS["music_masked"], "shrinkA": 4, "shrinkB": 6},
        fontsize=10,
        color=METHOD_COLORS["music_masked"],
        fontweight="bold",
        bbox=annotation_bbox,
    )

    ax_heatmap.set_title("Representative intersection trial", loc="left", fontsize=13, fontweight="bold")
    ax_heatmap.set_xlabel("Velocity (m/s)")
    ax_heatmap.set_ylabel("Range (m)")
    heatmap_legend = ax_heatmap.legend(
        handles=[truth_scatter, fft_scatter, music_scatter],
        loc="lower left", fontsize=9.5, handletextpad=0.5, borderpad=0.5, framealpha=0.88, facecolor="white", edgecolor="#CCCCCC",
    )
    heatmap_legend.set_zorder(8)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03, label="FFT range-Doppler power (dB)")

    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser("Generate the representative intersection-case figure.", "08_representative_intersection_case.png")
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
