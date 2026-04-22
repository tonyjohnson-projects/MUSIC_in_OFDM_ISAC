#!/usr/bin/env python3
"""Generate figures/08_representative_intersection_case.png."""

from __future__ import annotations

from collections import defaultdict

from matplotlib.lines import Line2D
from matplotlib.patheffects import Normal, Stroke

from common import METHOD_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, save_figure, trim_figure_whitespace
from aisle_isac.estimators import _estimate_music_model_order, azimuth_steering_matrix, fbss_covariance, fft_search_bounds, music_pseudospectrum
from aisle_isac.masked_observation import extract_known_symbol_cube
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
    cfg, trial = reconstruct_nominal_trial("intersection", trial_index)

    truth_targets = trial.masked_observation.snapshot.scenario.targets
    fft_detections = trial.estimates["fft_masked"].detections
    music_detections = trial.estimates["music_masked"].detections

    velocity_axis = trial.fft_cube.velocity_axis_mps
    range_axis = trial.fft_cube.range_axis_m
    velocity_min, velocity_max = float(velocity_axis[0]), float(velocity_axis[-1])
    range_min, range_max = float(range_axis[0]), float(range_axis[-1])
    velocity_span, range_span = velocity_max - velocity_min, range_max - range_min
    range_doppler_db = 10.0 * np.log10(np.maximum(np.max(trial.fft_cube.power_cube, axis=0), 1.0e-12))

    known_cube = extract_known_symbol_cube(trial.masked_observation)
    global_matrix = known_cube.reshape(known_cube.shape[0], -1)
    spatial_cov = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
    search_bounds = fft_search_bounds(trial.fft_cube)
    estimated_model_order = _estimate_music_model_order(spatial_cov, global_matrix.shape[1], cfg)
    spectrum_target_order = max(1, cfg.expected_target_count, estimated_model_order)
    azimuth_grid = np.linspace(
        max(-80.0, search_bounds.azimuth_min_deg + 0.5),
        min(80.0, search_bounds.azimuth_max_deg - 0.5),
        cfg.runtime_profile.music_grid_points * 3,
    )
    steering_matrix = azimuth_steering_matrix(cfg.effective_horizontal_positions_m[: cfg.fbss_subarray_len], azimuth_grid, cfg.wavelength_m)
    azimuth_spectrum = music_pseudospectrum(spatial_cov, n_targets=spectrum_target_order, steering_matrix=steering_matrix)
    azimuth_spectrum_db = 10.0 * np.log10(np.maximum(azimuth_spectrum / np.max(azimuth_spectrum), 1.0e-12))
    fft_azimuth_grid = trial.fft_cube.azimuth_axis_deg
    fft_azimuth_spectrum = np.max(trial.fft_cube.power_cube, axis=(1, 2))
    fft_azimuth_spectrum_db = 10.0 * np.log10(np.maximum(fft_azimuth_spectrum / np.max(fft_azimuth_spectrum), 1.0e-12))

    truth_azimuths = sorted(target.azimuth_deg for target in truth_targets)

    fig = plt.figure(figsize=(13.2, 5.6))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.40)
    ax_heatmap = fig.add_subplot(grid[0, 0])
    ax_azimuth = fig.add_subplot(grid[0, 1])

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
        "FFT false branch",
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
        "MUSIC lands on\nboth movers",
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

    ax_heatmap.set_title(f"Saved nominal intersection trial {trial_index}", loc="left", fontsize=13, fontweight="bold")
    ax_heatmap.set_xlabel("Velocity (m/s)")
    ax_heatmap.set_ylabel("Range (m)")
    heatmap_legend = ax_heatmap.legend(
        handles=[truth_scatter, fft_scatter, music_scatter],
        loc="lower left", fontsize=9.5, handletextpad=0.5, borderpad=0.5, framealpha=0.88, facecolor="white", edgecolor="#CCCCCC",
    )
    heatmap_legend.set_zorder(8)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03, label="FFT range-Doppler power (dB)")

    fft_color = METHOD_COLORS["fft_masked"]
    music_color = METHOD_COLORS["music_masked"]
    truth_color = "#1E8A57"
    fft_line, = ax_azimuth.plot(
        fft_azimuth_grid,
        fft_azimuth_spectrum_db,
        color=fft_color,
        linewidth=1.9,
        linestyle=(0, (5.0, 2.0)),
        alpha=0.9,
        label="FFT spectrum",
        zorder=3,
    )
    music_line, = ax_azimuth.plot(
        azimuth_grid,
        azimuth_spectrum_db,
        color=music_color,
        linewidth=2.4,
        label="MUSIC spectrum",
        zorder=4,
    )

    azimuth_min, azimuth_max = min(truth_azimuths), max(truth_azimuths)
    azimuth_margin = max(8.0, 1.5 * (azimuth_max - azimuth_min))
    ax_azimuth.set_xlim(azimuth_min - azimuth_margin, azimuth_max + azimuth_margin)
    ax_azimuth.set_ylim(
        min(-42.0, float(np.min(fft_azimuth_spectrum_db)), float(np.min(azimuth_spectrum_db)) - 1.0),
        2.0,
    )

    for azimuth in truth_azimuths:
        ax_azimuth.axvspan(azimuth - 0.25, azimuth + 0.25, color=truth_color, alpha=0.28, zorder=1)
        ax_azimuth.axvline(azimuth, color=truth_color, linewidth=2.0, alpha=0.95, zorder=2)

    ax_azimuth.set_xlabel("Azimuth (deg)")
    ax_azimuth.set_ylabel("Relative level (dB)")
    ax_azimuth.set_title("Marginal azimuth spectra", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax_azimuth.grid(True, alpha=0.22)

    truth_handle = Line2D([0], [0], color=truth_color, linewidth=3.0, alpha=0.95, label="Truth azimuth")
    ax_azimuth.legend(
        handles=[fft_line, music_line, truth_handle],
        loc="lower left", fontsize=9, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", handlelength=2.8, handletextpad=0.6,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser("Generate the representative intersection-case figure.", "08_representative_intersection_case.png")
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
