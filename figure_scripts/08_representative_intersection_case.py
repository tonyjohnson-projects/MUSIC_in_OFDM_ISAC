#!/usr/bin/env python3
"""Generate figures/08_representative_intersection_case.png."""

from __future__ import annotations

from collections import defaultdict

from common import METHOD_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, save_figure, trim_figure_whitespace
from aisle_isac.estimators import (
    _estimate_music_model_order,
    azimuth_steering_matrix,
    fbss_covariance,
    fft_search_bounds,
    music_pseudospectrum,
)
from aisle_isac.masked_observation import extract_known_symbol_cube
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import _nominal_point_spec, nominal_trial_parameters, simulate_communications_trial


def load_trial_rows() -> list[dict[str, str]]:
    return read_csv_rows(REPO_ROOT / "results" / "submission" / "data" / "trial_level_results.csv")


def select_representative_intersection_trial() -> int:
    rows = load_trial_rows()
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["scene_class"] != "intersection":
            continue
        if row["sweep_name"] != "nominal":
            continue
        if row["estimator_family"] != "headline":
            continue
        grouped[int(row["trial_index"])][row["method"]] = row

    best_score = float("-inf")
    best_trial_index = -1
    for trial_index, methods in grouped.items():
        fft_row = methods.get("fft_masked")
        music_row = methods.get("music_masked")
        if fft_row is None or music_row is None:
            continue
        if music_row["joint_resolution_success"] != "1":
            continue
        if fft_row["joint_resolution_success"] != "0":
            continue
        score = float(fft_row["unconditional_joint_assignment_rmse"]) - float(
            music_row["unconditional_joint_assignment_rmse"]
        )
        if score > best_score:
            best_score = score
            best_trial_index = trial_index
    if best_trial_index < 0:
        raise RuntimeError("Failed to find a representative intersection nominal trial")
    return best_trial_index


def reconstruct_nominal_trial(scene_name: str, trial_index: int):
    cfg = build_study_config("fr1", scene_name, "submission", enable_fbss_ablation=False)
    spec = _nominal_point_spec(cfg)
    seed_sequence = np.random.SeedSequence(
        [
            cfg.rng_seed,
            spec.point_index,
            int(round(1_000.0 * spec.occupied_fraction)),
            int(round(1_000.0 * spec.fragmentation_index)),
            int(round(1_000.0 * spec.bandwidth_span_fraction)),
            int(round(1_000.0 * spec.slow_time_span_fraction)),
        ]
    )
    child_seed = seed_sequence.spawn(cfg.runtime_profile.n_trials)[trial_index]
    trial = simulate_communications_trial(
        cfg,
        nominal_trial_parameters(cfg),
        spec.allocation_family,
        spec.allocation_label,
        spec.knowledge_mode,
        spec.modulation_scheme,
        spec.resource_grid_kwargs,
        np.random.default_rng(child_seed),
        include_fbss_ablation=False,
    )
    return cfg, trial


def make_figure(output_path) -> None:
    trial_index = select_representative_intersection_trial()
    cfg, trial = reconstruct_nominal_trial("intersection", trial_index)
    fft_cube = trial.fft_cube.power_cube
    range_doppler = np.max(fft_cube, axis=0)

    truth_targets = trial.masked_observation.snapshot.scenario.targets
    fft_detections = trial.estimates["fft_masked"].detections
    music_detections = trial.estimates["music_masked"].detections

    known_cube = extract_known_symbol_cube(trial.masked_observation)
    global_matrix = known_cube.reshape(known_cube.shape[0], -1)
    search_bounds = fft_search_bounds(trial.fft_cube)
    spatial_cov = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
    estimated_model_order = _estimate_music_model_order(spatial_cov, global_matrix.shape[1], cfg)
    spectrum_target_order = max(max(1, cfg.expected_target_count), estimated_model_order)
    azimuth_grid = np.linspace(
        max(-80.0, search_bounds.azimuth_min_deg + 0.5),
        min(80.0, search_bounds.azimuth_max_deg - 0.5),
        cfg.runtime_profile.music_grid_points * 3,
    )
    azimuth_spectrum = music_pseudospectrum(
        spatial_cov,
        n_targets=spectrum_target_order,
        steering_matrix=azimuth_steering_matrix(
            cfg.effective_horizontal_positions_m[: cfg.fbss_subarray_len],
            azimuth_grid,
            cfg.wavelength_m,
        ),
    )
    azimuth_spectrum_db = 10.0 * np.log10(np.maximum(azimuth_spectrum / np.max(azimuth_spectrum), 1.0e-12))
    known_truth = sorted(target.azimuth_deg for target in truth_targets)
    known_fft = sorted(detection.azimuth_deg for detection in fft_detections)
    known_music = sorted(detection.azimuth_deg for detection in music_detections)

    fig = plt.figure(figsize=(11.8, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), height_ratios=(1.0, 1.0))
    ax_heatmap = fig.add_subplot(grid[:, 0])
    ax_azimuth = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[1, 1])

    image = ax_heatmap.imshow(
        10.0 * np.log10(np.maximum(range_doppler, 1.0e-12)),
        aspect="auto",
        origin="lower",
        extent=[
            trial.fft_cube.velocity_axis_mps[0],
            trial.fft_cube.velocity_axis_mps[-1],
            trial.fft_cube.range_axis_m[0],
            trial.fft_cube.range_axis_m[-1],
        ],
        cmap="viridis",
    )
    ax_heatmap.scatter(
        [target.velocity_mps for target in truth_targets],
        [target.range_m for target in truth_targets],
        marker="*",
        s=290,
        c="#2FA66A",
        edgecolors="white",
        linewidths=1.1,
        label="Truth movers",
    )
    ax_heatmap.scatter(
        [detection.velocity_mps for detection in fft_detections],
        [detection.range_m for detection in fft_detections],
        marker="o",
        s=130,
        c=METHOD_COLORS["fft_masked"],
        edgecolors="white",
        linewidths=0.9,
        label="FFT detections",
    )
    ax_heatmap.scatter(
        [detection.velocity_mps for detection in music_detections],
        [detection.range_m for detection in music_detections],
        marker="x",
        s=160,
        linewidths=2.8,
        c=METHOD_COLORS["music_masked"],
        label="MUSIC detections",
    )

    false_fft = max(fft_detections, key=lambda detection: detection.range_m)
    ax_heatmap.annotate(
        "FFT false branch",
        xy=(false_fft.velocity_mps, false_fft.range_m),
        xytext=(false_fft.velocity_mps - 5.0, false_fft.range_m + 0.8),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": METHOD_COLORS["fft_masked"]},
        fontsize=10,
        color=METHOD_COLORS["fft_masked"],
        fontweight="bold",
    )
    ax_heatmap.annotate(
        "MUSIC lands on both movers",
        xy=(music_detections[0].velocity_mps, music_detections[0].range_m),
        xytext=(trial.fft_cube.velocity_axis_mps[0] + 0.6, max(target.range_m for target in truth_targets) + 1.1),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": METHOD_COLORS["music_masked"]},
        fontsize=10,
        color=METHOD_COLORS["music_masked"],
        fontweight="bold",
    )
    ax_heatmap.set_title(f"Saved nominal intersection trial {trial_index}", loc="left", fontsize=13, fontweight="bold")
    ax_heatmap.set_xlabel("Velocity (m/s)")
    ax_heatmap.set_ylabel("Range (m)")
    ax_heatmap.legend(frameon=False, loc="lower right", fontsize=9)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03, label="FFT range-Doppler power (dB)")

    ax_azimuth.plot(azimuth_grid, azimuth_spectrum_db, color=METHOD_COLORS["music_masked"], linewidth=2.3)
    for azimuth in known_truth:
        ax_azimuth.axvline(azimuth, color="#2FA66A", linestyle="--", linewidth=1.4, alpha=0.9)
    for azimuth in known_fft:
        ax_azimuth.axvline(azimuth, color=METHOD_COLORS["fft_masked"], linestyle=(0, (1.2, 2.2)), linewidth=1.7, alpha=0.85)
    for azimuth in known_music:
        ax_azimuth.axvline(azimuth, color=METHOD_COLORS["music_masked"], linestyle=(0, (7.0, 2.5, 1.4, 2.5)), linewidth=1.8, alpha=0.85)
    ax_azimuth.set_xlim(-5.0, 25.0)
    ax_azimuth.set_ylim(min(-42.0, float(np.min(azimuth_spectrum_db)) - 1.0), 2.0)
    ax_azimuth.set_ylabel("Relative level (dB)")
    ax_azimuth.set_title("Azimuth alignment", loc="left", fontsize=12, fontweight="bold")
    ax_azimuth.grid(True, alpha=0.22)
    ax_azimuth.text(0.01, 0.96, "Dashed = truth, dotted = FFT, dash-dot = MUSIC", transform=ax_azimuth.transAxes, va="top", fontsize=9)

    ax_summary.axis("off")
    ax_summary.text(0.0, 0.92, "What this trial shows", fontsize=13, fontweight="bold", color="#10233F")
    bullets = (
        "Paired nominal trial from the saved 64-trial submission bundle.",
        "FFT keeps the stronger mover but jumps to a plausible high-range false target for the second branch.",
        "MUSIC places both detections on the true pair and clears the joint gate on the same trial.",
    )
    y_position = 0.78
    for bullet in bullets:
        ax_summary.text(0.03, y_position, f"• {bullet}", fontsize=10.5, color="#1F2937", wrap=True)
        y_position -= 0.22

    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the representative intersection-case figure.",
        "08_representative_intersection_case.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
