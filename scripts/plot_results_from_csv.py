#!/usr/bin/env python3
"""Generate compact story-first figures from saved CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT + Local Refinement",
    "music_masked": "Masked Staged MUSIC + FBSS",
}
METHOD_COLORS = {
    "fft_masked": "#C44E52",
    "music_masked": "#4C72B0",
}
SCENE_ORDER = ("intersection", "open_aisle", "rack_aisle")
SCENE_COLORS = {
    "intersection": "#2F6B9A",
    "open_aisle": "#A65E2E",
    "rack_aisle": "#6B6B6B",
}
SWEEP_ORDER = (
    "allocation_family",
    "occupied_fraction",
    "fragmentation",
    "bandwidth_span",
    "slow_time_span",
    "range_separation",
    "velocity_separation",
    "angle_separation",
)
SWEEP_LABELS = {
    "allocation_family": "Allocation",
    "occupied_fraction": "Occupancy",
    "fragmentation": "Fragmentation",
    "bandwidth_span": "Bandwidth",
    "slow_time_span": "Slow-Time",
    "range_separation": "Range Sep.",
    "velocity_separation": "Velocity Sep.",
    "angle_separation": "Angle Sep.",
}
SUPPORT_SWEEP_ORDER = (
    "allocation_family",
    "occupied_fraction",
    "fragmentation",
    "bandwidth_span",
    "slow_time_span",
)
SEPARATION_SWEEP_ORDER = (
    "range_separation",
    "velocity_separation",
    "angle_separation",
)
STRONG_WIN_THRESHOLD = 0.10
STORY_FIGURE_NAMES = (
    "story_nominal_verdict_from_csv.png",
    "story_intersection_resolution_from_csv.png",
    "story_regime_map_from_csv.png",
    "story_coherence_overlap_from_csv.png",
    "story_pilot_only_collapse_from_csv.png",
    "story_trial_delta_from_csv.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create story-driven figures from saved CSV outputs.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results") / "quick",
        help="Result root containing a data/ directory, or the data/ directory itself.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to <input-root>/figures_from_csv.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before writing figures.",
    )
    return parser.parse_args()


def _resolve_paths(input_root: Path, output_dir: Path | None) -> tuple[Path, Path]:
    input_root = input_root.resolve()
    data_dir = input_root if input_root.name == "data" else input_root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    if output_dir is None:
        base_root = data_dir.parent if data_dir.name == "data" else data_dir
        output_dir = base_root / "figures_from_csv"
    return data_dir, output_dir.resolve()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_output_dir(output_dir: Path, clean_output: bool) -> None:
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _to_float(value: str, default: float | None = None) -> float | None:
    if value == "" or value is None:
        return default
    return float(value)


def _scene_key(scene_class: str) -> tuple[int, str]:
    if scene_class in SCENE_ORDER:
        return (SCENE_ORDER.index(scene_class), scene_class)
    return (len(SCENE_ORDER), scene_class)


def _scene_label_from_rows(rows: list[dict[str, str]], scene_class: str) -> str:
    for row in rows:
        if row["scene_class"] == scene_class:
            return row.get("scene_label", scene_class.replace("_", " ").title())
    return scene_class.replace("_", " ").title()


def _nominal_joint_deltas(rows: list[dict[str, str]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for scene_class in {row["scene_class"] for row in rows}:
        fft_row = next((row for row in rows if row["scene_class"] == scene_class and row["method"] == "fft_masked"), None)
        music_row = next((row for row in rows if row["scene_class"] == scene_class and row["method"] == "music_masked"), None)
        if fft_row is None or music_row is None:
            continue
        deltas[scene_class] = float(music_row["joint_resolution_probability"]) - float(fft_row["joint_resolution_probability"])
    return deltas


def _nominal_headline_rows(rows: list[dict[str, str]], *, method_name: str | None = None) -> list[dict[str, str]]:
    filtered = [
        row
        for row in rows
        if row["estimator_family"] == "headline"
        and row["sweep_name"] == "nominal"
        and row["knowledge_mode"] == "known_symbols"
    ]
    if method_name is not None:
        filtered = [row for row in filtered if row["method"] == method_name]
    return filtered


def _normalize_db(values: np.ndarray) -> np.ndarray:
    finite_values = np.asarray(values, dtype=float)
    finite_values = np.maximum(finite_values, np.max(finite_values) - 80.0)
    return finite_values - np.max(finite_values)


def _story_nominal_verdict(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    scene_classes = sorted({row["scene_class"] for row in rows}, key=_scene_key)
    scene_deltas = _nominal_joint_deltas(rows)
    scene_classes.sort(
        key=lambda scene_class: float(
            next(
                row["joint_resolution_probability"]
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "music_masked"
            )
        )
        - float(
            next(
                row["joint_resolution_probability"]
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "fft_masked"
            )
        ),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(scene_classes), dtype=float)
    for scene_index, scene_class in enumerate(scene_classes):
        fft_row = next(row for row in rows if row["scene_class"] == scene_class and row["method"] == "fft_masked")
        music_row = next(row for row in rows if row["scene_class"] == scene_class and row["method"] == "music_masked")
        fft_value = float(fft_row["joint_resolution_probability"])
        music_value = float(music_row["joint_resolution_probability"])
        fft_ci = (
            float(fft_row["joint_resolution_probability_ci95_lower"]),
            float(fft_row["joint_resolution_probability_ci95_upper"]),
        )
        music_ci = (
            float(music_row["joint_resolution_probability_ci95_lower"]),
            float(music_row["joint_resolution_probability_ci95_upper"]),
        )
        y_value = y_positions[scene_index]
        ax.hlines(y_value, min(fft_value, music_value), max(fft_value, music_value), color="#B8B8B8", linewidth=3.0)
        ax.errorbar(
            fft_value,
            y_value,
            xerr=np.array([[fft_value - fft_ci[0]], [fft_ci[1] - fft_value]]),
            fmt="o",
            markersize=9,
            color=METHOD_COLORS["fft_masked"],
            capsize=4,
            linewidth=2.0,
            label=METHOD_LABELS["fft_masked"] if scene_index == 0 else None,
        )
        ax.errorbar(
            music_value,
            y_value,
            xerr=np.array([[music_value - music_ci[0]], [music_ci[1] - music_value]]),
            fmt="s",
            markersize=9,
            color=METHOD_COLORS["music_masked"],
            capsize=4,
            linewidth=2.0,
            label=METHOD_LABELS["music_masked"] if scene_index == 0 else None,
        )
        delta = music_value - fft_value
        if delta >= 0.05:
            delta_label = f"MUSIC +{delta:.2f}"
            delta_color = METHOD_COLORS["music_masked"]
        elif delta <= -0.05:
            delta_label = f"FFT +{abs(delta):.2f}"
            delta_color = METHOD_COLORS["fft_masked"]
        else:
            delta_label = f"Near tie ({delta:+.2f})"
            delta_color = "#444444"
        ax.text(1.03, y_value, delta_label, va="center", ha="left", fontsize=10, color=delta_color, fontweight="bold")
        ax.text(
            -0.02,
            y_value,
            _scene_label_from_rows(rows, scene_class),
            va="center",
            ha="right",
            fontsize=11,
            color=SCENE_COLORS.get(scene_class, "#333333"),
            fontweight="bold",
        )
    ax.axvline(0.5, color="#D8D8D8", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, 1.14)
    ax.set_ylim(-0.7, len(scene_classes) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Nominal Joint-Resolution Probability")
    if {"intersection", "open_aisle", "rack_aisle"}.issubset(scene_deltas):
        title = "MUSIC only wins the nominal intersection scene\n64-trial FR1 nominal point with 95% Wilson intervals"
    elif len(scene_classes) == 1:
        scene_class = scene_classes[0]
        delta = scene_deltas.get(scene_class, 0.0)
        winner = "MUSIC" if delta > 0.05 else "FFT" if delta < -0.05 else "Neither method"
        verb = "wins" if abs(delta) > 0.05 else "separates"
        title = f"{winner} {verb} this nominal {_scene_label_from_rows(rows, scene_class)} case\nSingle available scene with 95% Wilson intervals"
    else:
        title = "Nominal scene verdicts\nSaved 64-trial FR1 point with 95% Wilson intervals"
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.20)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "story_nominal_verdict_from_csv.png", dpi=180)
    plt.close(fig)


def _story_intersection_resolution(
    nominal_rows: list[dict[str, str]],
    range_doppler_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, str]],
    music_spectrum_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    scene_deltas = _nominal_joint_deltas(nominal_rows)
    geometry_scene_classes = {row["scene_class"] for row in geometry_rows}
    azimuth_scene_classes = {row["scene_class"] for row in music_spectrum_rows if row["dimension"] == "azimuth"}
    available_scenes: list[str] = []
    for candidate_scene in sorted({row["scene_class"] for row in range_doppler_rows}, key=_scene_key):
        if candidate_scene not in geometry_scene_classes or candidate_scene not in azimuth_scene_classes:
            continue
        has_truth = any(row["scene_class"] == candidate_scene and row["entity_kind"] == "truth_target" for row in geometry_rows)
        has_fft = any(row["scene_class"] == candidate_scene and row["entity_group"] == "fft_masked" for row in geometry_rows)
        has_music = any(row["scene_class"] == candidate_scene and row["entity_group"] == "music_masked" for row in geometry_rows)
        if has_truth and has_fft and has_music:
            available_scenes.append(candidate_scene)
    if not available_scenes:
        return
    if "intersection" in available_scenes:
        scene_class = "intersection"
    else:
        positive_scenes = [scene for scene in available_scenes if scene_deltas.get(scene, 0.0) > 0.05]
        scene_class = max(positive_scenes, key=lambda scene: scene_deltas.get(scene, float("-inf"))) if positive_scenes else available_scenes[0]
    scene_label = _scene_label_from_rows(geometry_rows, scene_class)
    heatmap_rows = [row for row in range_doppler_rows if row["scene_class"] == scene_class]
    overlay_rows = [row for row in geometry_rows if row["scene_class"] == scene_class]
    if not heatmap_rows or not overlay_rows:
        return

    range_values = np.array(sorted({float(row["range_m"]) for row in heatmap_rows}), dtype=float)
    velocity_values = np.array(sorted({float(row["velocity_mps"]) for row in heatmap_rows}), dtype=float)
    power_grid = np.full((range_values.size, velocity_values.size), np.nan, dtype=float)
    range_index = {value: index for index, value in enumerate(range_values.tolist())}
    velocity_index = {value: index for index, value in enumerate(velocity_values.tolist())}
    for row in heatmap_rows:
        power_grid[range_index[float(row["range_m"])], velocity_index[float(row["velocity_mps"])]] = float(row["power_db"])

    truth_rows = [row for row in overlay_rows if row["entity_kind"] == "truth_target"]
    fft_rows = [row for row in overlay_rows if row["entity_group"] == "fft_masked"]
    music_rows = [row for row in overlay_rows if row["entity_group"] == "music_masked"]
    relevant_ranges = [float(row["range_m"]) for row in truth_rows + fft_rows + music_rows]
    relevant_velocities = [float(row["velocity_mps"]) for row in truth_rows + fft_rows + music_rows]
    min_range = max(np.min(range_values), min(relevant_ranges) - 2.0)
    max_range = min(np.max(range_values), max(relevant_ranges) + 2.0)
    min_velocity = max(np.min(velocity_values), min(relevant_velocities) - 2.5)
    max_velocity = min(np.max(velocity_values), max(relevant_velocities) + 2.5)

    zoom_range_mask = (range_values >= min_range) & (range_values <= max_range)
    zoom_velocity_mask = (velocity_values >= min_velocity) & (velocity_values <= max_velocity)
    zoom_grid = power_grid[np.ix_(zoom_range_mask, zoom_velocity_mask)]
    zoom_ranges = range_values[zoom_range_mask]
    zoom_velocities = velocity_values[zoom_velocity_mask]

    azimuth_rows = [row for row in music_spectrum_rows if row["scene_class"] == scene_class and row["dimension"] == "azimuth"]
    if not azimuth_rows:
        return
    azimuth_rows.sort(key=lambda row: float(row["coordinate_value"]))
    azimuth_x = np.array([float(row["coordinate_value"]) for row in azimuth_rows], dtype=float)
    azimuth_y = np.array([float(row["spectrum_db_rel"]) for row in azimuth_rows], dtype=float)

    fig = plt.figure(figsize=(12.0, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), height_ratios=(1.0, 1.0))
    ax_heatmap = fig.add_subplot(grid[:, 0])
    ax_azimuth = fig.add_subplot(grid[0, 1])
    ax_localization = fig.add_subplot(grid[1, 1])

    image = ax_heatmap.imshow(
        zoom_grid,
        aspect="auto",
        origin="lower",
        extent=[zoom_velocities[0], zoom_velocities[-1], zoom_ranges[0], zoom_ranges[-1]],
        cmap="viridis",
    )
    ax_heatmap.scatter(
        [float(row["velocity_mps"]) for row in truth_rows],
        [float(row["range_m"]) for row in truth_rows],
        c="#2FA66A",
        marker="*",
        s=180,
        edgecolors="white",
        linewidths=0.8,
        label="Truth movers",
    )
    ax_heatmap.scatter(
        [float(row["velocity_mps"]) for row in fft_rows],
        [float(row["range_m"]) for row in fft_rows],
        c=METHOD_COLORS["fft_masked"],
        marker="o",
        s=72,
        label="FFT detections",
    )
    ax_heatmap.scatter(
        [float(row["velocity_mps"]) for row in music_rows],
        [float(row["range_m"]) for row in music_rows],
        c=METHOD_COLORS["music_masked"],
        marker="x",
        s=92,
        linewidths=2.0,
        label="MUSIC detections",
    )
    fft_false_row = max(fft_rows, key=lambda row: float(row["range_m"]))
    ax_heatmap.annotate(
        "False FFT target",
        xy=(float(fft_false_row["velocity_mps"]), float(fft_false_row["range_m"])),
        xytext=(float(fft_false_row["velocity_mps"]) - 3.8, float(fft_false_row["range_m"]) + 0.8),
        arrowprops={"arrowstyle": "->", "color": METHOD_COLORS["fft_masked"], "lw": 1.5},
        color=METHOD_COLORS["fft_masked"],
        fontsize=10,
        fontweight="bold",
    )
    ax_heatmap.annotate(
        "MUSIC lands on both movers",
        xy=(float(music_rows[0]["velocity_mps"]), float(music_rows[0]["range_m"])),
        xytext=(min_velocity + 0.1, max_range - 0.25),
        arrowprops={"arrowstyle": "->", "color": METHOD_COLORS["music_masked"], "lw": 1.5},
        color=METHOD_COLORS["music_masked"],
        fontsize=10,
        fontweight="bold",
    )
    ax_heatmap.set_xlabel("Velocity (m/s)")
    ax_heatmap.set_ylabel("Range (m)")
    ax_heatmap.legend(frameon=False, loc="lower right", fontsize=9)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03, label="FFT Range-Doppler Power (dB)")

    ax_azimuth.plot(azimuth_x, azimuth_y, color=METHOD_COLORS["music_masked"], linewidth=2.3)
    for row in truth_rows:
        ax_azimuth.axvline(float(row["azimuth_deg"]), color="#2FA66A", linestyle="--", linewidth=1.4, alpha=0.9)
    for row in fft_rows:
        ax_azimuth.axvline(float(row["azimuth_deg"]), color=METHOD_COLORS["fft_masked"], linestyle=":", linewidth=1.4, alpha=0.75)
    for row in music_rows:
        ax_azimuth.axvline(float(row["azimuth_deg"]), color=METHOD_COLORS["music_masked"], linestyle="-.", linewidth=1.4, alpha=0.75)
    ax_azimuth.set_xlim(-30.0, 35.0)
    ax_azimuth.set_ylim(min(-42.0, float(np.min(azimuth_y)) - 1.0), 2.0)
    ax_azimuth.set_ylabel("Relative Level (dB)")
    ax_azimuth.set_title("Conditioned MUSIC azimuth slice", loc="left", fontsize=12, fontweight="bold")
    ax_azimuth.grid(True, alpha=0.22)
    ax_azimuth.text(0.01, 0.96, "Dashed = truth, dotted = FFT, dash-dot = MUSIC", transform=ax_azimuth.transAxes, va="top", fontsize=9)
    ax_localization.scatter(
        [float(row["azimuth_deg"]) for row in truth_rows],
        [float(row["range_m"]) for row in truth_rows],
        c="#2FA66A",
        marker="*",
        s=170,
        edgecolors="white",
        linewidths=0.8,
        label="Truth movers",
    )
    ax_localization.scatter(
        [float(row["azimuth_deg"]) for row in fft_rows],
        [float(row["range_m"]) for row in fft_rows],
        c=METHOD_COLORS["fft_masked"],
        marker="o",
        s=72,
        label="FFT detections",
    )
    ax_localization.scatter(
        [float(row["azimuth_deg"]) for row in music_rows],
        [float(row["range_m"]) for row in music_rows],
        c=METHOD_COLORS["music_masked"],
        marker="x",
        s=92,
        linewidths=2.0,
        label="MUSIC detections",
    )
    ax_localization.annotate(
        "FFT duplicate",
        xy=(float(fft_false_row["azimuth_deg"]), float(fft_false_row["range_m"])),
        xytext=(float(fft_false_row["azimuth_deg"]) - 11.0, float(fft_false_row["range_m"]) + 0.8),
        arrowprops={"arrowstyle": "->", "color": METHOD_COLORS["fft_masked"], "lw": 1.4},
        color=METHOD_COLORS["fft_masked"],
        fontsize=10,
        fontweight="bold",
    )
    ax_localization.set_xlim(min(float(row["azimuth_deg"]) for row in truth_rows + fft_rows + music_rows) - 5.0, max(float(row["azimuth_deg"]) for row in truth_rows + fft_rows + music_rows) + 5.0)
    ax_localization.set_ylim(min_range, max_range)
    ax_localization.set_xlabel("Azimuth (deg)")
    ax_localization.set_ylabel("Range (m)")
    ax_localization.set_title("Recovered target locations in range-azimuth space", loc="left", fontsize=12, fontweight="bold")
    ax_localization.grid(True, alpha=0.22)
    ax_localization.legend(frameon=False, loc="lower right", fontsize=9)

    scene_delta = scene_deltas.get(scene_class, 0.0)
    if scene_class == "intersection" and scene_delta > 0.05:
        figure_title = f"Why MUSIC wins this {scene_label} case\nFFT duplicates the faster mover, while MUSIC recovers both movers"
    elif scene_delta < -0.05:
        figure_title = f"Representative {scene_label} failure case\nSaved spectra and detections from one deterministic trial"
    else:
        figure_title = f"Representative {scene_label} comparison\nSaved spectra and detections from one deterministic trial"
    fig.suptitle(figure_title, x=0.06, y=0.97, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.88, wspace=0.28, hspace=0.38)
    fig.savefig(output_dir / "story_intersection_resolution_from_csv.png", dpi=180)
    plt.close(fig)


def _story_regime_map(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    scene_classes = sorted({row["scene_class"] for row in rows}, key=_scene_key)
    panel_specs = (
        ("Support-Limited Families", SUPPORT_SWEEP_ORDER, "Joint-resolution metric"),
        ("Axis-Isolated Separation Families", SEPARATION_SWEEP_ORDER, "Axis-specific metric"),
    )
    panel_data: list[tuple[np.ndarray, list[list[str]]]] = []
    max_abs_delta = 0.0
    for _title, sweep_names, _subtitle in panel_specs:
        matrix = np.zeros((len(scene_classes), len(sweep_names)), dtype=float)
        labels = [["" for _ in sweep_names] for _ in scene_classes]
        for row_index, scene_class in enumerate(scene_classes):
            for col_index, sweep_name in enumerate(sweep_names):
                sweep_rows = [row for row in rows if row["scene_class"] == scene_class and row["sweep_name"] == sweep_name]
                if not sweep_rows:
                    continue
                deltas = np.asarray([float(row["music_minus_fft"]) for row in sweep_rows], dtype=float)
                matrix[row_index, col_index] = float(np.mean(deltas))
                strong_wins = int(np.sum(deltas >= STRONG_WIN_THRESHOLD))
                labels[row_index][col_index] = f"{np.mean(deltas):+.2f}\n{strong_wins}/{len(deltas)}"
                max_abs_delta = max(max_abs_delta, float(np.max(np.abs(deltas))))
        panel_data.append((matrix, labels))
    max_abs_delta = max(max_abs_delta, 0.01)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs_delta, vmax=max_abs_delta)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": [len(SUPPORT_SWEEP_ORDER), len(SEPARATION_SWEEP_ORDER)]},
    )
    image = None
    for axis_index, (ax, (panel_title, sweep_names, panel_subtitle), (matrix, labels)) in enumerate(
        zip(axes, panel_specs, panel_data, strict=True)
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", norm=norm)
        ax.set_xticks(np.arange(len(sweep_names)))
        ax.set_xticklabels([SWEEP_LABELS[name] for name in sweep_names], rotation=20, ha="right")
        ax.set_yticks(np.arange(len(scene_classes)))
        if axis_index == 0:
            ax.set_yticklabels([_scene_label_from_rows(rows, scene_class) for scene_class in scene_classes])
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{panel_title}\n{panel_subtitle}", fontsize=11.5, fontweight="bold")
        for row_index in range(len(scene_classes)):
            for col_index in range(len(sweep_names)):
                value = matrix[row_index, col_index]
                if labels[row_index][col_index] == "":
                    continue
                text_color = "white" if abs(value) >= 0.18 else "#222222"
                ax.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=9, color=text_color)
    fig.suptitle(
        "MUSIC gains cluster in a few sweep families, not across the whole study\n"
        "Cell color = mean headline-metric delta; text = strong-win count (>= +0.10)",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.06, 0.03, "Support-limited and separation sweeps are split because they use different headline metrics.", fontsize=9)
    if image is not None:
        fig.colorbar(image, ax=axes, label="Mean Headline-Metric Delta", fraction=0.035, pad=0.02)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.17, top=0.82, wspace=0.16)
    fig.savefig(output_dir / "story_regime_map_from_csv.png", dpi=180)
    plt.close(fig)


def _story_coherence_overlap(rows: list[dict[str, str]], output_dir: Path) -> None:
    nominal_rows = _nominal_headline_rows(rows, method_name="fft_masked")
    if not nominal_rows:
        return
    scene_classes = sorted({row["scene_class"] for row in nominal_rows}, key=_scene_key)
    empirical_by_scene = [
        np.asarray(
            [float(row["empirical_target_coherence"]) for row in nominal_rows if row["scene_class"] == scene_class],
            dtype=float,
        )
        for scene_class in scene_classes
    ]
    configured_by_scene = {
        scene_class: float(next(row["configured_target_coherence"] for row in nominal_rows if row["scene_class"] == scene_class))
        for scene_class in scene_classes
    }
    positions = np.arange(len(scene_classes), dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    box = ax.boxplot(
        empirical_by_scene,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
    )
    for patch, scene_class in zip(box["boxes"], scene_classes, strict=True):
        patch.set_facecolor(SCENE_COLORS.get(scene_class, "#777777"))
        patch.set_alpha(0.26)
        patch.set_edgecolor(SCENE_COLORS.get(scene_class, "#777777"))
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    rng = np.random.default_rng(20260406)
    for position, scene_class, values in zip(positions, scene_classes, empirical_by_scene, strict=True):
        jitter = rng.uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=20,
            alpha=0.55,
            color=SCENE_COLORS.get(scene_class, "#777777"),
            edgecolors="none",
        )
        configured_value = configured_by_scene[scene_class]
        ax.scatter(
            position,
            configured_value,
            marker="D",
            s=54,
            color="#111111",
            zorder=4,
        )
        label_y = min(1.0, float(np.max(values)) + 0.04)
        ax.text(position, label_y, f"mean {np.mean(values):.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([_scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Target Coherence")
    ax.set_title(
        "Empirical nominal coherence overlaps strongly across scenes\n"
        "Configured coherence alone does not isolate the operating regimes",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.96,
        "Circles = empirical finite-snapshot coherence per nominal trial; diamond = configured scene coherence",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_dir / "story_coherence_overlap_from_csv.png", dpi=180)
    plt.close(fig)


def _story_pilot_only_collapse(
    nominal_rows: list[dict[str, str]],
    pilot_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    if not nominal_rows or not pilot_rows:
        return
    scene_classes = sorted({row["scene_class"] for row in nominal_rows}, key=_scene_key)
    nominal_index = {(row["scene_class"], row["method"]): row for row in nominal_rows}
    pilot_index = {(row["scene_class"], row["method"]): row for row in pilot_rows}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
    for axis_index, method_name in enumerate(METHOD_ORDER):
        ax = axes[axis_index]
        y_positions = np.arange(len(scene_classes), dtype=float)
        for row_index, scene_class in enumerate(scene_classes):
            nominal_value = float(nominal_index[(scene_class, method_name)]["joint_resolution_probability"])
            pilot_value = float(pilot_index[(scene_class, method_name)]["joint_resolution_probability"])
            color = SCENE_COLORS.get(scene_class, "#444444")
            ax.plot([pilot_value, nominal_value], [row_index, row_index], color=color, linewidth=3.0, alpha=0.90)
            ax.scatter(pilot_value, row_index, color=color, s=54, marker="o", zorder=3)
            ax.scatter(nominal_value, row_index, color=color, s=70, marker="s", zorder=3)
            ax.text(nominal_value + 0.02, row_index, f"{nominal_value:.2f}", va="center", ha="left", fontsize=9, color=color)
        ax.set_xlim(0.0, 1.02)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(METHOD_LABELS[method_name], fontsize=10.5, fontweight="bold", pad=10)
        ax.grid(True, axis="x", alpha=0.22)
        ax.axvline(0.0, color="#D0D0D0", linewidth=1.0)
    axes[0].set_yticks(np.arange(len(scene_classes)))
    axes[0].set_yticklabels([_scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Pjoint")
    axes[1].set_xlabel("Pjoint")
    fig.text(0.08, 0.84, "Circle = pilot-only, square = known-symbols", fontsize=9)
    fig.suptitle(
        "Removing oracle symbol knowledge collapses both methods\nNominal joint-resolution probability drops to near zero under pilot-only sensing",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(output_dir / "story_pilot_only_collapse_from_csv.png", dpi=180)
    plt.close(fig)


def _story_trial_delta(rows: list[dict[str, str]], output_dir: Path) -> None:
    nominal_rows = _nominal_headline_rows(rows)
    if not nominal_rows:
        return
    paired_by_scene: dict[str, list[float]] = defaultdict(list)
    for scene_class in {row["scene_class"] for row in nominal_rows}:
        fft_by_trial = {
            row["trial_index"]: float(row["unconditional_joint_assignment_rmse"])
            for row in nominal_rows
            if row["scene_class"] == scene_class and row["method"] == "fft_masked"
        }
        music_by_trial = {
            row["trial_index"]: float(row["unconditional_joint_assignment_rmse"])
            for row in nominal_rows
            if row["scene_class"] == scene_class and row["method"] == "music_masked"
        }
        common_trials = sorted(set(fft_by_trial) & set(music_by_trial), key=int)
        paired_by_scene[scene_class] = [fft_by_trial[trial_index] - music_by_trial[trial_index] for trial_index in common_trials]
    scene_classes = sorted(paired_by_scene, key=_scene_key)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axhspan(0.0, max(max(values) for values in paired_by_scene.values()) + 1.0e-9, color="#EAF2FB", alpha=0.65)
    ax.axhspan(min(min(values) for values in paired_by_scene.values()) - 1.0e-9, 0.0, color="#FBECEC", alpha=0.65)
    ax.axhline(0.0, color="#333333", linewidth=1.2)
    positions = np.arange(len(scene_classes), dtype=float)
    box = ax.boxplot(
        [paired_by_scene[scene_class] for scene_class in scene_classes],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
    )
    for patch, scene_class in zip(box["boxes"], scene_classes, strict=True):
        patch.set_facecolor(SCENE_COLORS.get(scene_class, "#777777"))
        patch.set_alpha(0.28)
        patch.set_edgecolor(SCENE_COLORS.get(scene_class, "#777777"))
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    rng = np.random.default_rng(20260404)
    for position, scene_class in zip(positions, scene_classes, strict=True):
        values = np.asarray(paired_by_scene[scene_class], dtype=float)
        jitter = rng.uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=22,
            alpha=0.60,
            color=SCENE_COLORS.get(scene_class, "#777777"),
            edgecolors="none",
        )
        win_fraction = float(np.mean(values > 0.0))
        ax.text(position, np.percentile(values, 85) + 0.02, f"{win_fraction:.0%} trials favor MUSIC", ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([_scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    ax.set_ylabel("FFT RMSE - MUSIC RMSE\n(positive means MUSIC is better)")
    win_fractions = {scene_class: float(np.mean(np.asarray(paired_by_scene[scene_class], dtype=float) > 0.0)) for scene_class in scene_classes}
    if {"intersection", "open_aisle"}.issubset(scene_classes):
        title = "Intersection gains are broad, while open-aisle losses are systematic\nPaired nominal trial deltas at the 64-trial FR1 nominal point"
    elif len(scene_classes) == 1:
        scene_class = scene_classes[0]
        if win_fractions[scene_class] > 0.55:
            verdict = "mostly favor MUSIC"
        elif win_fractions[scene_class] < 0.45:
            verdict = "mostly favor FFT"
        else:
            verdict = "split almost evenly"
        title = f"{_scene_label_from_rows(nominal_rows, scene_class)} trial deltas {verdict}\nPaired nominal trials at the saved FR1 point"
    else:
        title = "Paired nominal trial deltas by scene\nPositive values mean MUSIC reduces assignment RMSE"
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.20)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.87)
    fig.savefig(output_dir / "story_trial_delta_from_csv.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir, output_dir = _resolve_paths(args.input_root, args.output_dir)
    _prepare_output_dir(output_dir, clean_output=args.clean_output)

    nominal_rows = _read_csv_rows(data_dir / "nominal_summary.csv")
    pilot_rows = _read_csv_rows(data_dir / "pilot_only_nominal_summary.csv")
    usefulness_rows = _read_csv_rows(data_dir / "usefulness_windows.csv")
    trial_rows = _read_csv_rows(data_dir / "trial_level_results.csv")
    geometry_rows = _read_csv_rows(data_dir / "representative_scene_geometry.csv")
    range_doppler_rows = _read_csv_rows(data_dir / "representative_range_doppler.csv")
    music_spectrum_rows = _read_csv_rows(data_dir / "representative_music_spectra.csv")

    _story_nominal_verdict(nominal_rows, output_dir)
    _story_intersection_resolution(nominal_rows, range_doppler_rows, geometry_rows, music_spectrum_rows, output_dir)
    _story_regime_map(usefulness_rows, output_dir)
    _story_coherence_overlap(trial_rows, output_dir)
    _story_pilot_only_collapse(nominal_rows, pilot_rows, output_dir)
    _story_trial_delta(trial_rows, output_dir)

    print(f"Input data directory: {data_dir}")
    print(f"Wrote story figures to: {output_dir}")
    for filename in STORY_FIGURE_NAMES:
        print(f"- {output_dir / filename}")


if __name__ == "__main__":
    main()
