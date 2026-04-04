#!/usr/bin/env python3
"""Generate figures from saved CSV artifacts for the MUSIC OFDM ISAC study."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT + Local Refinement",
    "music_masked": "Masked Staged MUSIC + FBSS",
}
METHOD_COLORS = {
    "fft_masked": "#C44E52",
    "music_masked": "#4C72B0",
}
FBSS_ABLATION_ORDER = (
    "fbss_spatial_only",
    "fbss_spatial_range",
    "fbss_spatial_doppler",
    "fbss_spatial_range_doppler",
)
FBSS_ABLATION_LABELS = {
    "fbss_spatial_only": "Spatial FBSS Only",
    "fbss_spatial_range": "Spatial + Range FBSS",
    "fbss_spatial_doppler": "Spatial + Doppler FBSS",
    "fbss_spatial_range_doppler": "Spatial + Range + Doppler FBSS",
}
FBSS_ABLATION_COLORS = {
    "fbss_spatial_only": "#7F7F7F",
    "fbss_spatial_range": "#55A868",
    "fbss_spatial_doppler": "#DD8452",
    "fbss_spatial_range_doppler": "#4C72B0",
}
SWEEP_METRICS = {
    "range_separation": ("range_resolution_probability", "Range Resolution Probability"),
    "velocity_separation": ("velocity_resolution_probability", "Doppler Resolution Probability"),
    "angle_separation": ("angle_resolution_probability", "Angle Resolution Probability"),
}
ENTITY_COLORS = {
    "truth_target": "#55A868",
    "static_clutter": "#7F7F7F",
    "multipath": "#B07AA1",
    "nuisance": "#999999",
    "fft_masked": METHOD_COLORS["fft_masked"],
    "music_masked": METHOD_COLORS["music_masked"],
}
ROLE_CMAP = ListedColormap(["#F0F0F0", "#4C72B0", "#55A868", "#C44E52"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures from saved CSV outputs.")
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


def _to_float(value: str, default: float | None = None) -> float | None:
    if value == "" or value is None:
        return default
    return float(value)


def _unique_preserving(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["anchor"], row.get("anchor_label", row["anchor"]), row["scene_class"], row.get("scene_label", row["scene_class"]))


def _group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    return dict(grouped)


def _metric_for_sweep(sweep_name: str) -> tuple[str, str]:
    return SWEEP_METRICS.get(sweep_name, ("joint_resolution_probability", "Joint Resolution Probability"))


def _prepare_output_dir(output_dir: Path, clean_output: bool) -> None:
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _plot_nominal_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    metrics = [
        ("joint_detection_probability", "Pdet"),
        ("joint_resolution_probability", "Pjoint"),
        ("range_resolution_probability", "Prange"),
        ("velocity_resolution_probability", "Pvel"),
        ("angle_resolution_probability", "Pangle"),
    ]
    grouped = _group_rows(rows)
    group_labels = [f"{anchor_label}\n{scene_label}" for _anchor, anchor_label, _scene, scene_label in grouped]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x_positions = np.arange(len(group_labels), dtype=float)
    bar_width = 0.35
    for ax, (metric_name, metric_label) in zip(axes, metrics, strict=True):
        for method_index, method_name in enumerate(METHOD_ORDER):
            values = []
            for key in grouped:
                method_rows = [row for row in grouped[key] if row["method"] == method_name]
                values.append(_to_float(method_rows[0][metric_name], 0.0) if method_rows else 0.0)
            ax.bar(
                x_positions + (method_index - 0.5) * bar_width,
                values,
                width=bar_width,
                color=METHOD_COLORS[method_name],
                label=METHOD_LABELS[method_name],
            )
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=8)
    fig.suptitle("Nominal Detection and Resolution Summary")
    fig.tight_layout()
    fig.savefig(output_dir / "nominal_summary_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_pilot_only_nominal(
    nominal_rows: list[dict[str, str]],
    pilot_only_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    if not nominal_rows or not pilot_only_rows:
        return
    metrics = [
        ("joint_detection_probability", "Pdet"),
        ("joint_resolution_probability", "Pjoint"),
    ]
    nominal_index = {
        (row["anchor"], row["scene_class"], row["method"]): row
        for row in nominal_rows
    }
    pilot_index = {
        (row["anchor"], row["scene_class"], row["method"]): row
        for row in pilot_only_rows
    }
    group_keys = _unique_preserving([f"{row['anchor']}::{row['scene_class']}" for row in nominal_rows])
    group_labels = []
    for key in group_keys:
        anchor, scene_class = key.split("::", maxsplit=1)
        row = next(row for row in nominal_rows if row["anchor"] == anchor and row["scene_class"] == scene_class)
        group_labels.append(f"{row['anchor_label']}\n{row['scene_label']}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x_positions = np.arange(len(group_keys), dtype=float)
    bar_width = 0.18
    series = (
        ("known_symbols", nominal_index, -1.5, 0.95),
        ("pilot_only", pilot_index, -0.5, 0.55),
    )
    for ax, (metric_name, metric_label) in zip(axes, metrics, strict=True):
        for method_index, method_name in enumerate(METHOD_ORDER):
            for knowledge_mode, row_index, offset_base, alpha in series:
                values = []
                for key in group_keys:
                    anchor, scene_class = key.split("::", maxsplit=1)
                    row = row_index.get((anchor, scene_class, method_name))
                    values.append(_to_float(row[metric_name], 0.0) if row is not None else 0.0)
                ax.bar(
                    x_positions + (offset_base + 2.0 * method_index) * bar_width,
                    values,
                    width=bar_width,
                    color=METHOD_COLORS[method_name],
                    alpha=alpha,
                    label=f"{METHOD_LABELS[method_name]} / {knowledge_mode.replace('_', '-')}",
                )
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=8)
    fig.suptitle("Oracle Known-Symbols vs Pilot-Only Nominal Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "pilot_only_nominal_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_trial_level_joint_rmse(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    nominal_rows = [
        row
        for row in rows
        if row["estimator_family"] == "headline" and row["sweep_name"] in ("nominal", "pilot_only_nominal")
    ]
    grouped = _group_rows(nominal_rows)
    if not grouped:
        return
    fig, axes = plt.subplots(len(grouped), 1, figsize=(9.5, 3.8 * len(grouped)), squeeze=False)
    for axis_index, (group_key, group_rows) in enumerate(grouped.items()):
        ax = axes[axis_index, 0]
        _anchor, anchor_label, _scene, scene_label = group_key
        labels = []
        series = []
        positions = []
        position_index = 0
        for knowledge_mode in ("known_symbols", "pilot_only"):
            for method_name in METHOD_ORDER:
                method_rows = [
                    row
                    for row in group_rows
                    if row["knowledge_mode"] == knowledge_mode and row["method"] == method_name
                ]
                if not method_rows:
                    continue
                labels.append(f"{method_name}\n{knowledge_mode.replace('_', '-')}")
                series.append([float(row["unconditional_joint_assignment_rmse"]) for row in method_rows])
                positions.append(position_index)
                position_index += 1
        if not series:
            continue
        box = ax.boxplot(series, positions=positions, widths=0.6, patch_artist=True)
        for patch, label in zip(box["boxes"], labels, strict=True):
            method_name = label.split("\n", maxsplit=1)[0]
            patch.set_facecolor(METHOD_COLORS[method_name])
            patch.set_alpha(0.65 if "known-symbols" in label else 0.35)
        for median in box["medians"]:
            median.set_color("#222222")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Unconditional Joint RMSE")
        ax.set_title(f"Trial-Level Nominal Error Spread: {anchor_label} / {scene_label}")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trial_level_joint_rmse_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_sweep_figures(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    sweep_names = _unique_preserving([row["sweep_name"] for row in rows])
    for sweep_name in sweep_names:
        sweep_rows = [row for row in rows if row["sweep_name"] == sweep_name]
        grouped = _group_rows(sweep_rows)
        if not grouped:
            continue
        metric_name, metric_label = _metric_for_sweep(sweep_name)
        fig, axes = plt.subplots(len(grouped), 1, figsize=(10, 4.2 * len(grouped)), squeeze=False)
        for axis_index, (group_key, group_rows) in enumerate(grouped.items()):
            ax = axes[axis_index, 0]
            _anchor, anchor_label, _scene, scene_label = group_key
            has_numeric_x = all(row["parameter_numeric_value"] != "" for row in group_rows)
            if has_numeric_x:
                for method_name in METHOD_ORDER:
                    method_rows = [row for row in group_rows if row["method"] == method_name]
                    method_rows.sort(key=lambda row: float(row["parameter_numeric_value"]))
                    x_values = np.asarray([float(row["parameter_numeric_value"]) for row in method_rows], dtype=float)
                    y_values = np.asarray([float(row[metric_name]) for row in method_rows], dtype=float)
                    ax.plot(
                        x_values,
                        y_values,
                        marker="o" if method_name == "fft_masked" else "s",
                        linewidth=2.0,
                        color=METHOD_COLORS[method_name],
                        label=METHOD_LABELS[method_name],
                    )
                ax.set_xlabel(group_rows[0]["parameter_label"])
            else:
                parameter_order = _unique_preserving([row["parameter_value"] for row in group_rows])
                positions = np.arange(len(parameter_order), dtype=float)
                bar_width = 0.35
                for method_index, method_name in enumerate(METHOD_ORDER):
                    method_map = {
                        row["parameter_value"]: float(row[metric_name])
                        for row in group_rows
                        if row["method"] == method_name
                    }
                    values = [method_map.get(label, 0.0) for label in parameter_order]
                    ax.bar(
                        positions + (method_index - 0.5) * bar_width,
                        values,
                        width=bar_width,
                        color=METHOD_COLORS[method_name],
                        label=METHOD_LABELS[method_name],
                    )
                ax.set_xticks(positions)
                ax.set_xticklabels(parameter_order, rotation=20, ha="right")
                ax.set_xlabel(group_rows[0]["parameter_label"])
            ax.set_ylabel(metric_label)
            ax.set_ylim(0.0, 1.05)
            ax.set_title(f"{anchor_label} / {scene_label}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sweep_{sweep_name}_from_csv.png", dpi=180)
        plt.close(fig)


def _plot_usefulness_overview(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    grouped = _group_rows(rows)
    row_labels = [f"{anchor_label} / {scene_label}" for _anchor, anchor_label, _scene, scene_label in grouped]
    sweep_names = _unique_preserving([row["sweep_name"] for row in rows])
    matrix = np.zeros((len(grouped), len(sweep_names)), dtype=float)
    for row_index, group_key in enumerate(grouped):
        group_rows = grouped[group_key]
        for sweep_index, sweep_name in enumerate(sweep_names):
            sweep_rows = [row for row in group_rows if row["sweep_name"] == sweep_name]
            if not sweep_rows:
                continue
            matrix[row_index, sweep_index] = np.mean([float(row["usefulness_window"]) for row in sweep_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 * len(sweep_names) + 3, 0.8 * len(grouped) + 2.5))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(sweep_names)))
    ax.set_xticklabels(sweep_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Fraction of Sweep Points with Meaningful MUSIC Gain")
    fig.colorbar(image, ax=ax, label="Usefulness Window Fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "usefulness_overview_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_failure_overview(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    grouped = _group_rows(rows)
    labels = [f"{anchor_label} / {scene_label}" for _anchor, anchor_label, _scene, scene_label in grouped]
    x_positions = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, 2.2 * len(labels)), 4.8))
    bar_width = 0.35
    for method_index, method_name in enumerate(METHOD_ORDER):
        counts = []
        for group_key in grouped:
            counts.append(sum(row["method"] == method_name for row in grouped[group_key]))
        ax.bar(
            x_positions + (method_index - 0.5) * bar_width,
            counts,
            width=bar_width,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count of Sub-Perfect Sweep Points")
    ax.set_title("Failure-Mode Burden by Representative Study")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_overview_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_resource_masks(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        n_subcarriers = max(int(row["subcarrier_index"]) for row in group_rows) + 1
        n_symbols = max(int(row["symbol_index"]) for row in group_rows) + 1
        role_grid = np.zeros((n_subcarriers, n_symbols), dtype=int)
        for row in group_rows:
            role_grid[int(row["subcarrier_index"]), int(row["symbol_index"])] = int(row["role_code"])
        fig, ax = plt.subplots(figsize=(8, 4))
        image = ax.imshow(role_grid.T, aspect="auto", origin="lower", cmap=ROLE_CMAP, interpolation="nearest")
        ax.set_xlabel("Subcarrier Index")
        ax.set_ylabel("OFDM Symbol Index")
        ax.set_title(f"Resource Mask from CSV: {anchor_label} / {scene_label}")
        cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Muted", "Pilot", "Data", "Punctured"])
        fig.tight_layout()
        fig.savefig(output_dir / f"resource_mask_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def _plot_scene_geometry(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        for entity_group in ("static_clutter", "multipath", "truth_target", "fft_masked", "music_masked"):
            entity_rows = [row for row in group_rows if row["entity_group"] == entity_group]
            if not entity_rows:
                continue
            x_values = [float(row["x_m"]) for row in entity_rows]
            y_values = [float(row["y_m"]) for row in entity_rows]
            marker = {"truth_target": "*", "fft_masked": "o", "music_masked": "x"}.get(entity_group, ".")
            size = 110 if entity_group == "truth_target" else 55
            ax.scatter(
                x_values,
                y_values,
                c=ENTITY_COLORS[entity_group],
                marker=marker,
                s=size,
                alpha=0.9 if entity_group in ("truth_target", "music_masked") else 0.7,
                label=entity_group.replace("_", " "),
            )
        for row in [row for row in group_rows if row["entity_kind"] in ("truth_target", "detection")]:
            ax.annotate(
                row["entity_label"],
                (float(row["x_m"]), float(row["y_m"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Cross-Range x (m)")
        ax.set_ylabel("Down-Range y (m)")
        ax.set_title(f"Representative Scene Geometry: {anchor_label} / {scene_label}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"scene_geometry_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def _plot_scene_geometry_3d(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for entity_group in ("truth_target", "static_clutter", "multipath", "fft_masked", "music_masked"):
            entity_rows = [row for row in group_rows if row["entity_group"] == entity_group]
            if not entity_rows:
                continue
            ax.scatter(
                [float(row["range_m"]) for row in entity_rows],
                [float(row["azimuth_deg"]) for row in entity_rows],
                [float(row["velocity_mps"]) for row in entity_rows],
                c=ENTITY_COLORS[entity_group],
                marker={"truth_target": "*", "fft_masked": "o", "music_masked": "x"}.get(entity_group, "."),
                s=90 if entity_group == "truth_target" else 45,
                label=entity_group.replace("_", " "),
            )
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Azimuth (deg)")
        ax.set_zlabel("Velocity (m/s)")
        ax.set_title(f"Range / Angle / Doppler Geometry: {anchor_label} / {scene_label}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"scene_geometry_3d_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def _plot_range_doppler(rows: list[dict[str, str]], geometry_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    geometry_by_group = _group_rows(geometry_rows)
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        velocity_values = np.array(sorted({_to_float(row["velocity_mps"], 0.0) for row in group_rows}), dtype=float)
        range_values = np.array(sorted({_to_float(row["range_m"], 0.0) for row in group_rows}), dtype=float)
        image_grid = np.full((range_values.size, velocity_values.size), np.nan, dtype=float)
        range_index = {value: index for index, value in enumerate(range_values.tolist())}
        velocity_index = {value: index for index, value in enumerate(velocity_values.tolist())}
        for row in group_rows:
            image_grid[range_index[float(row["range_m"])], velocity_index[float(row["velocity_mps"])]] = float(row["power_db"])
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        image = ax.imshow(
            image_grid,
            aspect="auto",
            origin="lower",
            extent=[velocity_values[0], velocity_values[-1], range_values[0], range_values[-1]],
            cmap="viridis",
        )
        overlay_rows = geometry_by_group.get(group_key, [])
        truth_rows = [row for row in overlay_rows if row["entity_kind"] == "truth_target"]
        fft_rows = [row for row in overlay_rows if row["entity_group"] == "fft_masked"]
        music_rows = [row for row in overlay_rows if row["entity_group"] == "music_masked"]
        if truth_rows:
            ax.scatter(
                [float(row["velocity_mps"]) for row in truth_rows],
                [float(row["range_m"]) for row in truth_rows],
                c=ENTITY_COLORS["truth_target"],
                marker="*",
                s=130,
                label="Truth targets",
            )
        if fft_rows:
            ax.scatter(
                [float(row["velocity_mps"]) for row in fft_rows],
                [float(row["range_m"]) for row in fft_rows],
                c=METHOD_COLORS["fft_masked"],
                marker="o",
                s=55,
                label=METHOD_LABELS["fft_masked"],
            )
        if music_rows:
            ax.scatter(
                [float(row["velocity_mps"]) for row in music_rows],
                [float(row["range_m"]) for row in music_rows],
                c=METHOD_COLORS["music_masked"],
                marker="x",
                s=75,
                label=METHOD_LABELS["music_masked"],
            )
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Range (m)")
        ax.set_title(f"Representative Range-Doppler Heatmap: {anchor_label} / {scene_label}")
        ax.legend(fontsize=8)
        fig.colorbar(image, ax=ax, label="Power (dB)")
        fig.tight_layout()
        fig.savefig(output_dir / f"range_doppler_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def _plot_music_spectra(rows: list[dict[str, str]], geometry_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    geometry_by_group = _group_rows(geometry_rows)
    dimension_info = (
        ("azimuth", "Azimuth (deg)", "azimuth_deg"),
        ("range", "Range (m)", "range_m"),
        ("doppler", "Velocity (m/s)", "velocity_mps"),
    )
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.4), sharex=False)
        overlay_rows = geometry_by_group.get(group_key, [])
        truth_rows = [row for row in overlay_rows if row["entity_kind"] == "truth_target"]
        fft_rows = [row for row in overlay_rows if row["entity_group"] == "fft_masked"]
        music_rows = [row for row in overlay_rows if row["entity_group"] == "music_masked"]
        for ax, (dimension_name, xlabel, geometry_field) in zip(axes, dimension_info, strict=True):
            dim_rows = [row for row in group_rows if row["dimension"] == dimension_name]
            if not dim_rows:
                continue
            dim_rows.sort(key=lambda row: float(row["coordinate_value"]))
            x_values = np.asarray([float(row["coordinate_value"]) for row in dim_rows], dtype=float)
            y_values = np.asarray([float(row["spectrum_db_rel"]) for row in dim_rows], dtype=float)
            ax.plot(x_values, y_values, color=METHOD_COLORS["music_masked"], linewidth=2.0)
            for row in truth_rows:
                ax.axvline(float(row[geometry_field]), color=ENTITY_COLORS["truth_target"], linestyle="--", alpha=0.75)
            for row in fft_rows:
                ax.axvline(float(row[geometry_field]), color=METHOD_COLORS["fft_masked"], linestyle=":", alpha=0.5)
            for row in music_rows:
                ax.axvline(float(row[geometry_field]), color=METHOD_COLORS["music_masked"], linestyle="-.", alpha=0.6)
            ax.set_ylabel("Spectrum (dB rel.)")
            ax.set_xlabel(xlabel)
            ax.set_title(f"{dimension_name.title()} MUSIC Spectrum")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Representative MUSIC Spectra: {anchor_label} / {scene_label}")
        fig.tight_layout()
        fig.savefig(output_dir / f"music_spectra_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def _plot_fbss_ablation_nominal(rows: list[dict[str, str]], output_dir: Path) -> None:
    nominal_rows = [row for row in rows if row["sweep_name"] == "nominal"]
    if not nominal_rows:
        return
    metrics = [
        ("joint_detection_probability", "Pdet"),
        ("joint_resolution_probability", "Pjoint"),
        ("range_resolution_probability", "Prange"),
        ("velocity_resolution_probability", "Pvel"),
        ("angle_resolution_probability", "Pangle"),
    ]
    grouped = _group_rows(nominal_rows)
    group_labels = [f"{anchor_label}\n{scene_label}" for _anchor, anchor_label, _scene, scene_label in grouped]
    x_positions = np.arange(len(group_labels), dtype=float)
    bar_width = 0.18
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.8), sharey=True)
    for ax, (metric_name, metric_label) in zip(axes, metrics, strict=True):
        for variant_index, variant_name in enumerate(FBSS_ABLATION_ORDER):
            values = []
            for key in grouped:
                variant_rows = [row for row in grouped[key] if row["method"] == variant_name]
                values.append(_to_float(variant_rows[0][metric_name], 0.0) if variant_rows else 0.0)
            ax.bar(
                x_positions + (variant_index - 1.5) * bar_width,
                values,
                width=bar_width,
                color=FBSS_ABLATION_COLORS[variant_name],
                label=FBSS_ABLATION_LABELS[variant_name],
            )
        ax.set_title(metric_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=8)
    fig.suptitle("FBSS Ablation at the Nominal Point")
    fig.tight_layout()
    fig.savefig(output_dir / "fbss_ablation_nominal_from_csv.png", dpi=180)
    plt.close(fig)


def _plot_fbss_ablation_support_sweeps(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    metric_pairs = {
        "bandwidth_span": (
            ("joint_resolution_probability", "Joint Resolution Probability"),
            ("range_resolution_probability", "Range Resolution Probability"),
        ),
        "slow_time_span": (
            ("joint_resolution_probability", "Joint Resolution Probability"),
            ("velocity_resolution_probability", "Doppler Resolution Probability"),
        ),
    }
    for sweep_name, metrics in metric_pairs.items():
        sweep_rows = [row for row in rows if row["sweep_name"] == sweep_name]
        grouped = _group_rows(sweep_rows)
        if not grouped:
            continue
        fig, axes = plt.subplots(len(grouped), len(metrics), figsize=(5.6 * len(metrics), 3.8 * len(grouped)), squeeze=False)
        for row_index, (group_key, group_rows) in enumerate(grouped.items()):
            _anchor, anchor_label, _scene, scene_label = group_key
            for col_index, (metric_name, metric_label) in enumerate(metrics):
                ax = axes[row_index, col_index]
                for variant_name in FBSS_ABLATION_ORDER:
                    variant_rows = [row for row in group_rows if row["method"] == variant_name]
                    if not variant_rows:
                        continue
                    variant_rows.sort(key=lambda row: float(row["parameter_numeric_value"]))
                    x_values = np.asarray([float(row["parameter_numeric_value"]) for row in variant_rows], dtype=float)
                    y_values = np.asarray([float(row[metric_name]) for row in variant_rows], dtype=float)
                    ax.plot(
                        x_values,
                        y_values,
                        marker="o",
                        linewidth=2.0,
                        color=FBSS_ABLATION_COLORS[variant_name],
                        label=FBSS_ABLATION_LABELS[variant_name],
                    )
                ax.set_title(f"{anchor_label} / {scene_label} / {metric_label}")
                ax.set_xlabel(group_rows[0]["parameter_label"])
                ax.set_ylabel("Probability")
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, alpha=0.25)
                if row_index == 0 and col_index == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"fbss_ablation_{sweep_name}_from_csv.png", dpi=180)
        plt.close(fig)


def _plot_fbss_ablation_spectra(rows: list[dict[str, str]], geometry_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    geometry_by_group = _group_rows(geometry_rows)
    dimension_info = (
        ("azimuth", "Azimuth (deg)", "azimuth_deg"),
        ("range", "Range (m)", "range_m"),
        ("doppler", "Velocity (m/s)", "velocity_mps"),
    )
    for group_key, group_rows in _group_rows(rows).items():
        anchor, anchor_label, scene_class, scene_label = group_key
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.8), sharex=False)
        truth_rows = [row for row in geometry_by_group.get(group_key, []) if row["entity_kind"] == "truth_target"]
        for ax, (dimension_name, xlabel, truth_field) in zip(axes, dimension_info, strict=True):
            dim_rows = [row for row in group_rows if row["dimension"] == dimension_name]
            if not dim_rows:
                continue
            for variant_name in FBSS_ABLATION_ORDER:
                variant_rows = [row for row in dim_rows if row["series_name"] == variant_name]
                if not variant_rows:
                    continue
                variant_rows.sort(key=lambda row: float(row["coordinate_value"]))
                x_values = np.asarray([float(row["coordinate_value"]) for row in variant_rows], dtype=float)
                y_values = np.asarray([float(row["spectrum_db_rel"]) for row in variant_rows], dtype=float)
                ax.plot(
                    x_values,
                    y_values,
                    linewidth=2.0,
                    color=FBSS_ABLATION_COLORS[variant_name],
                    label=FBSS_ABLATION_LABELS[variant_name],
                )
            for row in truth_rows:
                ax.axvline(float(row[truth_field]), color=ENTITY_COLORS["truth_target"], linestyle="--", alpha=0.75)
            ax.set_title(f"{dimension_name.title()} MUSIC Spectrum")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Spectrum (dB rel.)")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Representative FBSS Ablation Spectra: {anchor_label} / {scene_label}")
        fig.tight_layout()
        fig.savefig(output_dir / f"fbss_ablation_spectra_{anchor}_{scene_class}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir, output_dir = _resolve_paths(args.input_root, args.output_dir)
    _prepare_output_dir(output_dir, clean_output=args.clean_output)

    all_sweep_rows = _read_csv_rows(data_dir / "all_sweep_results.csv")
    trial_level_rows = _read_csv_rows(data_dir / "trial_level_results.csv")
    nominal_rows = _read_csv_rows(data_dir / "nominal_summary.csv")
    pilot_only_nominal_rows = _read_csv_rows(data_dir / "pilot_only_nominal_summary.csv")
    usefulness_rows = _read_csv_rows(data_dir / "usefulness_windows.csv")
    fbss_ablation_rows = _read_csv_rows(data_dir / "fbss_ablation_results.csv")
    failure_rows = _read_csv_rows(data_dir / "failure_modes.csv")
    mask_rows = _read_csv_rows(data_dir / "representative_resource_mask.csv")
    geometry_rows = _read_csv_rows(data_dir / "representative_scene_geometry.csv")
    range_doppler_rows = _read_csv_rows(data_dir / "representative_range_doppler.csv")
    music_spectrum_rows = _read_csv_rows(data_dir / "representative_music_spectra.csv")
    fbss_ablation_spectrum_rows = _read_csv_rows(data_dir / "representative_fbss_ablation_spectra.csv")

    _plot_nominal_summary(nominal_rows, output_dir)
    _plot_pilot_only_nominal(nominal_rows, pilot_only_nominal_rows, output_dir)
    _plot_trial_level_joint_rmse(trial_level_rows, output_dir)
    _plot_sweep_figures(all_sweep_rows, output_dir)
    _plot_usefulness_overview(usefulness_rows, output_dir)
    _plot_failure_overview(failure_rows, output_dir)
    _plot_resource_masks(mask_rows, output_dir)
    _plot_scene_geometry(geometry_rows, output_dir)
    _plot_scene_geometry_3d(geometry_rows, output_dir)
    _plot_range_doppler(range_doppler_rows, geometry_rows, output_dir)
    _plot_music_spectra(music_spectrum_rows, geometry_rows, output_dir)
    _plot_fbss_ablation_nominal(fbss_ablation_rows, output_dir)
    _plot_fbss_ablation_support_sweeps(fbss_ablation_rows, output_dir)
    _plot_fbss_ablation_spectra(fbss_ablation_spectrum_rows, geometry_rows, output_dir)

    print(f"Input data directory: {data_dir}")
    print(f"Wrote CSV-driven figures to: {output_dir}")


if __name__ == "__main__":
    main()
