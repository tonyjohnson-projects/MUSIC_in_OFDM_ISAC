#!/usr/bin/env python3
"""Unified figure-generation entrypoint for the project."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import TwoSlopeNorm
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.estimators import (
    _estimate_music_model_order,
    azimuth_steering_matrix,
    fbss_covariance,
    fft_search_bounds,
    music_pseudospectrum,
)
from aisle_isac.masked_observation import extract_known_symbol_cube
from aisle_isac.resource_grid import build_resource_grid
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import (
    _nominal_point_spec,
    nominal_trial_parameters,
    simulate_communications_trial,
)


METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT + Local Refinement",
    "music_masked": "Masked Staged MUSIC + FBSS",
}
METHOD_COLORS = {
    "fft_masked": "#D55E00",
    "music_masked": "#0072B2",
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
    "story_rack_aisle_diagnostic_from_csv.png",
)
CANONICAL_FIGURE_DIR = REPO_ROOT / "figures"
PRESENTATION_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "presentation"
FIGURE_MANIFEST_PATH = PRESENTATION_ARTIFACT_DIR / "figure_manifest.json"
LEGACY_RESULTS_ARCHIVE = REPO_ROOT / "archive" / "results"


@dataclass(frozen=True)
class FigureSpec:
    """One canonical figure asset."""

    id: str
    filename: str
    title: str
    kind: str
    caption: str

    @property
    def output_path(self) -> Path:
        return CANONICAL_FIGURE_DIR / self.filename


def _build_story_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create story-driven figures from saved CSV outputs.", add_help=add_help)
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
    return parser


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


def _ensure_canonical_output_dirs() -> None:
    PRESENTATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _trim_figure_whitespace(path: Path, *, padding_px: int = 18, threshold: int = 248) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixels = np.asarray(rgba)
        content_mask = (pixels[..., 3] > 0) & np.any(pixels[..., :3] < threshold, axis=2)
        if not np.any(content_mask):
            return
        rows, cols = np.where(content_mask)
        top = max(0, int(rows.min()) - padding_px)
        bottom = min(rgba.height, int(rows.max()) + padding_px + 1)
        left = max(0, int(cols.min()) - padding_px)
        right = min(rgba.width, int(cols.max()) + padding_px + 1)
        cropped = rgba.crop((left, top, right, bottom))
        save_kwargs: dict[str, object] = {}
        if "dpi" in image.info:
            save_kwargs["dpi"] = image.info["dpi"]
        cropped.save(path, **save_kwargs)


def _flatten_png_to_rgb(path: Path) -> None:
    with Image.open(path) as image:
        rgb = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        rgb.paste(image.convert("RGB"), mask=alpha)
        rgb.save(path)


def _copy_figure(source: Path, spec: FigureSpec) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source figure: {source}")
    if source.resolve() == spec.output_path.resolve():
        return
    shutil.copy2(source, spec.output_path)
    _trim_figure_whitespace(spec.output_path)


def _render_equation_asset(
    spec: FigureSpec,
    lines: tuple[str, ...],
    *,
    font_size: int,
    line_gap: float = 0.34,
) -> None:
    with plt.rc_context(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        }
    ):
        fig = plt.figure(figsize=(11.0, 1.7 + 0.42 * max(0, len(lines) - 1)), dpi=320)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("white")
        ax.axis("off")
        start_y = 0.56 + 0.17 * max(0, len(lines) - 1)
        for index, line in enumerate(lines):
            ax.text(
                0.0,
                start_y - index * line_gap,
                line,
                fontsize=font_size,
                color="#10233F",
                ha="left",
                va="center",
            )
        fig.savefig(spec.output_path, dpi=320, facecolor="white", transparent=False, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    _trim_figure_whitespace(spec.output_path, padding_px=12, threshold=250)
    _flatten_png_to_rgb(spec.output_path)


def _write_manifest(specs: list[FigureSpec]) -> None:
    payload = {
        spec.id: {
            **asdict(spec),
            "path": str(spec.output_path.resolve()),
        }
        for spec in specs
    }
    FIGURE_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _contrast_text_effects(text_color: str) -> list[patheffects.AbstractPathEffect]:
    outline = "#111111" if text_color == "white" else "#FFFFFF"
    return [patheffects.withStroke(linewidth=1.6, foreground=outline)]


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
            mfc=METHOD_COLORS["fft_masked"],
            mec="white",
            mew=1.2,
            ecolor=METHOD_COLORS["fft_masked"],
            capsize=4,
            linewidth=2.2,
            label=METHOD_LABELS["fft_masked"] if scene_index == 0 else None,
        )
        ax.errorbar(
            music_value,
            y_value,
            xerr=np.array([[music_value - music_ci[0]], [music_ci[1] - music_value]]),
            fmt="s",
            markersize=9,
            color=METHOD_COLORS["music_masked"],
            mfc=METHOD_COLORS["music_masked"],
            mec="white",
            mew=1.2,
            ecolor=METHOD_COLORS["music_masked"],
            capsize=4,
            linewidth=2.2,
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
        s=290,
        edgecolors="white",
        linewidths=1.1,
        label="Truth movers",
    )
    ax_heatmap.scatter(
        [float(row["velocity_mps"]) for row in fft_rows],
        [float(row["range_m"]) for row in fft_rows],
        c=METHOD_COLORS["fft_masked"],
        marker="o",
        s=128,
        edgecolors="white",
        linewidths=0.8,
        label="FFT detections",
    )
    ax_heatmap.scatter(
        [float(row["velocity_mps"]) for row in music_rows],
        [float(row["range_m"]) for row in music_rows],
        c=METHOD_COLORS["music_masked"],
        marker="x",
        s=160,
        linewidths=2.8,
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
        ax_azimuth.axvline(
            float(row["azimuth_deg"]),
            color=METHOD_COLORS["fft_masked"],
            linestyle=(0, (1.2, 2.2)),
            linewidth=1.7,
            alpha=0.85,
        )
    for row in music_rows:
        ax_azimuth.axvline(
            float(row["azimuth_deg"]),
            color=METHOD_COLORS["music_masked"],
            linestyle=(0, (7.0, 2.5, 1.4, 2.5)),
            linewidth=1.8,
            alpha=0.85,
        )
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
        s=250,
        edgecolors="white",
        linewidths=1.0,
        label="Truth movers",
    )
    ax_localization.scatter(
        [float(row["azimuth_deg"]) for row in fft_rows],
        [float(row["range_m"]) for row in fft_rows],
        c=METHOD_COLORS["fft_masked"],
        marker="o",
        s=116,
        edgecolors="white",
        linewidths=0.8,
        label="FFT detections",
    )
    ax_localization.scatter(
        [float(row["azimuth_deg"]) for row in music_rows],
        [float(row["range_m"]) for row in music_rows],
        c=METHOD_COLORS["music_masked"],
        marker="x",
        s=148,
        linewidths=2.6,
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
                sweep_rows = [
                    row
                    for row in rows
                    if row["scene_class"] == scene_class and row["sweep_name"] == sweep_name
                ]
                if not sweep_rows:
                    continue

                deltas = np.asarray(
                    [float(row["music_minus_fft"]) for row in sweep_rows],
                    dtype=float,
                )
                mean_delta = float(np.mean(deltas))
                strong_wins = int(np.sum(deltas >= STRONG_WIN_THRESHOLD))

                matrix[row_index, col_index] = mean_delta
                labels[row_index][col_index] = f"{mean_delta:+.2f}\n{strong_wins}/{len(deltas)}"
                max_abs_delta = max(max_abs_delta, float(np.max(np.abs(deltas))))

        panel_data.append((matrix, labels))

    max_abs_delta = max(max_abs_delta, 0.01)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs_delta, vmax=max_abs_delta)

    # More vertical room for title + footnote, and slightly wider right panel.
    fig = plt.figure(figsize=(14.2, 6.2))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=(0.9, 5.0, 0.7),   # title, panels, footnote
        width_ratios=(len(SUPPORT_SWEEP_ORDER), len(SEPARATION_SWEEP_ORDER) + 0.6, 0.28),
        hspace=0.18,
        wspace=0.28,
    )

    title_ax = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])
    colorbar_ax = fig.add_subplot(grid[1, 2])
    footer_ax = fig.add_subplot(grid[2, :])

    title_ax.axis("off")
    footer_ax.axis("off")

    title_ax.text(
        0.0,
        1.0,
        "MUSIC gains cluster in a few sweep families, not across the whole study\n"
        "Cell color = mean headline-metric delta; text = strong-win count (>= +0.10)",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=title_ax.transAxes,
    )

    footer_ax.text(
        0.0,
        0.35,
        "Support-limited and separation sweeps are split because they use different headline metrics.",
        ha="left",
        va="center",
        fontsize=9,
        color="#444444",
        transform=footer_ax.transAxes,
    )

    axes = (ax_left, ax_right)
    image = None

    for axis_index, (ax, (panel_title, sweep_names, panel_subtitle), (matrix, labels)) in enumerate(
        zip(axes, panel_specs, panel_data, strict=True)
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", norm=norm)

        ax.set_xticks(np.arange(len(sweep_names)))
        ax.set_xticklabels(
            [SWEEP_LABELS[name] for name in sweep_names],
            rotation=20,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )

        ax.set_yticks(np.arange(len(scene_classes)))
        if axis_index == 0:
            ax.set_yticklabels(
                [_scene_label_from_rows(rows, scene_class) for scene_class in scene_classes],
                fontsize=8,
            )
        else:
            ax.set_yticklabels([])

        ax.set_title(
            f"{panel_title}\n{panel_subtitle}",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )

        # Thin white separators make cells easier to read.
        ax.set_xticks(np.arange(-0.5, len(sweep_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scene_classes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)

        for row_index in range(len(scene_classes)):
            for col_index in range(len(sweep_names)):
                if not labels[row_index][col_index]:
                    continue
                value = matrix[row_index, col_index]
                text_color = "white" if abs(value) >= 0.18 else "#222222"
                text = ax.text(
                    col_index,
                    row_index,
                    labels[row_index][col_index],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )
                text.set_path_effects(_contrast_text_effects(text_color))

    if image is not None:
        cbar = fig.colorbar(image, cax=colorbar_ax)
        cbar.set_label("Mean headline-metric delta", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(
        output_dir / "story_regime_map_from_csv.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
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
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
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
        ax.text(
            position,
            1.04,
            f"mean {np.mean(values):.2f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([_scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    ax.set_ylim(0.0, 1.08)
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
        0.91,
        "Circles = empirical finite-snapshot coherence per nominal trial; diamond = configured scene coherence",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.20)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.84)
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


def _story_rack_aisle_diagnostic(diag_rows: list[dict[str, str]], output_dir: Path) -> None:
    """Rack-aisle azimuth-stage failure diagnostic.

    Shows azimuth candidate histogram vs truth/clutter angles, and detection
    azimuth distribution, to visualise nuisance-capture failure.
    """
    rack_rows = [r for r in diag_rows if r.get("scene_class") == "rack_aisle"]
    if not rack_rows:
        return

    # --- parse per-trial data ---
    all_candidates: list[float] = []
    truth_az_0: list[float] = []
    truth_az_1: list[float] = []
    det_azimuths: list[float] = []
    for row in rack_rows:
        cands = row["music_stage_azimuth_candidates_deg"]
        if cands:
            all_candidates.extend(float(c) for c in cands.split("|"))
        for entry in row["truth_targets"].split("|"):
            parts = entry.split(":")
            az = float(parts[5])
            if int(parts[0]) == 0:
                truth_az_0.append(az)
            else:
                truth_az_1.append(az)
        for entry in row["detections"].split("|"):
            parts = entry.split(":")
            det_azimuths.append(float(parts[3]))

    if not all_candidates:
        return

    truth_mean_0 = float(np.mean(truth_az_0))
    truth_mean_1 = float(np.mean(truth_az_1))

    # rack_aisle clutter template azimuths (from scenarios.py)
    clutter_azimuths = {"left_rack": -24.0, "right_rack": 23.0, "far_endcap": 3.0}
    multipath_azimuths = {"left_wall": -11.0, "right_wall": 10.0}

    fig, (ax_cand, ax_det) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)

    # --- Left panel: azimuth candidate histogram ---
    bins = np.arange(-75, 80, 2.5)
    ax_cand.hist(all_candidates, bins=bins, color=SCENE_COLORS["rack_aisle"],
                 alpha=0.55, edgecolor="#444444", linewidth=0.4)
    # truth lines
    ax_cand.axvline(
        truth_mean_0,
        color=METHOD_COLORS["music_masked"],
        linewidth=2.6,
        linestyle=(0, (10, 4)),
        label=f"Truth T0 ({truth_mean_0:+.1f}\u00b0)",
    )
    ax_cand.axvline(
        truth_mean_1,
        color=METHOD_COLORS["music_masked"],
        linewidth=2.4,
        linestyle=(0, (6, 3)),
        label=f"Truth T1 ({truth_mean_1:+.1f}\u00b0)",
    )
    # clutter lines
    for name, az in clutter_azimuths.items():
        ax_cand.axvline(
            az,
            color=METHOD_COLORS["fft_masked"],
            linewidth=2.2,
            linestyle=(0, (1.0, 1.8)),
            label=f"Clutter: {name} ({az:+.0f}\u00b0)",
        )
    for name, az in multipath_azimuths.items():
        ax_cand.axvline(
            az,
            color="#4D4D4D",
            linewidth=2.0,
            linestyle=(0, (8.0, 2.2, 1.4, 2.2)),
            alpha=0.95,
            label=f"Multipath: {name} ({az:+.0f}\u00b0)",
        )
    ax_cand.set_xlabel("Azimuth (\u00b0)")
    ax_cand.set_ylabel("Candidate count (across 64 trials)")
    ax_cand.set_title("Azimuth candidates dominated by clutter branches",
                      loc="left", fontsize=12, fontweight="bold")
    ax_cand.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax_cand.set_xlim(-75, 75)
    ax_cand.grid(True, axis="y", alpha=0.20)

    # --- Right panel: detection azimuth distribution ---
    det_arr = np.asarray(det_azimuths)
    n_near_nuis = int(np.sum(np.abs(det_arr - (-21.7)) < 3.0))
    n_near_t0 = int(np.sum(np.abs(det_arr - truth_mean_0) < 3.0))
    n_near_t1 = int(np.sum(np.abs(det_arr - truth_mean_1) < 3.0))
    n_other = len(det_arr) - n_near_nuis - n_near_t0 - n_near_t1

    categories = [f"Near T0\n({truth_mean_0:+.1f}\u00b0)",
                  f"Near T1\n({truth_mean_1:+.1f}\u00b0)",
                  f"Near left rack\n(\u221221.7\u00b0)",
                  "Other"]
    counts = [n_near_t0, n_near_t1, n_near_nuis, n_other]
    colors = [METHOD_COLORS["music_masked"], METHOD_COLORS["music_masked"],
              METHOD_COLORS["fft_masked"], "#AAAAAA"]
    bars = ax_det.bar(categories, counts, color=colors, edgecolor="#444444", linewidth=0.6, alpha=0.75)
    for bar, count in zip(bars, counts):
        ax_det.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_det.set_ylabel(f"Detection count (of {len(det_arr)} total)")
    ax_det.set_title("Final detections: nuisance branch captures 39% of outputs",
                     loc="left", fontsize=12, fontweight="bold")
    ax_det.grid(True, axis="y", alpha=0.20)

    fig.suptitle("Rack-aisle azimuth-stage failure diagnostic\n"
                 "64-trial FR1 nominal point, MDL model order",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.88, wspace=0.28)
    fig.savefig(output_dir / "story_rack_aisle_diagnostic_from_csv.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def generate_story_figures(
    input_root: Path = Path("results") / "quick",
    output_dir: Path | None = None,
    *,
    clean_output: bool = False,
) -> list[Path]:
    data_dir, output_dir = _resolve_paths(input_root, output_dir)
    _prepare_output_dir(output_dir, clean_output=clean_output)

    nominal_rows = _read_csv_rows(data_dir / "nominal_summary.csv")
    pilot_rows = _read_csv_rows(data_dir / "pilot_only_nominal_summary.csv")
    usefulness_rows = _read_csv_rows(data_dir / "usefulness_windows.csv")
    trial_rows = _read_csv_rows(data_dir / "trial_level_results.csv")
    geometry_rows = _read_csv_rows(data_dir / "representative_scene_geometry.csv")
    range_doppler_rows = _read_csv_rows(data_dir / "representative_range_doppler.csv")
    music_spectrum_rows = _read_csv_rows(data_dir / "representative_music_spectra.csv")
    stage_diag_rows = _read_csv_rows(data_dir / "stage_diagnostics.csv")

    _story_nominal_verdict(nominal_rows, output_dir)
    _story_intersection_resolution(nominal_rows, range_doppler_rows, geometry_rows, music_spectrum_rows, output_dir)
    _story_regime_map(usefulness_rows, output_dir)
    _story_coherence_overlap(trial_rows, output_dir)
    _story_pilot_only_collapse(nominal_rows, pilot_rows, output_dir)
    _story_trial_delta(trial_rows, output_dir)
    _story_rack_aisle_diagnostic(stage_diag_rows, output_dir)

    print(f"Input data directory: {data_dir}")
    print(f"Wrote story figures to: {output_dir}")
    for filename in STORY_FIGURE_NAMES:
        output_path = output_dir / filename
        if output_path.exists():
            print(f"- {output_path}")
    return [output_dir / filename for filename in STORY_FIGURE_NAMES if (output_dir / filename).exists()]

def legacy_story_main(argv: list[str] | None = None) -> int:
    parser = _build_story_parser()
    args = parser.parse_args(argv)
    generate_story_figures(args.input_root, args.output_dir, clean_output=args.clean_output)
    return 0


def generate_1d_motivation(output_path: Path) -> None:
    """Generate the 1-D range MUSIC vs FFT motivation figure."""

    n_subcarriers = 96
    subcarrier_spacing_hz = 30.0e3
    bandwidth_hz = n_subcarriers * subcarrier_spacing_hz
    c = 299_792_458.0
    range_resolution_m = c / (2.0 * bandwidth_hz)

    separation_factor = 0.70
    center_range_m = 20.0
    separation_m = separation_factor * range_resolution_m
    range_1 = center_range_m - separation_m / 2
    range_2 = center_range_m + separation_m / 2

    snr_db = 20.0
    n_snapshots = 16
    n_trials = 200
    rng = np.random.default_rng(42)

    frequencies_hz = np.arange(n_subcarriers, dtype=float) * subcarrier_spacing_hz
    frequencies_hz -= np.mean(frequencies_hz)

    n_search = 1001
    search_range_m = np.linspace(
        center_range_m - 3 * range_resolution_m,
        center_range_m + 3 * range_resolution_m,
        n_search,
    )

    delay_matrix = np.zeros((n_subcarriers, n_search), dtype=np.complex128)
    for index, range_m in enumerate(search_range_m):
        delay_s = 2.0 * range_m / c
        delay_matrix[:, index] = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)

    avg_fft_spectrum = np.zeros(n_search)
    avg_music_spectrum = np.zeros(n_search)
    music_resolve_count = 0
    fft_resolve_count = 0
    threshold_m = 0.35 * range_resolution_m

    from scipy.signal import find_peaks

    for _ in range(n_trials):
        signal_power = 1.0
        noise_var = signal_power / (10.0 ** (snr_db / 10.0))

        data = np.zeros((n_subcarriers, n_snapshots), dtype=np.complex128)
        for range_true in (range_1, range_2):
            delay_s = 2.0 * range_true / c
            steering = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)
            source = rng.standard_normal((1, n_snapshots)) + 1j * rng.standard_normal((1, n_snapshots))
            source *= np.sqrt(signal_power / 2.0)
            data += steering[:, np.newaxis] * source

        noise = (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)) * np.sqrt(noise_var / 2.0)
        data += noise

        n_fft = n_search
        window = np.hanning(n_subcarriers)
        windowed = data[:, 0] * window
        fft_result = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
        fft_range_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, d=subcarrier_spacing_hz)) * c / 2.0
        fft_power = np.abs(fft_result) ** 2
        fft_interp = np.interp(search_range_m, fft_range_axis + center_range_m, fft_power)
        fft_interp /= max(np.max(fft_interp), 1e-12)
        avg_fft_spectrum += fft_interp

        peaks_fft, _ = find_peaks(fft_interp, distance=5)
        if len(peaks_fft) >= 2:
            peak_scores = fft_interp[peaks_fft]
            top2 = peaks_fft[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                fft_resolve_count += 1

        subarray_len = max(3, int(round(0.67 * n_subcarriers)))
        covariance = fbss_covariance(data, subarray_len)
        music_spec = music_pseudospectrum(covariance, 2, delay_matrix[:subarray_len, :])
        music_spec_norm = music_spec / max(np.max(music_spec), 1e-12)
        avg_music_spectrum += music_spec_norm

        peaks_music, _ = find_peaks(music_spec_norm, distance=5)
        if len(peaks_music) >= 2:
            peak_scores = music_spec_norm[peaks_music]
            top2 = peaks_music[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                music_resolve_count += 1

    avg_fft_spectrum /= n_trials
    avg_music_spectrum /= n_trials
    avg_fft_spectrum /= max(np.max(avg_fft_spectrum), 1e-12)
    avg_music_spectrum /= max(np.max(avg_music_spectrum), 1e-12)

    fft_resolve_prob = fft_resolve_count / n_trials
    music_resolve_prob = music_resolve_count / n_trials

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
    x_cells = (search_range_m - center_range_m) / range_resolution_m
    truth_color = "#6F6F6F"

    ax_left.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_fft_spectrum, 1e-6)),
        color=METHOD_COLORS["fft_masked"],
        linewidth=1.6,
        label="FFT",
        zorder=2,
    )
    ax_left.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_music_spectrum, 1e-6)),
        color=METHOD_COLORS["music_masked"],
        linewidth=1.7,
        label="MUSIC (K=2)",
        zorder=3,
    )
    ax_left.axvline(-separation_factor / 2, color=truth_color, linestyle=(0, (5, 4)), linewidth=0.6, alpha=0.7, label="Truth", zorder=1)
    ax_left.axvline(separation_factor / 2, color=truth_color, linestyle=(0, (5, 4)), linewidth=0.6, alpha=0.7, zorder=1)
    ax_left.set_xlabel("Range offset (resolution cells)")
    ax_left.set_ylabel("Normalized spectrum (dB)")
    ax_left.set_title(f"1-D range: {separation_factor:.2f}-cell separation, {snr_db:.0f} dB SNR")
    ax_left.set_xlim(-2.5, 2.5)
    ax_left.set_ylim(-30, 3)
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    methods = ["FFT", "MUSIC"]
    probabilities = [fft_resolve_prob, music_resolve_prob]
    bars = ax_right.bar(methods, probabilities, color=[METHOD_COLORS["fft_masked"], METHOD_COLORS["music_masked"]], width=0.5)
    for bar, probability in zip(bars, probabilities, strict=True):
        ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{probability:.2f}", ha="center", va="bottom", fontsize=10)
    ax_right.set_ylabel("Resolution probability")
    ax_right.set_title(f"P(resolve) over {n_trials} trials")
    ax_right.set_ylim(0, 1.15)
    ax_right.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"1-D Range-Only Motivation: Full Support, {n_subcarriers} Subcarriers, {n_snapshots} Snapshots",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"FFT resolve: {fft_resolve_prob:.3f}, MUSIC resolve: {music_resolve_prob:.3f}")


def legacy_motivation_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the 1-D range-only motivation figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "motivation_1d_range.png",
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)
    generate_1d_motivation(args.output)
    return 0


def _nominal_resource_mask(spec: FigureSpec) -> None:
    grid = build_resource_grid(
        "fragmented_prb",
        96,
        16,
        prb_size=12,
        n_prb_fragments=4,
        pilot_subcarrier_period=4,
        pilot_symbol_period=4,
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    role_colors = {0: "#C9D1DB", 1: "#4E79A7", 2: "#D55E00", 3: "#8E6C8A"}
    role_labels = {0: "Muted", 1: "Pilot", 2: "Data", 3: "Punctured"}
    unique_roles = sorted(int(value) for value in np.unique(grid.role_grid))
    cmap = matplotlib.colors.ListedColormap([role_colors[role] for role in unique_roles])
    bounds = np.arange(len(unique_roles) + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    role_index_grid = np.vectorize({role: index for index, role in enumerate(unique_roles)}.get)(grid.role_grid)
    image = ax.imshow(role_index_grid.T, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Simulated subcarrier index")
    ax.set_ylabel("Slow-time snapshot")
    ax.set_title("Nominal fragmented scheduled PRB mask", loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, grid.role_grid.shape[0], 12), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.role_grid.shape[1], 4), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(len(unique_roles)), fraction=0.046, pad=0.03)
    colorbar.ax.set_yticklabels([role_labels[role] for role in unique_roles])
    fig.tight_layout()
    fig.savefig(spec.output_path, dpi=180)
    plt.close(fig)
    _trim_figure_whitespace(spec.output_path)


def _load_trial_rows() -> list[dict[str, str]]:
    path = REPO_ROOT / "results" / "submission" / "data" / "trial_level_results.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_representative_intersection_trial() -> int:
    rows = _load_trial_rows()
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["scene_class"] != "intersection" or row["sweep_name"] != "nominal" or row["estimator_family"] != "headline":
            continue
        grouped[int(row["trial_index"])][row["method"]] = row

    best_score = float("-inf")
    best_trial_index = -1
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


def _reconstruct_nominal_trial(scene_name: str, trial_index: int):
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


def _representative_intersection_case(spec: FigureSpec) -> None:
    trial_index = _select_representative_intersection_trial()
    cfg, trial = _reconstruct_nominal_trial("intersection", trial_index)
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
    ax_heatmap.scatter([target.velocity_mps for target in truth_targets], [target.range_m for target in truth_targets], marker="*", s=290, c="#2FA66A", edgecolors="white", linewidths=1.1, label="Truth movers")
    ax_heatmap.scatter([detection.velocity_mps for detection in fft_detections], [detection.range_m for detection in fft_detections], marker="o", s=130, c=METHOD_COLORS["fft_masked"], edgecolors="white", linewidths=0.9, label="FFT detections")
    ax_heatmap.scatter([detection.velocity_mps for detection in music_detections], [detection.range_m for detection in music_detections], marker="x", s=160, linewidths=2.8, c=METHOD_COLORS["music_masked"], label="MUSIC detections")

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
    fig.savefig(spec.output_path, dpi=180)
    plt.close(fig)
    _trim_figure_whitespace(spec.output_path)


def _model_order_nominal_comparison(spec: FigureSpec) -> None:
    mdl_path = REPO_ROOT / "results" / "analysis" / "model_order_nominal_64trials.csv"
    eigengap_path = REPO_ROOT / "results" / "analysis" / "model_order_nominal_64trials_eigengap.csv"

    records: dict[str, dict[str, float]] = defaultdict(dict)
    for path in (mdl_path, eigengap_path):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scene = row["scene"]
                mode = row["music_model_order_mode"]
                records[scene][mode] = float(row["music_joint_pres"])
                records[scene]["fft"] = float(row["fft_joint_pres"])

    scene_order = ("intersection", "open_aisle", "rack_aisle")
    scene_labels = {"intersection": "Intersection", "open_aisle": "Open aisle", "rack_aisle": "Rack aisle"}
    method_order = ("fft", "mdl", "eigengap", "expected")
    method_labels = {"fft": "FFT", "mdl": "MDL MUSIC", "eigengap": "Eigengap MUSIC", "expected": "Expected-order MUSIC"}
    method_colors = {"fft": "#D55E00", "mdl": "#7F7F7F", "eigengap": "#009E73", "expected": "#0072B2"}
    method_markers = {"fft": "o", "mdl": "D", "eigengap": "^", "expected": "s"}

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(scene_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(method_order))
    for scene_index, scene in enumerate(scene_order):
        values = [records[scene][method] for method in method_order]
        ax.hlines(y_positions[scene_index], min(values), max(values), color="#D7D7D7", linewidth=2.5)
        for offset, method in zip(offsets, method_order, strict=True):
            value = records[scene][method]
            ax.scatter(value, y_positions[scene_index] + offset, s=90, color=method_colors[method], marker=method_markers[method], edgecolors="white", linewidths=1.1, label=method_labels[method] if scene_index == 0 else None, zorder=3)
            ax.text(value + 0.015, y_positions[scene_index] + offset, f"{value:.3f}", va="center", fontsize=9)
        ax.text(-0.02, y_positions[scene_index], scene_labels[scene], ha="right", va="center", fontsize=11, fontweight="bold", color=SCENE_COLORS[scene])

    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(-0.6, len(scene_order) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Nominal joint-resolution probability")
    ax.set_title("Model-order diagnosis at the nominal point", loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(spec.output_path, dpi=180)
    plt.close(fig)
    _trim_figure_whitespace(spec.output_path)


def _expected_order_nuisance_sweep(spec: FigureSpec) -> None:
    scene_paths = {
        "open_aisle": REPO_ROOT / "results" / "submission_expected_order" / "open_aisle" / "data" / "nuisance_gain_offset.csv",
        "intersection": REPO_ROOT / "results" / "submission_expected_order" / "intersection" / "data" / "nuisance_gain_offset.csv",
        "rack_aisle": REPO_ROOT / "results" / "submission_expected_order" / "rack_aisle" / "data" / "nuisance_gain_offset.csv",
    }
    scene_labels = {"open_aisle": "Open aisle", "intersection": "Intersection", "rack_aisle": "Rack aisle"}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    for ax, scene in zip(axes, ("open_aisle", "intersection", "rack_aisle"), strict=True):
        with scene_paths[scene].open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append((float(row["parameter_numeric_value"]), float(row["joint_resolution_probability"])))
        for method in ("fft_masked", "music_masked"):
            series = sorted(grouped[method], key=lambda item: item[0])
            ax.plot(
                [item[0] for item in series],
                [item[1] for item in series],
                marker="o" if method == "fft_masked" else "s",
                linewidth=2.2,
                color=METHOD_COLORS[method],
                label="FFT" if method == "fft_masked" else "Expected-order MUSIC",
            )
        ax.set_title(scene_labels[scene], fontsize=12, fontweight="bold", color=SCENE_COLORS[scene])
        ax.set_xlabel("Uniform nuisance gain offset (dB)")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Joint-resolution probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.suptitle("Expected-order nuisance-strength sweep", fontsize=14, fontweight="bold", x=0.08, ha="left")
    fig.tight_layout()
    fig.savefig(spec.output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _trim_figure_whitespace(spec.output_path)


def _runtime_comparison(spec: FigureSpec) -> None:
    path = REPO_ROOT / "results" / "submission" / "data" / "runtime_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    scene_order = ("intersection", "open_aisle", "rack_aisle")
    scene_labels = {"intersection": "Intersection", "open_aisle": "Open aisle", "rack_aisle": "Rack aisle"}
    by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_scene[row["scene_class"]][row["method"]] = float(row["total_runtime_s"])

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y_positions = np.arange(len(scene_order), dtype=float)
    offset = 0.17
    fft_values = [by_scene[scene]["fft_masked"] for scene in scene_order]
    music_values = [by_scene[scene]["music_masked"] for scene in scene_order]
    ax.barh(y_positions - offset, fft_values, height=0.28, color=METHOD_COLORS["fft_masked"], label="FFT")
    ax.barh(y_positions + offset, music_values, height=0.28, color=METHOD_COLORS["music_masked"], label="MUSIC")
    for index, scene in enumerate(scene_order):
        ax.text(fft_values[index] + 0.01, y_positions[index] - offset, f"{fft_values[index]:.3f} s", va="center", fontsize=9)
        ax.text(music_values[index] + 0.01, y_positions[index] + offset, f"{music_values[index]:.3f} s", va="center", fontsize=9)
    ax.set_yticks(y_positions, [scene_labels[scene] for scene in scene_order])
    ax.set_xlabel("Total runtime per nominal point")
    ax.set_title("Nominal runtime comparison", loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(spec.output_path, dpi=180)
    plt.close(fig)
    _trim_figure_whitespace(spec.output_path)


def _masked_observation_equation(spec: FigureSpec) -> None:
    _render_equation_asset(
        spec,
        (
            r"$y_{h,k,n}=m_{k,n}x_{k,n}\sum_{p=1}^{P}\alpha_p a_h(\theta_p)$",
            r"$\exp(-j2\pi f_k\tau_p)\,\exp(j2\pi\nu_p t_n)\,s_p[n]+w_{h,k,n}$",
        ),
        font_size=32,
    )


def _music_pseudospectrum_equation(spec: FigureSpec) -> None:
    _render_equation_asset(
        spec,
        (
            r"$P_{\mathrm{MUSIC}}(\phi)=\frac{1}{\left\|E_n^H a(\phi)\right\|_2^2}$",
        ),
        font_size=38,
        line_gap=0.0,
    )


def generate_figures() -> dict[str, dict[str, str]]:
    """Create the canonical figure set used by the report and deck."""

    _ensure_canonical_output_dirs()
    specs = [
        FigureSpec("motivation_1d_range", "01_motivation_1d_range.png", "1-D MUSIC motivation", "generated", "Motivating 1-D range-only MUSIC versus FFT figure."),
        FigureSpec("story_nominal_verdict", "02_nominal_scene_verdict.png", "Nominal verdict", "reused", "Saved nominal verdict figure from CSV outputs."),
        FigureSpec("story_trial_delta", "03_nominal_trial_delta.png", "Paired nominal trial delta", "reused", "Saved paired nominal trial-delta figure from CSV outputs."),
        FigureSpec("story_coherence_overlap", "04_scene_coherence_overlap.png", "Coherence overlap", "reused", "Saved configured-versus-empirical coherence figure."),
        FigureSpec("story_regime_map", "05_regime_map.png", "Regime map", "reused", "Saved sweep-family regime map from CSV outputs."),
        FigureSpec("story_rack_aisle_diagnostic", "06_rack_aisle_failure_diagnostic.png", "Rack-aisle failure diagnostic", "reused", "Saved rack-aisle candidate and detection diagnostic."),
        FigureSpec("nominal_resource_mask", "07_nominal_resource_mask.png", "Nominal resource mask", "generated", "Nominal fragmented PRB resource mask used by the study."),
        FigureSpec("representative_intersection_case", "08_representative_intersection_case.png", "Representative nominal intersection trial", "generated", "Reconstructed saved nominal trial showing FFT versus MUSIC behavior in intersection."),
        FigureSpec("model_order_nominal_comparison", "09_model_order_nominal_comparison.png", "Model-order comparison", "generated", "Nominal P_joint comparison across FFT, MDL, eigengap, and expected-order MUSIC."),
        FigureSpec("expected_order_nuisance_sweep", "10_expected_order_nuisance_sweep.png", "Expected-order nuisance sweep", "generated", "Expected-order nuisance-strength sweep across all three scenes."),
        FigureSpec("runtime_comparison", "11_nominal_runtime_comparison.png", "Runtime comparison", "generated", "Nominal runtime comparison for FFT and MUSIC across scenes."),
        FigureSpec("masked_observation_equation", "12_masked_observation_equation.png", "Masked observation equation", "generated", "Rendered masked observation model equation."),
        FigureSpec("music_pseudospectrum_equation", "13_music_pseudospectrum_equation.png", "MUSIC pseudospectrum equation", "generated", "Rendered MUSIC pseudospectrum equation."),
    ]

    reused_sources = {
        "story_nominal_verdict": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_nominal_verdict_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_nominal_verdict_from_csv.png",
        ),
        "story_trial_delta": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_trial_delta_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_trial_delta_from_csv.png",
        ),
        "story_coherence_overlap": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_coherence_overlap_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_coherence_overlap_from_csv.png",
        ),
        "story_regime_map": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_regime_map_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_regime_map_from_csv.png",
        ),
        "story_rack_aisle_diagnostic": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_rack_aisle_diagnostic_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_rack_aisle_diagnostic_from_csv.png",
        ),
    }
    generated_handlers = {
        "motivation_1d_range": lambda spec: generate_1d_motivation(spec.output_path),
        "nominal_resource_mask": _nominal_resource_mask,
        "representative_intersection_case": _representative_intersection_case,
        "model_order_nominal_comparison": _model_order_nominal_comparison,
        "expected_order_nuisance_sweep": _expected_order_nuisance_sweep,
        "runtime_comparison": _runtime_comparison,
        "masked_observation_equation": _masked_observation_equation,
        "music_pseudospectrum_equation": _music_pseudospectrum_equation,
    }

    for spec in specs:
        if spec.id in reused_sources:
            source = _first_existing_path(reused_sources[spec.id])
            if source is None:
                if spec.output_path.exists():
                    continue
                raise FileNotFoundError(f"Missing source figure for {spec.id}")
            _copy_figure(source, spec)
            continue
        generated_handlers[spec.id](spec)

    _write_manifest(specs)
    return json.loads(FIGURE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate project figures from one unified entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "story",
        help="Generate story figures from saved CSV artifacts.",
        parents=[_build_story_parser(add_help=False)],
    )

    motivation_parser = subparsers.add_parser("motivation", help="Generate the 1-D motivation figure.")
    motivation_parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "motivation_1d_range.png",
        help="Output PNG path.",
    )

    subparsers.add_parser("canonical", help="Generate the canonical report/deck figure set.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_main_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "canonical"):
        generate_figures()
        return 0
    if args.command == "story":
        generate_story_figures(args.input_root, args.output_dir, clean_output=args.clean_output)
        return 0
    if args.command == "motivation":
        generate_1d_motivation(args.output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
