"""Figure generation for the final presentation deck."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "presentation"
FIGURE_OUTPUT_DIR = REPO_ROOT / "figures"
FIGURE_MANIFEST_PATH = OUTPUT_ROOT / "figure_manifest.json"
LEGACY_RESULTS_ARCHIVE = REPO_ROOT / "archive" / "results"

METHOD_COLORS = {
    "fft_masked": "#D55E00",
    "music_masked": "#0072B2",
}
SCENE_COLORS = {
    "intersection": "#2F6B9A",
    "open_aisle": "#A65E2E",
    "rack_aisle": "#6B6B6B",
}


@dataclass(frozen=True)
class FigureSpec:
    """One generated or reused figure asset."""

    id: str
    filename: str
    title: str
    kind: str
    caption: str

    @property
    def output_path(self) -> Path:
        return FIGURE_OUTPUT_DIR / self.filename


def _ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _copy_figure(source: Path, spec: FigureSpec) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source figure: {source}")
    if source.resolve() == spec.output_path.resolve():
        return
    shutil.copy2(source, spec.output_path)
    _trim_figure_whitespace(spec.output_path)


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
        save_kwargs = {}
        if "dpi" in image.info:
            save_kwargs["dpi"] = image.info["dpi"]
        cropped.save(path, **save_kwargs)


def _flatten_png_to_rgb(path: Path) -> None:
    with Image.open(path) as image:
        rgb = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        rgb.paste(image.convert("RGB"), mask=alpha)
        rgb.save(path)


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
    role_colors = {
        0: "#C9D1DB",  # muted REs: darker than near-white so they still show on projectors
        1: "#4E79A7",
        2: "#D55E00",
        3: "#8E6C8A",
    }
    role_labels = {
        0: "Muted",
        1: "Pilot",
        2: "Data",
        3: "Punctured",
    }
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
        if music_row["joint_resolution_success"] != "1" or fft_row["joint_resolution_success"] != "0":
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
    known_truth = sorted((target.azimuth_deg for target in truth_targets))
    known_fft = sorted((detection.azimuth_deg for detection in fft_detections))
    known_music = sorted((detection.azimuth_deg for detection in music_detections))

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
        [d.velocity_mps for d in fft_detections],
        [d.range_m for d in fft_detections],
        marker="o",
        s=130,
        c=METHOD_COLORS["fft_masked"],
        edgecolors="white",
        linewidths=0.9,
        label="FFT detections",
    )
    ax_heatmap.scatter(
        [d.velocity_mps for d in music_detections],
        [d.range_m for d in music_detections],
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
    ax_heatmap.set_title("Saved nominal intersection trial 55", loc="left", fontsize=13, fontweight="bold")
    ax_heatmap.set_xlabel("Velocity (m/s)")
    ax_heatmap.set_ylabel("Range (m)")
    ax_heatmap.legend(frameon=False, loc="lower right", fontsize=9)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.03, label="FFT range-Doppler power (dB)")

    ax_azimuth.plot(azimuth_grid, azimuth_spectrum_db, color=METHOD_COLORS["music_masked"], linewidth=2.3)
    for azimuth in known_truth:
        ax_azimuth.axvline(azimuth, color="#2FA66A", linestyle="--", linewidth=1.4, alpha=0.9)
    for azimuth in known_fft:
        ax_azimuth.axvline(
            azimuth,
            color=METHOD_COLORS["fft_masked"],
            linestyle=(0, (1.2, 2.2)),
            linewidth=1.7,
            alpha=0.85,
        )
    for azimuth in known_music:
        ax_azimuth.axvline(
            azimuth,
            color=METHOD_COLORS["music_masked"],
            linestyle=(0, (7.0, 2.5, 1.4, 2.5)),
            linewidth=1.8,
            alpha=0.85,
        )
    ax_azimuth.set_xlim(-5.0, 25.0)
    ax_azimuth.set_ylim(min(-42.0, float(np.min(azimuth_spectrum_db)) - 1.0), 2.0)
    ax_azimuth.set_ylabel("Relative level (dB)")
    ax_azimuth.set_title("Azimuth alignment", loc="left", fontsize=12, fontweight="bold")
    ax_azimuth.grid(True, alpha=0.22)
    ax_azimuth.text(
        0.01,
        0.96,
        "Dashed = truth, dotted = FFT, dash-dot = MUSIC",
        transform=ax_azimuth.transAxes,
        va="top",
        fontsize=9,
    )

    ax_summary.axis("off")
    ax_summary.text(
        0.0,
        0.92,
        "What this trial shows",
        fontsize=13,
        fontweight="bold",
        color="#10233F",
    )
    bullets = (
        "Paired nominal trial from the saved 64-trial submission bundle.",
        "FFT keeps the stronger mover but jumps to a plausible high-range false target for the second branch.",
        "MUSIC places both detections on the true pair and clears the joint gate on the same trial.",
    )
    y = 0.78
    for bullet in bullets:
        ax_summary.text(0.03, y, f"• {bullet}", fontsize=10.5, color="#1F2937", wrap=True)
        y -= 0.22

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
    scene_labels = {
        "intersection": "Intersection",
        "open_aisle": "Open aisle",
        "rack_aisle": "Rack aisle",
    }
    method_order = ("fft", "mdl", "eigengap", "expected")
    method_labels = {
        "fft": "FFT",
        "mdl": "MDL MUSIC",
        "eigengap": "Eigengap MUSIC",
        "expected": "Expected-order MUSIC",
    }
    method_colors = {
        "fft": "#D55E00",
        "mdl": "#7F7F7F",
        "eigengap": "#009E73",
        "expected": "#0072B2",
    }
    method_markers = {
        "fft": "o",
        "mdl": "D",
        "eigengap": "^",
        "expected": "s",
    }

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(scene_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(method_order))
    for scene_index, scene in enumerate(scene_order):
        values = [records[scene][method] for method in method_order]
        ax.hlines(y_positions[scene_index], min(values), max(values), color="#D7D7D7", linewidth=2.5)
        for offset, method in zip(offsets, method_order):
            value = records[scene][method]
            ax.scatter(
                value,
                y_positions[scene_index] + offset,
                s=90,
                color=method_colors[method],
                marker=method_markers[method],
                edgecolors="white",
                linewidths=1.1,
                label=method_labels[method] if scene_index == 0 else None,
                zorder=3,
            )
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
    scene_labels = {
        "open_aisle": "Open aisle",
        "intersection": "Intersection",
        "rack_aisle": "Rack aisle",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    for ax, scene in zip(axes, ("open_aisle", "intersection", "rack_aisle")):
        with scene_paths[scene].open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            method = row["method"]
            grouped[method].append(
                (
                    float(row["parameter_numeric_value"]),
                    float(row["joint_resolution_probability"]),
                )
            )
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
    scene_labels = {
        "intersection": "Intersection",
        "open_aisle": "Open aisle",
        "rack_aisle": "Rack aisle",
    }
    by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_scene[row["scene_class"]][row["method"]] = float(row["total_runtime_s"])

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(scene_order), dtype=float)
    offset = 0.17
    fft_values = [by_scene[scene]["fft_masked"] for scene in scene_order]
    music_values = [by_scene[scene]["music_masked"] for scene in scene_order]
    ax.barh(y - offset, fft_values, height=0.28, color=METHOD_COLORS["fft_masked"], label="FFT")
    ax.barh(y + offset, music_values, height=0.28, color=METHOD_COLORS["music_masked"], label="MUSIC")
    for idx, scene in enumerate(scene_order):
        ax.text(fft_values[idx] + 0.01, y[idx] - offset, f"{fft_values[idx]:.3f} s", va="center", fontsize=9)
        ax.text(music_values[idx] + 0.01, y[idx] + offset, f"{music_values[idx]:.3f} s", va="center", fontsize=9)
    ax.set_yticks(y, [scene_labels[scene] for scene in scene_order])
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
    """Create all figure assets required by the deck and return a manifest."""

    _ensure_output_dirs()
    specs = [
        FigureSpec(
            id="motivation_1d_range",
            filename="01_motivation_1d_range.png",
            title="1-D MUSIC motivation",
            kind="reused",
            caption="Motivating 1-D range-only MUSIC versus FFT figure.",
        ),
        FigureSpec(
            id="story_nominal_verdict",
            filename="02_nominal_scene_verdict.png",
            title="Nominal verdict",
            kind="reused",
            caption="Saved nominal verdict figure from CSV outputs.",
        ),
        FigureSpec(
            id="story_trial_delta",
            filename="03_nominal_trial_delta.png",
            title="Paired nominal trial delta",
            kind="reused",
            caption="Saved paired nominal trial-delta figure from CSV outputs.",
        ),
        FigureSpec(
            id="story_regime_map",
            filename="05_regime_map.png",
            title="Regime map",
            kind="reused",
            caption="Saved sweep-family regime map from CSV outputs.",
        ),
        FigureSpec(
            id="story_rack_aisle_diagnostic",
            filename="06_rack_aisle_failure_diagnostic.png",
            title="Rack-aisle failure diagnostic",
            kind="reused",
            caption="Saved rack-aisle candidate and detection diagnostic.",
        ),
        FigureSpec(
            id="story_coherence_overlap",
            filename="04_scene_coherence_overlap.png",
            title="Coherence overlap",
            kind="reused",
            caption="Saved configured-versus-empirical coherence figure.",
        ),
        FigureSpec(
            id="nominal_resource_mask",
            filename="07_nominal_resource_mask.png",
            title="Nominal resource mask",
            kind="generated",
            caption="Nominal fragmented PRB resource mask used by the study.",
        ),
        FigureSpec(
            id="representative_intersection_case",
            filename="08_representative_intersection_case.png",
            title="Representative nominal intersection trial",
            kind="generated",
            caption="Reconstructed saved nominal trial showing FFT versus MUSIC behavior in intersection.",
        ),
        FigureSpec(
            id="model_order_nominal_comparison",
            filename="09_model_order_nominal_comparison.png",
            title="Model-order comparison",
            kind="generated",
            caption="Nominal P_joint comparison across FFT, MDL, eigengap, and expected-order MUSIC.",
        ),
        FigureSpec(
            id="expected_order_nuisance_sweep",
            filename="10_expected_order_nuisance_sweep.png",
            title="Expected-order nuisance sweep",
            kind="generated",
            caption="Expected-order nuisance-strength sweep across all three scenes.",
        ),
        FigureSpec(
            id="runtime_comparison",
            filename="11_nominal_runtime_comparison.png",
            title="Runtime comparison",
            kind="generated",
            caption="Nominal runtime comparison for FFT and MUSIC across scenes.",
        ),
        FigureSpec(
            id="masked_observation_equation",
            filename="12_masked_observation_equation.png",
            title="Masked observation equation",
            kind="generated",
            caption="Rendered masked observation model equation.",
        ),
        FigureSpec(
            id="music_pseudospectrum_equation",
            filename="13_music_pseudospectrum_equation.png",
            title="MUSIC pseudospectrum equation",
            kind="generated",
            caption="Rendered MUSIC pseudospectrum equation.",
        ),
    ]

    reused_sources = {
        "motivation_1d_range": (
            REPO_ROOT / "results" / "figures" / "motivation_1d_range.png",
            LEGACY_RESULTS_ARCHIVE / "results_figures" / "motivation_1d_range.png",
        ),
        "story_nominal_verdict": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_nominal_verdict_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_nominal_verdict_from_csv.png",
        ),
        "story_trial_delta": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_trial_delta_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_trial_delta_from_csv.png",
        ),
        "story_regime_map": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_regime_map_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_regime_map_from_csv.png",
        ),
        "story_rack_aisle_diagnostic": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_rack_aisle_diagnostic_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_rack_aisle_diagnostic_from_csv.png",
        ),
        "story_coherence_overlap": (
            REPO_ROOT / "results" / "submission" / "figures_from_csv" / "story_coherence_overlap_from_csv.png",
            LEGACY_RESULTS_ARCHIVE / "submission" / "figures_from_csv" / "story_coherence_overlap_from_csv.png",
        ),
    }

    generated_handlers = {
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
        else:
            generated_handlers[spec.id](spec)

    _write_manifest(specs)
    return json.loads(FIGURE_MANIFEST_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    generate_figures()
