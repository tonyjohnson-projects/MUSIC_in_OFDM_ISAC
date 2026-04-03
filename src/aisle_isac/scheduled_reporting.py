"""Reporting for the communications-limited MUSIC study."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from aisle_isac.estimators_music import METHOD_LABELS, METHOD_ORDER
from aisle_isac.scheduled_study import CommunicationsStudyResult, SweepResult


METHOD_COLORS = {
    "fft_masked": "#C44E52",
    "music_masked": "#4C72B0",
}

METHOD_MARKERS = {
    "fft_masked": "o",
    "music_masked": "s",
}


def prepare_output_directories(root_dir: Path, clean_outputs: bool) -> tuple[Path, Path]:
    if clean_outputs and root_dir.exists():
        shutil.rmtree(root_dir)
    data_dir = root_dir / "data"
    figures_dir = root_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figures_dir


def _wilson_interval(success_count: int, trial_count: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    if trial_count <= 0:
        return 0.0, 0.0
    proportion = success_count / trial_count
    z_squared = z_value**2
    denominator = 1.0 + z_squared / trial_count
    center = (proportion + z_squared / (2.0 * trial_count)) / denominator
    margin = (
        z_value
        * np.sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * trial_count)) / trial_count)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary_probability_fields(summary) -> dict[str, str]:
    detection_lower, detection_upper = _wilson_interval(summary.joint_detection_success_count, summary.trial_count)
    resolution_lower, resolution_upper = _wilson_interval(summary.joint_resolution_success_count, summary.trial_count)
    return {
        "trial_count": f"{summary.trial_count:d}",
        "joint_detection_probability_ci95_lower": f"{detection_lower:.6f}",
        "joint_detection_probability_ci95_upper": f"{detection_upper:.6f}",
        "joint_resolution_probability_ci95_lower": f"{resolution_lower:.6f}",
        "joint_resolution_probability_ci95_upper": f"{resolution_upper:.6f}",
    }


def _sweep_rows(sweep: SweepResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for point in sweep.points:
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            row = {
                "anchor": point.anchor_name,
                "anchor_label": point.anchor_label,
                "scene_class": point.scene_class_name,
                "scene_label": point.scene_label,
                "sweep_name": point.sweep_name,
                "parameter_name": point.parameter_name,
                "parameter_label": point.parameter_label,
                "parameter_value": point.parameter_value,
                "parameter_numeric_value": (
                    f"{point.parameter_numeric_value:.6f}" if point.parameter_numeric_value is not None else ""
                ),
                "allocation_family": point.allocation_family,
                "allocation_label": point.allocation_label,
                "knowledge_mode": point.knowledge_mode,
                "modulation_scheme": point.modulation_scheme,
                "occupied_fraction": f"{point.occupied_fraction:.6f}",
                "pilot_fraction": f"{point.pilot_fraction:.6f}",
                "fragmentation_index": f"{point.fragmentation_index:.6f}",
                "bandwidth_span_fraction": f"{point.bandwidth_span_fraction:.6f}",
                "slow_time_span_fraction": f"{point.slow_time_span_fraction:.6f}",
                "burst_profile": point.burst_profile_name,
                "aperture_size": f"{point.aperture_size:d}",
                "target_pair": point.target_pair,
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                "scene_cost": f"{summary.scene_cost:.6f}",
                "unconditional_range_rmse_m": f"{summary.unconditional_range_rmse_m:.6f}",
                "unconditional_velocity_rmse_mps": f"{summary.unconditional_velocity_rmse_mps:.6f}",
                "unconditional_angle_rmse_deg": f"{summary.unconditional_angle_rmse_deg:.6f}",
                "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                "miss_probability": f"{summary.miss_probability:.6f}",
                "reported_target_count_accuracy": f"{summary.reported_target_count_accuracy:.6f}",
                "frontend_runtime_s": f"{summary.frontend_runtime_s:.6f}",
                "incremental_runtime_s": f"{summary.incremental_runtime_s:.6f}",
                "total_runtime_s": f"{summary.total_runtime_s:.6f}",
            }
            row.update(_summary_probability_fields(summary))
            rows.append(row)
    return rows


def _write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _categorical_positions(labels: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    positions = np.arange(len(labels), dtype=float)
    mapping = {label: position for label, position in zip(labels, positions, strict=True)}
    return positions, mapping


def _plot_sweep_figure(output_path: Path, sweep_name: str, studies: list[CommunicationsStudyResult]) -> None:
    matching_sweeps = [sweep for study in studies for sweep in study.sweeps if sweep.sweep_name == sweep_name]
    if not matching_sweeps:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    is_numeric = all(point.parameter_numeric_value is not None for sweep in matching_sweeps for point in sweep.points)
    if is_numeric:
        for sweep in matching_sweeps:
            for method_name in METHOD_ORDER:
                x_values = np.asarray([point.parameter_numeric_value for point in sweep.points], dtype=float)
                y_values = np.asarray(
                    [point.method_summaries[method_name].joint_detection_probability for point in sweep.points],
                    dtype=float,
                )
                label = f"{METHOD_LABELS[method_name]} / {sweep.anchor_label} / {sweep.scene_label}"
                ax.plot(
                    x_values,
                    y_values,
                    color=METHOD_COLORS[method_name],
                    marker=METHOD_MARKERS[method_name],
                    linewidth=2.0,
                    label=label,
                )
        ax.set_xlabel(matching_sweeps[0].parameter_label)
    else:
        labels = [point.parameter_value for point in matching_sweeps[0].points]
        positions, mapping = _categorical_positions(labels)
        series_count = len(METHOD_ORDER) * len(matching_sweeps)
        bar_width = 0.8 / max(series_count, 1)
        series_index = 0
        for sweep in matching_sweeps:
            for method_name in METHOD_ORDER:
                x_values = positions - 0.4 + bar_width * series_index + 0.5 * bar_width
                y_values = np.asarray(
                    [point.method_summaries[method_name].joint_detection_probability for point in sweep.points],
                    dtype=float,
                )
                ax.bar(
                    x_values,
                    y_values,
                    width=bar_width,
                    color=METHOD_COLORS[method_name],
                    alpha=0.7 if sweep.anchor_name == matching_sweeps[0].anchor_name else 0.45,
                    label=f"{METHOD_LABELS[method_name]} / {sweep.anchor_label} / {sweep.scene_label}",
                )
                series_index += 1
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel(matching_sweeps[0].parameter_label)
    ax.set_ylabel("Joint Detection Probability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _nominal_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "anchor": study.anchor_name,
                    "anchor_label": study.anchor_label,
                    "scene_class": study.scene_class_name,
                    "scene_label": study.scene_label,
                    "allocation_family": point.allocation_family,
                    "allocation_label": point.allocation_label,
                    "knowledge_mode": point.knowledge_mode,
                    "occupied_fraction": f"{point.occupied_fraction:.6f}",
                    "pilot_fraction": f"{point.pilot_fraction:.6f}",
                    "fragmentation_index": f"{point.fragmentation_index:.6f}",
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                    "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                    "scene_cost": f"{summary.scene_cost:.6f}",
                    "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _runtime_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "anchor": study.anchor_name,
                    "scene_class": study.scene_class_name,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "frontend_runtime_s": f"{summary.frontend_runtime_s:.6f}",
                    "incremental_runtime_s": f"{summary.incremental_runtime_s:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _failure_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        for sweep in study.sweeps:
            for point in sweep.points:
                for method_name in METHOD_ORDER:
                    summary = point.method_summaries[method_name]
                    if summary.joint_detection_probability >= 0.999:
                        continue
                    rows.append(
                        {
                            "anchor": study.anchor_name,
                            "scene_class": study.scene_class_name,
                            "sweep_name": point.sweep_name,
                            "parameter_value": point.parameter_value,
                            "allocation_family": point.allocation_family,
                            "knowledge_mode": point.knowledge_mode,
                            "method": method_name,
                            "method_label": METHOD_LABELS[method_name],
                            "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                            "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                            "scene_cost": f"{summary.scene_cost:.6f}",
                            "miss_probability": f"{summary.miss_probability:.6f}",
                        }
                    )
    rows.sort(key=lambda row: float(row["scene_cost"]), reverse=True)
    return rows


def _comparison_rows(studies: list[CommunicationsStudyResult], key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "comparison_key": key,
                    "anchor": study.anchor_name,
                    "scene_class": study.scene_class_name,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                    "scene_cost": f"{summary.scene_cost:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _plot_runtime_summary(output_path: Path, studies: list[CommunicationsStudyResult]) -> None:
    rows = _runtime_rows(studies)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [row["method_label"] for row in rows]
    values = [float(row["total_runtime_s"]) for row in rows]
    ax.bar(np.arange(len(labels)), values, color=[METHOD_COLORS[row["method"]] for row in rows])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Nominal Runtime Summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_representative_resource_mask(output_path: Path, study: CommunicationsStudyResult) -> None:
    role_grid = study.representative_trial.masked_observation.resource_grid.role_grid
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = ListedColormap(["#F0F0F0", "#4C72B0", "#55A868", "#C44E52"])
    image = ax.imshow(role_grid.T, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("OFDM Symbol Index")
    ax.set_title("Representative Resource Mask")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Muted", "Pilot", "Data", "Punctured"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_representative_spectrum(output_path: Path, study: CommunicationsStudyResult) -> None:
    fft_cube = study.representative_trial.fft_cube.power_cube
    range_doppler = np.max(fft_cube, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        10.0 * np.log10(np.maximum(range_doppler, 1.0e-12)),
        aspect="auto",
        origin="lower",
        extent=[
            study.representative_trial.fft_cube.velocity_axis_mps[0],
            study.representative_trial.fft_cube.velocity_axis_mps[-1],
            study.representative_trial.fft_cube.range_axis_m[0],
            study.representative_trial.fft_cube.range_axis_m[-1],
        ],
        cmap="viridis",
    )
    for method_name, estimate in study.representative_trial.estimates.items():
        ax.scatter(
            [detection.velocity_mps for detection in estimate.detections],
            [detection.range_m for detection in estimate.detections],
            marker="o" if method_name == "fft_masked" else "x",
            s=60,
            c=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Representative Range-Doppler Spectrum")
    ax.legend()
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_all_outputs(
    studies: list[CommunicationsStudyResult],
    output_root: Path,
    *,
    clean_outputs: bool,
    sweep_names: tuple[str, ...],
    include_scene_comparison: bool,
    include_anchor_comparison: bool,
) -> None:
    """Write CSVs and figures for the communications-scheduled study."""

    data_dir, figures_dir = prepare_output_directories(output_root, clean_outputs=clean_outputs)

    for sweep_name in sweep_names:
        sweep_rows = [row for study in studies for sweep in study.sweeps if sweep.sweep_name == sweep_name for row in _sweep_rows(sweep)]
        _write_csv(data_dir / f"{sweep_name}.csv", sweep_rows)
        _plot_sweep_figure(figures_dir / f"{sweep_name}.png", sweep_name, studies)

    _write_csv(data_dir / "nominal_summary.csv", _nominal_rows(studies))
    _write_csv(data_dir / "runtime_summary.csv", _runtime_rows(studies))
    _write_csv(data_dir / "failure_modes.csv", _failure_rows(studies))
    _plot_runtime_summary(figures_dir / "runtime_summary.png", studies)

    if include_scene_comparison:
        _write_csv(data_dir / "scene_comparison.csv", _comparison_rows(studies, key="scene"))
    if include_anchor_comparison:
        _write_csv(data_dir / "anchor_comparison.csv", _comparison_rows(studies, key="anchor"))

    if studies:
        _plot_representative_resource_mask(figures_dir / "representative_resource_mask.png", studies[0])
        _plot_representative_spectrum(figures_dir / "representative_spectrum.png", studies[0])
