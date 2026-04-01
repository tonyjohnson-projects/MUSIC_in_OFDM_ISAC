"""CSV export and figure generation for the angle-range-Doppler study."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aisle_isac.study import METHOD_LABELS, METHOD_ORDER, PUBLIC_SWEEP_NAMES, SceneStudyResult, SweepResult


METHOD_COLORS = {
    "fft": "#C44E52",
    "music": "#4C72B0",
    "fbss": "#55A868",
    "music_full": "#8172B2",
}

METHOD_MARKERS = {
    "fft": "o",
    "music": "s",
    "fbss": "^",
    "music_full": "D",
}


def prepare_output_directories(root_dir: Path, clean_outputs: bool) -> tuple[Path, Path]:
    """Create the output tree, optionally removing stale artifacts first."""

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
    detection_lower, detection_upper = _wilson_interval(
        summary.joint_detection_success_count,
        summary.trial_count,
    )
    resolution_lower, resolution_upper = _wilson_interval(
        summary.joint_resolution_success_count,
        summary.trial_count,
    )
    false_alarm_lower, false_alarm_upper = _wilson_interval(
        summary.false_alarm_event_count,
        summary.trial_count,
    )
    miss_lower, miss_upper = _wilson_interval(
        summary.miss_event_count,
        summary.trial_count,
    )
    return {
        "trial_count": f"{summary.trial_count:d}",
        "joint_detection_probability_ci95_lower": f"{detection_lower:.6f}",
        "joint_detection_probability_ci95_upper": f"{detection_upper:.6f}",
        "joint_resolution_probability_ci95_lower": f"{resolution_lower:.6f}",
        "joint_resolution_probability_ci95_upper": f"{resolution_upper:.6f}",
        "false_alarm_probability_ci95_lower": f"{false_alarm_lower:.6f}",
        "false_alarm_probability_ci95_upper": f"{false_alarm_upper:.6f}",
        "miss_probability_ci95_lower": f"{miss_lower:.6f}",
        "miss_probability_ci95_upper": f"{miss_upper:.6f}",
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
                "burst_profile": point.burst_profile_name,
                "burst_profile_label": point.burst_profile_label,
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
                "unconditional_rmse_over_crb": f"{summary.unconditional_rmse_over_crb:.6f}",
                "conditional_range_rmse_m": (
                    f"{summary.conditional_range_rmse_m:.6f}" if summary.conditional_range_rmse_m is not None else ""
                ),
                "conditional_velocity_rmse_mps": (
                    f"{summary.conditional_velocity_rmse_mps:.6f}" if summary.conditional_velocity_rmse_mps is not None else ""
                ),
                "conditional_angle_rmse_deg": (
                    f"{summary.conditional_angle_rmse_deg:.6f}" if summary.conditional_angle_rmse_deg is not None else ""
                ),
                "conditional_joint_assignment_rmse": (
                    f"{summary.conditional_joint_assignment_rmse:.6f}" if summary.conditional_joint_assignment_rmse is not None else ""
                ),
                "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                "miss_probability": f"{summary.miss_probability:.6f}",
                "model_order_accuracy": f"{summary.model_order_accuracy:.6f}",
                "frontend_runtime_s": f"{summary.frontend_runtime_s:.6f}",
                "incremental_runtime_s": f"{summary.incremental_runtime_s:.6f}",
                "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                "conditional_rmse_over_crb": (
                    f"{summary.conditional_rmse_over_crb:.6f}" if summary.conditional_rmse_over_crb is not None else ""
                ),
            }
            row.update(_summary_probability_fields(summary))
            rows.append(row)
    return rows


def write_tidy_sweep_csv(output_path: Path, sweeps: list[SweepResult]) -> None:
    rows = [row for sweep in sweeps for row in _sweep_rows(sweep)]
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scene_comparison_rows(studies: list[SceneStudyResult]) -> list[dict[str, str]]:
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
                    "comparison_view": "scene_comparison",
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
                    "unconditional_rmse_over_crb": f"{summary.unconditional_rmse_over_crb:.6f}",
                    "conditional_range_rmse_m": (
                        f"{summary.conditional_range_rmse_m:.6f}" if summary.conditional_range_rmse_m is not None else ""
                    ),
                    "conditional_velocity_rmse_mps": (
                        f"{summary.conditional_velocity_rmse_mps:.6f}" if summary.conditional_velocity_rmse_mps is not None else ""
                    ),
                    "conditional_angle_rmse_deg": (
                        f"{summary.conditional_angle_rmse_deg:.6f}" if summary.conditional_angle_rmse_deg is not None else ""
                    ),
                    "conditional_joint_assignment_rmse": (
                        f"{summary.conditional_joint_assignment_rmse:.6f}" if summary.conditional_joint_assignment_rmse is not None else ""
                    ),
                    "conditional_rmse_over_crb": (
                        f"{summary.conditional_rmse_over_crb:.6f}" if summary.conditional_rmse_over_crb is not None else ""
                    ),
                    "model_order_accuracy": f"{summary.model_order_accuracy:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _fr1_vs_fr2_rows(studies: list[SceneStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "scene_class": study.scene_class_name,
                    "scene_label": study.scene_label,
                    "anchor": study.anchor_name,
                    "anchor_label": study.anchor_label,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                    "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                    "scene_cost": f"{summary.scene_cost:.6f}",
                    "unconditional_range_rmse_m": f"{summary.unconditional_range_rmse_m:.6f}",
                    "unconditional_velocity_rmse_mps": f"{summary.unconditional_velocity_rmse_mps:.6f}",
                    "unconditional_angle_rmse_deg": f"{summary.unconditional_angle_rmse_deg:.6f}",
                    "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                    "unconditional_rmse_over_crb": f"{summary.unconditional_rmse_over_crb:.6f}",
                    "conditional_range_rmse_m": (
                        f"{summary.conditional_range_rmse_m:.6f}" if summary.conditional_range_rmse_m is not None else ""
                    ),
                    "conditional_velocity_rmse_mps": (
                        f"{summary.conditional_velocity_rmse_mps:.6f}" if summary.conditional_velocity_rmse_mps is not None else ""
                    ),
                    "conditional_angle_rmse_deg": (
                        f"{summary.conditional_angle_rmse_deg:.6f}" if summary.conditional_angle_rmse_deg is not None else ""
                    ),
                    "conditional_joint_assignment_rmse": (
                        f"{summary.conditional_joint_assignment_rmse:.6f}" if summary.conditional_joint_assignment_rmse is not None else ""
                    ),
                    "conditional_rmse_over_crb": (
                        f"{summary.conditional_rmse_over_crb:.6f}" if summary.conditional_rmse_over_crb is not None else ""
                    ),
                    "model_order_accuracy": f"{summary.model_order_accuracy:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _crb_gap_rows(studies: list[SceneStudyResult]) -> list[dict[str, str]]:
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
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "scene_cost": f"{summary.scene_cost:.6f}",
                    "unconditional_rmse_over_crb": f"{summary.unconditional_rmse_over_crb:.6f}",
                    "conditional_rmse_over_crb": (
                        f"{summary.conditional_rmse_over_crb:.6f}" if summary.conditional_rmse_over_crb is not None else ""
                    ),
                    "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                    "conditional_joint_assignment_rmse": (
                        f"{summary.conditional_joint_assignment_rmse:.6f}" if summary.conditional_joint_assignment_rmse is not None else ""
                    ),
                    "unconditional_range_rmse_m": f"{summary.unconditional_range_rmse_m:.6f}",
                    "unconditional_velocity_rmse_mps": f"{summary.unconditional_velocity_rmse_mps:.6f}",
                    "unconditional_angle_rmse_deg": f"{summary.unconditional_angle_rmse_deg:.6f}",
                    "model_order_accuracy": f"{summary.model_order_accuracy:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
            )
    return rows


def _write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_scene_comparison_csv(output_path: Path, studies: list[SceneStudyResult]) -> None:
    _write_rows(output_path, _scene_comparison_rows(studies))


def write_fr1_vs_fr2_csv(output_path: Path, studies: list[SceneStudyResult]) -> None:
    _write_rows(output_path, _fr1_vs_fr2_rows(studies))


def write_crb_gap_csv(output_path: Path, studies: list[SceneStudyResult]) -> None:
    _write_rows(output_path, _crb_gap_rows(studies))


def _grouped_studies(studies: list[SceneStudyResult]) -> tuple[list[str], list[str], dict[tuple[str, str], SceneStudyResult]]:
    anchors = sorted({study.anchor_name for study in studies})
    scenes = sorted({study.scene_class_name for study in studies})
    study_map = {(study.anchor_name, study.scene_class_name): study for study in studies}
    return anchors, scenes, study_map


def _scene_label(scene_name: str, study_map: dict[tuple[str, str], SceneStudyResult]) -> str:
    for (_, candidate_scene), study in study_map.items():
        if candidate_scene == scene_name:
            return study.scene_label
    return scene_name


def _anchor_label(anchor_name: str, study_map: dict[tuple[str, str], SceneStudyResult]) -> str:
    for (candidate_anchor, _), study in study_map.items():
        if candidate_anchor == anchor_name:
            return study.anchor_label
    return anchor_name.upper()


def _plot_sweep_grid(output_path: Path, studies: list[SceneStudyResult], sweep_name: str) -> None:
    anchors, scenes, study_map = _grouped_studies(studies)
    fig, axes = plt.subplots(len(scenes), len(anchors), figsize=(5.5 * len(anchors), 3.8 * len(scenes)), squeeze=False)

    for row_index, scene_name in enumerate(scenes):
        for col_index, anchor_name in enumerate(anchors):
            axis = axes[row_index, col_index]
            study = study_map.get((anchor_name, scene_name))
            if study is None:
                axis.axis("off")
                continue
            sweep = next(result for result in study.sweeps if result.sweep_name == sweep_name)
            is_numeric = all(point.parameter_numeric_value is not None for point in sweep.points)
            if is_numeric:
                x_values = np.array([point.parameter_numeric_value for point in sweep.points], dtype=float)
                axis.set_xticks(x_values)
            else:
                x_values = np.arange(len(sweep.points), dtype=float)
                axis.set_xticks(x_values)
                axis.set_xticklabels([point.parameter_value for point in sweep.points], rotation=20)

            for method_name in METHOD_ORDER:
                y_values = np.array(
                    [point.method_summaries[method_name].joint_resolution_probability for point in sweep.points],
                    dtype=float,
                )
                axis.plot(
                    x_values,
                    y_values,
                    marker=METHOD_MARKERS[method_name],
                    linewidth=2.0,
                    color=METHOD_COLORS[method_name],
                    label=METHOD_LABELS[method_name],
                )
            axis.set_title(f"{study.scene_label} / {study.anchor_label}")
            axis.set_ylabel("Joint Resolution Probability")
            axis.set_xlabel(sweep.parameter_label)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)

    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scene_comparison(output_path: Path, studies: list[SceneStudyResult]) -> None:
    anchors, scenes, study_map = _grouped_studies(studies)
    fig, axes = plt.subplots(1, len(anchors), figsize=(6.0 * len(anchors), 4.5), squeeze=False)
    bar_width = 0.22
    x_base = np.arange(len(scenes), dtype=float)

    for col_index, anchor_name in enumerate(anchors):
        axis = axes[0, col_index]
        for method_offset, method_name in enumerate(METHOD_ORDER):
            y_values = []
            for scene_name in scenes:
                study = study_map.get((anchor_name, scene_name))
                if study is None:
                    y_values.append(np.nan)
                else:
                    y_values.append(study.nominal_point.method_summaries[method_name].joint_resolution_probability)
            axis.bar(
                x_base + (method_offset - 1) * bar_width,
                y_values,
                width=bar_width,
                color=METHOD_COLORS[method_name],
                label=METHOD_LABELS[method_name],
            )
        axis.set_xticks(x_base)
        axis.set_xticklabels([_scene_label(scene, study_map) for scene in scenes], rotation=15)
        axis.set_title(_anchor_label(anchor_name, study_map))
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Nominal Joint Resolution Probability")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fr1_vs_fr2(output_path: Path, studies: list[SceneStudyResult]) -> None:
    anchors, scenes, study_map = _grouped_studies(studies)
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.0 * len(scenes), 4.5), squeeze=False)
    x_base = np.arange(len(anchors), dtype=float)
    bar_width = 0.22

    for col_index, scene_name in enumerate(scenes):
        axis = axes[0, col_index]
        for method_offset, method_name in enumerate(METHOD_ORDER):
            y_values = []
            for anchor_name in anchors:
                study = study_map.get((anchor_name, scene_name))
                y_values.append(
                    study.nominal_point.method_summaries[method_name].joint_resolution_probability if study is not None else np.nan
                )
            axis.bar(
                x_base + (method_offset - 1) * bar_width,
                y_values,
                width=bar_width,
                color=METHOD_COLORS[method_name],
                label=METHOD_LABELS[method_name],
            )
        axis.set_xticks(x_base)
        axis.set_xticklabels([_anchor_label(anchor, study_map) for anchor in anchors])
        axis.set_title(_scene_label(scene_name, study_map))
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Nominal Joint Resolution Probability")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_crb_gap(output_path: Path, studies: list[SceneStudyResult]) -> None:
    anchors, scenes, study_map = _grouped_studies(studies)
    fig, axes = plt.subplots(len(scenes), len(anchors), figsize=(5.5 * len(anchors), 3.8 * len(scenes)), squeeze=False)
    x_positions = np.arange(len(METHOD_ORDER), dtype=float)

    for row_index, scene_name in enumerate(scenes):
        for col_index, anchor_name in enumerate(anchors):
            axis = axes[row_index, col_index]
            study = study_map.get((anchor_name, scene_name))
            if study is None:
                axis.axis("off")
                continue
            y_values = [
                study.nominal_point.method_summaries[method_name].unconditional_rmse_over_crb
                for method_name in METHOD_ORDER
            ]
            axis.bar(x_positions, y_values, color=[METHOD_COLORS[method_name] for method_name in METHOD_ORDER])
            axis.set_xticks(x_positions)
            axis.set_xticklabels([METHOD_LABELS[method_name] for method_name in METHOD_ORDER], rotation=15)
            axis.set_title(f"{study.scene_label} / {study.anchor_label}")
            axis.set_ylabel("Unconditional Normalized RMSE / CRB")
            axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_cube_slices(output_path: Path, studies: list[SceneStudyResult]) -> None:
    preferred = next(
        (study for study in studies if study.anchor_name == "fr1" and study.scene_class_name == "rack_aisle"),
        studies[0],
    )
    trial = preferred.representative_trial
    fft_cube = trial.fft_cube
    range_doppler = np.sum(fft_cube.power_cube, axis=0)
    angle_range = np.sum(fft_cube.power_cube, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    rd_image = 10.0 * np.log10(np.maximum(range_doppler.T, 1.0e-12) / np.max(range_doppler))
    axes[0].imshow(
        rd_image,
        aspect="auto",
        origin="lower",
        extent=(
            fft_cube.range_axis_m[0],
            fft_cube.range_axis_m[-1],
            fft_cube.velocity_axis_mps[0],
            fft_cube.velocity_axis_mps[-1],
        ),
        cmap="viridis",
    )
    axes[0].set_title("Representative Range-Doppler Slice")
    axes[0].set_xlabel("Range (m)")
    axes[0].set_ylabel("Velocity (m/s)")

    ar_image = 10.0 * np.log10(np.maximum(angle_range, 1.0e-12) / np.max(angle_range))
    axes[1].imshow(
        ar_image,
        aspect="auto",
        origin="lower",
        extent=(
            fft_cube.range_axis_m[0],
            fft_cube.range_axis_m[-1],
            fft_cube.azimuth_axis_deg[0],
            fft_cube.azimuth_axis_deg[-1],
        ),
        cmap="magma",
    )
    axes[1].set_title("Representative Angle-Range Slice")
    axes[1].set_xlabel("Range (m)")
    axes[1].set_ylabel("Azimuth (deg)")

    for target in trial.snapshot.scenario.targets:
        axes[2].scatter(target.azimuth_deg, target.range_m, marker="x", s=80, color="black", label="Truth")
    for method_name in METHOD_ORDER:
        detections = trial.estimates[method_name].detections
        axes[2].scatter(
            [detection.azimuth_deg for detection in detections],
            [detection.range_m for detection in detections],
            marker=METHOD_MARKERS[method_name],
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    handles, labels = axes[2].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[2].legend(unique.values(), unique.keys())
    axes[2].set_title("Truth and Estimated Azimuth-Range Points")
    axes[2].set_xlabel("Azimuth (deg)")
    axes[2].set_ylabel("Range (m)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_all_outputs(
    studies: list[SceneStudyResult],
    root_dir: Path,
    clean_outputs: bool,
    sweep_names: tuple[str, ...] | None = None,
    include_scene_comparison: bool = True,
    include_fr1_vs_fr2: bool = True,
    include_crb_gap: bool = True,
    include_representative_cube_slices: bool = True,
) -> None:
    """Write all public CSVs and figures for the requested study bundle."""

    data_dir, figures_dir = prepare_output_directories(root_dir, clean_outputs=clean_outputs)
    selected_sweep_names = sweep_names or PUBLIC_SWEEP_NAMES
    for sweep_name in selected_sweep_names:
        sweeps = [
            sweep
            for study in studies
            for sweep in study.sweeps
            if sweep.sweep_name == sweep_name
        ]
        write_tidy_sweep_csv(data_dir / f"{sweep_name}.csv", sweeps)

    if include_scene_comparison:
        write_scene_comparison_csv(data_dir / "scene_comparison.csv", studies)
    if include_fr1_vs_fr2:
        write_fr1_vs_fr2_csv(data_dir / "fr1_vs_fr2.csv", studies)
    if include_crb_gap:
        write_crb_gap_csv(data_dir / "crb_gap.csv", studies)

    for sweep_name in selected_sweep_names:
        _plot_sweep_grid(figures_dir / f"{sweep_name}.png", studies, sweep_name)
    if include_scene_comparison:
        plot_scene_comparison(figures_dir / "scene_comparison.png", studies)
    if include_fr1_vs_fr2:
        plot_fr1_vs_fr2(figures_dir / "fr1_vs_fr2.png", studies)
    if include_crb_gap:
        plot_crb_gap(figures_dir / "crb_gap.png", studies)
    if include_representative_cube_slices:
        plot_representative_cube_slices(figures_dir / "representative_cube_slices.png", studies)
