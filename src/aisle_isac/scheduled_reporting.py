"""Reporting for the communications-limited MUSIC study."""

from __future__ import annotations

import csv
from dataclasses import replace
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from aisle_isac.estimators import (
    _estimate_music_model_order,
    _doppler_music_covariance,
    _range_search_upper_m,
    _range_music_covariance,
    azimuth_steering_matrix,
    covariance_matrix,
    doppler_steering_matrix,
    fft_search_bounds,
    fbss_covariance,
    music_pseudospectrum,
    range_steering_matrix,
)
from aisle_isac.estimators_music import FBSS_ABLATION_FLAGS, FBSS_ABLATION_LABELS, FBSS_ABLATION_ORDER, METHOD_LABELS, METHOD_ORDER
from aisle_isac.masked_observation import extract_known_symbol_cube
from aisle_isac.resource_grid import ROLE_LABELS
from aisle_isac.scheduled_study import CommunicationsStudyResult, FBSS_ABLATION_SWEEP_NAMES, SweepResult


SCHEMA_VERSION = "2.0"
ESTIMATOR_SET = ",".join(METHOD_ORDER)
FBSS_ABLATION_SET = ",".join(FBSS_ABLATION_ORDER)
USEFULNESS_DELTA_THRESHOLD = 0.10
ROLE_CODE_LABELS = {int(role): label for role, label in ROLE_LABELS.items()}

METHOD_COLORS = {
    "fft_masked": "#C44E52",
    "music_masked": "#4C72B0",
}

METHOD_MARKERS = {
    "fft_masked": "o",
    "music_masked": "s",
}
FBSS_ABLATION_COLORS = {
    "fbss_spatial_only": "#7F7F7F",
    "fbss_spatial_range": "#55A868",
    "fbss_spatial_doppler": "#DD8452",
    "fbss_spatial_range_doppler": "#4C72B0",
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


def _metric_probability_fields(summary) -> dict[str, str]:
    interval_specs = (
        ("joint_detection_probability", summary.joint_detection_success_count),
        ("joint_resolution_probability", summary.joint_resolution_success_count),
        ("range_resolution_probability", summary.range_resolution_success_count),
        ("velocity_resolution_probability", summary.velocity_resolution_success_count),
        ("angle_resolution_probability", summary.angle_resolution_success_count),
    )
    fields = {"trial_count": f"{summary.trial_count:d}"}
    for metric_name, success_count in interval_specs:
        lower, upper = _wilson_interval(success_count, summary.trial_count)
        fields[f"{metric_name}_ci95_lower"] = f"{lower:.6f}"
        fields[f"{metric_name}_ci95_upper"] = f"{upper:.6f}"
    return fields


def _sweep_metric(sweep_name: str) -> tuple[str, str]:
    if sweep_name == "range_separation":
        return "range_resolution_probability", "Range Resolution Probability"
    if sweep_name == "velocity_separation":
        return "velocity_resolution_probability", "Doppler Resolution Probability"
    if sweep_name == "angle_separation":
        return "angle_resolution_probability", "Angle Resolution Probability"
    return "joint_resolution_probability", "Joint Resolution Probability"


def _study_sweep_rows(study: CommunicationsStudyResult, sweep: SweepResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for point in sweep.points:
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            row = {
                "schema_version": SCHEMA_VERSION,
                "evidence_profile": study.evidence_profile_name,
                "estimator_set": ESTIMATOR_SET,
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
                "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
                "unconditional_range_rmse_m": f"{summary.unconditional_range_rmse_m:.6f}",
                "unconditional_velocity_rmse_mps": f"{summary.unconditional_velocity_rmse_mps:.6f}",
                "unconditional_angle_rmse_deg": f"{summary.unconditional_angle_rmse_deg:.6f}",
                "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
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
                "reported_target_count_accuracy": f"{summary.reported_target_count_accuracy:.6f}",
                "mean_estimated_model_order": (
                    f"{summary.mean_estimated_model_order:.6f}" if summary.mean_estimated_model_order is not None else ""
                ),
                "estimated_model_order_accuracy": (
                    f"{summary.estimated_model_order_accuracy:.6f}" if summary.estimated_model_order_accuracy is not None else ""
                ),
                "frontend_runtime_s": f"{summary.frontend_runtime_s:.6f}",
                "incremental_runtime_s": f"{summary.incremental_runtime_s:.6f}",
                "total_runtime_s": f"{summary.total_runtime_s:.6f}",
            }
            row.update(_metric_probability_fields(summary))
            rows.append(row)
    return rows


def _write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_optional_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def _study_reference_fields(study: CommunicationsStudyResult) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_profile": study.evidence_profile_name,
        "estimator_set": ESTIMATOR_SET,
        "anchor": study.anchor_name,
        "anchor_label": study.anchor_label,
        "scene_class": study.scene_class_name,
        "scene_label": study.scene_label,
        "burst_profile": study.config.burst_profile.name,
        "aperture_size": f"{study.config.array_geometry.n_rx_cols:d}",
        "representative_seed": f"{study.config.rng_seed:d}",
        "music_model_order_mode": study.config.music_model_order_mode,
        "music_fixed_model_order": (
            f"{study.config.music_fixed_model_order:d}" if study.config.music_fixed_model_order is not None else ""
        ),
        "fbss_ablation_enabled": f"{int(study.config.enable_fbss_ablation):d}",
    }


def _all_sweep_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    return [
        row
        for study in studies
        for sweep in study.sweeps
        for row in _study_sweep_rows(study, sweep)
    ]


def _trial_level_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point_sources = [study.nominal_point]
        if study.pilot_only_nominal_point is not None:
            point_sources.append(study.pilot_only_nominal_point)
        point_sources.extend(point for sweep in study.sweeps for point in sweep.points)
        for point in point_sources:
            for trial_row in point.trial_rows:
                estimator_set = ESTIMATOR_SET if trial_row["estimator_family"] == "headline" else FBSS_ABLATION_SET
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "estimator_set": estimator_set,
                        **trial_row,
                    }
                )
    return rows


def _stage_diagnostic_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    """Extract azimuth-stage diagnostics for MUSIC nominal trials."""
    rows: list[dict[str, str]] = []
    for study in studies:
        for trial_row in study.nominal_point.trial_rows:
            if trial_row.get("method") != "music_masked":
                continue
            if trial_row.get("estimator_family") != "headline":
                continue
            rows.append({
                "anchor": trial_row["anchor"],
                "anchor_label": trial_row["anchor_label"],
                "scene_class": trial_row["scene_class"],
                "scene_label": trial_row["scene_label"],
                "trial_index": trial_row["trial_index"],
                "music_model_order_mode": trial_row.get("music_model_order_mode", ""),
                "music_fixed_model_order": trial_row.get("music_fixed_model_order", ""),
                "estimated_model_order": trial_row.get("estimated_model_order", ""),
                "music_stage_azimuth_peak_count": trial_row.get("music_stage_azimuth_peak_count", ""),
                "music_stage_azimuth_candidate_count": trial_row.get("music_stage_azimuth_candidate_count", ""),
                "music_stage_azimuth_candidates_deg": trial_row.get("music_stage_azimuth_candidates_deg", ""),
                "music_stage_coarse_candidate_count": trial_row.get("music_stage_coarse_candidate_count", ""),
                "music_stage_coarse_candidates": trial_row.get("music_stage_coarse_candidates", ""),
                "joint_detection_success": trial_row["joint_detection_success"],
                "joint_resolution_success": trial_row["joint_resolution_success"],
                "truth_targets": trial_row["truth_targets"],
                "detections": trial_row["detections"],
            })
    return rows


def _fbss_variant_cfg(cfg, variant_name: str):
    range_fbss_enabled, doppler_fbss_enabled = FBSS_ABLATION_FLAGS[variant_name]
    return replace(
        cfg,
        music_range_fbss_fraction=cfg.music_range_fbss_fraction if range_fbss_enabled else 0.0,
        music_doppler_fbss_fraction=cfg.music_doppler_fbss_fraction if doppler_fbss_enabled else 0.0,
    )


def _polar_to_xy(range_m: float, azimuth_deg: float) -> tuple[float, float]:
    azimuth_rad = np.deg2rad(azimuth_deg)
    x_m = float(range_m * np.sin(azimuth_rad))
    y_m = float(range_m * np.cos(azimuth_rad))
    return x_m, y_m


def _representative_resource_mask_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.representative_trial is None:
            continue
        reference = _study_reference_fields(study)
        masked_observation = study.representative_trial.masked_observation
        role_grid = masked_observation.resource_grid.role_grid
        available_mask = masked_observation.availability_mask
        known_mask = masked_observation.known_symbol_mask
        symbols = masked_observation.symbol_map.symbols
        for subcarrier_index in range(role_grid.shape[0]):
            for symbol_index in range(role_grid.shape[1]):
                role_code = int(role_grid[subcarrier_index, symbol_index])
                rows.append(
                    {
                        **reference,
                        "subcarrier_index": f"{subcarrier_index:d}",
                        "symbol_index": f"{symbol_index:d}",
                        "role_code": f"{role_code:d}",
                        "role_label": ROLE_CODE_LABELS.get(role_code, "unknown"),
                        "available_sensing": f"{int(available_mask[subcarrier_index, symbol_index]):d}",
                        "known_symbol": f"{int(known_mask[subcarrier_index, symbol_index]):d}",
                        "symbol_real": f"{float(np.real(symbols[subcarrier_index, symbol_index])):.6f}",
                        "symbol_imag": f"{float(np.imag(symbols[subcarrier_index, symbol_index])):.6f}",
                    }
                )
    return rows


def _representative_scene_geometry_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.representative_trial is None:
            continue
        reference = _study_reference_fields(study)
        scenario = study.representative_trial.masked_observation.snapshot.scenario
        static_labels = {scatterer.label for scatterer in study.config.scene_class.static_clutter}
        multipath_labels = {scatterer.label for scatterer in study.config.scene_class.multipath}

        for truth_index, target in enumerate(scenario.targets):
            x_m, y_m = _polar_to_xy(target.range_m, target.azimuth_deg)
            rows.append(
                {
                    **reference,
                    "entity_kind": "truth_target",
                    "entity_group": "truth_target",
                    "entity_label": target.label,
                    "method": "",
                    "method_label": "",
                    "entity_index": f"{truth_index:d}",
                    "range_m": f"{target.range_m:.6f}",
                    "velocity_mps": f"{target.velocity_mps:.6f}",
                    "azimuth_deg": f"{target.azimuth_deg:.6f}",
                    "x_m": f"{x_m:.6f}",
                    "y_m": f"{y_m:.6f}",
                    "amplitude_db": f"{target.amplitude_db:.6f}",
                    "path_gain_linear": f"{target.path_gain_linear:.6e}",
                    "score": "",
                }
            )

        for nuisance_index, nuisance in enumerate(scenario.nuisance):
            x_m, y_m = _polar_to_xy(nuisance.range_m, nuisance.azimuth_deg)
            if nuisance.label in static_labels:
                nuisance_group = "static_clutter"
            elif nuisance.label in multipath_labels:
                nuisance_group = "multipath"
            else:
                nuisance_group = "nuisance"
            rows.append(
                {
                    **reference,
                    "entity_kind": "nuisance",
                    "entity_group": nuisance_group,
                    "entity_label": nuisance.label,
                    "method": "",
                    "method_label": "",
                    "entity_index": f"{nuisance_index:d}",
                    "range_m": f"{nuisance.range_m:.6f}",
                    "velocity_mps": f"{nuisance.velocity_mps:.6f}",
                    "azimuth_deg": f"{nuisance.azimuth_deg:.6f}",
                    "x_m": f"{x_m:.6f}",
                    "y_m": f"{y_m:.6f}",
                    "amplitude_db": f"{nuisance.amplitude_db:.6f}",
                    "path_gain_linear": f"{nuisance.path_gain_linear:.6e}",
                    "score": "",
                }
            )

        for method_name, estimate in study.representative_trial.estimates.items():
            for detection_index, detection in enumerate(estimate.detections):
                x_m, y_m = _polar_to_xy(detection.range_m, detection.azimuth_deg)
                rows.append(
                    {
                        **reference,
                        "entity_kind": "detection",
                        "entity_group": method_name,
                        "entity_label": f"{METHOD_LABELS[method_name]} detection {detection_index + 1:d}",
                        "method": method_name,
                        "method_label": METHOD_LABELS[method_name],
                        "entity_index": f"{detection_index:d}",
                        "range_m": f"{detection.range_m:.6f}",
                        "velocity_mps": f"{detection.velocity_mps:.6f}",
                        "azimuth_deg": f"{detection.azimuth_deg:.6f}",
                        "x_m": f"{x_m:.6f}",
                        "y_m": f"{y_m:.6f}",
                        "amplitude_db": "",
                        "path_gain_linear": "",
                        "score": f"{detection.score:.6f}",
                    }
                )
    return rows


def _representative_range_doppler_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.representative_trial is None:
            continue
        reference = _study_reference_fields(study)
        fft_cube = study.representative_trial.fft_cube
        power_cube = fft_cube.power_cube
        range_doppler = np.max(power_cube, axis=0)
        azimuth_peak_indices = np.argmax(power_cube, axis=0)
        for range_index, range_m in enumerate(fft_cube.range_axis_m):
            for velocity_index, velocity_mps in enumerate(fft_cube.velocity_axis_mps):
                power_linear = float(range_doppler[range_index, velocity_index])
                peak_azimuth_deg = float(fft_cube.azimuth_axis_deg[azimuth_peak_indices[range_index, velocity_index]])
                rows.append(
                    {
                        **reference,
                        "range_index": f"{range_index:d}",
                        "velocity_index": f"{velocity_index:d}",
                        "range_m": f"{float(range_m):.6f}",
                        "velocity_mps": f"{float(velocity_mps):.6f}",
                        "peak_azimuth_deg": f"{peak_azimuth_deg:.6f}",
                        "power_linear": f"{power_linear:.6e}",
                        "power_db": f"{10.0 * np.log10(max(power_linear, 1.0e-12)):.6f}",
                    }
                )
    return rows


def _representative_music_spectrum_rows_for_series(
    study: CommunicationsStudyResult,
    *,
    cfg,
    series_name: str,
    series_label: str,
    detections: tuple,
    estimator_set: str,
) -> list[dict[str, str]]:
    if study.representative_trial is None:
        return []
    reference = _study_reference_fields(study)
    reference["estimator_set"] = estimator_set
    known_cube = extract_known_symbol_cube(study.representative_trial.masked_observation)
    known_mask = study.representative_trial.masked_observation.known_symbol_mask
    global_matrix = known_cube.reshape(known_cube.shape[0], -1)
    search_bounds = fft_search_bounds(study.representative_trial.fft_cube)
    range_fbss_enabled = cfg.music_range_fbss_fraction > 0.0
    doppler_fbss_enabled = cfg.music_doppler_fbss_fraction > 0.0

    spatial_cov = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
    spatial_positions = cfg.effective_horizontal_positions_m[: cfg.fbss_subarray_len]
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
        steering_matrix=azimuth_steering_matrix(spatial_positions, azimuth_grid, cfg.wavelength_m),
    )
    azimuth_spectrum_db = 10.0 * np.log10(np.maximum(azimuth_spectrum / np.max(azimuth_spectrum), 1.0e-12))

    if detections:
        representative_azimuth_deg = float(detections[0].azimuth_deg)
    else:
        representative_azimuth_deg = float(azimuth_grid[int(np.argmax(azimuth_spectrum))])
    azimuth_weights = azimuth_steering_matrix(
        cfg.effective_horizontal_positions_m,
        np.array([representative_azimuth_deg]),
        cfg.wavelength_m,
    )[:, 0]
    azimuth_weights /= np.sqrt(max(1, azimuth_weights.size))
    beamformed = np.einsum("h,hft->ft", azimuth_weights.conj(), known_cube, optimize=True)

    range_search_upper = _range_search_upper_m(cfg, search_bounds)
    range_grid_step_m = max(0.5 * cfg.range_resolution_m, 0.25)
    n_range_grid = max(
        cfg.runtime_profile.music_grid_points * 2,
        int(np.ceil((range_search_upper - max(0.5, search_bounds.range_min_m)) / range_grid_step_m)) + 1,
    )
    range_grid = np.linspace(max(0.5, search_bounds.range_min_m), range_search_upper, n_range_grid)
    range_cov, range_frequencies_hz, _ = _range_music_covariance(cfg, beamformed, known_mask)
    range_spectrum = music_pseudospectrum(
        range_cov,
        n_targets=1,
        steering_matrix=range_steering_matrix(range_frequencies_hz, range_grid),
    )
    range_spectrum_db = 10.0 * np.log10(np.maximum(range_spectrum / np.max(range_spectrum), 1.0e-12))

    if detections:
        representative_range_m = float(detections[0].range_m)
    else:
        representative_range_m = float(range_grid[int(np.argmax(range_spectrum))])
    range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([representative_range_m]))[:, 0]
    doppler_grid = np.linspace(
        search_bounds.velocity_min_mps,
        search_bounds.velocity_max_mps,
        cfg.runtime_profile.music_grid_points,
    )
    doppler_signal = (beamformed * range_weights.conj()[:, np.newaxis]).T
    doppler_cov, doppler_times_s, _ = _doppler_music_covariance(cfg, doppler_signal, known_mask)
    doppler_spectrum = music_pseudospectrum(
        doppler_cov,
        n_targets=1,
        steering_matrix=doppler_steering_matrix(doppler_times_s, doppler_grid, cfg.wavelength_m),
    )
    doppler_spectrum_db = 10.0 * np.log10(np.maximum(doppler_spectrum / np.max(doppler_spectrum), 1.0e-12))

    rows: list[dict[str, str]] = []
    for dimension_name, coordinate_name, coordinates, spectrum_linear, spectrum_db, conditioning_azimuth_deg, conditioning_range_m in (
        ("azimuth", "azimuth_deg", azimuth_grid, azimuth_spectrum, azimuth_spectrum_db, None, None),
        ("range", "range_m", range_grid, range_spectrum, range_spectrum_db, representative_azimuth_deg, None),
        ("doppler", "velocity_mps", doppler_grid, doppler_spectrum, doppler_spectrum_db, representative_azimuth_deg, representative_range_m),
    ):
        for coordinate_value, spectrum_linear_value, spectrum_db_value in zip(
            coordinates.tolist(),
            spectrum_linear.tolist(),
            spectrum_db.tolist(),
            strict=True,
        ):
            rows.append(
                {
                    **reference,
                    "series_name": series_name,
                    "series_label": series_label,
                    "spatial_fbss": "1",
                    "range_fbss": f"{int(range_fbss_enabled):d}",
                    "doppler_fbss": f"{int(doppler_fbss_enabled):d}",
                    "dimension": dimension_name,
                    "coordinate_name": coordinate_name,
                    "coordinate_value": f"{float(coordinate_value):.6f}",
                    "spectrum_linear": f"{float(spectrum_linear_value):.6e}",
                    "spectrum_db_rel": f"{float(spectrum_db_value):.6f}",
                    "conditioning_azimuth_deg": _format_optional_float(conditioning_azimuth_deg),
                    "conditioning_range_m": _format_optional_float(conditioning_range_m),
                    "estimated_model_order": f"{estimated_model_order:d}",
                    "spectrum_target_order": f"{spectrum_target_order:d}",
                }
            )
    return rows


def _representative_music_spectrum_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.representative_trial is None:
            continue
        rows.extend(
            _representative_music_spectrum_rows_for_series(
                study,
                cfg=study.config,
                series_name="music_masked",
                series_label=METHOD_LABELS["music_masked"],
                detections=study.representative_trial.estimates["music_masked"].detections,
                estimator_set=ESTIMATOR_SET,
            )
        )
    return rows


def _representative_fbss_ablation_spectrum_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.representative_trial is None:
            continue
        if study.representative_trial.fbss_ablation_estimates is None:
            continue
        for variant_name in FBSS_ABLATION_ORDER:
            rows.extend(
                _representative_music_spectrum_rows_for_series(
                    study,
                    cfg=_fbss_variant_cfg(study.config, variant_name),
                    series_name=variant_name,
                    series_label=FBSS_ABLATION_LABELS[variant_name],
                    detections=study.representative_trial.fbss_ablation_estimates[variant_name].detections,
                    estimator_set=FBSS_ABLATION_SET,
                )
            )
    return rows


def _categorical_positions(labels: list[str]) -> np.ndarray:
    return np.arange(len(labels), dtype=float)


def _plot_sweep_figure(output_path: Path, sweep_name: str, studies: list[CommunicationsStudyResult]) -> None:
    matching_sweeps = [sweep for study in studies for sweep in study.sweeps if sweep.sweep_name == sweep_name]
    if not matching_sweeps:
        return

    metric_name, metric_label = _sweep_metric(sweep_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    is_numeric = all(point.parameter_numeric_value is not None for sweep in matching_sweeps for point in sweep.points)
    if is_numeric:
        for sweep in matching_sweeps:
            for method_name in METHOD_ORDER:
                x_values = np.asarray([point.parameter_numeric_value for point in sweep.points], dtype=float)
                y_values = np.asarray(
                    [getattr(point.method_summaries[method_name], metric_name) for point in sweep.points],
                    dtype=float,
                )
                ax.plot(
                    x_values,
                    y_values,
                    color=METHOD_COLORS[method_name],
                    marker=METHOD_MARKERS[method_name],
                    linewidth=2.0,
                    label=f"{METHOD_LABELS[method_name]} / {sweep.anchor_label} / {sweep.scene_label}",
                )
        ax.set_xlabel(matching_sweeps[0].parameter_label)
    else:
        labels = [point.parameter_value for point in matching_sweeps[0].points]
        positions = _categorical_positions(labels)
        series_count = len(METHOD_ORDER) * len(matching_sweeps)
        bar_width = 0.8 / max(series_count, 1)
        series_index = 0
        for sweep in matching_sweeps:
            for method_name in METHOD_ORDER:
                x_values = positions - 0.4 + bar_width * series_index + 0.5 * bar_width
                y_values = np.asarray(
                    [getattr(point.method_summaries[method_name], metric_name) for point in sweep.points],
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
    ax.set_ylabel(metric_label)
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
            row = {
                "schema_version": SCHEMA_VERSION,
                "evidence_profile": study.evidence_profile_name,
                "estimator_set": ESTIMATOR_SET,
                "anchor": study.anchor_name,
                "anchor_label": study.anchor_label,
                "scene_class": study.scene_class_name,
                "scene_label": study.scene_label,
                "allocation_family": point.allocation_family,
                "allocation_label": point.allocation_label,
                "occupied_fraction": f"{point.occupied_fraction:.6f}",
                "fragmentation_index": f"{point.fragmentation_index:.6f}",
                "bandwidth_span_fraction": f"{point.bandwidth_span_fraction:.6f}",
                "slow_time_span_fraction": f"{point.slow_time_span_fraction:.6f}",
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
                "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                "miss_probability": f"{summary.miss_probability:.6f}",
                "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                "mean_estimated_model_order": (
                    f"{summary.mean_estimated_model_order:.6f}" if summary.mean_estimated_model_order is not None else ""
                ),
                "estimated_model_order_accuracy": (
                    f"{summary.estimated_model_order_accuracy:.6f}" if summary.estimated_model_order_accuracy is not None else ""
                ),
                "total_runtime_s": f"{summary.total_runtime_s:.6f}",
            }
            row.update(_metric_probability_fields(summary))
            rows.append(row)
    return rows


def _pilot_only_nominal_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        if study.pilot_only_nominal_point is None:
            continue
        point = study.pilot_only_nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            row = {
                "schema_version": SCHEMA_VERSION,
                "evidence_profile": study.evidence_profile_name,
                "estimator_set": ESTIMATOR_SET,
                "anchor": study.anchor_name,
                "anchor_label": study.anchor_label,
                "scene_class": study.scene_class_name,
                "scene_label": study.scene_label,
                "knowledge_mode": point.knowledge_mode,
                "allocation_family": point.allocation_family,
                "allocation_label": point.allocation_label,
                "occupied_fraction": f"{point.occupied_fraction:.6f}",
                "fragmentation_index": f"{point.fragmentation_index:.6f}",
                "bandwidth_span_fraction": f"{point.bandwidth_span_fraction:.6f}",
                "slow_time_span_fraction": f"{point.slow_time_span_fraction:.6f}",
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
                "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                "miss_probability": f"{summary.miss_probability:.6f}",
                "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                "total_runtime_s": f"{summary.total_runtime_s:.6f}",
            }
            row.update(_metric_probability_fields(summary))
            rows.append(row)
    return rows


def _runtime_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_profile": study.evidence_profile_name,
                    "estimator_set": ESTIMATOR_SET,
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
            metric_name, _metric_label = _sweep_metric(sweep.sweep_name)
            for point in sweep.points:
                for method_name in METHOD_ORDER:
                    summary = point.method_summaries[method_name]
                    metric_value = getattr(summary, metric_name)
                    if metric_value >= 0.999:
                        continue
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "evidence_profile": study.evidence_profile_name,
                            "estimator_set": ESTIMATOR_SET,
                            "anchor": study.anchor_name,
                            "scene_class": study.scene_class_name,
                            "sweep_name": point.sweep_name,
                            "parameter_value": point.parameter_value,
                            "allocation_family": point.allocation_family,
                            "method": method_name,
                            "method_label": METHOD_LABELS[method_name],
                            "headline_metric": metric_name,
                            "headline_metric_value": f"{metric_value:.6f}",
                            "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                            "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                            "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                            "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                            "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
                            "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                            "miss_probability": f"{summary.miss_probability:.6f}",
                        }
                    )
    rows.sort(key=lambda row: (float(row["headline_metric_value"]), -float(row["miss_probability"])))
    return rows


def _usefulness_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        for sweep in study.sweeps:
            metric_name, _metric_label = _sweep_metric(sweep.sweep_name)
            for point in sweep.points:
                fft_summary = point.method_summaries["fft_masked"]
                music_summary = point.method_summaries["music_masked"]
                fft_value = float(getattr(fft_summary, metric_name))
                music_value = float(getattr(music_summary, metric_name))
                delta = music_value - fft_value
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "evidence_profile": study.evidence_profile_name,
                        "estimator_set": ESTIMATOR_SET,
                        "anchor": study.anchor_name,
                        "scene_class": study.scene_class_name,
                        "scene_label": study.scene_label,
                        "sweep_name": sweep.sweep_name,
                        "parameter_value": point.parameter_value,
                        "parameter_numeric_value": (
                            f"{point.parameter_numeric_value:.6f}" if point.parameter_numeric_value is not None else ""
                        ),
                        "metric_name": metric_name,
                        "fft_value": f"{fft_value:.6f}",
                        "music_value": f"{music_value:.6f}",
                        "music_minus_fft": f"{delta:.6f}",
                        "usefulness_window": "1" if delta >= USEFULNESS_DELTA_THRESHOLD else "0",
                    }
                )
    return rows


def _fbss_ablation_rows(studies: list[CommunicationsStudyResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        ablation_points = [study.nominal_point]
        ablation_points.extend(
            point
            for sweep in study.sweeps
            if sweep.sweep_name in FBSS_ABLATION_SWEEP_NAMES and sweep.sweep_name != "nominal"
            for point in sweep.points
        )
        for point in ablation_points:
            if point.fbss_ablation_summaries is None:
                continue
            for method_name in FBSS_ABLATION_ORDER:
                summary = point.fbss_ablation_summaries[method_name]
                range_fbss_enabled, doppler_fbss_enabled = FBSS_ABLATION_FLAGS[method_name]
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_profile": study.evidence_profile_name,
                    "estimator_set": FBSS_ABLATION_SET,
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
                    "occupied_fraction": f"{point.occupied_fraction:.6f}",
                    "fragmentation_index": f"{point.fragmentation_index:.6f}",
                    "bandwidth_span_fraction": f"{point.bandwidth_span_fraction:.6f}",
                    "slow_time_span_fraction": f"{point.slow_time_span_fraction:.6f}",
                    "method": method_name,
                    "method_label": FBSS_ABLATION_LABELS[method_name],
                    "spatial_fbss": "1",
                    "range_fbss": f"{int(range_fbss_enabled):d}",
                    "doppler_fbss": f"{int(doppler_fbss_enabled):d}",
                    "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                    "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                    "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                    "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                    "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
                    "false_alarm_probability": f"{summary.false_alarm_probability:.6f}",
                    "miss_probability": f"{summary.miss_probability:.6f}",
                    "unconditional_joint_assignment_rmse": f"{summary.unconditional_joint_assignment_rmse:.6f}",
                    "frontend_runtime_s": f"{summary.frontend_runtime_s:.6f}",
                    "incremental_runtime_s": f"{summary.incremental_runtime_s:.6f}",
                    "total_runtime_s": f"{summary.total_runtime_s:.6f}",
                }
                row.update(_metric_probability_fields(summary))
                rows.append(row)
    return rows


def _comparison_rows(studies: list[CommunicationsStudyResult], key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for study in studies:
        point = study.nominal_point
        for method_name in METHOD_ORDER:
            summary = point.method_summaries[method_name]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "evidence_profile": study.evidence_profile_name,
                    "estimator_set": ESTIMATOR_SET,
                    "comparison_key": key,
                    "anchor": study.anchor_name,
                    "scene_class": study.scene_class_name,
                    "method": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "joint_detection_probability": f"{summary.joint_detection_probability:.6f}",
                    "joint_resolution_probability": f"{summary.joint_resolution_probability:.6f}",
                    "range_resolution_probability": f"{summary.range_resolution_probability:.6f}",
                    "velocity_resolution_probability": f"{summary.velocity_resolution_probability:.6f}",
                    "angle_resolution_probability": f"{summary.angle_resolution_probability:.6f}",
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
    if study.representative_trial is None:
        return
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
    if study.representative_trial is None:
        return
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
    all_sweep_rows = _all_sweep_rows(studies)

    for sweep_name in sweep_names:
        sweep_rows = [
            row
            for row in all_sweep_rows
            if row["sweep_name"] == sweep_name
        ]
        _write_csv(data_dir / f"{sweep_name}.csv", sweep_rows)
        _plot_sweep_figure(figures_dir / f"{sweep_name}.png", sweep_name, studies)

    _write_csv(data_dir / "all_sweep_results.csv", all_sweep_rows)
    _write_csv(data_dir / "trial_level_results.csv", _trial_level_rows(studies))
    _write_csv(data_dir / "nominal_summary.csv", _nominal_rows(studies))
    pilot_only_rows = _pilot_only_nominal_rows(studies)
    if pilot_only_rows:
        _write_csv(data_dir / "pilot_only_nominal_summary.csv", pilot_only_rows)
    _write_csv(data_dir / "runtime_summary.csv", _runtime_rows(studies))
    _write_csv(data_dir / "failure_modes.csv", _failure_rows(studies))
    _write_csv(data_dir / "usefulness_windows.csv", _usefulness_rows(studies))
    stage_diag_rows = _stage_diagnostic_rows(studies)
    if stage_diag_rows:
        _write_csv(data_dir / "stage_diagnostics.csv", stage_diag_rows)
    fbss_rows = _fbss_ablation_rows(studies)
    if fbss_rows:
        _write_csv(data_dir / "fbss_ablation_results.csv", fbss_rows)
    representative_resource_rows = _representative_resource_mask_rows(studies)
    if representative_resource_rows:
        _write_csv(data_dir / "representative_resource_mask.csv", representative_resource_rows)
    representative_geometry_rows = _representative_scene_geometry_rows(studies)
    if representative_geometry_rows:
        _write_csv(data_dir / "representative_scene_geometry.csv", representative_geometry_rows)
    representative_range_doppler_rows = _representative_range_doppler_rows(studies)
    if representative_range_doppler_rows:
        _write_csv(data_dir / "representative_range_doppler.csv", representative_range_doppler_rows)
    representative_music_rows = _representative_music_spectrum_rows(studies)
    if representative_music_rows:
        _write_csv(data_dir / "representative_music_spectra.csv", representative_music_rows)
    representative_fbss_rows = _representative_fbss_ablation_spectrum_rows(studies)
    if representative_fbss_rows:
        _write_csv(data_dir / "representative_fbss_ablation_spectra.csv", representative_fbss_rows)
    _plot_runtime_summary(figures_dir / "runtime_summary.png", studies)

    if include_scene_comparison:
        _write_csv(data_dir / "scene_comparison.csv", _comparison_rows(studies, key="scene"))
    if include_anchor_comparison:
        _write_csv(data_dir / "anchor_comparison.csv", _comparison_rows(studies, key="anchor"))

    if studies and studies[0].representative_trial is not None:
        _plot_representative_resource_mask(figures_dir / "representative_resource_mask.png", studies[0])
        _plot_representative_spectrum(figures_dir / "representative_spectrum.png", studies[0])
