"""Radar-only metrics for the angle-range-Doppler study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from aisle_isac.channel_models import TargetState
from aisle_isac.config import StudyConfig


@dataclass(frozen=True)
class TrialAssignment:
    """One truth-to-detection assignment."""

    truth_index: int
    detection_index: int
    range_error_m: float
    velocity_error_mps: float
    angle_error_deg: float
    normalized_error: float
    within_detection_gate: bool
    within_resolution_gate: bool
    rmse_over_crb: float | None


@dataclass(frozen=True)
class MethodTrialMetrics:
    """Per-trial sensing outcomes for one estimator."""

    detections: np.ndarray
    reported_target_count: int
    matched_target_count: int
    assignments: tuple[TrialAssignment, ...]
    joint_detection_success: bool
    joint_resolution_success: bool
    false_alarm_count: int
    miss_count: int
    any_false_alarm: bool
    any_miss: bool
    unconditional_range_rmse_m: float
    unconditional_velocity_rmse_mps: float
    unconditional_angle_rmse_deg: float
    unconditional_joint_assignment_rmse: float
    unconditional_rmse_over_crb: float
    conditional_range_rmse_m: float | None
    conditional_velocity_rmse_mps: float | None
    conditional_angle_rmse_deg: float | None
    conditional_joint_assignment_rmse: float | None
    conditional_rmse_over_crb: float | None
    scene_cost: float
    frontend_runtime_s: float
    incremental_runtime_s: float
    total_runtime_s: float


@dataclass(frozen=True)
class MethodPointSummary:
    """Aggregated sweep-point metrics for one estimator."""

    trial_count: int
    joint_detection_success_count: int
    joint_resolution_success_count: int
    false_alarm_event_count: int
    miss_event_count: int
    joint_detection_probability: float
    joint_resolution_probability: float
    unconditional_range_rmse_m: float
    unconditional_velocity_rmse_mps: float
    unconditional_angle_rmse_deg: float
    unconditional_joint_assignment_rmse: float
    unconditional_rmse_over_crb: float
    conditional_range_rmse_m: float | None
    conditional_velocity_rmse_mps: float | None
    conditional_angle_rmse_deg: float | None
    conditional_joint_assignment_rmse: float | None
    conditional_rmse_over_crb: float | None
    scene_cost: float
    false_alarm_probability: float
    miss_probability: float
    reported_target_count_accuracy: float
    frontend_runtime_s: float
    incremental_runtime_s: float
    total_runtime_s: float


def _normalized_components(
    cfg: StudyConfig,
    range_error_m: float,
    velocity_error_mps: float,
    angle_error_deg: float,
) -> tuple[float, float, float]:
    return (
        range_error_m / max(cfg.range_resolution_m, 1.0e-9),
        velocity_error_mps / max(cfg.velocity_resolution_mps, 1.0e-9),
        angle_error_deg / max(cfg.azimuth_resolution_deg, 1.0e-9),
    )


def _joint_target_crb(
    cfg: StudyConfig,
    targets: tuple[TargetState, ...],
    noise_variance: float,
) -> list[float]:
    """Joint multi-target CRB returning per-target normalized CRB norms.

    Builds the full joint Fisher Information Matrix for all target
    parameters (range, velocity, azimuth per target).  Cross-terms
    between closely-spaced targets increase the CRB relative to the
    single-target bound, correctly reflecting the resolution limit.
    """

    positions = cfg.effective_horizontal_positions_m
    frequencies = cfg.frequencies_hz
    times = cfg.snapshot_times_s
    wavelength = cfg.wavelength_m
    c = 299_792_458.0

    n_targets = len(targets)
    n_params = 3 * n_targets

    grad_list: list[np.ndarray] = []
    for target in targets:
        a_h = np.exp(-1j * 2 * np.pi * positions * np.sin(np.deg2rad(target.azimuth_deg)) / wavelength)
        a_f = np.exp(-1j * 2 * np.pi * frequencies * 2 * target.range_m / c)
        a_t = np.exp(1j * 2 * np.pi * times * 2 * target.velocity_mps / wavelength)

        g_k = target.path_gain_linear * a_h[:, None, None] * a_f[None, :, None] * a_t[None, None, :]

        dr = g_k * (-1j * 4 * np.pi / c * frequencies)[None, :, None]
        dv = g_k * (1j * 4 * np.pi / wavelength * times)[None, None, :]
        daz = g_k * (
            -1j * 2 * np.pi / wavelength
            * np.cos(np.deg2rad(target.azimuth_deg))
            * (np.pi / 180.0)
            * positions
        )[:, None, None]

        grad_list.extend([dr.ravel(), dv.ravel(), daz.ravel()])

    grad = np.array(grad_list)
    fim = (2.0 / max(noise_variance, 1.0e-12)) * np.real(grad @ grad.conj().T)

    try:
        crb_diag = np.diag(np.linalg.inv(fim))
    except np.linalg.LinAlgError:
        crb_diag = np.full(n_params, 1.0)

    results: list[float] = []
    for k in range(n_targets):
        range_std = np.sqrt(max(crb_diag[3 * k + 0], 0.0))
        vel_std = np.sqrt(max(crb_diag[3 * k + 1], 0.0))
        az_std_deg = np.sqrt(max(crb_diag[3 * k + 2], 0.0))

        normalized = np.asarray(
            [
                range_std / max(cfg.range_resolution_m, 1.0e-9),
                vel_std / max(cfg.velocity_resolution_mps, 1.0e-9),
                az_std_deg / max(cfg.azimuth_resolution_deg, 1.0e-9),
            ],
            dtype=float,
        )
        results.append(float(np.sqrt(np.mean(normalized**2))))

    return results


def evaluate_trial(
    cfg: StudyConfig,
    truth_targets: tuple[TargetState, ...],
    detections: tuple,
    reported_target_count: int,
    noise_variance: float,
    frontend_runtime_s: float,
    incremental_runtime_s: float,
    total_runtime_s: float,
) -> MethodTrialMetrics:
    """Assign detections to truth and compute radar-only metrics."""

    detection_array = np.asarray(
        [[detection.range_m, detection.velocity_mps, detection.azimuth_deg] for detection in detections],
        dtype=float,
    )
    truth_array = np.asarray(
        [[target.range_m, target.velocity_mps, target.azimuth_deg] for target in truth_targets],
        dtype=float,
    )

    crb_norms = _joint_target_crb(cfg, truth_targets, noise_variance)

    assignments: list[TrialAssignment] = []
    matched_detection_indices: set[int] = set()
    matched_truth_indices: set[int] = set()
    if detection_array.size and truth_array.size:
        cost_matrix = np.zeros((truth_array.shape[0], detection_array.shape[0]), dtype=float)
        for truth_index, truth in enumerate(truth_array):
            for detection_index, detection in enumerate(detection_array):
                normalized = _normalized_components(
                    cfg,
                    abs(detection[0] - truth[0]),
                    abs(detection[1] - truth[1]),
                    abs(detection[2] - truth[2]),
                )
                cost_matrix[truth_index, detection_index] = float(np.mean(np.square(normalized)))
        truth_indices, detection_indices = linear_sum_assignment(cost_matrix)
        for truth_index, detection_index in zip(truth_indices.tolist(), detection_indices.tolist(), strict=True):
            range_error_m = abs(detection_array[detection_index, 0] - truth_array[truth_index, 0])
            velocity_error_mps = abs(detection_array[detection_index, 1] - truth_array[truth_index, 1])
            angle_error_deg = abs(detection_array[detection_index, 2] - truth_array[truth_index, 2])
            norm_range, norm_velocity, norm_angle = _normalized_components(
                cfg,
                range_error_m,
                velocity_error_mps,
                angle_error_deg,
            )
            within_detection_gate = (
                norm_range <= 1.0 and norm_velocity <= 1.0 and norm_angle <= 1.0
            )
            within_resolution_gate = (
                norm_range <= cfg.resolution_cell_fraction
                and norm_velocity <= cfg.resolution_cell_fraction
                and norm_angle <= cfg.resolution_cell_fraction
            )
            if within_detection_gate:
                matched_detection_indices.add(detection_index)
                matched_truth_indices.add(truth_index)
            normalized_error = float(np.sqrt(np.mean([norm_range**2, norm_velocity**2, norm_angle**2])))
            crb_norm = crb_norms[truth_index]
            assignments.append(
                TrialAssignment(
                    truth_index=truth_index,
                    detection_index=detection_index,
                    range_error_m=range_error_m,
                    velocity_error_mps=velocity_error_mps,
                    angle_error_deg=angle_error_deg,
                    normalized_error=normalized_error,
                    within_detection_gate=within_detection_gate,
                    within_resolution_gate=within_resolution_gate,
                    rmse_over_crb=normalized_error / max(crb_norm, 1.0e-12),
                )
            )

    detection_gated_assignments = [assignment for assignment in assignments if assignment.within_detection_gate]
    resolution_assignments = [assignment for assignment in assignments if assignment.within_resolution_gate]
    conditional_range_rmse_m = (
        float(np.sqrt(np.mean([assignment.range_error_m**2 for assignment in detection_gated_assignments])))
        if detection_gated_assignments
        else None
    )
    conditional_velocity_rmse_mps = (
        float(np.sqrt(np.mean([assignment.velocity_error_mps**2 for assignment in detection_gated_assignments])))
        if detection_gated_assignments
        else None
    )
    conditional_angle_rmse_deg = (
        float(np.sqrt(np.mean([assignment.angle_error_deg**2 for assignment in detection_gated_assignments])))
        if detection_gated_assignments
        else None
    )
    conditional_joint_assignment_rmse = (
        float(np.sqrt(np.mean([assignment.normalized_error**2 for assignment in detection_gated_assignments])))
        if detection_gated_assignments
        else None
    )
    conditional_rmse_over_crb = (
        float(np.mean([assignment.rmse_over_crb for assignment in detection_gated_assignments if assignment.rmse_over_crb is not None]))
        if detection_gated_assignments
        else None
    )
    false_alarm_count = max(0, len(detections) - len(matched_detection_indices))
    miss_count = max(0, len(truth_targets) - len(matched_truth_indices))

    assignment_by_truth_index = {assignment.truth_index: assignment for assignment in assignments}
    unconditional_range_errors: list[float] = []
    unconditional_velocity_errors: list[float] = []
    unconditional_angle_errors: list[float] = []
    unconditional_normalized_errors: list[float] = []
    unconditional_rmse_over_crb_values: list[float] = []
    for truth_index, truth_target in enumerate(truth_targets):
        assignment = assignment_by_truth_index.get(truth_index)
        crb_norm = crb_norms[truth_index]
        if assignment is None or not assignment.within_detection_gate:
            range_error_m = cfg.range_resolution_m
            velocity_error_mps = cfg.velocity_resolution_mps
            angle_error_deg = cfg.azimuth_resolution_deg
        else:
            range_error_m = assignment.range_error_m
            velocity_error_mps = assignment.velocity_error_mps
            angle_error_deg = assignment.angle_error_deg

        norm_range, norm_velocity, norm_angle = _normalized_components(
            cfg,
            range_error_m,
            velocity_error_mps,
            angle_error_deg,
        )
        normalized_error = float(np.sqrt(np.mean([norm_range**2, norm_velocity**2, norm_angle**2])))
        unconditional_range_errors.append(range_error_m)
        unconditional_velocity_errors.append(velocity_error_mps)
        unconditional_angle_errors.append(angle_error_deg)
        unconditional_normalized_errors.append(normalized_error)
        unconditional_rmse_over_crb_values.append(normalized_error / max(crb_norm, 1.0e-12))

    unconditional_range_rmse_m = float(np.sqrt(np.mean(np.square(unconditional_range_errors))))
    unconditional_velocity_rmse_mps = float(np.sqrt(np.mean(np.square(unconditional_velocity_errors))))
    unconditional_angle_rmse_deg = float(np.sqrt(np.mean(np.square(unconditional_angle_errors))))
    unconditional_joint_assignment_rmse = float(np.sqrt(np.mean(np.square(unconditional_normalized_errors))))
    unconditional_rmse_over_crb = float(np.mean(unconditional_rmse_over_crb_values))
    scene_cost = float(
        (np.sum(unconditional_normalized_errors) + false_alarm_count)
        / max(len(truth_targets), 1)
    )
    return MethodTrialMetrics(
        detections=detection_array,
        reported_target_count=reported_target_count,
        matched_target_count=len(detection_gated_assignments),
        assignments=tuple(assignments),
        joint_detection_success=len(detection_gated_assignments) == len(truth_targets),
        joint_resolution_success=len(resolution_assignments) == len(truth_targets),
        false_alarm_count=false_alarm_count,
        miss_count=miss_count,
        any_false_alarm=false_alarm_count > 0,
        any_miss=miss_count > 0,
        unconditional_range_rmse_m=unconditional_range_rmse_m,
        unconditional_velocity_rmse_mps=unconditional_velocity_rmse_mps,
        unconditional_angle_rmse_deg=unconditional_angle_rmse_deg,
        unconditional_joint_assignment_rmse=unconditional_joint_assignment_rmse,
        unconditional_rmse_over_crb=unconditional_rmse_over_crb,
        conditional_range_rmse_m=conditional_range_rmse_m,
        conditional_velocity_rmse_mps=conditional_velocity_rmse_mps,
        conditional_angle_rmse_deg=conditional_angle_rmse_deg,
        conditional_joint_assignment_rmse=conditional_joint_assignment_rmse,
        conditional_rmse_over_crb=conditional_rmse_over_crb,
        scene_cost=scene_cost,
        frontend_runtime_s=frontend_runtime_s,
        incremental_runtime_s=incremental_runtime_s,
        total_runtime_s=total_runtime_s,
    )


def summarize_method_metrics(
    metrics: list[MethodTrialMetrics],
    expected_target_count: int,
) -> MethodPointSummary:
    """Aggregate per-trial metrics over one sweep point."""

    trial_count = len(metrics)
    if trial_count == 0:
        raise ValueError("Cannot summarize an empty metrics list")

    detection_success_count = sum(metric.joint_detection_success for metric in metrics)
    resolution_success_count = sum(metric.joint_resolution_success for metric in metrics)
    false_alarm_event_count = sum(metric.any_false_alarm for metric in metrics)
    miss_event_count = sum(metric.any_miss for metric in metrics)
    reported_target_count_accuracy = float(
        np.mean([metric.reported_target_count == expected_target_count for metric in metrics], dtype=float)
    )

    def _aggregate(values: list[float | None]) -> float | None:
        finite_values = [value for value in values if value is not None]
        if not finite_values:
            return None
        return float(np.mean(finite_values))

    return MethodPointSummary(
        trial_count=trial_count,
        joint_detection_success_count=detection_success_count,
        joint_resolution_success_count=resolution_success_count,
        false_alarm_event_count=false_alarm_event_count,
        miss_event_count=miss_event_count,
        joint_detection_probability=detection_success_count / trial_count,
        joint_resolution_probability=resolution_success_count / trial_count,
        unconditional_range_rmse_m=float(np.mean([metric.unconditional_range_rmse_m for metric in metrics])),
        unconditional_velocity_rmse_mps=float(np.mean([metric.unconditional_velocity_rmse_mps for metric in metrics])),
        unconditional_angle_rmse_deg=float(np.mean([metric.unconditional_angle_rmse_deg for metric in metrics])),
        unconditional_joint_assignment_rmse=float(np.mean([metric.unconditional_joint_assignment_rmse for metric in metrics])),
        unconditional_rmse_over_crb=float(np.mean([metric.unconditional_rmse_over_crb for metric in metrics])),
        conditional_range_rmse_m=_aggregate([metric.conditional_range_rmse_m for metric in metrics]),
        conditional_velocity_rmse_mps=_aggregate([metric.conditional_velocity_rmse_mps for metric in metrics]),
        conditional_angle_rmse_deg=_aggregate([metric.conditional_angle_rmse_deg for metric in metrics]),
        conditional_joint_assignment_rmse=_aggregate([metric.conditional_joint_assignment_rmse for metric in metrics]),
        conditional_rmse_over_crb=_aggregate([metric.conditional_rmse_over_crb for metric in metrics]),
        scene_cost=float(np.mean([metric.scene_cost for metric in metrics])),
        false_alarm_probability=false_alarm_event_count / trial_count,
        miss_probability=miss_event_count / trial_count,
        reported_target_count_accuracy=reported_target_count_accuracy,
        frontend_runtime_s=float(np.mean([metric.frontend_runtime_s for metric in metrics])),
        incremental_runtime_s=float(np.mean([metric.incremental_runtime_s for metric in metrics])),
        total_runtime_s=float(np.mean([metric.total_runtime_s for metric in metrics])),
    )
