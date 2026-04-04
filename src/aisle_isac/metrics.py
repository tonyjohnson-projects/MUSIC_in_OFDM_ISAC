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
    within_joint_resolution_gate: bool
    within_range_resolution_gate: bool
    within_velocity_resolution_gate: bool
    within_angle_resolution_gate: bool


@dataclass(frozen=True)
class MethodTrialMetrics:
    """Per-trial sensing outcomes for one estimator."""

    detections: np.ndarray
    reported_target_count: int
    estimated_model_order: int | None
    matched_target_count: int
    assignments: tuple[TrialAssignment, ...]
    joint_detection_success: bool
    joint_resolution_success: bool
    range_resolution_success: bool
    velocity_resolution_success: bool
    angle_resolution_success: bool
    false_alarm_count: int
    miss_count: int
    any_false_alarm: bool
    any_miss: bool
    unconditional_range_rmse_m: float
    unconditional_velocity_rmse_mps: float
    unconditional_angle_rmse_deg: float
    unconditional_joint_assignment_rmse: float
    conditional_range_rmse_m: float | None
    conditional_velocity_rmse_mps: float | None
    conditional_angle_rmse_deg: float | None
    conditional_joint_assignment_rmse: float | None
    frontend_runtime_s: float
    incremental_runtime_s: float
    total_runtime_s: float


@dataclass(frozen=True)
class MethodPointSummary:
    """Aggregated sweep-point metrics for one estimator."""

    trial_count: int
    joint_detection_success_count: int
    joint_resolution_success_count: int
    range_resolution_success_count: int
    velocity_resolution_success_count: int
    angle_resolution_success_count: int
    false_alarm_event_count: int
    miss_event_count: int
    joint_detection_probability: float
    joint_resolution_probability: float
    range_resolution_probability: float
    velocity_resolution_probability: float
    angle_resolution_probability: float
    unconditional_range_rmse_m: float
    unconditional_velocity_rmse_mps: float
    unconditional_angle_rmse_deg: float
    unconditional_joint_assignment_rmse: float
    conditional_range_rmse_m: float | None
    conditional_velocity_rmse_mps: float | None
    conditional_angle_rmse_deg: float | None
    conditional_joint_assignment_rmse: float | None
    false_alarm_probability: float
    miss_probability: float
    reported_target_count_accuracy: float
    mean_estimated_model_order: float | None
    estimated_model_order_accuracy: float | None
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


def evaluate_trial(
    cfg: StudyConfig,
    truth_targets: tuple[TargetState, ...],
    detections: tuple,
    reported_target_count: int,
    estimated_model_order: int | None,
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
            within_detection_gate = norm_range <= 1.0 and norm_velocity <= 1.0 and norm_angle <= 1.0
            within_joint_resolution_gate = (
                within_detection_gate
                and norm_range <= cfg.resolution_cell_fraction
                and norm_velocity <= cfg.resolution_cell_fraction
                and norm_angle <= cfg.resolution_cell_fraction
            )
            within_range_resolution_gate = within_detection_gate and norm_range <= cfg.resolution_cell_fraction
            within_velocity_resolution_gate = within_detection_gate and norm_velocity <= cfg.resolution_cell_fraction
            within_angle_resolution_gate = within_detection_gate and norm_angle <= cfg.resolution_cell_fraction
            if within_detection_gate:
                matched_detection_indices.add(detection_index)
                matched_truth_indices.add(truth_index)
            normalized_error = float(np.sqrt(np.mean([norm_range**2, norm_velocity**2, norm_angle**2])))
            assignments.append(
                TrialAssignment(
                    truth_index=truth_index,
                    detection_index=detection_index,
                    range_error_m=range_error_m,
                    velocity_error_mps=velocity_error_mps,
                    angle_error_deg=angle_error_deg,
                    normalized_error=normalized_error,
                    within_detection_gate=within_detection_gate,
                    within_joint_resolution_gate=within_joint_resolution_gate,
                    within_range_resolution_gate=within_range_resolution_gate,
                    within_velocity_resolution_gate=within_velocity_resolution_gate,
                    within_angle_resolution_gate=within_angle_resolution_gate,
                )
            )

    detection_gated_assignments = [assignment for assignment in assignments if assignment.within_detection_gate]
    joint_resolution_assignments = [assignment for assignment in assignments if assignment.within_joint_resolution_gate]
    range_resolution_assignments = [assignment for assignment in assignments if assignment.within_range_resolution_gate]
    velocity_resolution_assignments = [assignment for assignment in assignments if assignment.within_velocity_resolution_gate]
    angle_resolution_assignments = [assignment for assignment in assignments if assignment.within_angle_resolution_gate]

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
    false_alarm_count = max(0, len(detections) - len(matched_detection_indices))
    miss_count = max(0, len(truth_targets) - len(matched_truth_indices))

    assignment_by_truth_index = {assignment.truth_index: assignment for assignment in assignments}
    unconditional_range_errors: list[float] = []
    unconditional_velocity_errors: list[float] = []
    unconditional_angle_errors: list[float] = []
    unconditional_normalized_errors: list[float] = []
    for truth_index, _truth_target in enumerate(truth_targets):
        assignment = assignment_by_truth_index.get(truth_index)
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
        unconditional_range_errors.append(range_error_m)
        unconditional_velocity_errors.append(velocity_error_mps)
        unconditional_angle_errors.append(angle_error_deg)
        unconditional_normalized_errors.append(float(np.sqrt(np.mean([norm_range**2, norm_velocity**2, norm_angle**2]))))

    unconditional_range_rmse_m = float(np.sqrt(np.mean(np.square(unconditional_range_errors))))
    unconditional_velocity_rmse_mps = float(np.sqrt(np.mean(np.square(unconditional_velocity_errors))))
    unconditional_angle_rmse_deg = float(np.sqrt(np.mean(np.square(unconditional_angle_errors))))
    unconditional_joint_assignment_rmse = float(np.sqrt(np.mean(np.square(unconditional_normalized_errors))))

    expected_truth_count = len(truth_targets)
    return MethodTrialMetrics(
        detections=detection_array,
        reported_target_count=reported_target_count,
        estimated_model_order=estimated_model_order,
        matched_target_count=len(detection_gated_assignments),
        assignments=tuple(assignments),
        joint_detection_success=len(detection_gated_assignments) == expected_truth_count,
        joint_resolution_success=len(joint_resolution_assignments) == expected_truth_count,
        range_resolution_success=len(range_resolution_assignments) == expected_truth_count,
        velocity_resolution_success=len(velocity_resolution_assignments) == expected_truth_count,
        angle_resolution_success=len(angle_resolution_assignments) == expected_truth_count,
        false_alarm_count=false_alarm_count,
        miss_count=miss_count,
        any_false_alarm=false_alarm_count > 0,
        any_miss=miss_count > 0,
        unconditional_range_rmse_m=unconditional_range_rmse_m,
        unconditional_velocity_rmse_mps=unconditional_velocity_rmse_mps,
        unconditional_angle_rmse_deg=unconditional_angle_rmse_deg,
        unconditional_joint_assignment_rmse=unconditional_joint_assignment_rmse,
        conditional_range_rmse_m=conditional_range_rmse_m,
        conditional_velocity_rmse_mps=conditional_velocity_rmse_mps,
        conditional_angle_rmse_deg=conditional_angle_rmse_deg,
        conditional_joint_assignment_rmse=conditional_joint_assignment_rmse,
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

    def _aggregate(values: list[float | None]) -> float | None:
        finite_values = [value for value in values if value is not None]
        if not finite_values:
            return None
        return float(np.mean(finite_values))

    estimated_orders = [metric.estimated_model_order for metric in metrics if metric.estimated_model_order is not None]
    return MethodPointSummary(
        trial_count=trial_count,
        joint_detection_success_count=sum(metric.joint_detection_success for metric in metrics),
        joint_resolution_success_count=sum(metric.joint_resolution_success for metric in metrics),
        range_resolution_success_count=sum(metric.range_resolution_success for metric in metrics),
        velocity_resolution_success_count=sum(metric.velocity_resolution_success for metric in metrics),
        angle_resolution_success_count=sum(metric.angle_resolution_success for metric in metrics),
        false_alarm_event_count=sum(metric.any_false_alarm for metric in metrics),
        miss_event_count=sum(metric.any_miss for metric in metrics),
        joint_detection_probability=float(np.mean([metric.joint_detection_success for metric in metrics], dtype=float)),
        joint_resolution_probability=float(np.mean([metric.joint_resolution_success for metric in metrics], dtype=float)),
        range_resolution_probability=float(np.mean([metric.range_resolution_success for metric in metrics], dtype=float)),
        velocity_resolution_probability=float(np.mean([metric.velocity_resolution_success for metric in metrics], dtype=float)),
        angle_resolution_probability=float(np.mean([metric.angle_resolution_success for metric in metrics], dtype=float)),
        unconditional_range_rmse_m=float(np.mean([metric.unconditional_range_rmse_m for metric in metrics])),
        unconditional_velocity_rmse_mps=float(np.mean([metric.unconditional_velocity_rmse_mps for metric in metrics])),
        unconditional_angle_rmse_deg=float(np.mean([metric.unconditional_angle_rmse_deg for metric in metrics])),
        unconditional_joint_assignment_rmse=float(np.mean([metric.unconditional_joint_assignment_rmse for metric in metrics])),
        conditional_range_rmse_m=_aggregate([metric.conditional_range_rmse_m for metric in metrics]),
        conditional_velocity_rmse_mps=_aggregate([metric.conditional_velocity_rmse_mps for metric in metrics]),
        conditional_angle_rmse_deg=_aggregate([metric.conditional_angle_rmse_deg for metric in metrics]),
        conditional_joint_assignment_rmse=_aggregate([metric.conditional_joint_assignment_rmse for metric in metrics]),
        false_alarm_probability=float(np.mean([metric.any_false_alarm for metric in metrics], dtype=float)),
        miss_probability=float(np.mean([metric.any_miss for metric in metrics], dtype=float)),
        reported_target_count_accuracy=float(
            np.mean([metric.reported_target_count == expected_target_count for metric in metrics], dtype=float)
        ),
        mean_estimated_model_order=_aggregate([float(value) for value in estimated_orders]) if estimated_orders else None,
        estimated_model_order_accuracy=(
            float(np.mean([value == expected_target_count for value in estimated_orders], dtype=float))
            if estimated_orders
            else None
        ),
        frontend_runtime_s=float(np.mean([metric.frontend_runtime_s for metric in metrics])),
        incremental_runtime_s=float(np.mean([metric.incremental_runtime_s for metric in metrics])),
        total_runtime_s=float(np.mean([metric.total_runtime_s for metric in metrics])),
    )
