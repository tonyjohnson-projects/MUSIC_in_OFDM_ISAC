"""Allocation-driven MUSIC study harness for communications-limited OFDM ISAC."""

from __future__ import annotations

import concurrent.futures as futures
from dataclasses import dataclass
import os

import numpy as np

from aisle_isac.allocation_metrics import AllocationSummary, summarize_allocation
from aisle_isac.channel_models import TrialParameters
from aisle_isac.config import StudyConfig
from aisle_isac.estimators import FftCubeResult, MethodEstimate, config_search_bounds, validate_targets_within_search_bounds
from aisle_isac.estimators_fft_masked import prepare_masked_frontend
from aisle_isac.estimators_music import (
    FBSS_ABLATION_LABELS,
    FBSS_ABLATION_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    run_masked_estimators,
    run_masked_estimators_with_fbss_ablation,
)
from aisle_isac.masked_observation import MaskedObservation, simulate_masked_observation
from aisle_isac.metrics import MethodPointSummary, MethodTrialMetrics, evaluate_trial, summarize_method_metrics
from aisle_isac.resource_grid import ResourceGrid, ResourceElementRole, build_resource_grid
from aisle_isac.scenarios import build_study_config


PUBLIC_SWEEP_NAMES = (
    "allocation_family",
    "occupied_fraction",
    "fragmentation",
    "bandwidth_span",
    "slow_time_span",
    "range_separation",
    "velocity_separation",
    "angle_separation",
)

SUBMISSION_SWEEP_NAMES = PUBLIC_SWEEP_NAMES
AXIS_ISOLATION_MULTIPLIER = 2.4
FBSS_ABLATION_SWEEP_NAMES = ("nominal", "bandwidth_span", "slow_time_span")


@dataclass(frozen=True)
class CommunicationsTrialResult:
    """One communications-scheduled trial bundle."""

    masked_observation: MaskedObservation
    allocation_summary: AllocationSummary
    fft_cube: FftCubeResult
    estimates: dict[str, MethodEstimate]
    metrics: dict[str, MethodTrialMetrics]
    fbss_ablation_estimates: dict[str, MethodEstimate] | None = None
    fbss_ablation_metrics: dict[str, MethodTrialMetrics] | None = None


@dataclass(frozen=True)
class SweepPointSpec:
    """Serializable definition of one allocation-driven sweep point."""

    point_index: int
    sweep_name: str
    parameter_name: str
    parameter_label: str
    parameter_value: str
    parameter_numeric_value: float | None
    anchor_name: str
    scene_name: str
    profile_name: str
    trial_count: int
    suite: str
    burst_profile_name: str
    rx_columns: int
    center_range_m: float
    range_separation_m: float
    velocity_separation_mps: float
    angle_separation_deg: float
    allocation_family: str
    allocation_label: str
    knowledge_mode: str
    modulation_scheme: str
    resource_grid_kwargs: dict[str, object]
    occupied_fraction: float
    pilot_fraction: float
    fragmentation_index: float
    bandwidth_span_fraction: float
    slow_time_span_fraction: float


@dataclass(frozen=True)
class SweepPointResult:
    """Aggregated summary for one communications-scheduled sweep point."""

    sweep_name: str
    parameter_name: str
    parameter_label: str
    parameter_value: str
    parameter_numeric_value: float | None
    anchor_name: str
    anchor_label: str
    scene_class_name: str
    scene_label: str
    burst_profile_name: str
    burst_profile_label: str
    aperture_size: int
    target_pair: str
    allocation_family: str
    allocation_label: str
    knowledge_mode: str
    modulation_scheme: str
    occupied_fraction: float
    pilot_fraction: float
    fragmentation_index: float
    bandwidth_span_fraction: float
    slow_time_span_fraction: float
    method_summaries: dict[str, MethodPointSummary]
    fbss_ablation_summaries: dict[str, MethodPointSummary] | None = None
    trial_rows: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class SweepResult:
    """All points for one communications-scheduled sweep family."""

    sweep_name: str
    parameter_name: str
    parameter_label: str
    anchor_name: str
    anchor_label: str
    scene_class_name: str
    scene_label: str
    points: list[SweepPointResult]


@dataclass(frozen=True)
class CommunicationsStudyResult:
    """Complete allocation-driven study for one anchor and scene."""

    config: StudyConfig
    evidence_profile_name: str
    anchor_name: str
    anchor_label: str
    scene_class_name: str
    scene_label: str
    sweeps: list[SweepResult]
    nominal_point: SweepPointResult
    pilot_only_nominal_point: SweepPointResult
    representative_trial: CommunicationsTrialResult


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def _suite_values(suite: str, coarse: tuple, dense: tuple) -> tuple:
    return dense if suite == "full" else coarse


def nominal_trial_parameters(cfg: StudyConfig) -> TrialParameters:
    return TrialParameters(
        center_range_m=cfg.scene_class.nominal_range_m,
        range_separation_m=cfg.scene_class.default_range_separation_cells * cfg.range_resolution_m,
        velocity_separation_mps=cfg.scene_class.default_velocity_separation_cells * cfg.velocity_resolution_mps,
        angle_separation_deg=cfg.scene_class.default_angle_separation_cells * cfg.azimuth_resolution_deg,
    )


def _axis_isolated_trial_parameters(
    cfg: StudyConfig,
    *,
    range_separation_m: float | None = None,
    velocity_separation_mps: float | None = None,
    angle_separation_deg: float | None = None,
) -> TrialParameters:
    return TrialParameters(
        center_range_m=cfg.scene_class.nominal_range_m,
        range_separation_m=(
            range_separation_m
            if range_separation_m is not None
            else AXIS_ISOLATION_MULTIPLIER * cfg.range_resolution_m
        ),
        velocity_separation_mps=(
            velocity_separation_mps
            if velocity_separation_mps is not None
            else AXIS_ISOLATION_MULTIPLIER * cfg.velocity_resolution_mps
        ),
        angle_separation_deg=(
            angle_separation_deg
            if angle_separation_deg is not None
            else AXIS_ISOLATION_MULTIPLIER * cfg.azimuth_resolution_deg
        ),
    )


def _build_resource_grid_and_summary(
    cfg: StudyConfig,
    allocation_family: str,
    resource_grid_kwargs: dict[str, object],
) -> tuple[ResourceGrid, AllocationSummary]:
    resource_grid = build_resource_grid(
        allocation_family,
        cfg.n_subcarriers,
        cfg.burst_profile.n_snapshots,
        **resource_grid_kwargs,
    )
    return resource_grid, summarize_allocation(resource_grid)


def _make_point_spec(
    cfg: StudyConfig,
    *,
    point_index: int,
    sweep_name: str,
    parameter_name: str,
    parameter_label: str,
    parameter_value: str,
    parameter_numeric_value: float | None,
    params: TrialParameters,
    allocation_family: str,
    allocation_label: str,
    knowledge_mode: str,
    modulation_scheme: str,
    resource_grid_kwargs: dict[str, object],
) -> SweepPointSpec:
    _, allocation_summary = _build_resource_grid_and_summary(cfg, allocation_family, resource_grid_kwargs)
    return SweepPointSpec(
        point_index=point_index,
        sweep_name=sweep_name,
        parameter_name=parameter_name,
        parameter_label=parameter_label,
        parameter_value=parameter_value,
        parameter_numeric_value=parameter_numeric_value,
        anchor_name=cfg.anchor.name,
        scene_name=cfg.scene_class.name,
        profile_name=cfg.runtime_profile.name,
        trial_count=cfg.runtime_profile.n_trials,
        suite=cfg.sweep_suite,
        burst_profile_name=cfg.burst_profile.name,
        rx_columns=cfg.array_geometry.n_rx_cols,
        center_range_m=params.center_range_m,
        range_separation_m=params.range_separation_m,
        velocity_separation_mps=params.velocity_separation_mps,
        angle_separation_deg=params.angle_separation_deg,
        allocation_family=allocation_family,
        allocation_label=allocation_label,
        knowledge_mode=knowledge_mode,
        modulation_scheme=modulation_scheme,
        resource_grid_kwargs=dict(resource_grid_kwargs),
        occupied_fraction=allocation_summary.occupied_fraction,
        pilot_fraction=allocation_summary.pilot_fraction,
        fragmentation_index=allocation_summary.fragmentation_index,
        bandwidth_span_fraction=allocation_summary.contiguous_bandwidth_span_fraction,
        slow_time_span_fraction=allocation_summary.slow_time_span_fraction,
    )


def _nominal_allocation_settings() -> tuple[str, str, str, str, dict[str, object]]:
    return (
        "fragmented_prb",
        "Fragmented Scheduled PRB",
        "known_symbols",
        "qpsk",
        {
            "prb_size": 12,
            "n_prb_fragments": 4,
            "pilot_subcarrier_period": 4,
            "pilot_symbol_period": 4,
        },
    )


def _pilot_only_nominal_point_spec(cfg: StudyConfig) -> SweepPointSpec:
    params = nominal_trial_parameters(cfg)
    allocation_family, allocation_label, _knowledge_mode, modulation_scheme, resource_grid_kwargs = _nominal_allocation_settings()
    return _make_point_spec(
        cfg,
        point_index=0,
        sweep_name="pilot_only_nominal",
        parameter_name="pilot_only_nominal_point",
        parameter_label="Pilot-Only Nominal Point",
        parameter_value="pilot_only_nominal",
        parameter_numeric_value=None,
        params=params,
        allocation_family=allocation_family,
        allocation_label=f"{allocation_label} / Pilot-Only Knowledge",
        knowledge_mode="pilot_only",
        modulation_scheme=modulation_scheme,
        resource_grid_kwargs=resource_grid_kwargs,
    )


def _seed_spawn_key(seed_sequence: np.random.SeedSequence) -> str:
    if not seed_sequence.spawn_key:
        return "root"
    return ".".join(str(value) for value in seed_sequence.spawn_key)


def _serialize_truth_targets(trial_result: CommunicationsTrialResult) -> str:
    return "|".join(
        (
            f"{target_index}:{target.label}:{target.target_class_name}:"
            f"{target.range_m:.3f}:{target.velocity_mps:.3f}:{target.azimuth_deg:.3f}:{target.path_gain_linear:.6e}"
        )
        for target_index, target in enumerate(trial_result.masked_observation.snapshot.scenario.targets)
    )


def _serialize_detections(estimate: MethodEstimate) -> str:
    return "|".join(
        (
            f"{detection_index}:"
            f"{detection.range_m:.3f}:{detection.velocity_mps:.3f}:{detection.azimuth_deg:.3f}:{detection.score:.6e}"
        )
        for detection_index, detection in enumerate(estimate.detections)
    )


def _serialize_assignments(metric: MethodTrialMetrics) -> str:
    return "|".join(
        (
            f"{assignment.truth_index}->{assignment.detection_index}:"
            f"{assignment.range_error_m:.3f}:{assignment.velocity_error_mps:.3f}:{assignment.angle_error_deg:.3f}:"
            f"{int(assignment.within_detection_gate)}:{int(assignment.within_joint_resolution_gate)}"
        )
        for assignment in metric.assignments
    )


def _trial_row(
    cfg: StudyConfig,
    task: SweepPointSpec,
    *,
    trial_index: int,
    child_seed: np.random.SeedSequence,
    trial_result: CommunicationsTrialResult,
    method_name: str,
    method_label: str,
    estimate: MethodEstimate,
    metric: MethodTrialMetrics,
    estimator_family: str,
) -> dict[str, str]:
    scenario = trial_result.masked_observation.snapshot.scenario
    truth_targets = scenario.targets
    return {
        "evidence_profile": cfg.evidence_profile_name,
        "anchor": cfg.anchor.name,
        "anchor_label": cfg.anchor.label,
        "scene_class": cfg.scene_class.name,
        "scene_label": cfg.scene_class.label,
        "sweep_name": task.sweep_name,
        "parameter_name": task.parameter_name,
        "parameter_label": task.parameter_label,
        "parameter_value": task.parameter_value,
        "parameter_numeric_value": f"{task.parameter_numeric_value:.6f}" if task.parameter_numeric_value is not None else "",
        "allocation_family": task.allocation_family,
        "allocation_label": task.allocation_label,
        "knowledge_mode": task.knowledge_mode,
        "modulation_scheme": task.modulation_scheme,
        "estimator_family": estimator_family,
        "method": method_name,
        "method_label": method_label,
        "trial_index": f"{trial_index:d}",
        "trial_spawn_key": _seed_spawn_key(child_seed),
        "truth_target_count": f"{len(truth_targets):d}",
        "reported_target_count": f"{metric.reported_target_count:d}",
        "estimated_model_order": f"{metric.estimated_model_order:d}" if metric.estimated_model_order is not None else "",
        "matched_target_count": f"{metric.matched_target_count:d}",
        "joint_detection_success": f"{int(metric.joint_detection_success):d}",
        "joint_resolution_success": f"{int(metric.joint_resolution_success):d}",
        "range_resolution_success": f"{int(metric.range_resolution_success):d}",
        "velocity_resolution_success": f"{int(metric.velocity_resolution_success):d}",
        "angle_resolution_success": f"{int(metric.angle_resolution_success):d}",
        "false_alarm_count": f"{metric.false_alarm_count:d}",
        "miss_count": f"{metric.miss_count:d}",
        "any_false_alarm": f"{int(metric.any_false_alarm):d}",
        "any_miss": f"{int(metric.any_miss):d}",
        "unconditional_range_rmse_m": f"{metric.unconditional_range_rmse_m:.6f}",
        "unconditional_velocity_rmse_mps": f"{metric.unconditional_velocity_rmse_mps:.6f}",
        "unconditional_angle_rmse_deg": f"{metric.unconditional_angle_rmse_deg:.6f}",
        "unconditional_joint_assignment_rmse": f"{metric.unconditional_joint_assignment_rmse:.6f}",
        "conditional_range_rmse_m": f"{metric.conditional_range_rmse_m:.6f}" if metric.conditional_range_rmse_m is not None else "",
        "conditional_velocity_rmse_mps": (
            f"{metric.conditional_velocity_rmse_mps:.6f}" if metric.conditional_velocity_rmse_mps is not None else ""
        ),
        "conditional_angle_rmse_deg": f"{metric.conditional_angle_rmse_deg:.6f}" if metric.conditional_angle_rmse_deg is not None else "",
        "conditional_joint_assignment_rmse": (
            f"{metric.conditional_joint_assignment_rmse:.6f}"
            if metric.conditional_joint_assignment_rmse is not None
            else ""
        ),
        "frontend_runtime_s": f"{metric.frontend_runtime_s:.6f}",
        "incremental_runtime_s": f"{metric.incremental_runtime_s:.6f}",
        "total_runtime_s": f"{metric.total_runtime_s:.6f}",
        "realized_center_range_m": f"{scenario.trial_parameters.center_range_m:.6f}",
        "realized_range_separation_m": f"{scenario.trial_parameters.range_separation_m:.6f}",
        "realized_velocity_separation_mps": f"{scenario.trial_parameters.velocity_separation_mps:.6f}",
        "realized_angle_separation_deg": f"{scenario.trial_parameters.angle_separation_deg:.6f}",
        "center_range_offset_m": f"{scenario.trial_jitter.center_range_offset_m:.6f}",
        "range_separation_offset_m": f"{scenario.trial_jitter.range_separation_offset_m:.6f}",
        "velocity_separation_offset_mps": f"{scenario.trial_jitter.velocity_separation_offset_mps:.6f}",
        "angle_separation_offset_deg": f"{scenario.trial_jitter.angle_separation_offset_deg:.6f}",
        "source_mode": scenario.source_model.mode,
        "configured_target_coherence": f"{scenario.source_model.configured_target_coherence:.6f}",
        "empirical_target_coherence": f"{scenario.source_model.empirical_target_coherence:.6f}",
        "temporal_correlation": f"{scenario.source_model.temporal_correlation:.6f}",
        "truth_targets": _serialize_truth_targets(trial_result),
        "detections": _serialize_detections(estimate),
        "assignments": _serialize_assignments(metric),
    }


def _build_bandwidth_span_points(base_cfg: StudyConfig, params: TrialParameters, suite: str) -> list[SweepPointSpec]:
    width_fractions = _suite_values(suite, (0.25, 0.50, 0.75, 1.00), (0.20, 0.35, 0.50, 0.70, 0.85, 1.00))
    return [
        _make_point_spec(
            base_cfg,
            point_index=point_index,
            sweep_name="bandwidth_span",
            parameter_name="bandwidth_span_fraction",
            parameter_label="Bandwidth Span Fraction",
            parameter_value=f"{width_fraction:.3f}",
            parameter_numeric_value=width_fraction,
            params=params,
            allocation_family="block_pilot",
            allocation_label="Bandwidth-Limited Block",
            knowledge_mode="known_symbols",
            modulation_scheme="qpsk",
            resource_grid_kwargs={
                "block_width_subcarriers": max(12, int(round(width_fraction * base_cfg.n_subcarriers))),
                "block_symbol_span": base_cfg.burst_profile.n_snapshots,
                "n_frequency_blocks": 1,
            },
        )
        for point_index, width_fraction in enumerate(width_fractions)
    ]


def _build_slow_time_span_points(base_cfg: StudyConfig, params: TrialParameters, suite: str) -> list[SweepPointSpec]:
    span_fractions = _suite_values(suite, (0.25, 0.50, 0.75, 1.00), (0.20, 0.35, 0.50, 0.70, 0.85, 1.00))
    point_specs: list[SweepPointSpec] = []
    for point_index, span_fraction in enumerate(span_fractions):
        span_symbols = max(2, int(round(span_fraction * base_cfg.burst_profile.n_snapshots)))
        span_symbols = min(span_symbols, base_cfg.burst_profile.n_snapshots)
        point_specs.append(
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name="slow_time_span",
                parameter_name="slow_time_span_fraction",
                parameter_label="Slow-Time Span Fraction",
                parameter_value=f"{span_symbols / base_cfg.burst_profile.n_snapshots:.3f}",
                parameter_numeric_value=span_symbols / base_cfg.burst_profile.n_snapshots,
                params=params,
                allocation_family="fragmented_prb",
                allocation_label="Fragmented Scheduled PRB",
                knowledge_mode="known_symbols",
                modulation_scheme="qpsk",
                resource_grid_kwargs={
                    "prb_size": 12,
                    "n_prb_fragments": 4,
                    "pilot_subcarrier_period": 4,
                    "pilot_symbol_period": 4,
                    "active_symbol_indices": tuple(range(span_symbols)),
                },
            )
        )
    return point_specs


def _build_sweep_point_specs(
    anchor_name: str,
    scene_name: str,
    profile_name: str,
    trial_count: int,
    suite: str,
    sweep_name: str,
) -> list[SweepPointSpec]:
    base_cfg = build_study_config(
        anchor_name,
        scene_name,
        profile_name,
        trial_count_override=trial_count,
        suite=suite,
    )
    params = nominal_trial_parameters(base_cfg)

    if sweep_name == "allocation_family":
        families = (
            ("full_grid", "Radar-Full Grid", "known_symbols", "qpsk", {"full_grid_role": ResourceElementRole.PILOT}),
            ("comb_pilot", "Comb Pilot Grid", "known_symbols", "qpsk", {"pilot_subcarrier_period": 4, "pilot_symbol_period": 2}),
            ("block_pilot", "Block Pilot Grid", "known_symbols", "qpsk", {"block_width_subcarriers": 48, "block_symbol_span": 16, "n_frequency_blocks": 1}),
            ("fragmented_prb", "Fragmented Scheduled PRB", "known_symbols", "qpsk", {"prb_size": 12, "n_prb_fragments": 4, "pilot_subcarrier_period": 4, "pilot_symbol_period": 4}),
            ("punctured_grid", "Punctured Scheduled PRB", "known_symbols", "qpsk", {"puncture_fraction": 0.15, "puncture_base_family": "fragmented_prb", "pilot_subcarrier_period": 4, "pilot_symbol_period": 4, "prb_size": 12, "n_prb_fragments": 4}),
        )
        return [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="allocation_family",
                parameter_label="Allocation Family",
                parameter_value=allocation_label,
                parameter_numeric_value=None,
                params=params,
                allocation_family=allocation_family,
                allocation_label=allocation_label,
                knowledge_mode=knowledge_mode,
                modulation_scheme=modulation_scheme,
                resource_grid_kwargs=kwargs,
            )
            for point_index, (allocation_family, allocation_label, knowledge_mode, modulation_scheme, kwargs) in enumerate(families)
        ]

    nominal_family, nominal_label, nominal_knowledge, nominal_modulation, nominal_kwargs = _nominal_allocation_settings()

    if sweep_name == "occupied_fraction":
        fragment_counts = _suite_values(suite, (1, 2, 4, 6, 8), (1, 2, 3, 4, 5, 6, 8))
        points = [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="occupied_fraction",
                parameter_label="Occupied RE Fraction",
                parameter_value="",
                parameter_numeric_value=None,
                params=params,
                allocation_family="fragmented_prb",
                allocation_label="Fragmented Scheduled PRB",
                knowledge_mode="known_symbols",
                modulation_scheme="qpsk",
                resource_grid_kwargs={
                    "prb_size": 12,
                    "n_prb_fragments": int(fragment_count),
                    "pilot_subcarrier_period": 4,
                    "pilot_symbol_period": 4,
                },
            )
            for point_index, fragment_count in enumerate(fragment_counts)
        ]
        return [
            SweepPointSpec(
                **{
                    **point.__dict__,
                    "parameter_value": f"{point.occupied_fraction:.3f}",
                    "parameter_numeric_value": point.occupied_fraction,
                }
            )
            for point in sorted(points, key=lambda point: point.occupied_fraction)
        ]

    if sweep_name == "fragmentation":
        pattern_points = [
            ("block_pilot", "Low Fragmentation", {"block_width_subcarriers": 48, "block_symbol_span": 16, "n_frequency_blocks": 1}),
            ("fragmented_prb", "Medium Fragmentation", {"prb_size": 12, "n_prb_fragments": 4, "pilot_subcarrier_period": 4, "pilot_symbol_period": 4}),
            ("punctured_grid", "Punctured Fragmentation", {"puncture_fraction": 0.15, "puncture_base_family": "fragmented_prb", "prb_size": 12, "n_prb_fragments": 4, "pilot_subcarrier_period": 4, "pilot_symbol_period": 4}),
            ("comb_pilot", "High Fragmentation", {"pilot_subcarrier_period": 2, "pilot_symbol_period": 1}),
        ]
        points = [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="fragmentation_index",
                parameter_label="Fragmentation Index",
                parameter_value=allocation_label,
                parameter_numeric_value=None,
                params=params,
                allocation_family=allocation_family,
                allocation_label=allocation_label,
                knowledge_mode="known_symbols",
                modulation_scheme="qpsk",
                resource_grid_kwargs=kwargs,
            )
            for point_index, (allocation_family, allocation_label, kwargs) in enumerate(pattern_points)
        ]
        return [
            SweepPointSpec(
                **{
                    **point.__dict__,
                    "parameter_numeric_value": point.fragmentation_index,
                }
            )
            for point in sorted(points, key=lambda point: point.fragmentation_index)
        ]

    if sweep_name == "bandwidth_span":
        return _build_bandwidth_span_points(base_cfg, params, suite)

    if sweep_name == "slow_time_span":
        return _build_slow_time_span_points(base_cfg, params, suite)

    if sweep_name == "range_separation":
        multipliers = _suite_values(suite, (0.60, 0.85, 1.00, 1.20, 1.50), (0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80))
        return [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="range_separation_m",
                parameter_label="Range Separation (m)",
                parameter_value=f"{multiplier * base_cfg.range_resolution_m:.3f}",
                parameter_numeric_value=multiplier * base_cfg.range_resolution_m,
                params=_axis_isolated_trial_parameters(
                    base_cfg,
                    range_separation_m=multiplier * base_cfg.range_resolution_m,
                ),
                allocation_family=nominal_family,
                allocation_label=nominal_label,
                knowledge_mode=nominal_knowledge,
                modulation_scheme=nominal_modulation,
                resource_grid_kwargs=nominal_kwargs,
            )
            for point_index, multiplier in enumerate(multipliers)
        ]

    if sweep_name == "velocity_separation":
        multipliers = _suite_values(suite, (0.50, 0.75, 1.00, 1.25, 1.50), (0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 1.80))
        return [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="velocity_separation_mps",
                parameter_label="Velocity Separation (m/s)",
                parameter_value=f"{multiplier * base_cfg.velocity_resolution_mps:.3f}",
                parameter_numeric_value=multiplier * base_cfg.velocity_resolution_mps,
                params=_axis_isolated_trial_parameters(
                    base_cfg,
                    velocity_separation_mps=multiplier * base_cfg.velocity_resolution_mps,
                ),
                allocation_family=nominal_family,
                allocation_label=nominal_label,
                knowledge_mode=nominal_knowledge,
                modulation_scheme=nominal_modulation,
                resource_grid_kwargs=nominal_kwargs,
            )
            for point_index, multiplier in enumerate(multipliers)
        ]

    if sweep_name == "angle_separation":
        multipliers = _suite_values(suite, (0.60, 0.85, 1.00, 1.20, 1.50), (0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80))
        return [
            _make_point_spec(
                base_cfg,
                point_index=point_index,
                sweep_name=sweep_name,
                parameter_name="angle_separation_deg",
                parameter_label="Angle Separation (deg)",
                parameter_value=f"{multiplier * base_cfg.azimuth_resolution_deg:.3f}",
                parameter_numeric_value=multiplier * base_cfg.azimuth_resolution_deg,
                params=_axis_isolated_trial_parameters(
                    base_cfg,
                    angle_separation_deg=multiplier * base_cfg.azimuth_resolution_deg,
                ),
                allocation_family=nominal_family,
                allocation_label=nominal_label,
                knowledge_mode=nominal_knowledge,
                modulation_scheme=nominal_modulation,
                resource_grid_kwargs=nominal_kwargs,
            )
            for point_index, multiplier in enumerate(multipliers)
        ]

    raise ValueError(f"Unsupported sweep_name: {sweep_name}")


def simulate_communications_trial(
    cfg: StudyConfig,
    params: TrialParameters,
    allocation_family: str,
    allocation_label: str,
    knowledge_mode: str,
    modulation_scheme: str,
    resource_grid_kwargs: dict[str, object],
    rng: np.random.Generator,
    *,
    include_fbss_ablation: bool = False,
) -> CommunicationsTrialResult:
    """Run one communications-scheduled trial end to end."""

    resource_grid, allocation_summary = _build_resource_grid_and_summary(cfg, allocation_family, resource_grid_kwargs)
    masked_observation = simulate_masked_observation(
        cfg,
        params,
        resource_grid,
        rng=rng,
        modulation_scheme=modulation_scheme,
        knowledge_mode=knowledge_mode,
    )
    validate_targets_within_search_bounds(masked_observation.snapshot.scenario.targets, config_search_bounds(cfg))
    frontend = prepare_masked_frontend(cfg, masked_observation, embedding_mode="weighted")
    fbss_ablation_estimates: dict[str, MethodEstimate] | None = None
    if include_fbss_ablation:
        estimates, fbss_ablation_estimates = run_masked_estimators_with_fbss_ablation(cfg, masked_observation, frontend)
    else:
        estimates = run_masked_estimators(cfg, masked_observation, frontend)
    metrics = {
        method_name: evaluate_trial(
            cfg,
            truth_targets=masked_observation.snapshot.scenario.targets,
            detections=estimate.detections,
            reported_target_count=estimate.reported_target_count,
            estimated_model_order=estimate.estimated_model_order,
            frontend_runtime_s=estimate.frontend_runtime_s,
            incremental_runtime_s=estimate.incremental_runtime_s,
            total_runtime_s=estimate.total_runtime_s,
        )
        for method_name, estimate in estimates.items()
    }
    fbss_ablation_metrics = (
        {
            method_name: evaluate_trial(
                cfg,
                truth_targets=masked_observation.snapshot.scenario.targets,
                detections=estimate.detections,
                reported_target_count=estimate.reported_target_count,
                estimated_model_order=estimate.estimated_model_order,
                frontend_runtime_s=estimate.frontend_runtime_s,
                incremental_runtime_s=estimate.incremental_runtime_s,
                total_runtime_s=estimate.total_runtime_s,
            )
            for method_name, estimate in fbss_ablation_estimates.items()
        }
        if fbss_ablation_estimates is not None
        else None
    )
    return CommunicationsTrialResult(
        masked_observation=masked_observation,
        allocation_summary=allocation_summary,
        fft_cube=frontend.fft_cube,
        estimates=estimates,
        metrics=metrics,
        fbss_ablation_estimates=fbss_ablation_estimates,
        fbss_ablation_metrics=fbss_ablation_metrics,
    )


def _trial_rng(seed_sequence: np.random.SeedSequence) -> np.random.Generator:
    return np.random.default_rng(seed_sequence)


def _evaluate_point_task(task: SweepPointSpec) -> SweepPointResult:
    cfg = build_study_config(
        task.anchor_name,
        task.scene_name,
        task.profile_name,
        burst_profile_name=task.burst_profile_name,
        rx_columns=task.rx_columns,
        suite=task.suite,
        trial_count_override=task.trial_count,
    )
    params = TrialParameters(
        center_range_m=task.center_range_m,
        range_separation_m=task.range_separation_m,
        velocity_separation_mps=task.velocity_separation_mps,
        angle_separation_deg=task.angle_separation_deg,
    )
    seed_sequence = np.random.SeedSequence(
        [
            cfg.rng_seed,
            task.point_index,
            int(round(1_000.0 * task.occupied_fraction)),
            int(round(1_000.0 * task.fragmentation_index)),
            int(round(1_000.0 * task.bandwidth_span_fraction)),
            int(round(1_000.0 * task.slow_time_span_fraction)),
        ]
    )
    trial_metrics: dict[str, list[MethodTrialMetrics]] = {method_name: [] for method_name in METHOD_ORDER}
    include_fbss_ablation = task.sweep_name in FBSS_ABLATION_SWEEP_NAMES
    fbss_ablation_trial_metrics = (
        {method_name: [] for method_name in FBSS_ABLATION_ORDER}
        if include_fbss_ablation
        else None
    )
    trial_rows: list[dict[str, str]] = []
    for trial_index, child_seed in enumerate(seed_sequence.spawn(cfg.runtime_profile.n_trials)):
        trial_result = simulate_communications_trial(
            cfg,
            params,
            task.allocation_family,
            task.allocation_label,
            task.knowledge_mode,
            task.modulation_scheme,
            task.resource_grid_kwargs,
            _trial_rng(child_seed),
            include_fbss_ablation=include_fbss_ablation,
        )
        for method_name in METHOD_ORDER:
            trial_metrics[method_name].append(trial_result.metrics[method_name])
            trial_rows.append(
                _trial_row(
                    cfg,
                    task,
                    trial_index=trial_index,
                    child_seed=child_seed,
                    trial_result=trial_result,
                    method_name=method_name,
                    method_label=METHOD_LABELS[method_name],
                    estimate=trial_result.estimates[method_name],
                    metric=trial_result.metrics[method_name],
                    estimator_family="headline",
                )
            )
        if fbss_ablation_trial_metrics is not None and trial_result.fbss_ablation_metrics is not None:
            for method_name in FBSS_ABLATION_ORDER:
                fbss_ablation_trial_metrics[method_name].append(trial_result.fbss_ablation_metrics[method_name])
                trial_rows.append(
                    _trial_row(
                        cfg,
                        task,
                        trial_index=trial_index,
                        child_seed=child_seed,
                        trial_result=trial_result,
                        method_name=method_name,
                        method_label=FBSS_ABLATION_LABELS[method_name],
                        estimate=trial_result.fbss_ablation_estimates[method_name],
                        metric=trial_result.fbss_ablation_metrics[method_name],
                        estimator_family="fbss_ablation",
                    )
                )

    return SweepPointResult(
        sweep_name=task.sweep_name,
        parameter_name=task.parameter_name,
        parameter_label=task.parameter_label,
        parameter_value=task.parameter_value,
        parameter_numeric_value=task.parameter_numeric_value,
        anchor_name=cfg.anchor.name,
        anchor_label=cfg.anchor.label,
        scene_class_name=cfg.scene_class.name,
        scene_label=cfg.scene_class.label,
        burst_profile_name=cfg.burst_profile.name,
        burst_profile_label=cfg.burst_profile.label,
        aperture_size=cfg.array_geometry.n_rx_cols,
        target_pair="-".join(cfg.scene_class.target_pair).upper(),
        allocation_family=task.allocation_family,
        allocation_label=task.allocation_label,
        knowledge_mode=task.knowledge_mode,
        modulation_scheme=task.modulation_scheme,
        occupied_fraction=task.occupied_fraction,
        pilot_fraction=task.pilot_fraction,
        fragmentation_index=task.fragmentation_index,
        bandwidth_span_fraction=task.bandwidth_span_fraction,
        slow_time_span_fraction=task.slow_time_span_fraction,
        method_summaries={
            method_name: summarize_method_metrics(trial_metrics[method_name], cfg.expected_target_count)
            for method_name in METHOD_ORDER
        },
        fbss_ablation_summaries=(
            {
                method_name: summarize_method_metrics(fbss_ablation_trial_metrics[method_name], cfg.expected_target_count)
                for method_name in FBSS_ABLATION_ORDER
            }
            if fbss_ablation_trial_metrics is not None
            else None
        ),
        trial_rows=tuple(trial_rows),
    )


def _run_sweep(
    cfg: StudyConfig,
    sweep_name: str,
    *,
    show_progress: bool,
    max_workers: int,
) -> SweepResult:
    specs = _build_sweep_point_specs(
        cfg.anchor.name,
        cfg.scene_class.name,
        cfg.runtime_profile.name,
        cfg.runtime_profile.n_trials,
        cfg.sweep_suite,
        sweep_name,
    )
    if max_workers == 1:
        points = [_evaluate_point_task(spec) for spec in specs]
    else:
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            points = list(executor.map(_evaluate_point_task, specs))
    points.sort(key=lambda point: (point.parameter_numeric_value is None, point.parameter_numeric_value, point.parameter_value))
    if show_progress:
        print(f"[{cfg.anchor.name}/{cfg.scene_class.name}] completed {sweep_name} ({len(points)} points)")
    return SweepResult(
        sweep_name=sweep_name,
        parameter_name=points[0].parameter_name,
        parameter_label=points[0].parameter_label,
        anchor_name=cfg.anchor.name,
        anchor_label=cfg.anchor.label,
        scene_class_name=cfg.scene_class.name,
        scene_label=cfg.scene_class.label,
        points=points,
    )


def _nominal_point_spec(cfg: StudyConfig) -> SweepPointSpec:
    params = nominal_trial_parameters(cfg)
    allocation_family, allocation_label, knowledge_mode, modulation_scheme, resource_grid_kwargs = _nominal_allocation_settings()
    return _make_point_spec(
        cfg,
        point_index=0,
        sweep_name="nominal",
        parameter_name="nominal_point",
        parameter_label="Nominal Point",
        parameter_value="nominal",
        parameter_numeric_value=None,
        params=params,
        allocation_family=allocation_family,
        allocation_label=allocation_label,
        knowledge_mode=knowledge_mode,
        modulation_scheme=modulation_scheme,
        resource_grid_kwargs=resource_grid_kwargs,
    )


def run_communications_study(
    cfg: StudyConfig,
    *,
    show_progress: bool = False,
    max_workers: int | None = None,
    suite: str | None = None,
    sweep_names: tuple[str, ...] | None = None,
) -> CommunicationsStudyResult:
    """Run the allocation-driven communications-scheduled study."""

    if suite is None:
        suite = cfg.sweep_suite
    if max_workers is None:
        max_workers = _default_max_workers()
    selected_sweeps = sweep_names or (SUBMISSION_SWEEP_NAMES if cfg.runtime_profile.name == "submission" else PUBLIC_SWEEP_NAMES)

    study_cfg = build_study_config(
        cfg.anchor.name,
        cfg.scene_class.name,
        cfg.runtime_profile.name,
        burst_profile_name=cfg.burst_profile.name,
        rx_columns=cfg.array_geometry.n_rx_cols,
        suite=suite,
        trial_count_override=cfg.runtime_profile.n_trials,
    )

    sweeps = [
        _run_sweep(
            study_cfg,
            sweep_name,
            show_progress=show_progress,
            max_workers=max_workers,
        )
        for sweep_name in selected_sweeps
    ]
    nominal_spec = _nominal_point_spec(study_cfg)
    nominal_point = _evaluate_point_task(nominal_spec)
    pilot_only_nominal_point = _evaluate_point_task(_pilot_only_nominal_point_spec(study_cfg))
    representative_trial = simulate_communications_trial(
        study_cfg,
        nominal_trial_parameters(study_cfg),
        nominal_spec.allocation_family,
        nominal_spec.allocation_label,
        nominal_spec.knowledge_mode,
        nominal_spec.modulation_scheme,
        nominal_spec.resource_grid_kwargs,
        np.random.default_rng(study_cfg.rng_seed),
        include_fbss_ablation=True,
    )

    return CommunicationsStudyResult(
        config=study_cfg,
        evidence_profile_name=study_cfg.evidence_profile_name,
        anchor_name=study_cfg.anchor.name,
        anchor_label=study_cfg.anchor.label,
        scene_class_name=study_cfg.scene_class.name,
        scene_label=study_cfg.scene_class.label,
        sweeps=sweeps,
        nominal_point=nominal_point,
        pilot_only_nominal_point=pilot_only_nominal_point,
        representative_trial=representative_trial,
    )
