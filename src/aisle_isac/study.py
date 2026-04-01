"""Study execution for the private-5G angle-range-Doppler simulator."""

from __future__ import annotations

import concurrent.futures as futures
from dataclasses import dataclass, replace
import os
import time

import numpy as np

from aisle_isac.channel_models import CubeSnapshot, TrialParameters, build_truth_targets, simulate_radar_cube
from aisle_isac.config import StudyConfig
from aisle_isac.estimators import (
    FftCubeResult,
    MethodEstimate,
    config_search_bounds,
    prepare_frontend,
    run_estimators,
    validate_targets_within_search_bounds,
)
from aisle_isac.metrics import MethodPointSummary, MethodTrialMetrics, evaluate_trial, summarize_method_metrics
from aisle_isac.scenarios import build_study_config


METHOD_ORDER = ("fft", "music", "fbss", "music_full")
METHOD_LABELS = {
    "fft": "Angle-Range-Doppler FFT",
    "music": "FFT-Seeded Staged Azimuth MUSIC",
    "fbss": "FFT-Seeded Azimuth MUSIC + FBSS",
    "music_full": "Full-Search MUSIC",
}


@dataclass(frozen=True)
class TrialResult:
    """One simulated trial, including cube, detections, and metrics."""

    snapshot: CubeSnapshot
    fft_cube: FftCubeResult
    estimates: dict[str, MethodEstimate]
    metrics: dict[str, MethodTrialMetrics]


@dataclass(frozen=True)
class SweepPointSpec:
    """Serializable description of one sweep point."""

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
    n_subcarriers_override: int | None = None


@dataclass(frozen=True)
class SweepPointResult:
    """Aggregated summary for one public sweep point."""

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
    method_summaries: dict[str, MethodPointSummary]


@dataclass(frozen=True)
class SweepResult:
    """All points for one sweep family in one anchor/scene pair."""

    sweep_name: str
    parameter_name: str
    parameter_label: str
    anchor_name: str
    anchor_label: str
    scene_class_name: str
    scene_label: str
    points: list[SweepPointResult]


@dataclass(frozen=True)
class SceneStudyResult:
    """Complete study bundle for one anchor and scene class."""

    anchor_name: str
    anchor_label: str
    scene_class_name: str
    scene_label: str
    sweeps: list[SweepResult]
    nominal_point: SweepPointResult
    representative_trial: TrialResult


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def _suite_values(suite: str, coarse: tuple[float, ...], dense: tuple[float, ...]) -> tuple[float, ...]:
    return dense if suite == "full" else coarse


def nominal_trial_parameters(cfg: StudyConfig) -> TrialParameters:
    """Return the deployment-facing nominal geometry for the active config."""

    return TrialParameters(
        center_range_m=cfg.scene_class.nominal_range_m,
        range_separation_m=cfg.scene_class.default_range_separation_cells * cfg.range_resolution_m,
        velocity_separation_mps=cfg.scene_class.default_velocity_separation_cells * cfg.velocity_resolution_mps,
        angle_separation_deg=cfg.scene_class.default_angle_separation_cells * cfg.azimuth_resolution_deg,
    )


def _build_sweep_point_specs(
    anchor_name: str,
    scene_name: str,
    profile_name: str,
    trial_count: int,
    suite: str,
    sweep_name: str,
) -> list[SweepPointSpec]:
    """Build point specifications for one public sweep family."""

    point_specs: list[SweepPointSpec] = []
    if sweep_name == "range_separation":
        multipliers = _suite_values(suite, (0.60, 0.85, 1.00, 1.20, 1.50), (0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80))
        for point_index, multiplier in enumerate(multipliers):
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="range_separation_m",
                    parameter_label="Range Separation (m)",
                    parameter_value=f"{multiplier * cfg.range_resolution_m:.3f}",
                    parameter_numeric_value=multiplier * cfg.range_resolution_m,
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=8,
                    center_range_m=params.center_range_m,
                    range_separation_m=multiplier * cfg.range_resolution_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=params.angle_separation_deg,
                )
            )
        return point_specs

    if sweep_name == "velocity_separation":
        multipliers = _suite_values(suite, (0.50, 0.75, 1.00, 1.25, 1.50), (0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 1.80))
        for point_index, multiplier in enumerate(multipliers):
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="velocity_separation_mps",
                    parameter_label="Velocity Separation (m/s)",
                    parameter_value=f"{multiplier * cfg.velocity_resolution_mps:.3f}",
                    parameter_numeric_value=multiplier * cfg.velocity_resolution_mps,
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=8,
                    center_range_m=params.center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=multiplier * cfg.velocity_resolution_mps,
                    angle_separation_deg=params.angle_separation_deg,
                )
            )
        return point_specs

    if sweep_name == "angle_separation":
        multipliers = _suite_values(suite, (0.60, 0.85, 1.00, 1.20, 1.50), (0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80))
        for point_index, multiplier in enumerate(multipliers):
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="angle_separation_deg",
                    parameter_label="Angle Separation (deg)",
                    parameter_value=f"{multiplier * cfg.azimuth_resolution_deg:.3f}",
                    parameter_numeric_value=multiplier * cfg.azimuth_resolution_deg,
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=8,
                    center_range_m=params.center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=multiplier * cfg.azimuth_resolution_deg,
                )
            )
        return point_specs

    if sweep_name == "absolute_range":
        absolute_ranges = _suite_values(suite, (12.0, 18.0, 24.0, 30.0), (10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0))
        for point_index, center_range_m in enumerate(absolute_ranges):
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="absolute_range_m",
                    parameter_label="Absolute Target Range (m)",
                    parameter_value=f"{center_range_m:.3f}",
                    parameter_numeric_value=center_range_m,
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=8,
                    center_range_m=center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=params.angle_separation_deg,
                )
            )
        return point_specs

    if sweep_name == "burst_profile":
        burst_profiles = ("short_cpi", "balanced_cpi", "long_cpi")
        for point_index, burst_profile_name in enumerate(burst_profiles):
            cfg = build_study_config(anchor_name, scene_name, profile_name, burst_profile_name, 8, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="burst_profile",
                    parameter_label="Burst Profile",
                    parameter_value=burst_profile_name,
                    parameter_numeric_value=None,
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name=burst_profile_name,
                    rx_columns=8,
                    center_range_m=params.center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=params.angle_separation_deg,
                )
            )
        return point_specs

    if sweep_name == "aperture":
        apertures = (8, 12, 16, 24)
        for point_index, rx_columns in enumerate(apertures):
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", rx_columns, suite)
            params = nominal_trial_parameters(cfg)
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="aperture_size",
                    parameter_label="Horizontal Rx Aperture",
                    parameter_value=str(rx_columns),
                    parameter_numeric_value=float(rx_columns),
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=rx_columns,
                    center_range_m=params.center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=params.angle_separation_deg,
                )
            )
        return point_specs

    if sweep_name == "resource_fraction":
        subcarrier_counts = _suite_values(
            suite,
            (32, 64, 96, 192, 384),
            (24, 48, 96, 192, 384, 768),
        )
        for point_index, n_sc in enumerate(subcarrier_counts):
            n_sc = int(n_sc)
            cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
            params = nominal_trial_parameters(cfg)
            physical_count = cfg.anchor.physical_subcarrier_count
            sensing_fraction = n_sc / physical_count
            point_specs.append(
                SweepPointSpec(
                    point_index=point_index,
                    sweep_name=sweep_name,
                    parameter_name="sensing_subcarriers",
                    parameter_label="Sensing Subcarriers",
                    parameter_value=f"{n_sc:d}",
                    parameter_numeric_value=float(n_sc),
                    anchor_name=anchor_name,
                    scene_name=scene_name,
                    profile_name=profile_name,
                    trial_count=trial_count,
                    suite=suite,
                    burst_profile_name="balanced_cpi",
                    rx_columns=8,
                    center_range_m=params.center_range_m,
                    range_separation_m=params.range_separation_m,
                    velocity_separation_mps=params.velocity_separation_mps,
                    angle_separation_deg=params.angle_separation_deg,
                    n_subcarriers_override=n_sc,
                )
            )
        return point_specs

    raise ValueError(f"Unsupported sweep_name: {sweep_name}")


def simulate_trial(
    cfg: StudyConfig,
    params: TrialParameters,
    rng: np.random.Generator,
) -> TrialResult:
    """Run one radar-only trial end to end."""

    snapshot = simulate_radar_cube(cfg, params, rng)
    search_bounds = config_search_bounds(cfg)
    validate_targets_within_search_bounds(snapshot.scenario.targets, search_bounds)
    frontend = prepare_frontend(cfg, snapshot.horizontal_cube)
    estimates = run_estimators(
        cfg,
        snapshot.horizontal_cube,
        fft_cube=frontend.fft_cube,
        coarse_candidates=frontend.coarse_candidates,
        frontend_runtime_s=frontend.frontend_runtime_s,
    )
    metrics = {
        method_name: evaluate_trial(
            cfg,
            truth_targets=snapshot.scenario.targets,
            detections=estimate.detections,
            estimated_model_order=estimate.estimated_model_order,
            noise_variance=snapshot.noise_variance,
            frontend_runtime_s=estimate.frontend_runtime_s,
            incremental_runtime_s=estimate.incremental_runtime_s,
            total_runtime_s=estimate.total_runtime_s,
        )
        for method_name, estimate in estimates.items()
    }
    return TrialResult(
        snapshot=snapshot,
        fft_cube=frontend.fft_cube,
        estimates=estimates,
        metrics=metrics,
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
    if task.n_subcarriers_override is not None:
        cfg = replace(cfg, runtime_profile=replace(
            cfg.runtime_profile, n_simulated_subcarriers=task.n_subcarriers_override,
        ))
    params = TrialParameters(
        center_range_m=task.center_range_m,
        range_separation_m=task.range_separation_m,
        velocity_separation_mps=task.velocity_separation_mps,
        angle_separation_deg=task.angle_separation_deg,
    )
    validate_targets_within_search_bounds(build_truth_targets(cfg, params), config_search_bounds(cfg))
    seed_sequence = np.random.SeedSequence([cfg.rng_seed, task.point_index, len(task.parameter_value), task.rx_columns])
    trial_metrics: dict[str, list[MethodTrialMetrics]] = {method_name: [] for method_name in METHOD_ORDER}
    for child_seed in seed_sequence.spawn(cfg.runtime_profile.n_trials):
        trial_result = simulate_trial(cfg, params, _trial_rng(child_seed))
        for method_name in METHOD_ORDER:
            trial_metrics[method_name].append(trial_result.metrics[method_name])

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
        method_summaries={
            method_name: summarize_method_metrics(trial_metrics[method_name], cfg.expected_target_count)
            for method_name in METHOD_ORDER
        },
    )


def _run_sweep(
    anchor_name: str,
    scene_name: str,
    profile_name: str,
    trial_count: int,
    suite: str,
    sweep_name: str,
    show_progress: bool,
    max_workers: int,
) -> SweepResult:
    tasks = _build_sweep_point_specs(anchor_name, scene_name, profile_name, trial_count, suite, sweep_name)
    if max_workers <= 1:
        results = [_evaluate_point_task(task) for task in tasks]
    else:
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_evaluate_point_task, tasks))

    if show_progress:
        print(f"[{anchor_name}/{scene_name}] completed {sweep_name} ({len(results)} points)")

    first = results[0]
    ordered_results = sorted(
        results,
        key=lambda point: (
            point.parameter_numeric_value is None,
            point.parameter_numeric_value if point.parameter_numeric_value is not None else point.parameter_value,
        ),
    )
    return SweepResult(
        sweep_name=sweep_name,
        parameter_name=first.parameter_name,
        parameter_label=first.parameter_label,
        anchor_name=first.anchor_name,
        anchor_label=first.anchor_label,
        scene_class_name=first.scene_class_name,
        scene_label=first.scene_label,
        points=ordered_results,
    )


def _representative_trial(anchor_name: str, scene_name: str, profile_name: str, suite: str) -> TrialResult:
    cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite)
    params = nominal_trial_parameters(cfg)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.rng_seed, 99_999]))
    return simulate_trial(cfg, params, rng)


def _nominal_point(anchor_name: str, scene_name: str, profile_name: str, trial_count: int, suite: str) -> SweepPointResult:
    cfg = build_study_config(anchor_name, scene_name, profile_name, "balanced_cpi", 8, suite, trial_count_override=trial_count)
    params = nominal_trial_parameters(cfg)
    task = SweepPointSpec(
        point_index=90_000,
        sweep_name="nominal_point",
        parameter_name="nominal_point",
        parameter_label="Nominal Point",
        parameter_value="nominal",
        parameter_numeric_value=None,
        anchor_name=anchor_name,
        scene_name=scene_name,
        profile_name=profile_name,
        trial_count=cfg.runtime_profile.n_trials,
        suite=suite,
        burst_profile_name="balanced_cpi",
        rx_columns=8,
        center_range_m=params.center_range_m,
        range_separation_m=params.range_separation_m,
        velocity_separation_mps=params.velocity_separation_mps,
        angle_separation_deg=params.angle_separation_deg,
    )
    return _evaluate_point_task(task)


def run_study(
    cfg: StudyConfig,
    show_progress: bool = False,
    max_workers: int | None = None,
    suite: str | None = None,
) -> SceneStudyResult:
    """Run all public sweeps for one anchor and scene class."""

    suite = suite or cfg.sweep_suite
    max_workers = max_workers if max_workers is not None else _default_max_workers()
    sweep_names = (
        "range_separation",
        "velocity_separation",
        "angle_separation",
        "absolute_range",
        "burst_profile",
        "aperture",
        "resource_fraction",
    )
    sweeps = [
        _run_sweep(
            cfg.anchor.name,
            cfg.scene_class.name,
            cfg.runtime_profile.name,
            cfg.runtime_profile.n_trials,
            suite,
            sweep_name,
            show_progress=show_progress,
            max_workers=max_workers,
        )
        for sweep_name in sweep_names
    ]
    return SceneStudyResult(
        anchor_name=cfg.anchor.name,
        anchor_label=cfg.anchor.label,
        scene_class_name=cfg.scene_class.name,
        scene_label=cfg.scene_class.label,
        sweeps=sweeps,
        nominal_point=_nominal_point(
            cfg.anchor.name,
            cfg.scene_class.name,
            cfg.runtime_profile.name,
            cfg.runtime_profile.n_trials,
            suite,
        ),
        representative_trial=_representative_trial(cfg.anchor.name, cfg.scene_class.name, cfg.runtime_profile.name, suite),
    )
