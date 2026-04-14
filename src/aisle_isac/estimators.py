"""Shared FFT and MUSIC utilities for communications-limited OFDM ISAC."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import maximum_filter

from aisle_isac.channel_models import TargetState
from aisle_isac.config import StudyConfig
from aisle_isac.ofdm import (
    azimuth_steering_matrix,
    doppler_steering_matrix,
    fft_azimuth_axis_deg,
    fft_range_axis_m,
    fft_velocity_axis_mps,
    range_steering_matrix,
    sparse_unambiguous_range_m,
)

FBSS_CONTIGUOUS_SUPPORT_FRACTION = 0.5


@dataclass(frozen=True)
class Detection:
    """One detected target hypothesis."""

    range_m: float
    velocity_mps: float
    azimuth_deg: float
    score: float


@dataclass(frozen=True)
class FftCubeResult:
    """FFT cube and its public axes."""

    power_cube: np.ndarray
    azimuth_axis_deg: np.ndarray
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    embedding_mode: str = "legacy"
    support_energy: float = 1.0
    full_support_energy: float = 1.0
    normalization_gain: float = 1.0
    known_fraction: float = 1.0


@dataclass(frozen=True)
class SearchBounds:
    """Search bounds implied by the FFT axes."""

    range_min_m: float
    range_max_m: float
    velocity_min_mps: float
    velocity_max_mps: float
    azimuth_min_deg: float
    azimuth_max_deg: float


@dataclass(frozen=True)
class FrontendArtifacts:
    """Shared FFT front-end used by every method."""

    fft_cube: FftCubeResult
    coarse_candidates: tuple[Detection, ...]
    search_bounds: SearchBounds
    frontend_runtime_s: float


@dataclass(frozen=True)
class MethodEstimate:
    """Output of one estimator family."""

    label: str
    detections: tuple[Detection, ...]
    reported_target_count: int
    estimated_model_order: int | None
    frontend_runtime_s: float
    incremental_runtime_s: float
    total_runtime_s: float
    stage_diagnostics: dict[str, str] | None = None


def covariance_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """Return the sample covariance for a sensors x snapshots matrix."""

    data_matrix = np.asarray(data_matrix, dtype=np.complex128)
    snapshot_count = max(1, data_matrix.shape[1])
    return np.einsum("ik,jk->ij", data_matrix, data_matrix.conj(), optimize=True) / snapshot_count


def _serialize_detection_sequence(detections: tuple[Detection, ...] | list[Detection]) -> str:
    return "|".join(
        (
            f"{detection_index}:"
            f"{detection.range_m:.3f}:{detection.velocity_mps:.3f}:{detection.azimuth_deg:.3f}:{detection.score:.6e}"
        )
        for detection_index, detection in enumerate(detections)
    )


def _fbss_subarray_len(n_sensors: int, fraction: float) -> int | None:
    if fraction <= 0.0 or n_sensors < 3:
        return None
    return max(3, min(n_sensors, int(round(fraction * n_sensors))))


def _longest_contiguous_run(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return None

    best_start = current_start = int(indices[0])
    best_stop = current_stop = int(indices[0]) + 1
    for index in indices[1:].tolist():
        if index == current_stop:
            current_stop += 1
            continue
        if current_stop - current_start > best_stop - best_start:
            best_start, best_stop = current_start, current_stop
        current_start = int(index)
        current_stop = int(index) + 1
    if current_stop - current_start > best_stop - best_start:
        best_start, best_stop = current_start, current_stop
    return best_start, best_stop


def fbss_covariance(data_matrix: np.ndarray, subarray_len: int) -> np.ndarray:
    """Forward-backward spatial smoothing for a uniform horizontal aperture."""

    n_sensors, snapshot_count = data_matrix.shape
    if not 1 < subarray_len <= n_sensors:
        raise ValueError("subarray_len must lie in (1, n_sensors]")
    n_subarrays = n_sensors - subarray_len + 1
    exchange_matrix = np.fliplr(np.eye(subarray_len, dtype=np.complex128))
    forward = np.zeros((subarray_len, subarray_len), dtype=np.complex128)
    for start_idx in range(n_subarrays):
        subarray = data_matrix[start_idx : start_idx + subarray_len]
        forward += np.einsum("ik,jk->ij", subarray, subarray.conj(), optimize=True) / snapshot_count
    forward /= n_subarrays
    backward = exchange_matrix @ forward.conj() @ exchange_matrix
    return 0.5 * (forward + backward)


def _range_music_covariance(
    cfg: StudyConfig,
    beamformed: np.ndarray,
    known_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if known_mask is None:
        return covariance_matrix(beamformed), cfg.frequencies_hz, False

    active_symbol_indices = np.flatnonzero(np.any(known_mask, axis=0))
    if active_symbol_indices.size < 2:
        return covariance_matrix(beamformed), cfg.frequencies_hz, False

    common_frequency_mask = np.all(known_mask[:, active_symbol_indices], axis=1)
    contiguous_run = _longest_contiguous_run(common_frequency_mask)
    if contiguous_run is None:
        return covariance_matrix(beamformed), cfg.frequencies_hz, False

    run_start, run_stop = contiguous_run
    run_len = run_stop - run_start
    common_frequency_count = int(np.count_nonzero(common_frequency_mask))
    subarray_len = _fbss_subarray_len(run_len, cfg.music_range_fbss_fraction)
    if (
        subarray_len is None
        or common_frequency_count <= 0
        or run_len / common_frequency_count < FBSS_CONTIGUOUS_SUPPORT_FRACTION
    ):
        return covariance_matrix(beamformed), cfg.frequencies_hz, False

    stable_symbol_mask = np.all(known_mask[run_start:run_stop, :], axis=0)
    stable_symbol_indices = np.flatnonzero(stable_symbol_mask)
    if stable_symbol_indices.size < 2:
        return covariance_matrix(beamformed), cfg.frequencies_hz, False

    support_matrix = beamformed[run_start:run_stop][:, stable_symbol_indices]
    return fbss_covariance(support_matrix, subarray_len), cfg.frequencies_hz[run_start : run_start + subarray_len], True


def _doppler_music_covariance(
    cfg: StudyConfig,
    doppler_signal: np.ndarray,
    known_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if known_mask is None:
        return covariance_matrix(doppler_signal), cfg.snapshot_times_s, False

    active_frequency_indices = np.flatnonzero(np.any(known_mask, axis=1))
    if active_frequency_indices.size < 2:
        return covariance_matrix(doppler_signal), cfg.snapshot_times_s, False

    common_symbol_mask = np.all(known_mask[active_frequency_indices, :], axis=0)
    contiguous_run = _longest_contiguous_run(common_symbol_mask)
    if contiguous_run is None:
        return covariance_matrix(doppler_signal), cfg.snapshot_times_s, False

    run_start, run_stop = contiguous_run
    run_len = run_stop - run_start
    common_symbol_count = int(np.count_nonzero(common_symbol_mask))
    subarray_len = _fbss_subarray_len(run_len, cfg.music_doppler_fbss_fraction)
    if (
        subarray_len is None
        or common_symbol_count <= 0
        or run_len / common_symbol_count < FBSS_CONTIGUOUS_SUPPORT_FRACTION
    ):
        return covariance_matrix(doppler_signal), cfg.snapshot_times_s, False

    stable_frequency_mask = np.all(known_mask[:, run_start:run_stop], axis=1)
    stable_frequency_indices = np.flatnonzero(stable_frequency_mask)
    if stable_frequency_indices.size < 2:
        return covariance_matrix(doppler_signal), cfg.snapshot_times_s, False

    support_matrix = doppler_signal[run_start:run_stop][:, stable_frequency_indices]
    return fbss_covariance(support_matrix, subarray_len), cfg.snapshot_times_s[run_start : run_start + subarray_len], True


def estimate_model_order_mdl(
    covariance: np.ndarray,
    snapshot_count: int,
    max_sources: int,
) -> int:
    """Estimate source count with the classical MDL criterion."""

    dimension = covariance.shape[0]
    if dimension <= 1 or snapshot_count <= 1:
        return 0
    max_sources = max(0, min(max_sources, dimension - 1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(np.maximum(eigenvalues.real, 1.0e-12))[::-1]

    best_order = 0
    best_value = float("inf")
    for candidate_order in range(max_sources + 1):
        noise_eigenvalues = eigenvalues[candidate_order:]
        if noise_eigenvalues.size == 0:
            continue
        arithmetic_mean = float(np.mean(noise_eigenvalues))
        geometric_mean = float(np.exp(np.mean(np.log(noise_eigenvalues))))
        if arithmetic_mean <= 0.0 or geometric_mean <= 0.0:
            continue
        mdl_value = (
            -snapshot_count
            * (dimension - candidate_order)
            * np.log(geometric_mean / arithmetic_mean)
            + 0.5 * candidate_order * (2 * dimension - candidate_order) * np.log(snapshot_count)
        )
        if mdl_value < best_value:
            best_value = float(mdl_value)
            best_order = candidate_order
    return best_order


def music_pseudospectrum(
    covariance: np.ndarray,
    n_targets: int,
    steering_matrix: np.ndarray,
) -> np.ndarray:
    """Evaluate the MUSIC pseudospectrum on a provided steering grid."""

    n_targets = max(1, min(n_targets, covariance.shape[0] - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    noise_subspace = eigenvectors[:, order][:, n_targets:]
    projections = np.einsum("ij,jk->ik", noise_subspace.conj().T, steering_matrix, optimize=True)
    denominator = np.sum(np.abs(projections) ** 2, axis=0)
    return 1.0 / np.maximum(denominator, 1.0e-12)


def _embed_frequency_grid(cfg: StudyConfig, radar_cube: np.ndarray) -> np.ndarray:
    embedded = np.zeros(
        (radar_cube.shape[0], cfg.anchor.physical_subcarrier_count, radar_cube.shape[2]),
        dtype=np.complex128,
    )
    embedded[:, cfg.simulated_subcarrier_indices, :] = radar_cube
    return embedded


def _cropped_positive_range_axis_m(cfg: StudyConfig) -> np.ndarray:
    full_range_axis_m = fft_range_axis_m(cfg)
    range_upper_m = sparse_unambiguous_range_m(cfg)
    keep_count = int(np.searchsorted(full_range_axis_m, range_upper_m + 1.0e-12, side="right"))
    keep_count = max(1, min(full_range_axis_m.size, keep_count))
    return full_range_axis_m[:keep_count]


def build_fft_cube(cfg: StudyConfig, radar_cube: np.ndarray) -> FftCubeResult:
    """Compute the coarse FFT cube over azimuth, range, and Doppler."""

    embedded_cube = _embed_frequency_grid(cfg, radar_cube)
    n_azimuth = cfg.runtime_profile.fft_angle_oversample * radar_cube.shape[0]
    n_range = cfg.runtime_profile.fft_range_oversample * cfg.anchor.physical_subcarrier_count
    n_velocity = cfg.runtime_profile.fft_doppler_oversample * radar_cube.shape[2]

    azimuth_window = np.hanning(radar_cube.shape[0]) if radar_cube.shape[0] > 1 else np.ones(1)
    frequency_window = (
        np.hanning(cfg.anchor.physical_subcarrier_count)
        if cfg.anchor.physical_subcarrier_count > 1
        else np.ones(1)
    )
    slow_time_window = np.hanning(radar_cube.shape[2]) if radar_cube.shape[2] > 1 else np.ones(1)
    windowed_cube = (
        embedded_cube
        * azimuth_window[:, np.newaxis, np.newaxis]
        * frequency_window[np.newaxis, :, np.newaxis]
        * slow_time_window[np.newaxis, np.newaxis, :]
    )

    azimuth_cube = np.fft.fftshift(np.fft.fft(windowed_cube, n=n_azimuth, axis=0), axes=0)
    range_cube = np.fft.ifft(np.fft.ifftshift(azimuth_cube, axes=1), n=n_range, axis=1)
    positive_range_axis_m = _cropped_positive_range_axis_m(cfg)
    range_cube = range_cube[:, : positive_range_axis_m.size, :]
    doppler_cube = np.fft.fftshift(np.fft.fft(range_cube, n=n_velocity, axis=2), axes=2)
    return FftCubeResult(
        power_cube=np.abs(doppler_cube) ** 2,
        azimuth_axis_deg=fft_azimuth_axis_deg(cfg),
        range_axis_m=positive_range_axis_m,
        velocity_axis_mps=fft_velocity_axis_mps(cfg),
        embedding_mode="legacy",
    )


def fft_search_bounds(fft_cube: FftCubeResult) -> SearchBounds:
    return SearchBounds(
        range_min_m=float(np.min(fft_cube.range_axis_m)),
        range_max_m=float(np.max(fft_cube.range_axis_m)),
        velocity_min_mps=float(np.min(fft_cube.velocity_axis_mps)),
        velocity_max_mps=float(np.max(fft_cube.velocity_axis_mps)),
        azimuth_min_deg=float(np.min(fft_cube.azimuth_axis_deg)),
        azimuth_max_deg=float(np.max(fft_cube.azimuth_axis_deg)),
    )


def config_search_bounds(cfg: StudyConfig) -> SearchBounds:
    return SearchBounds(
        range_min_m=float(np.min(fft_range_axis_m(cfg))),
        range_max_m=float(np.max(fft_range_axis_m(cfg))),
        velocity_min_mps=float(np.min(fft_velocity_axis_mps(cfg))),
        velocity_max_mps=float(np.max(fft_velocity_axis_mps(cfg))),
        azimuth_min_deg=float(np.min(fft_azimuth_axis_deg(cfg))),
        azimuth_max_deg=float(np.max(fft_azimuth_axis_deg(cfg))),
    )


def validate_targets_within_search_bounds(targets: tuple[TargetState, ...], search_bounds: SearchBounds) -> None:
    """Assert that every truth target lies inside the estimator search region."""

    for target in targets:
        if not search_bounds.range_min_m <= target.range_m <= search_bounds.range_max_m:
            raise ValueError(f"Target range {target.range_m:.3f} m falls outside the FFT range search bounds")
        if not search_bounds.velocity_min_mps <= target.velocity_mps <= search_bounds.velocity_max_mps:
            raise ValueError(f"Target velocity {target.velocity_mps:.3f} m/s falls outside the FFT Doppler search bounds")
        if not search_bounds.azimuth_min_deg <= target.azimuth_deg <= search_bounds.azimuth_max_deg:
            raise ValueError(f"Target azimuth {target.azimuth_deg:.3f} deg falls outside the FFT azimuth search bounds")


def _normalized_distance_cells_sq(cfg: StudyConfig, first: Detection, second: Detection) -> float:
    range_component = (first.range_m - second.range_m) / max(cfg.range_resolution_m, 1.0e-9)
    velocity_component = (first.velocity_mps - second.velocity_mps) / max(cfg.velocity_resolution_mps, 1.0e-9)
    angle_component = (first.azimuth_deg - second.azimuth_deg) / max(cfg.azimuth_resolution_deg, 1.0e-9)
    return float((range_component * range_component + velocity_component * velocity_component + angle_component * angle_component) / 3.0)


def _local_maxima_indices(power_cube: np.ndarray) -> np.ndarray:
    local_max_mask = power_cube == maximum_filter(power_cube, size=3, mode="nearest")
    return np.flatnonzero(local_max_mask)


def _sort_candidate_indices(power_cube: np.ndarray, candidate_indices: np.ndarray, top_pool_size: int) -> np.ndarray:
    if candidate_indices.size == 0:
        return candidate_indices
    candidate_scores = power_cube.ravel()[candidate_indices]
    if candidate_indices.size > top_pool_size:
        selection = np.argpartition(candidate_scores, -top_pool_size)[-top_pool_size:]
        candidate_indices = candidate_indices[selection]
        candidate_scores = candidate_scores[selection]
    sort_order = np.argsort(candidate_scores)[::-1]
    return candidate_indices[sort_order]


def _range_search_upper_m(cfg: StudyConfig, search_bounds: SearchBounds) -> float:
    return min(search_bounds.range_max_m, sparse_unambiguous_range_m(cfg))


def extract_candidates_from_fft(
    cfg: StudyConfig,
    fft_cube: FftCubeResult,
    max_candidates: int,
) -> tuple[Detection, ...]:
    """Extract coarse FFT candidates using a noise-floor threshold and backfill."""

    power_cube = fft_cube.power_cube
    all_local_maxima = _local_maxima_indices(power_cube)
    if all_local_maxima.size == 0:
        return ()

    range_upper_m = sparse_unambiguous_range_m(cfg)
    _, range_indices, _ = np.unravel_index(all_local_maxima, power_cube.shape)
    in_unambiguous_interval = fft_cube.range_axis_m[range_indices] <= range_upper_m + 1.0e-12
    candidate_indices = all_local_maxima[in_unambiguous_interval]
    if candidate_indices.size == 0:
        return ()

    top_pool_size = max(cfg.detector_backfill_pool_size, max_candidates * 32)
    sorted_local_maxima = _sort_candidate_indices(power_cube, candidate_indices, top_pool_size=top_pool_size)

    allowed_power = power_cube[:, fft_cube.range_axis_m <= range_upper_m + 1.0e-12, :]
    noise_floor = float(np.median(allowed_power))
    threshold = noise_floor * cfg.detector_threshold_scale
    selected_indices = [flat_index for flat_index in sorted_local_maxima.tolist() if float(power_cube.ravel()[flat_index]) >= threshold]

    minimum_required = min(max_candidates, max(cfg.expected_target_count, 1))
    if len(selected_indices) < minimum_required:
        seen = set(selected_indices)
        for flat_index in sorted_local_maxima.tolist():
            if flat_index in seen:
                continue
            selected_indices.append(flat_index)
            seen.add(flat_index)
            if len(selected_indices) >= minimum_required:
                break

    detections: list[Detection] = []
    nms_radius_sq = cfg.detection_nms_radius_cells * cfg.detection_nms_radius_cells
    for flat_index in selected_indices:
        score = float(power_cube.ravel()[flat_index])
        azimuth_index, range_index, velocity_index = np.unravel_index(flat_index, power_cube.shape)
        detection = Detection(
            range_m=float(fft_cube.range_axis_m[range_index]),
            velocity_mps=float(fft_cube.velocity_axis_mps[velocity_index]),
            azimuth_deg=float(fft_cube.azimuth_axis_deg[azimuth_index]),
            score=score,
        )
        if all(_normalized_distance_cells_sq(cfg, detection, existing) >= nms_radius_sq for existing in detections):
            detections.append(detection)
        if len(detections) >= max_candidates:
            break
    return tuple(detections)


def prepare_frontend(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    fft_cube: FftCubeResult | None = None,
    coarse_candidates: tuple[Detection, ...] | None = None,
) -> FrontendArtifacts:
    """Build the shared FFT front-end and time it once."""

    if fft_cube is not None and coarse_candidates is not None:
        return FrontendArtifacts(
            fft_cube=fft_cube,
            coarse_candidates=coarse_candidates,
            search_bounds=fft_search_bounds(fft_cube),
            frontend_runtime_s=0.0,
        )

    start_time = time.perf_counter()
    computed_fft_cube = build_fft_cube(cfg, radar_cube) if fft_cube is None else fft_cube
    computed_candidates = (
        extract_candidates_from_fft(cfg, computed_fft_cube, cfg.runtime_profile.coarse_candidate_count)
        if coarse_candidates is None
        else coarse_candidates
    )
    return FrontendArtifacts(
        fft_cube=computed_fft_cube,
        coarse_candidates=computed_candidates,
        search_bounds=fft_search_bounds(computed_fft_cube),
        frontend_runtime_s=time.perf_counter() - start_time,
    )


def _bounded_grid(center: float, half_span: float, lower: float, upper: float, n_points: int) -> np.ndarray:
    return np.linspace(max(lower, center - half_span), min(upper, center + half_span), n_points)


def _matched_filter_score(
    radar_cube: np.ndarray,
    azimuth_weights: np.ndarray,
    range_weights: np.ndarray,
    doppler_weights: np.ndarray,
) -> float:
    response = np.einsum(
        "h,hft,f,t->",
        azimuth_weights.conj(),
        radar_cube,
        range_weights.conj(),
        doppler_weights.conj(),
        optimize=True,
    )
    return float(np.abs(response))


def _max_output_detections(cfg: StudyConfig) -> int:
    return max(1, cfg.expected_target_count)


def _estimate_music_model_order(
    covariance: np.ndarray,
    snapshot_count: int,
    cfg: StudyConfig | None = None,
) -> int:
    if cfg is not None and cfg.music_model_order_mode == "expected":
        return max(1, cfg.expected_target_count)
    if cfg is not None and cfg.music_model_order_mode == "fixed":
        return int(cfg.music_fixed_model_order or max(1, cfg.expected_target_count))
    max_sources = min(max(1, covariance.shape[0] - 1), 6)
    return estimate_model_order_mdl(covariance, snapshot_count, max_sources)


def _build_detection(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    range_m: float,
    velocity_mps: float,
    azimuth_deg: float,
) -> Detection:
    azimuth_weights = azimuth_steering_matrix(
        cfg.effective_horizontal_positions_m,
        np.array([azimuth_deg]),
        cfg.wavelength_m,
    )[:, 0]
    azimuth_weights /= np.sqrt(max(1, azimuth_weights.size))
    range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([range_m]))[:, 0]
    doppler_weights = doppler_steering_matrix(
        cfg.snapshot_times_s,
        np.array([velocity_mps]),
        cfg.wavelength_m,
    )[:, 0]
    return Detection(
        range_m=float(range_m),
        velocity_mps=float(velocity_mps),
        azimuth_deg=float(azimuth_deg),
        score=_matched_filter_score(radar_cube, azimuth_weights, range_weights, doppler_weights),
    )


def _refine_detection_local(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    coarse_detection: Detection,
    search_bounds: SearchBounds,
    range_upper_bound_m: float | None = None,
) -> Detection:
    """Refine one coarse candidate with bounded matched-filter coordinate descent."""

    range_upper = search_bounds.range_max_m if range_upper_bound_m is None else min(
        search_bounds.range_max_m,
        range_upper_bound_m,
    )
    refined_range_m = coarse_detection.range_m
    refined_velocity_mps = coarse_detection.velocity_mps
    refined_azimuth_deg = coarse_detection.azimuth_deg

    def _score(azimuth_deg: float, range_m: float, velocity_mps: float) -> float:
        return _build_detection(
            cfg,
            radar_cube,
            range_m=range_m,
            velocity_mps=velocity_mps,
            azimuth_deg=azimuth_deg,
        ).score

    for _ in range(2):
        azimuth_grid = _bounded_grid(
            refined_azimuth_deg,
            cfg.azimuth_resolution_deg,
            max(-80.0, search_bounds.azimuth_min_deg),
            min(80.0, search_bounds.azimuth_max_deg),
            21,
        )
        refined_azimuth_deg = float(
            max(
                azimuth_grid,
                key=lambda azimuth_deg: _score(azimuth_deg, refined_range_m, refined_velocity_mps),
            )
        )

        range_grid = _bounded_grid(
            refined_range_m,
            2.5 * cfg.range_resolution_m,
            max(0.5, search_bounds.range_min_m),
            range_upper,
            41,
        )
        refined_range_m = float(
            max(
                range_grid,
                key=lambda range_m: _score(refined_azimuth_deg, range_m, refined_velocity_mps),
            )
        )

        velocity_grid = _bounded_grid(
            refined_velocity_mps,
            cfg.velocity_resolution_mps,
            search_bounds.velocity_min_mps,
            search_bounds.velocity_max_mps,
            21,
        )
        refined_velocity_mps = float(
            max(
                velocity_grid,
                key=lambda velocity_mps: _score(refined_azimuth_deg, refined_range_m, velocity_mps),
            )
        )

    return _build_detection(
        cfg,
        radar_cube,
        range_m=refined_range_m,
        velocity_mps=refined_velocity_mps,
        azimuth_deg=refined_azimuth_deg,
    )


def refine_detection_set_local(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    coarse_candidates: tuple[Detection, ...],
    search_bounds: SearchBounds,
    *,
    candidate_pool_size: int | None = None,
    range_upper_bound_m: float | None = None,
) -> tuple[Detection, ...]:
    """Refine a candidate set with local matched-filter coordinate descent and NMS."""

    if not coarse_candidates:
        return ()

    selected_candidates = coarse_candidates
    if candidate_pool_size is not None:
        selected_candidates = coarse_candidates[: max(1, candidate_pool_size)]

    refined_candidates = tuple(
        _refine_detection_local(
            cfg,
            radar_cube,
            coarse_detection=candidate,
            search_bounds=search_bounds,
            range_upper_bound_m=range_upper_bound_m,
        )
        for candidate in selected_candidates
    )
    merged: list[Detection] = []
    nms_radius_sq = cfg.detection_nms_radius_cells * cfg.detection_nms_radius_cells
    for candidate in sorted(refined_candidates, key=lambda item: item.score, reverse=True):
        if all(_normalized_distance_cells_sq(cfg, candidate, existing) >= nms_radius_sq for existing in merged):
            merged.append(candidate)
    return tuple(merged[: _max_output_detections(cfg)])


def _run_fft_estimator(
    cfg: StudyConfig,
    frontend: FrontendArtifacts,
) -> MethodEstimate:
    detections = frontend.coarse_candidates[: _max_output_detections(cfg)]
    return MethodEstimate(
        label="Angle-Range-Doppler FFT",
        detections=detections,
        reported_target_count=len(detections),
        estimated_model_order=None,
        frontend_runtime_s=frontend.frontend_runtime_s,
        incremental_runtime_s=0.0,
        total_runtime_s=frontend.frontend_runtime_s,
    )


def _run_staged_music(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    coarse_candidates: tuple[Detection, ...],
    search_bounds: SearchBounds,
    frontend_runtime_s: float,
    use_fbss: bool,
) -> MethodEstimate:
    start_time = time.perf_counter()
    estimated_model_order = None
    if not coarse_candidates:
        incremental_runtime_s = time.perf_counter() - start_time
        return MethodEstimate(
            label="FFT-Seeded Azimuth MUSIC + FBSS" if use_fbss else "FFT-Seeded Staged Azimuth MUSIC",
            detections=(),
            reported_target_count=0,
            estimated_model_order=estimated_model_order,
            frontend_runtime_s=frontend_runtime_s,
            incremental_runtime_s=incremental_runtime_s,
            total_runtime_s=frontend_runtime_s + incremental_runtime_s,
        )

    horizontal_positions = cfg.effective_horizontal_positions_m
    range_search_upper = _range_search_upper_m(cfg, search_bounds)
    refined_candidates: list[Detection] = []
    for coarse in coarse_candidates:
        coarse_range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([coarse.range_m]))[:, 0]
        coarse_doppler_weights = doppler_steering_matrix(
            cfg.snapshot_times_s,
            np.array([coarse.velocity_mps]),
            cfg.wavelength_m,
        )[:, 0]

        aligned_spatial = (
            radar_cube
            * coarse_range_weights.conj()[np.newaxis, :, np.newaxis]
            * coarse_doppler_weights.conj()[np.newaxis, np.newaxis, :]
        )
        spatial_matrix = aligned_spatial.reshape(radar_cube.shape[0], -1)

        if use_fbss:
            spatial_covariance = fbss_covariance(spatial_matrix, cfg.fbss_subarray_len)
            spatial_positions = horizontal_positions[: cfg.fbss_subarray_len]
        else:
            spatial_covariance = covariance_matrix(spatial_matrix)
            spatial_positions = horizontal_positions

        azimuth_grid = _bounded_grid(
            coarse.azimuth_deg,
            max(2.0 * cfg.azimuth_resolution_deg, 8.0),
            search_bounds.azimuth_min_deg,
            search_bounds.azimuth_max_deg,
            cfg.runtime_profile.music_grid_points,
        )
        azimuth_spectrum = music_pseudospectrum(
            spatial_covariance,
            n_targets=1,
            steering_matrix=azimuth_steering_matrix(spatial_positions, azimuth_grid, cfg.wavelength_m),
        )
        refined_azimuth_deg = float(azimuth_grid[int(np.argmax(azimuth_spectrum))])
        azimuth_weights = azimuth_steering_matrix(horizontal_positions, np.array([refined_azimuth_deg]), cfg.wavelength_m)[:, 0]
        azimuth_weights /= np.sqrt(max(1, azimuth_weights.size))

        beamformed_cube = np.einsum("h,hft->ft", azimuth_weights.conj(), radar_cube, optimize=True)

        range_grid = _bounded_grid(
            coarse.range_m,
            max(2.0 * cfg.range_resolution_m, 2.0),
            search_bounds.range_min_m,
            range_search_upper,
            cfg.runtime_profile.music_grid_points,
        )
        range_covariance = covariance_matrix(beamformed_cube * coarse_doppler_weights.conj()[np.newaxis, :])
        range_spectrum = music_pseudospectrum(
            range_covariance,
            n_targets=1,
            steering_matrix=range_steering_matrix(cfg.frequencies_hz, range_grid),
        )
        refined_range_m = float(range_grid[int(np.argmax(range_spectrum))])
        refined_range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([refined_range_m]))[:, 0]

        doppler_grid = _bounded_grid(
            coarse.velocity_mps,
            max(2.0 * cfg.velocity_resolution_mps, 0.75),
            search_bounds.velocity_min_mps,
            search_bounds.velocity_max_mps,
            cfg.runtime_profile.music_grid_points,
        )
        doppler_covariance = covariance_matrix((beamformed_cube * refined_range_weights.conj()[:, np.newaxis]).T)
        doppler_spectrum = music_pseudospectrum(
            doppler_covariance,
            n_targets=1,
            steering_matrix=doppler_steering_matrix(cfg.snapshot_times_s, doppler_grid, cfg.wavelength_m),
        )
        refined_velocity_mps = float(doppler_grid[int(np.argmax(doppler_spectrum))])
        refined_doppler_weights = doppler_steering_matrix(
            cfg.snapshot_times_s,
            np.array([refined_velocity_mps]),
            cfg.wavelength_m,
        )[:, 0]

        refined_candidates.append(
            Detection(
                range_m=refined_range_m,
                velocity_mps=refined_velocity_mps,
                azimuth_deg=refined_azimuth_deg,
                score=_matched_filter_score(
                    radar_cube,
                    azimuth_weights=azimuth_weights,
                    range_weights=refined_range_weights,
                    doppler_weights=refined_doppler_weights,
                ),
            )
        )

    merged: list[Detection] = []
    nms_radius_sq = cfg.detection_nms_radius_cells * cfg.detection_nms_radius_cells
    for candidate in sorted(refined_candidates, key=lambda item: item.score, reverse=True):
        if all(_normalized_distance_cells_sq(cfg, candidate, existing) >= nms_radius_sq for existing in merged):
            merged.append(candidate)

    incremental_runtime_s = time.perf_counter() - start_time
    detections = tuple(merged[: _max_output_detections(cfg)])
    return MethodEstimate(
        label="FFT-Seeded Azimuth MUSIC + FBSS" if use_fbss else "FFT-Seeded Staged Azimuth MUSIC",
        detections=detections,
        reported_target_count=len(detections),
        estimated_model_order=estimated_model_order,
        frontend_runtime_s=frontend_runtime_s,
        incremental_runtime_s=incremental_runtime_s,
        total_runtime_s=frontend_runtime_s + incremental_runtime_s,
    )


def _1d_peak_indices(spectrum: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """Find local-maximum indices in a 1-D spectrum."""

    from scipy.ndimage import maximum_filter1d

    local_max = spectrum == maximum_filter1d(spectrum, size=max(3, 2 * min_distance + 1), mode="nearest")
    return np.flatnonzero(local_max)


def _run_full_search_music(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    search_bounds: SearchBounds,
    frontend_runtime_s: float,
    use_fbss: bool,
    known_mask: np.ndarray | None = None,
) -> MethodEstimate:
    """Staged MUSIC with dense azimuth/range/Doppler grid search (not FFT-seeded).

    This estimator performs a dense azimuth search, then conditional range MUSIC,
    then conditional Doppler MUSIC, followed by local matched-filter refinement.
    It is a staged subspace pipeline rather than a true joint 3-D MUSIC search.
    """

    start_time = time.perf_counter()
    label = "Staged MUSIC + FBSS" if use_fbss else "Staged MUSIC"

    horizontal_positions = cfg.effective_horizontal_positions_m
    global_matrix = radar_cube.reshape(radar_cube.shape[0], -1)
    target_order = _max_output_detections(cfg)

    # --- 1. Full azimuth search ---
    n_az_grid = cfg.runtime_profile.music_grid_points * 3
    azimuth_grid = np.linspace(
        max(-80.0, search_bounds.azimuth_min_deg + 0.5),
        min(80.0, search_bounds.azimuth_max_deg - 0.5),
        n_az_grid,
    )

    if use_fbss:
        spatial_cov = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
        spatial_positions = horizontal_positions[: cfg.fbss_subarray_len]
    else:
        spatial_cov = covariance_matrix(global_matrix)
        spatial_positions = horizontal_positions

    estimated_model_order = _estimate_music_model_order(spatial_cov, global_matrix.shape[1], cfg)
    spectrum_target_order = max(target_order, estimated_model_order)
    azimuth_spectrum = music_pseudospectrum(
        spatial_cov,
        n_targets=spectrum_target_order,
        steering_matrix=azimuth_steering_matrix(spatial_positions, azimuth_grid, cfg.wavelength_m),
    )

    az_peak_idx = _1d_peak_indices(azimuth_spectrum, min_distance=max(1, n_az_grid // 20))
    if az_peak_idx.size == 0:
        az_peak_idx = np.array([int(np.argmax(azimuth_spectrum))])
    az_scores = azimuth_spectrum[az_peak_idx]
    top_k = min(
        max(target_order * cfg.music_azimuth_peak_factor, target_order + 2, estimated_model_order + 1),
        az_peak_idx.size,
    )
    top_az_idx = az_peak_idx[np.argsort(az_scores)[-top_k:]]
    azimuth_candidates = azimuth_grid[top_az_idx]

    # --- 2. Per-azimuth: range MUSIC + Doppler MUSIC ---
    range_search_upper = _range_search_upper_m(cfg, search_bounds)

    refined_candidates: list[Detection] = []
    coarse_stage_candidates: list[Detection] = []
    range_grid_step_m = max(0.5 * cfg.range_resolution_m, 0.25)
    n_range_grid = max(
        cfg.runtime_profile.music_grid_points * 2,
        int(np.ceil((range_search_upper - max(0.5, search_bounds.range_min_m)) / range_grid_step_m)) + 1,
    )
    for az_deg in azimuth_candidates:
        az_weights = azimuth_steering_matrix(
            horizontal_positions, np.array([az_deg]), cfg.wavelength_m
        )[:, 0]
        az_weights /= np.sqrt(max(1, az_weights.size))
        beamformed = np.einsum("h,hft->ft", az_weights.conj(), radar_cube, optimize=True)

        range_grid = np.linspace(
            max(0.5, search_bounds.range_min_m),
            range_search_upper,
            n_range_grid,
        )
        range_cov, range_frequencies_hz, _ = _range_music_covariance(cfg, beamformed, known_mask)
        range_spectrum = music_pseudospectrum(
            range_cov,
            n_targets=1,
            steering_matrix=range_steering_matrix(range_frequencies_hz, range_grid),
        )
        r_peak_idx = _1d_peak_indices(range_spectrum, min_distance=max(1, n_range_grid // 30))
        if r_peak_idx.size == 0:
            r_peak_idx = np.array([int(np.argmax(range_spectrum))])
        r_scores = range_spectrum[r_peak_idx]
        top_r = min(max(2, cfg.music_range_peak_pool), r_peak_idx.size)
        top_r_idx = r_peak_idx[np.argsort(r_scores)[-top_r:]]

        for ri in top_r_idx:
            r_m = float(range_grid[ri])
            range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([r_m]))[:, 0]

            doppler_grid = np.linspace(
                search_bounds.velocity_min_mps,
                search_bounds.velocity_max_mps,
                cfg.runtime_profile.music_grid_points,
            )
            doppler_signal = (beamformed * range_weights.conj()[:, np.newaxis]).T
            doppler_cov, doppler_times_s, _ = _doppler_music_covariance(
                cfg,
                doppler_signal,
                known_mask,
            )
            doppler_spectrum = music_pseudospectrum(
                doppler_cov,
                n_targets=1,
                steering_matrix=doppler_steering_matrix(doppler_times_s, doppler_grid, cfg.wavelength_m),
            )
            v_mps = float(doppler_grid[int(np.argmax(doppler_spectrum))])
            coarse_stage_candidate = Detection(
                range_m=r_m,
                velocity_mps=v_mps,
                azimuth_deg=float(az_deg),
                score=0.0,
            )
            coarse_stage_candidates.append(coarse_stage_candidate)

            if cfg.skip_local_refinement:
                refined_candidates.append(coarse_stage_candidate)
            else:
                refined_candidates.append(
                    _refine_detection_local(
                        cfg,
                        radar_cube,
                        coarse_detection=coarse_stage_candidate,
                        search_bounds=search_bounds,
                        range_upper_bound_m=range_search_upper,
                    )
                )

    # --- 3. NMS merge ---
    merged: list[Detection] = []
    nms_radius_sq = cfg.detection_nms_radius_cells * cfg.detection_nms_radius_cells
    for candidate in sorted(refined_candidates, key=lambda item: item.score, reverse=True):
        if all(_normalized_distance_cells_sq(cfg, candidate, existing) >= nms_radius_sq for existing in merged):
            merged.append(candidate)

    incremental_runtime_s = time.perf_counter() - start_time
    detections = tuple(merged[: _max_output_detections(cfg)])
    return MethodEstimate(
        label=label,
        detections=detections,
        reported_target_count=len(detections),
        estimated_model_order=estimated_model_order,
        frontend_runtime_s=frontend_runtime_s,
        incremental_runtime_s=incremental_runtime_s,
        total_runtime_s=frontend_runtime_s + incremental_runtime_s,
        stage_diagnostics={
            "azimuth_peak_count": f"{az_peak_idx.size:d}",
            "azimuth_candidate_count": f"{azimuth_candidates.size:d}",
            "azimuth_candidates_deg": "|".join(f"{float(value):.3f}" for value in azimuth_candidates.tolist()),
            "coarse_stage_candidate_count": f"{len(coarse_stage_candidates):d}",
            "coarse_stage_candidates": _serialize_detection_sequence(coarse_stage_candidates),
        },
    )


def run_estimators(
    cfg: StudyConfig,
    radar_cube: np.ndarray,
    fft_cube: FftCubeResult | None = None,
    coarse_candidates: tuple[Detection, ...] | None = None,
    frontend_runtime_s: float | None = None,
) -> dict[str, MethodEstimate]:
    """Run the full estimator family on one radar cube."""

    if fft_cube is not None and coarse_candidates is not None and frontend_runtime_s is not None:
        frontend = FrontendArtifacts(
            fft_cube=fft_cube,
            coarse_candidates=coarse_candidates,
            search_bounds=fft_search_bounds(fft_cube),
            frontend_runtime_s=frontend_runtime_s,
        )
    else:
        frontend = prepare_frontend(cfg, radar_cube, fft_cube=fft_cube, coarse_candidates=coarse_candidates)

    return {
        "fft": _run_fft_estimator(cfg, frontend),
        "music_full": _run_full_search_music(
            cfg,
            radar_cube,
            frontend.search_bounds,
            frontend.frontend_runtime_s,
            use_fbss=False,
        ),
    }
