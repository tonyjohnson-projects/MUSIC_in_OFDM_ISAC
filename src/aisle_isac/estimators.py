"""Estimators for the private-5G angle-range-Doppler study."""

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
)


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
    estimated_model_order: int
    frontend_runtime_s: float
    incremental_runtime_s: float
    total_runtime_s: float


def covariance_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """Return the sample covariance for a sensors x snapshots matrix."""

    data_matrix = np.asarray(data_matrix, dtype=np.complex128)
    snapshot_count = max(1, data_matrix.shape[1])
    return np.einsum("ik,jk->ij", data_matrix, data_matrix.conj(), optimize=True) / snapshot_count


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
    range_cube = range_cube[:, : n_range // 2, :]
    doppler_cube = np.fft.fftshift(np.fft.fft(range_cube, n=n_velocity, axis=2), axes=2)
    return FftCubeResult(
        power_cube=np.abs(doppler_cube) ** 2,
        azimuth_axis_deg=fft_azimuth_axis_deg(cfg),
        range_axis_m=fft_range_axis_m(cfg),
        velocity_axis_mps=fft_velocity_axis_mps(cfg),
    )


def fft_search_bounds(fft_cube: FftCubeResult) -> SearchBounds:
    return SearchBounds(
        range_min_m=float(fft_cube.range_axis_m[0]),
        range_max_m=float(fft_cube.range_axis_m[-1]),
        velocity_min_mps=float(fft_cube.velocity_axis_mps[0]),
        velocity_max_mps=float(fft_cube.velocity_axis_mps[-1]),
        azimuth_min_deg=float(fft_cube.azimuth_axis_deg[0]),
        azimuth_max_deg=float(fft_cube.azimuth_axis_deg[-1]),
    )


def config_search_bounds(cfg: StudyConfig) -> SearchBounds:
    return SearchBounds(
        range_min_m=float(fft_range_axis_m(cfg)[0]),
        range_max_m=float(fft_range_axis_m(cfg)[-1]),
        velocity_min_mps=float(fft_velocity_axis_mps(cfg)[0]),
        velocity_max_mps=float(fft_velocity_axis_mps(cfg)[-1]),
        azimuth_min_deg=float(fft_azimuth_axis_deg(cfg)[0]),
        azimuth_max_deg=float(fft_azimuth_axis_deg(cfg)[-1]),
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


def extract_candidates_from_fft(
    cfg: StudyConfig,
    fft_cube: FftCubeResult,
    max_candidates: int,
) -> tuple[Detection, ...]:
    """Extract coarse FFT candidates using a noise-floor threshold and backfill."""

    power_cube = fft_cube.power_cube
    all_local_maxima = _local_maxima_indices(power_cube)
    top_pool_size = max(cfg.detector_backfill_pool_size, max_candidates * 32)
    sorted_local_maxima = _sort_candidate_indices(power_cube, all_local_maxima, top_pool_size=top_pool_size)
    if sorted_local_maxima.size == 0:
        return ()

    noise_floor = float(np.median(power_cube))
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


def _run_fft_estimator(
    frontend: FrontendArtifacts,
) -> MethodEstimate:
    return MethodEstimate(
        label="Angle-Range-Doppler FFT",
        detections=frontend.coarse_candidates,
        estimated_model_order=len(frontend.coarse_candidates),
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
    if not coarse_candidates:
        incremental_runtime_s = time.perf_counter() - start_time
        return MethodEstimate(
            label="FFT-Seeded Azimuth MUSIC + FBSS" if use_fbss else "FFT-Seeded Staged Azimuth MUSIC",
            detections=(),
            estimated_model_order=0,
            frontend_runtime_s=frontend_runtime_s,
            incremental_runtime_s=incremental_runtime_s,
            total_runtime_s=frontend_runtime_s + incremental_runtime_s,
        )

    global_matrix = radar_cube.reshape(radar_cube.shape[0], -1)
    if use_fbss:
        global_covariance = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
        global_order = estimate_model_order_mdl(
            global_covariance,
            snapshot_count=global_matrix.shape[1],
            max_sources=min(4, global_covariance.shape[0] - 1),
        )
    else:
        global_covariance = covariance_matrix(global_matrix)
        global_order = estimate_model_order_mdl(
            global_covariance,
            snapshot_count=global_matrix.shape[1],
            max_sources=min(4, global_covariance.shape[0] - 1),
        )
    global_order = max(1, global_order)

    horizontal_positions = cfg.effective_horizontal_positions_m
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
            local_order = estimate_model_order_mdl(
                spatial_covariance,
                snapshot_count=spatial_matrix.shape[1],
                max_sources=min(4, spatial_covariance.shape[0] - 1),
            )
        else:
            spatial_covariance = covariance_matrix(spatial_matrix)
            spatial_positions = horizontal_positions
            local_order = estimate_model_order_mdl(
                spatial_covariance,
                snapshot_count=spatial_matrix.shape[1],
                max_sources=min(4, spatial_covariance.shape[0] - 1),
            )
        local_order = max(1, local_order)

        azimuth_grid = _bounded_grid(
            coarse.azimuth_deg,
            max(2.0 * cfg.azimuth_resolution_deg, 8.0),
            search_bounds.azimuth_min_deg,
            search_bounds.azimuth_max_deg,
            cfg.runtime_profile.music_grid_points,
        )
        azimuth_spectrum = music_pseudospectrum(
            spatial_covariance,
            n_targets=local_order,
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
            search_bounds.range_max_m,
            cfg.runtime_profile.music_grid_points,
        )
        range_covariance = covariance_matrix(beamformed_cube * coarse_doppler_weights.conj()[np.newaxis, :])
        range_spectrum = music_pseudospectrum(
            range_covariance,
            n_targets=local_order,
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
            n_targets=local_order,
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
    return MethodEstimate(
        label="FFT-Seeded Azimuth MUSIC + FBSS" if use_fbss else "FFT-Seeded Staged Azimuth MUSIC",
        detections=tuple(merged[: cfg.runtime_profile.coarse_candidate_count]),
        estimated_model_order=global_order,
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
) -> MethodEstimate:
    """MUSIC with dense independent grid search (not FFT-seeded).

    This estimator demonstrates true sub-resolution capability by
    searching the full parameter space with MUSIC pseudospectra
    rather than refining FFT candidates.
    """

    start_time = time.perf_counter()
    label = "Full-Search MUSIC + FBSS" if use_fbss else "Full-Search MUSIC"

    horizontal_positions = cfg.effective_horizontal_positions_m

    # --- 1. Global model-order estimation ---
    global_matrix = radar_cube.reshape(radar_cube.shape[0], -1)
    if use_fbss:
        global_cov = fbss_covariance(global_matrix, cfg.fbss_subarray_len)
    else:
        global_cov = covariance_matrix(global_matrix)
    global_order = max(
        1,
        estimate_model_order_mdl(
            global_cov,
            snapshot_count=global_matrix.shape[1],
            max_sources=min(4, global_cov.shape[0] - 1),
        ),
    )

    # --- 2. Full azimuth search ---
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

    azimuth_spectrum = music_pseudospectrum(
        spatial_cov,
        n_targets=global_order,
        steering_matrix=azimuth_steering_matrix(spatial_positions, azimuth_grid, cfg.wavelength_m),
    )

    az_peak_idx = _1d_peak_indices(azimuth_spectrum, min_distance=max(1, n_az_grid // 20))
    if az_peak_idx.size == 0:
        az_peak_idx = np.array([int(np.argmax(azimuth_spectrum))])
    az_scores = azimuth_spectrum[az_peak_idx]
    top_k = min(cfg.runtime_profile.coarse_candidate_count, az_peak_idx.size)
    top_az_idx = az_peak_idx[np.argsort(az_scores)[-top_k:]]
    azimuth_candidates = azimuth_grid[top_az_idx]

    # --- 3. Per-azimuth: range MUSIC + Doppler MUSIC ---
    # Compute unambiguous range from maximum subcarrier gap.
    # Sparse OFDM subcarriers create periodic range ambiguities at
    # R_unamb = c / (2 * max_gap_hz).  We must restrict the search
    # grid to stay within the first unambiguous interval.
    C_LIGHT = 299_792_458.0
    sorted_freqs = np.sort(cfg.frequencies_hz)
    if sorted_freqs.size > 1:
        max_gap_hz = float(np.max(np.diff(sorted_freqs)))
        max_unambiguous_range_m = C_LIGHT / (2.0 * max_gap_hz)
    else:
        max_unambiguous_range_m = search_bounds.range_max_m
    range_search_upper = min(search_bounds.range_max_m, max_unambiguous_range_m)

    refined_candidates: list[Detection] = []
    n_range_grid = cfg.runtime_profile.music_grid_points * 2
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
        range_cov = covariance_matrix(beamformed)
        range_order = max(
            1,
            estimate_model_order_mdl(
                range_cov,
                snapshot_count=beamformed.shape[1],
                max_sources=min(4, range_cov.shape[0] - 1),
            ),
        )
        range_spectrum = music_pseudospectrum(
            range_cov,
            n_targets=range_order,
            steering_matrix=range_steering_matrix(cfg.frequencies_hz, range_grid),
        )
        r_peak_idx = _1d_peak_indices(range_spectrum, min_distance=max(1, n_range_grid // 30))
        if r_peak_idx.size == 0:
            r_peak_idx = np.array([int(np.argmax(range_spectrum))])
        r_scores = range_spectrum[r_peak_idx]
        top_r = min(3, r_peak_idx.size)
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
            doppler_cov = covariance_matrix(doppler_signal)
            doppler_order = max(
                1,
                estimate_model_order_mdl(
                    doppler_cov,
                    snapshot_count=doppler_signal.shape[1],
                    max_sources=min(4, doppler_cov.shape[0] - 1),
                ),
            )
            doppler_spectrum = music_pseudospectrum(
                doppler_cov,
                n_targets=doppler_order,
                steering_matrix=doppler_steering_matrix(cfg.snapshot_times_s, doppler_grid, cfg.wavelength_m),
            )
            v_mps = float(doppler_grid[int(np.argmax(doppler_spectrum))])

            doppler_weights = doppler_steering_matrix(
                cfg.snapshot_times_s, np.array([v_mps]), cfg.wavelength_m
            )[:, 0]

            refined_candidates.append(
                Detection(
                    range_m=r_m,
                    velocity_mps=v_mps,
                    azimuth_deg=float(az_deg),
                    score=_matched_filter_score(radar_cube, az_weights, range_weights, doppler_weights),
                )
            )

    # --- 4. NMS merge ---
    merged: list[Detection] = []
    nms_radius_sq = cfg.detection_nms_radius_cells * cfg.detection_nms_radius_cells
    for candidate in sorted(refined_candidates, key=lambda item: item.score, reverse=True):
        if all(_normalized_distance_cells_sq(cfg, candidate, existing) >= nms_radius_sq for existing in merged):
            merged.append(candidate)

    incremental_runtime_s = time.perf_counter() - start_time
    return MethodEstimate(
        label=label,
        detections=tuple(merged[: cfg.runtime_profile.coarse_candidate_count]),
        estimated_model_order=global_order,
        frontend_runtime_s=frontend_runtime_s,
        incremental_runtime_s=incremental_runtime_s,
        total_runtime_s=frontend_runtime_s + incremental_runtime_s,
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
        "fft": _run_fft_estimator(frontend),
        "music": _run_staged_music(
            cfg,
            radar_cube,
            frontend.coarse_candidates,
            frontend.search_bounds,
            frontend.frontend_runtime_s,
            use_fbss=False,
        ),
        "fbss": _run_staged_music(
            cfg,
            radar_cube,
            frontend.coarse_candidates,
            frontend.search_bounds,
            frontend.frontend_runtime_s,
            use_fbss=True,
        ),
        "music_full": _run_full_search_music(
            cfg,
            radar_cube,
            frontend.search_bounds,
            frontend.frontend_runtime_s,
            use_fbss=False,
        ),
    }
