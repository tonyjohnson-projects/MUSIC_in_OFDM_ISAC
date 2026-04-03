"""Masked FFT baselines for communications-scheduled OFDM ISAC."""

from __future__ import annotations

import time

import numpy as np

from aisle_isac.config import StudyConfig
from aisle_isac.estimators import FftCubeResult, FrontendArtifacts, extract_candidates_from_fft, fft_search_bounds
from aisle_isac.masked_observation import MaskedObservation, extract_known_symbol_cube
from aisle_isac.ofdm import fft_azimuth_axis_deg, fft_range_axis_m, fft_velocity_axis_mps


MASKED_FFT_EMBEDDING_MODES = ("zero_fill", "weighted")


def _embed_frequency_grid(cfg: StudyConfig, radar_cube: np.ndarray) -> np.ndarray:
    embedded = np.zeros(
        (radar_cube.shape[0], cfg.anchor.physical_subcarrier_count, radar_cube.shape[2]),
        dtype=np.complex128,
    )
    embedded[:, cfg.simulated_subcarrier_indices, :] = radar_cube
    return embedded


def _fft_windows(
    cfg: StudyConfig,
    antenna_count: int,
    symbol_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_window = np.hanning(antenna_count) if antenna_count > 1 else np.ones(1)
    frequency_window = (
        np.hanning(cfg.anchor.physical_subcarrier_count)
        if cfg.anchor.physical_subcarrier_count > 1
        else np.ones(1)
    )
    slow_time_window = np.hanning(symbol_count) if symbol_count > 1 else np.ones(1)
    return azimuth_window, frequency_window, slow_time_window


def _fft_power_cube(
    cfg: StudyConfig,
    embedded_cube: np.ndarray,
    azimuth_window: np.ndarray,
    frequency_window: np.ndarray,
    slow_time_window: np.ndarray,
) -> np.ndarray:
    n_azimuth = cfg.runtime_profile.fft_angle_oversample * embedded_cube.shape[0]
    n_range = cfg.runtime_profile.fft_range_oversample * cfg.anchor.physical_subcarrier_count
    n_velocity = cfg.runtime_profile.fft_doppler_oversample * embedded_cube.shape[2]
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
    return np.abs(doppler_cube) ** 2


def _support_statistics(
    cfg: StudyConfig,
    known_mask: np.ndarray,
    azimuth_window: np.ndarray,
    frequency_window: np.ndarray,
    slow_time_window: np.ndarray,
) -> tuple[float, float, float]:
    known_mask = np.asarray(known_mask, dtype=bool)
    if known_mask.ndim != 2:
        raise ValueError("known_mask must be a 2D subcarrier-by-symbol mask")
    if known_mask.shape != (cfg.n_subcarriers, slow_time_window.size):
        raise ValueError("known_mask must use the simulated subcarrier dimension of the active config")

    simulated_frequency_weights = np.square(frequency_window[cfg.simulated_subcarrier_indices])[:, np.newaxis]
    slow_time_weights = np.square(slow_time_window)[np.newaxis, :]
    per_re_weights = simulated_frequency_weights * slow_time_weights
    azimuth_energy = float(np.sum(np.square(azimuth_window)))

    support_energy = azimuth_energy * float(np.sum(per_re_weights[known_mask]))
    full_support_energy = azimuth_energy * float(np.sum(per_re_weights))
    known_fraction = float(np.mean(known_mask))
    return support_energy, full_support_energy, known_fraction


def build_masked_fft_cube_from_cube(
    cfg: StudyConfig,
    known_cube: np.ndarray,
    known_mask: np.ndarray,
    *,
    embedding_mode: str = "weighted",
) -> FftCubeResult:
    """Compute a masked FFT baseline from a de-embedded known-symbol cube."""

    if embedding_mode not in MASKED_FFT_EMBEDDING_MODES:
        supported = ", ".join(MASKED_FFT_EMBEDDING_MODES)
        raise ValueError(f"embedding_mode must be one of {supported}")

    known_cube = np.asarray(known_cube, dtype=np.complex128)
    known_mask = np.asarray(known_mask, dtype=bool)
    if known_cube.ndim != 3:
        raise ValueError("known_cube must be antenna-by-subcarrier-by-symbol")
    if known_cube.shape[1:] != known_mask.shape:
        raise ValueError("known_mask must match the subcarrier and symbol dimensions of known_cube")
    if known_cube.shape[1] != cfg.n_subcarriers:
        raise ValueError("known_cube uses a different simulated subcarrier count than the active config")
    if known_cube.shape[2] != cfg.burst_profile.n_snapshots:
        raise ValueError("known_cube uses a different symbol count than the active config")

    azimuth_window, frequency_window, slow_time_window = _fft_windows(cfg, known_cube.shape[0], known_cube.shape[2])
    raw_power_cube = _fft_power_cube(
        cfg,
        _embed_frequency_grid(cfg, known_cube),
        azimuth_window,
        frequency_window,
        slow_time_window,
    )
    support_energy, full_support_energy, known_fraction = _support_statistics(
        cfg,
        known_mask,
        azimuth_window,
        frequency_window,
        slow_time_window,
    )
    if support_energy <= 0.0:
        raise ValueError("known_mask must contain at least one known resource element")

    normalization_gain = 1.0 if embedding_mode == "zero_fill" else full_support_energy / support_energy
    return FftCubeResult(
        power_cube=raw_power_cube * normalization_gain,
        azimuth_axis_deg=fft_azimuth_axis_deg(cfg),
        range_axis_m=fft_range_axis_m(cfg),
        velocity_axis_mps=fft_velocity_axis_mps(cfg),
        embedding_mode=embedding_mode,
        support_energy=support_energy,
        full_support_energy=full_support_energy,
        normalization_gain=normalization_gain,
        known_fraction=known_fraction,
    )


def build_masked_fft_cube(
    cfg: StudyConfig,
    masked_observation: MaskedObservation,
    *,
    embedding_mode: str = "weighted",
) -> FftCubeResult:
    """Compute a masked FFT cube directly from a masked observation."""

    known_cube = extract_known_symbol_cube(masked_observation)
    return build_masked_fft_cube_from_cube(
        cfg,
        known_cube,
        masked_observation.known_symbol_mask,
        embedding_mode=embedding_mode,
    )


def prepare_masked_frontend(
    cfg: StudyConfig,
    masked_observation: MaskedObservation,
    *,
    embedding_mode: str = "weighted",
    max_candidates: int | None = None,
) -> FrontendArtifacts:
    """Build the masked FFT front-end and coarse candidate list once."""

    start_time = time.perf_counter()
    fft_cube = build_masked_fft_cube(cfg, masked_observation, embedding_mode=embedding_mode)
    candidate_count = cfg.runtime_profile.coarse_candidate_count if max_candidates is None else max_candidates
    coarse_candidates = extract_candidates_from_fft(cfg, fft_cube, candidate_count)
    return FrontendArtifacts(
        fft_cube=fft_cube,
        coarse_candidates=coarse_candidates,
        search_bounds=fft_search_bounds(fft_cube),
        frontend_runtime_s=time.perf_counter() - start_time,
    )
