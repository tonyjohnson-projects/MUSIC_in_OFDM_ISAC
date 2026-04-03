"""NR-like waveform and steering helpers for the angle-range-Doppler study."""

from __future__ import annotations

import numpy as np

from aisle_isac.config import C_LIGHT_M_PER_S, StudyConfig


def range_steering_matrix(frequencies_hz: np.ndarray, ranges_m: np.ndarray) -> np.ndarray:
    """Return frequency-domain steering vectors for the requested ranges."""

    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    ranges_m = np.asarray(ranges_m, dtype=float)
    delays_s = 2.0 * ranges_m / C_LIGHT_M_PER_S
    return np.exp(-1j * 2.0 * np.pi * frequencies_hz[:, np.newaxis] * delays_s[np.newaxis, :])


def doppler_steering_matrix(
    times_s: np.ndarray,
    velocities_mps: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Return slow-time steering vectors for the requested radial velocities."""

    times_s = np.asarray(times_s, dtype=float)
    velocities_mps = np.asarray(velocities_mps, dtype=float)
    doppler_hz = 2.0 * velocities_mps / wavelength_m
    return np.exp(1j * 2.0 * np.pi * times_s[:, np.newaxis] * doppler_hz[np.newaxis, :])


def azimuth_steering_matrix(
    horizontal_positions_m: np.ndarray,
    azimuths_deg: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Return horizontal-array steering vectors for azimuth-only estimation."""

    horizontal_positions_m = np.asarray(horizontal_positions_m, dtype=float)
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)
    sin_azimuth = np.sin(np.deg2rad(azimuths_deg))
    return np.exp(
        -1j
        * 2.0
        * np.pi
        * horizontal_positions_m[:, np.newaxis]
        * sin_azimuth[np.newaxis, :]
        / wavelength_m
    )


def fft_range_axis_m(cfg: StudyConfig) -> np.ndarray:
    """Return the positive FFT range axis."""

    n_fft = cfg.runtime_profile.fft_range_oversample * cfg.anchor.physical_subcarrier_count
    range_axis_m = (
        np.arange(n_fft, dtype=float)
        * C_LIGHT_M_PER_S
        / (2.0 * cfg.anchor.occupied_bandwidth_hz * cfg.runtime_profile.fft_range_oversample)
    )
    return range_axis_m[: n_fft // 2]


def sparse_unambiguous_range_m(cfg: StudyConfig) -> float:
    """Return the first unambiguous range interval implied by the sampled tones."""

    sorted_frequencies_hz = np.sort(cfg.frequencies_hz)
    if sorted_frequencies_hz.size <= 1:
        return float(fft_range_axis_m(cfg)[-1])

    max_gap_hz = float(np.max(np.diff(sorted_frequencies_hz)))
    first_interval_m = C_LIGHT_M_PER_S / (2.0 * max(max_gap_hz, 1.0e-12))
    return min(float(fft_range_axis_m(cfg)[-1]), first_interval_m)


def fft_velocity_axis_mps(cfg: StudyConfig) -> np.ndarray:
    """Return the Doppler FFT velocity axis."""

    n_fft = cfg.runtime_profile.fft_doppler_oversample * cfg.burst_profile.n_snapshots
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=cfg.anchor.slot_duration_s))
    return doppler_hz * cfg.wavelength_m / 2.0


def fft_azimuth_axis_deg(cfg: StudyConfig) -> np.ndarray:
    """Return the spatial FFT azimuth axis for the effective horizontal aperture."""

    n_fft = cfg.runtime_profile.fft_angle_oversample * cfg.effective_horizontal_positions_m.size
    spatial_sine = 2.0 * np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0))
    # Match the steering-vector convention used throughout the MUSIC pipeline.
    return -np.rad2deg(np.arcsin(np.clip(spatial_sine, -0.999, 0.999)))
