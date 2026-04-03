"""Radar-cube generation for the communications-limited MUSIC study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from aisle_isac.config import ArrayGeometry, BurstProfile, C_LIGHT_M_PER_S, StudyConfig
from aisle_isac.scenarios import build_target_catalog


@dataclass(frozen=True)
class TrialParameters:
    """Geometry parameters for one Monte Carlo trial."""

    center_range_m: float
    range_separation_m: float
    velocity_separation_mps: float
    angle_separation_deg: float


@dataclass(frozen=True)
class TargetState:
    """Resolved truth target or nuisance scatterer state."""

    label: str
    target_class_name: str
    range_m: float
    velocity_mps: float
    azimuth_deg: float
    amplitude_db: float
    path_gain_linear: float


@dataclass(frozen=True)
class ScenarioState:
    """One complete scene realization."""

    anchor_name: str
    scene_class_name: str
    burst_profile_name: str
    aperture_size: int
    target_pair: tuple[str, str]
    targets: tuple[TargetState, ...]
    nuisance: tuple[TargetState, ...]
    nominal_snr_db: float


@dataclass(frozen=True)
class CubeSnapshot:
    """Simulated radar cube and the truth scenario it contains."""

    scenario: ScenarioState
    radar_cube: np.ndarray
    target_only_cube: np.ndarray
    noise_variance: float

    @property
    def horizontal_cube(self) -> np.ndarray:
        return self.radar_cube

    @property
    def target_only_horizontal_cube(self) -> np.ndarray:
        return self.target_only_cube


def _unit_modulus_scalar(rng: np.random.Generator) -> np.complex128:
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return np.complex128(np.exp(1j * phase))


def _build_target_reflectivities(cfg: StudyConfig, rng: np.random.Generator) -> np.ndarray:
    """Return one random unit-modulus reflectivity per mover.

    The project studies super-resolution over range, Doppler, and azimuth. To keep
    those tones physically clean inside one CPI, each mover is modeled with a
    trial-wise complex reflectivity scalar rather than a per-snapshot random
    source sequence.
    """

    return np.asarray([_unit_modulus_scalar(rng) for _ in range(cfg.expected_target_count)], dtype=np.complex128)


def path_amplitude(range_m: float, amplitude_db: float, wavelength_m: float = 1.0) -> float:
    """Monostatic radar-equation amplitude: sqrt(RCS) * lambda / R^2.

    Follows the radar range equation where received amplitude scales as
    lambda * sqrt(sigma) / R^2.  The wavelength term makes FR1-vs-FR2
    comparisons physically meaningful: shorter wavelength (FR2) sees
    higher free-space path loss for the same range.
    """

    return 10.0 ** (amplitude_db / 20.0) * wavelength_m / max(range_m, 1.0) ** 2


def build_truth_targets(cfg: StudyConfig, params: TrialParameters) -> tuple[TargetState, ...]:
    """Build the two mover truth states for one trial geometry."""

    catalog = build_target_catalog()
    scene = cfg.scene_class
    target_a = catalog[scene.target_pair[0]]
    target_b = catalog[scene.target_pair[1]]

    ranges_m = (
        params.center_range_m - 0.5 * params.range_separation_m,
        params.center_range_m + 0.5 * params.range_separation_m,
    )
    velocities_mps = (
        scene.nominal_center_velocity_mps - 0.5 * params.velocity_separation_mps,
        scene.nominal_center_velocity_mps + 0.5 * params.velocity_separation_mps,
    )
    azimuths_deg = (
        scene.nominal_azimuth_center_deg - 0.5 * params.angle_separation_deg,
        scene.nominal_azimuth_center_deg + 0.5 * params.angle_separation_deg,
    )

    return (
        TargetState(
            label=target_a.label,
            target_class_name=target_a.name,
            range_m=float(ranges_m[0]),
            velocity_mps=float(velocities_mps[0]),
            azimuth_deg=float(azimuths_deg[0]),
            amplitude_db=target_a.rcs_db,
            path_gain_linear=path_amplitude(float(ranges_m[0]), target_a.rcs_db, cfg.wavelength_m),
        ),
        TargetState(
            label=target_b.label,
            target_class_name=target_b.name,
            range_m=float(ranges_m[1]),
            velocity_mps=float(velocities_mps[1]),
            azimuth_deg=float(azimuths_deg[1]),
            amplitude_db=target_b.rcs_db + scene.second_target_power_offset_db,
            path_gain_linear=path_amplitude(
                float(ranges_m[1]),
                target_b.rcs_db + scene.second_target_power_offset_db,
                cfg.wavelength_m,
            ),
        ),
    )


def _build_nuisance(cfg: StudyConfig, params: TrialParameters) -> tuple[TargetState, ...]:
    scene = cfg.scene_class
    nuisance: list[TargetState] = []
    for template in scene.static_clutter + scene.multipath:
        range_m = max(2.0, params.center_range_m + template.range_offset_m)
        nuisance.append(
            TargetState(
                label=template.label,
                target_class_name="scatterer",
                range_m=range_m,
                velocity_mps=template.velocity_mps,
                azimuth_deg=scene.nominal_azimuth_center_deg + template.azimuth_offset_deg,
                amplitude_db=template.gain_db,
                path_gain_linear=path_amplitude(range_m, template.gain_db, cfg.wavelength_m),
            )
        )
    return tuple(nuisance)


def _azimuth_steering(horizontal_positions_m: np.ndarray, wavelength_m: float, azimuth_deg: float) -> np.ndarray:
    azimuth_rad = np.deg2rad(azimuth_deg)
    phase = horizontal_positions_m * np.sin(azimuth_rad)
    return np.exp(-1j * 2.0 * np.pi * phase / wavelength_m)


def _range_steering(frequencies_hz: np.ndarray, range_m: float) -> np.ndarray:
    delay_s = 2.0 * range_m / C_LIGHT_M_PER_S
    return np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)


def _doppler_steering(times_s: np.ndarray, wavelength_m: float, velocity_mps: float) -> np.ndarray:
    doppler_hz = 2.0 * velocity_mps / wavelength_m
    return np.exp(1j * 2.0 * np.pi * doppler_hz * times_s)


def _nominal_trial_parameters(cfg: StudyConfig) -> TrialParameters:
    return TrialParameters(
        center_range_m=cfg.scene_class.nominal_range_m,
        range_separation_m=cfg.scene_class.default_range_separation_cells * cfg.range_resolution_m,
        velocity_separation_mps=cfg.scene_class.default_velocity_separation_cells * cfg.velocity_resolution_mps,
        angle_separation_deg=cfg.scene_class.default_angle_separation_cells * cfg.azimuth_resolution_deg,
    )


def _baseline_noise_config(cfg: StudyConfig) -> StudyConfig:
    baseline_burst = BurstProfile(name="balanced_cpi", label="Balanced CPI", n_snapshots=16)
    baseline_geometry = ArrayGeometry(
        name="linear_2x8", n_tx=2, n_rx_rows=1, n_rx_cols=8,
        horizontal_spacing_lambda=0.5, tx_spacing_lambda=4.0,
    )
    baseline_profile = replace(cfg.runtime_profile, n_simulated_subcarriers=96)
    return replace(cfg, burst_profile=baseline_burst, array_geometry=baseline_geometry, runtime_profile=baseline_profile)


@lru_cache(maxsize=None)
def calibrated_noise_variance(cfg: StudyConfig) -> float:
    """Calibrate receiver noise once per anchor/scene at the nominal baseline point."""

    baseline_cfg = _baseline_noise_config(cfg)
    params = _nominal_trial_parameters(baseline_cfg)
    targets = build_truth_targets(baseline_cfg, params)
    positions_m = baseline_cfg.effective_horizontal_positions_m
    frequencies_hz = baseline_cfg.frequencies_hz
    times_s = baseline_cfg.snapshot_times_s
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                baseline_cfg.rng_seed,
                int(round(baseline_cfg.anchor.carrier_frequency_hz)),
                int(round(1_000.0 * baseline_cfg.scene_class.nominal_range_m)),
                baseline_cfg.array_geometry.n_rx_cols,
            ]
        )
    )
    reflectivities = _build_target_reflectivities(baseline_cfg, rng)

    target_only_cube = np.zeros(
        (positions_m.size, baseline_cfg.n_subcarriers, baseline_cfg.burst_profile.n_snapshots),
        dtype=np.complex128,
    )
    for target_index, target in enumerate(targets):
        spatial = _azimuth_steering(positions_m, baseline_cfg.wavelength_m, target.azimuth_deg)
        spectral = _range_steering(frequencies_hz, target.range_m)
        temporal = _doppler_steering(times_s, baseline_cfg.wavelength_m, target.velocity_mps)
        target_only_cube += (
            target.path_gain_linear
            * reflectivities[target_index]
            * spatial[:, np.newaxis, np.newaxis]
            * spectral[np.newaxis, :, np.newaxis]
            * temporal[np.newaxis, np.newaxis, :]
        )

    signal_power = float(np.mean(np.abs(target_only_cube) ** 2))
    snr_linear = 10.0 ** (baseline_cfg.scene_class.nominal_snr_db / 10.0)
    return max(signal_power / max(snr_linear, 1.0e-12), 1.0e-12)


def simulate_radar_cube(
    cfg: StudyConfig,
    params: TrialParameters,
    rng: np.random.Generator,
) -> CubeSnapshot:
    """Generate one noisy waveform-constrained MIMO-OFDM radar cube."""

    targets = build_truth_targets(cfg, params)
    nuisance = _build_nuisance(cfg, params)
    scenario = ScenarioState(
        anchor_name=cfg.anchor.name,
        scene_class_name=cfg.scene_class.name,
        burst_profile_name=cfg.burst_profile.name,
        aperture_size=cfg.array_geometry.n_rx_cols,
        target_pair=cfg.scene_class.target_pair,
        targets=targets,
        nuisance=nuisance,
        nominal_snr_db=cfg.scene_class.nominal_snr_db,
    )

    horizontal_positions_m = cfg.effective_horizontal_positions_m
    frequencies_hz = cfg.frequencies_hz
    times_s = cfg.snapshot_times_s
    target_reflectivities = _build_target_reflectivities(cfg, rng)

    target_only_cube = np.zeros(
        (horizontal_positions_m.size, cfg.n_subcarriers, cfg.burst_profile.n_snapshots),
        dtype=np.complex128,
    )
    for target_index, target in enumerate(targets):
        spatial = _azimuth_steering(horizontal_positions_m, cfg.wavelength_m, target.azimuth_deg)
        spectral = _range_steering(frequencies_hz, target.range_m)
        temporal = _doppler_steering(times_s, cfg.wavelength_m, target.velocity_mps)
        target_only_cube += (
            target.path_gain_linear
            * target_reflectivities[target_index]
            * spatial[:, np.newaxis, np.newaxis]
            * spectral[np.newaxis, :, np.newaxis]
            * temporal[np.newaxis, np.newaxis, :]
        )

    nuisance_cube = np.zeros_like(target_only_cube)
    for nuisance_index, path in enumerate(nuisance):
        spatial = _azimuth_steering(horizontal_positions_m, cfg.wavelength_m, path.azimuth_deg)
        spectral = _range_steering(frequencies_hz, path.range_m)
        temporal = _doppler_steering(times_s, cfg.wavelength_m, path.velocity_mps)
        template = (cfg.scene_class.static_clutter + cfg.scene_class.multipath)[nuisance_index]
        if template.coherent_with_target_index is not None:
            reflectivity = target_reflectivities[template.coherent_with_target_index]
        else:
            reflectivity = _unit_modulus_scalar(rng)
        nuisance_cube += (
            path.path_gain_linear
            * reflectivity
            * spatial[:, np.newaxis, np.newaxis]
            * spectral[np.newaxis, :, np.newaxis]
            * temporal[np.newaxis, np.newaxis, :]
        )

    noise_variance = calibrated_noise_variance(cfg)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.normal(size=target_only_cube.shape) + 1j * rng.normal(size=target_only_cube.shape)
    )
    radar_cube = target_only_cube + nuisance_cube + noise

    return CubeSnapshot(
        scenario=scenario,
        radar_cube=radar_cube,
        target_only_cube=target_only_cube,
        noise_variance=noise_variance,
    )
