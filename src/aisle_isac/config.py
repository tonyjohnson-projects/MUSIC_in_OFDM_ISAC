"""Configuration dataclasses for the communications-limited MUSIC study."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import math

import numpy as np


C_LIGHT_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class RuntimeProfile:
    """Runtime knobs that control study scale and estimator density."""

    name: str
    n_trials: int
    n_simulated_subcarriers: int
    fft_range_oversample: int
    fft_doppler_oversample: int
    fft_angle_oversample: int
    music_grid_points: int
    coarse_candidate_count: int


@dataclass(frozen=True)
class WaveformAnchor:
    """Private-5G waveform anchor used by the sensing study."""

    name: str
    label: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    subcarrier_spacing_hz: float

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT_M_PER_S / self.carrier_frequency_hz

    @property
    def physical_subcarrier_count(self) -> int:
        return int(round(self.bandwidth_hz / self.subcarrier_spacing_hz))

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.physical_subcarrier_count * self.subcarrier_spacing_hz

    @property
    def slot_duration_s(self) -> float:
        return 1.0e-3 * 15.0e3 / self.subcarrier_spacing_hz

    @property
    def range_resolution_m(self) -> float:
        return C_LIGHT_M_PER_S / (2.0 * self.occupied_bandwidth_hz)


@dataclass(frozen=True)
class BurstProfile:
    """Named NR-like burst profile expressed as reference-bearing snapshots."""

    name: str
    label: str
    n_snapshots: int

    def cpi_s(self, anchor: WaveformAnchor) -> float:
        return self.n_snapshots * anchor.slot_duration_s


@dataclass(frozen=True)
class ArrayGeometry:
    """Horizontal monostatic TDM-MIMO array geometry."""

    name: str
    n_tx: int
    n_rx_rows: int
    n_rx_cols: int
    horizontal_spacing_lambda: float = 0.5
    tx_spacing_lambda: float = 0.5

    def rx_positions_m(self, wavelength_m: float) -> np.ndarray:
        return np.arange(self.n_rx_cols, dtype=float) * self.horizontal_spacing_lambda * wavelength_m

    def tx_positions_m(self, wavelength_m: float) -> np.ndarray:
        return np.arange(self.n_tx, dtype=float) * self.tx_spacing_lambda * wavelength_m

    def virtual_positions_m(self, wavelength_m: float) -> np.ndarray:
        rx_positions = self.rx_positions_m(wavelength_m)
        tx_positions = self.tx_positions_m(wavelength_m)
        positions = []
        for tx_position in tx_positions:
            for rx_position in rx_positions:
                positions.append(tx_position + rx_position)
        return np.asarray(positions, dtype=float)

    def effective_horizontal_positions_m(self, wavelength_m: float) -> np.ndarray:
        x_positions = np.round(self.virtual_positions_m(wavelength_m), decimals=12)
        return np.unique(x_positions)


@dataclass(frozen=True)
class TargetClass:
    """Industrial mover class used to instantiate truth targets."""

    name: str
    label: str
    rcs_db: float
    height_m: float
    nominal_speed_mps: float


@dataclass(frozen=True)
class ScattererTemplate:
    """Static or dynamic nuisance reflector template."""

    label: str
    range_offset_m: float
    azimuth_offset_deg: float
    gain_db: float
    velocity_mps: float = 0.0
    coherent_with_target_index: int | None = None


@dataclass(frozen=True)
class SceneClass:
    """Indoor deployment class with fixed geometry and clutter semantics."""

    name: str
    label: str
    target_pair: tuple[str, str]
    nominal_range_m: float
    nominal_azimuth_center_deg: float
    nominal_center_velocity_mps: float
    nominal_snr_db: float
    default_range_separation_cells: float
    default_velocity_separation_cells: float
    default_angle_separation_cells: float
    second_target_power_offset_db: float
    base_station_height_m: float
    target_coherence: float
    center_range_jitter_m: float
    range_separation_jitter_cells: float
    velocity_separation_jitter_cells: float
    angle_separation_jitter_cells: float
    target_amplitude_jitter_db: float
    nuisance_gain_jitter_db: float
    static_clutter: tuple[ScattererTemplate, ...]
    multipath: tuple[ScattererTemplate, ...]


@dataclass(frozen=True)
class StudySweep:
    """Definition for one public study axis."""

    name: str
    label: str
    parameter_name: str
    parameter_label: str
    parameter_values: tuple[float | str, ...]


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem locations for generated artifacts."""

    root_dir: Path
    data_dirname: str = "data"
    figures_dirname: str = "figures"

    @property
    def data_dir(self) -> Path:
        return self.root_dir / self.data_dirname

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / self.figures_dirname


@dataclass(frozen=True)
class StudyConfig:
    """Resolved configuration for one anchor, scene, burst, and aperture choice."""

    anchor: WaveformAnchor
    burst_profile: BurstProfile
    array_geometry: ArrayGeometry
    scene_class: SceneClass
    runtime_profile: RuntimeProfile
    output_config: OutputConfig
    sweep_suite: str
    evidence_profile_name: str
    expected_target_count: int = 2
    fbss_fraction: float = 0.67
    detection_nms_radius_cells: float = 0.9
    detector_threshold_scale: float = 7.5
    detector_backfill_pool_size: int = 128
    resolution_cell_fraction: float = 0.35
    target_coherence: float = 0.0
    source_temporal_correlation: float = 0.985
    center_range_jitter_m: float = 0.0
    range_separation_jitter_cells: float = 0.0
    velocity_separation_jitter_cells: float = 0.0
    angle_separation_jitter_cells: float = 0.0
    target_amplitude_jitter_db: float = 0.0
    nuisance_gain_jitter_db: float = 0.0
    music_azimuth_peak_factor: int = 3
    music_range_peak_pool: int = 4
    music_range_fbss_fraction: float = 0.67
    music_doppler_fbss_fraction: float = 0.67
    rng_seed: int = 20_260_331

    @cached_property
    def n_subcarriers(self) -> int:
        return self.runtime_profile.n_simulated_subcarriers

    @cached_property
    def wavelength_m(self) -> float:
        return self.anchor.wavelength_m

    @cached_property
    def physical_frequencies_hz(self) -> np.ndarray:
        indices = np.arange(self.anchor.physical_subcarrier_count, dtype=float)
        centered = indices - 0.5 * (self.anchor.physical_subcarrier_count - 1)
        return centered * self.anchor.subcarrier_spacing_hz

    @cached_property
    def simulated_subcarrier_indices(self) -> np.ndarray:
        if self.n_subcarriers >= self.anchor.physical_subcarrier_count:
            return np.arange(self.anchor.physical_subcarrier_count, dtype=int)
        return np.unique(
            np.round(
                np.linspace(0, self.anchor.physical_subcarrier_count - 1, self.n_subcarriers),
            ).astype(int)
        )

    @cached_property
    def frequencies_hz(self) -> np.ndarray:
        return self.physical_frequencies_hz[self.simulated_subcarrier_indices]

    @cached_property
    def sampled_occupied_bandwidth_hz(self) -> float:
        if self.frequencies_hz.size <= 1:
            return self.anchor.subcarrier_spacing_hz
        return float(np.max(self.frequencies_hz) - np.min(self.frequencies_hz) + self.anchor.subcarrier_spacing_hz)

    @cached_property
    def snapshot_times_s(self) -> np.ndarray:
        return np.arange(self.burst_profile.n_snapshots, dtype=float) * self.anchor.slot_duration_s

    @cached_property
    def cpi_s(self) -> float:
        return self.burst_profile.cpi_s(self.anchor)

    @cached_property
    def range_resolution_m(self) -> float:
        return C_LIGHT_M_PER_S / (2.0 * max(self.sampled_occupied_bandwidth_hz, self.anchor.subcarrier_spacing_hz))

    @cached_property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * max(self.cpi_s, 1.0e-12))

    @cached_property
    def virtual_positions_m(self) -> np.ndarray:
        return self.array_geometry.virtual_positions_m(self.wavelength_m)

    @cached_property
    def effective_horizontal_positions_m(self) -> np.ndarray:
        return np.unique(np.round(self.virtual_positions_m, decimals=12))

    @cached_property
    def horizontal_spacing_m(self) -> float:
        positions = self.effective_horizontal_positions_m
        if positions.size < 2:
            return 0.5 * self.wavelength_m
        return float(np.median(np.diff(positions)))

    @cached_property
    def horizontal_aperture_m(self) -> float:
        positions = self.effective_horizontal_positions_m
        if positions.size < 2:
            return self.horizontal_spacing_m
        return float(positions[-1] - positions[0])

    @cached_property
    def azimuth_resolution_deg(self) -> float:
        aperture = max(self.horizontal_aperture_m, 1.0e-9)
        beamwidth_rad = 0.886 * self.wavelength_m / aperture
        return float(np.rad2deg(max(beamwidth_rad, math.radians(0.5))))

    @cached_property
    def fbss_subarray_len(self) -> int:
        n_channels = self.effective_horizontal_positions_m.size
        return max(3, min(n_channels - 1, int(round(self.fbss_fraction * n_channels))))
