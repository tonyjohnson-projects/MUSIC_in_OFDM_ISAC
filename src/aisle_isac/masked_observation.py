"""Masked observation synthesis for communications-scheduled OFDM ISAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisle_isac.channel_models import CubeSnapshot, TrialParameters, simulate_radar_cube
from aisle_isac.config import StudyConfig
from aisle_isac.modulation import CommunicationSymbolMap, generate_symbol_map
from aisle_isac.resource_grid import ResourceGrid


@dataclass(frozen=True)
class MaskedObservation:
    """Irregular-grid observation derived from a full MIMO-OFDM cube."""

    snapshot: CubeSnapshot
    resource_grid: ResourceGrid
    symbol_map: CommunicationSymbolMap
    measurement_cube: np.ndarray
    target_only_measurement_cube: np.ndarray

    def __post_init__(self) -> None:
        measurement_cube = np.asarray(self.measurement_cube, dtype=np.complex128)
        target_only_measurement_cube = np.asarray(self.target_only_measurement_cube, dtype=np.complex128)
        if measurement_cube.ndim != 3:
            raise ValueError("measurement_cube must be antenna-by-subcarrier-by-symbol")
        if target_only_measurement_cube.shape != measurement_cube.shape:
            raise ValueError("target_only_measurement_cube must match measurement_cube")
        if measurement_cube.shape[1:] != self.resource_grid.shape:
            raise ValueError("resource_grid shape must match the subcarrier and symbol dimensions")
        if self.symbol_map.symbols.shape != self.resource_grid.shape:
            raise ValueError("symbol_map shape must match resource_grid")
        object.__setattr__(self, "measurement_cube", measurement_cube.copy())
        object.__setattr__(self, "target_only_measurement_cube", target_only_measurement_cube.copy())

    @property
    def availability_mask(self) -> np.ndarray:
        return self.resource_grid.available_sensing_mask

    @property
    def known_symbol_mask(self) -> np.ndarray:
        return self.symbol_map.known_symbol_mask

    @property
    def noise_variance(self) -> float:
        return float(self.snapshot.noise_variance)


def apply_resource_grid(
    radar_cube: np.ndarray,
    resource_grid: ResourceGrid,
    symbol_map: CommunicationSymbolMap,
) -> np.ndarray:
    """Apply occupancy and modulation to a full OFDM radar cube."""

    radar_cube = np.asarray(radar_cube, dtype=np.complex128)
    if radar_cube.ndim != 3:
        raise ValueError("radar_cube must be antenna-by-subcarrier-by-symbol")
    if radar_cube.shape[1:] != resource_grid.shape:
        raise ValueError("resource_grid shape must match the cube subcarrier and symbol dimensions")
    if symbol_map.symbols.shape != resource_grid.shape:
        raise ValueError("symbol_map shape must match resource_grid")
    return (
        radar_cube
        * resource_grid.available_sensing_mask[np.newaxis, :, :]
        * symbol_map.symbols[np.newaxis, :, :]
    )


def extract_known_symbol_cube(masked_observation: MaskedObservation, *, fill_value: complex = 0.0j) -> np.ndarray:
    """De-embed known transmit symbols and zero-fill unknown REs."""

    recovered_cube = np.full_like(masked_observation.measurement_cube, np.complex128(fill_value))
    if not np.any(masked_observation.known_symbol_mask):
        return recovered_cube
    known_symbols = masked_observation.symbol_map.symbols[masked_observation.known_symbol_mask]
    recovered_cube[:, masked_observation.known_symbol_mask] = (
        masked_observation.measurement_cube[:, masked_observation.known_symbol_mask]
        / known_symbols[np.newaxis, :]
    )
    return recovered_cube


def simulate_masked_observation(
    cfg: StudyConfig,
    params: TrialParameters,
    resource_grid: ResourceGrid,
    *,
    rng: np.random.Generator,
    modulation_scheme: str = "qpsk",
    knowledge_mode: str = "known_symbols",
) -> MaskedObservation:
    """Generate one communications-scheduled masked observation."""

    if resource_grid.shape != (cfg.n_subcarriers, cfg.burst_profile.n_snapshots):
        raise ValueError("resource_grid shape must match the active study configuration")

    full_snapshot = simulate_radar_cube(cfg, params, rng)
    symbol_map = generate_symbol_map(
        resource_grid,
        rng=rng,
        modulation_scheme=modulation_scheme,
        knowledge_mode=knowledge_mode,
    )
    measurement_cube = apply_resource_grid(full_snapshot.radar_cube, resource_grid, symbol_map)
    target_only_measurement_cube = apply_resource_grid(full_snapshot.target_only_cube, resource_grid, symbol_map)
    return MaskedObservation(
        snapshot=full_snapshot,
        resource_grid=resource_grid,
        symbol_map=symbol_map,
        measurement_cube=measurement_cube,
        target_only_measurement_cube=target_only_measurement_cube,
    )

