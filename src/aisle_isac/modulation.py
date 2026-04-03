"""Transmit-symbol helpers for monostatic OFDM ISAC studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisle_isac.resource_grid import ResourceGrid


SUPPORTED_MODULATION_SCHEMES = ("qpsk", "16qam")
SUPPORTED_KNOWLEDGE_MODES = ("known_symbols",)


@dataclass(frozen=True)
class CommunicationSymbolMap:
    """Transmit-symbol assignments and estimator knowledge flags."""

    modulation_scheme: str
    knowledge_mode: str
    symbols: np.ndarray
    known_symbol_mask: np.ndarray

    def __post_init__(self) -> None:
        symbols = np.asarray(self.symbols, dtype=np.complex128)
        known_symbol_mask = np.asarray(self.known_symbol_mask, dtype=bool)
        if symbols.ndim != 2:
            raise ValueError("symbols must be a 2D subcarrier-by-symbol array")
        if known_symbol_mask.shape != symbols.shape:
            raise ValueError("known_symbol_mask must match the symbol grid shape")
        object.__setattr__(self, "symbols", symbols.copy())
        object.__setattr__(self, "known_symbol_mask", known_symbol_mask.copy())


def constellation_points(modulation_scheme: str) -> np.ndarray:
    modulation_scheme = modulation_scheme.lower()
    if modulation_scheme == "qpsk":
        return np.asarray([1.0 + 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    if modulation_scheme == "16qam":
        levels = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=float)
        grid_real, grid_imag = np.meshgrid(levels, levels, indexing="ij")
        return (grid_real + 1j * grid_imag).reshape(-1).astype(np.complex128) / np.sqrt(10.0)
    supported = ", ".join(SUPPORTED_MODULATION_SCHEMES)
    raise ValueError(f"modulation_scheme must be one of {supported}")


def generate_symbol_map(
    resource_grid: ResourceGrid,
    *,
    rng: np.random.Generator,
    modulation_scheme: str = "qpsk",
    knowledge_mode: str = "known_symbols",
    pilot_symbol: complex = 1.0 + 0.0j,
) -> CommunicationSymbolMap:
    """Return the symbol field carried on one resource grid."""

    if knowledge_mode != "known_symbols":
        raise ValueError("The active thesis path supports only known_symbols sensing")

    symbol_grid = np.zeros(resource_grid.shape, dtype=np.complex128)
    symbol_grid[resource_grid.pilot_mask] = np.complex128(pilot_symbol)

    data_positions = np.count_nonzero(resource_grid.data_mask)
    if data_positions:
        points = constellation_points(modulation_scheme)
        symbol_grid[resource_grid.data_mask] = rng.choice(points, size=data_positions)

    known_symbol_mask = resource_grid.occupied_mask

    return CommunicationSymbolMap(
        modulation_scheme=modulation_scheme,
        knowledge_mode=knowledge_mode,
        symbols=symbol_grid,
        known_symbol_mask=known_symbol_mask,
    )
