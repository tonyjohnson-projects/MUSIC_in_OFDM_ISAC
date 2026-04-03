"""Communications-scheduled OFDM resource-grid abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


ALLOCATION_FAMILIES = (
    "full_grid",
    "comb_pilot",
    "block_pilot",
    "fragmented_prb",
    "pilot_plus_data",
    "punctured_grid",
)


class ResourceElementRole(IntEnum):
    """Discrete tag assigned to one OFDM resource element."""

    MUTED = 0
    PILOT = 1
    DATA = 2
    PUNCTURED = 3


ROLE_LABELS = {
    ResourceElementRole.MUTED: "muted",
    ResourceElementRole.PILOT: "pilot",
    ResourceElementRole.DATA: "data",
    ResourceElementRole.PUNCTURED: "punctured",
}
_VALID_ROLE_CODES = tuple(int(role) for role in ResourceElementRole)
_OCCUPIED_ROLE_CODES = np.asarray(
    [int(ResourceElementRole.PILOT), int(ResourceElementRole.DATA)],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ResourceGrid:
    """2D communications scheduling mask over subcarriers and OFDM symbols."""

    allocation_family: str
    role_grid: np.ndarray

    def __post_init__(self) -> None:
        role_grid = np.asarray(self.role_grid, dtype=np.uint8)
        if role_grid.ndim != 2:
            raise ValueError("role_grid must be a 2D array over subcarrier and symbol index")
        if role_grid.size == 0:
            raise ValueError("role_grid must not be empty")
        if np.any(~np.isin(role_grid, _VALID_ROLE_CODES)):
            raise ValueError("role_grid contains unsupported resource-element roles")
        object.__setattr__(self, "role_grid", role_grid.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.role_grid.shape

    @property
    def n_subcarriers(self) -> int:
        return int(self.role_grid.shape[0])

    @property
    def n_symbols(self) -> int:
        return int(self.role_grid.shape[1])

    @property
    def pilot_mask(self) -> np.ndarray:
        return self.role_grid == int(ResourceElementRole.PILOT)

    @property
    def data_mask(self) -> np.ndarray:
        return self.role_grid == int(ResourceElementRole.DATA)

    @property
    def muted_mask(self) -> np.ndarray:
        return self.role_grid == int(ResourceElementRole.MUTED)

    @property
    def punctured_mask(self) -> np.ndarray:
        return self.role_grid == int(ResourceElementRole.PUNCTURED)

    @property
    def occupied_mask(self) -> np.ndarray:
        return np.isin(self.role_grid, _OCCUPIED_ROLE_CODES)

    @property
    def available_sensing_mask(self) -> np.ndarray:
        return self.occupied_mask

    def role_counts(self) -> dict[str, int]:
        return {
            "pilot": int(np.count_nonzero(self.pilot_mask)),
            "data": int(np.count_nonzero(self.data_mask)),
            "muted": int(np.count_nonzero(self.muted_mask)),
            "punctured": int(np.count_nonzero(self.punctured_mask)),
        }


def _new_role_grid(n_subcarriers: int, n_symbols: int) -> np.ndarray:
    if n_subcarriers < 1 or n_symbols < 1:
        raise ValueError("n_subcarriers and n_symbols must both be positive")
    return np.full(
        (n_subcarriers, n_symbols),
        fill_value=np.uint8(ResourceElementRole.MUTED),
        dtype=np.uint8,
    )


def _periodic_indices(length: int, period: int, offset: int = 0) -> np.ndarray:
    if period < 1:
        raise ValueError("period must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return np.arange(offset, length, period, dtype=int)


def _evenly_spaced_starts(length: int, block_width: int, count: int) -> np.ndarray:
    if block_width < 1:
        raise ValueError("block_width must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    if block_width > length:
        raise ValueError("block_width cannot exceed the available length")
    if count == 1:
        return np.asarray([0], dtype=int)
    max_start = length - block_width
    return np.unique(np.round(np.linspace(0, max_start, num=count)).astype(int))


def _comb_pilot_mask(
    n_subcarriers: int,
    n_symbols: int,
    pilot_subcarrier_period: int,
    pilot_symbol_period: int,
    pilot_subcarrier_offset: int = 0,
    pilot_symbol_offset: int = 0,
) -> np.ndarray:
    pilot_mask = np.zeros((n_subcarriers, n_symbols), dtype=bool)
    subcarrier_indices = _periodic_indices(n_subcarriers, pilot_subcarrier_period, pilot_subcarrier_offset)
    symbol_indices = _periodic_indices(n_symbols, pilot_symbol_period, pilot_symbol_offset)
    if subcarrier_indices.size == 0 or symbol_indices.size == 0:
        return pilot_mask
    pilot_mask[np.ix_(subcarrier_indices, symbol_indices)] = True
    return pilot_mask


def build_full_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    role: ResourceElementRole = ResourceElementRole.PILOT,
) -> ResourceGrid:
    """Return a dense sensing-friendly grid with one role everywhere."""

    if role not in (ResourceElementRole.PILOT, ResourceElementRole.DATA):
        raise ValueError("full-grid role must be either PILOT or DATA")
    role_grid = _new_role_grid(n_subcarriers, n_symbols)
    role_grid[:, :] = np.uint8(role)
    return ResourceGrid(allocation_family="full_grid", role_grid=role_grid)


def build_comb_pilot_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    pilot_subcarrier_period: int = 4,
    pilot_symbol_period: int = 4,
    pilot_subcarrier_offset: int = 0,
    pilot_symbol_offset: int = 0,
) -> ResourceGrid:
    """Return a sparse grid with pilots on a regular comb pattern."""

    role_grid = _new_role_grid(n_subcarriers, n_symbols)
    pilot_mask = _comb_pilot_mask(
        n_subcarriers,
        n_symbols,
        pilot_subcarrier_period,
        pilot_symbol_period,
        pilot_subcarrier_offset,
        pilot_symbol_offset,
    )
    role_grid[pilot_mask] = np.uint8(ResourceElementRole.PILOT)
    return ResourceGrid(allocation_family="comb_pilot", role_grid=role_grid)


def build_block_pilot_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    block_width_subcarriers: int = 12,
    block_symbol_span: int = 4,
    n_frequency_blocks: int = 2,
) -> ResourceGrid:
    """Return a sparse grid with contiguous pilot blocks."""

    if block_symbol_span < 1:
        raise ValueError("block_symbol_span must be positive")
    role_grid = _new_role_grid(n_subcarriers, n_symbols)
    symbol_stop = min(block_symbol_span, n_symbols)
    starts = _evenly_spaced_starts(n_subcarriers, block_width_subcarriers, n_frequency_blocks)
    for start in starts:
        stop = min(start + block_width_subcarriers, n_subcarriers)
        role_grid[start:stop, :symbol_stop] = np.uint8(ResourceElementRole.PILOT)
    return ResourceGrid(allocation_family="block_pilot", role_grid=role_grid)


def build_fragmented_prb_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    prb_size: int = 12,
    n_prb_fragments: int = 3,
    pilot_subcarrier_period: int = 4,
    pilot_symbol_period: int = 4,
    active_symbol_indices: tuple[int, ...] | None = None,
) -> ResourceGrid:
    """Return a communications-style fragmented PRB allocation."""

    if prb_size < 1:
        raise ValueError("prb_size must be positive")
    if n_prb_fragments < 1:
        raise ValueError("n_prb_fragments must be positive")
    role_grid = _new_role_grid(n_subcarriers, n_symbols)
    total_prbs = int(np.ceil(n_subcarriers / prb_size))
    chosen_prbs = _evenly_spaced_starts(total_prbs, 1, min(n_prb_fragments, total_prbs))
    if active_symbol_indices is None:
        symbol_indices = np.arange(n_symbols, dtype=int)
    else:
        symbol_indices = np.asarray(sorted(set(active_symbol_indices)), dtype=int)
        if symbol_indices.size == 0:
            raise ValueError("active_symbol_indices must not be empty")
        if np.any(symbol_indices < 0) or np.any(symbol_indices >= n_symbols):
            raise ValueError("active_symbol_indices contains an out-of-range symbol index")
    for prb_index in chosen_prbs:
        start = int(prb_index * prb_size)
        stop = min(start + prb_size, n_subcarriers)
        subcarrier_indices = np.arange(start, stop, dtype=int)
        role_grid[np.ix_(subcarrier_indices, symbol_indices)] = np.uint8(ResourceElementRole.DATA)

    pilot_mask = _comb_pilot_mask(
        n_subcarriers,
        n_symbols,
        pilot_subcarrier_period,
        pilot_symbol_period,
    ) & (role_grid == int(ResourceElementRole.DATA))
    role_grid[pilot_mask] = np.uint8(ResourceElementRole.PILOT)
    return ResourceGrid(allocation_family="fragmented_prb", role_grid=role_grid)


def build_pilot_plus_data_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    pilot_subcarrier_period: int = 4,
    pilot_symbol_period: int = 4,
) -> ResourceGrid:
    """Return a dense allocation with pilot REs embedded in data REs."""

    role_grid = _new_role_grid(n_subcarriers, n_symbols)
    role_grid[:, :] = np.uint8(ResourceElementRole.DATA)
    pilot_mask = _comb_pilot_mask(
        n_subcarriers,
        n_symbols,
        pilot_subcarrier_period,
        pilot_symbol_period,
    )
    role_grid[pilot_mask] = np.uint8(ResourceElementRole.PILOT)
    return ResourceGrid(allocation_family="pilot_plus_data", role_grid=role_grid)


def build_punctured_grid(
    n_subcarriers: int,
    n_symbols: int,
    *,
    puncture_fraction: float = 0.15,
    puncture_base_family: str = "pilot_plus_data",
    pilot_subcarrier_period: int = 4,
    pilot_symbol_period: int = 4,
    prb_size: int = 12,
    n_prb_fragments: int = 3,
) -> ResourceGrid:
    """Return an occupied grid with deterministic holes punched into it."""

    if puncture_fraction < 0.0 or puncture_fraction > 1.0:
        raise ValueError("puncture_fraction must lie in [0, 1]")
    if puncture_base_family == "punctured_grid":
        raise ValueError("puncture_base_family must not recurse into punctured_grid")

    base_grid = build_resource_grid(
        puncture_base_family,
        n_subcarriers,
        n_symbols,
        pilot_subcarrier_period=pilot_subcarrier_period,
        pilot_symbol_period=pilot_symbol_period,
        prb_size=prb_size,
        n_prb_fragments=n_prb_fragments,
    )
    role_grid = base_grid.role_grid.copy()
    occupied_indices = np.argwhere(base_grid.occupied_mask)
    if occupied_indices.size == 0 or puncture_fraction == 0.0:
        return ResourceGrid(allocation_family="punctured_grid", role_grid=role_grid)

    puncture_count = int(round(puncture_fraction * occupied_indices.shape[0]))
    if puncture_count == 0:
        puncture_count = 1
    puncture_count = min(puncture_count, occupied_indices.shape[0])
    selected_positions = np.unique(
        np.round(np.linspace(0, occupied_indices.shape[0] - 1, num=puncture_count)).astype(int)
    )
    punctures = occupied_indices[selected_positions]
    role_grid[punctures[:, 0], punctures[:, 1]] = np.uint8(ResourceElementRole.PUNCTURED)
    return ResourceGrid(allocation_family="punctured_grid", role_grid=role_grid)


def build_resource_grid(
    allocation_family: str,
    n_subcarriers: int,
    n_symbols: int,
    *,
    pilot_subcarrier_period: int = 4,
    pilot_symbol_period: int = 4,
    pilot_subcarrier_offset: int = 0,
    pilot_symbol_offset: int = 0,
    block_width_subcarriers: int = 12,
    block_symbol_span: int = 4,
    n_frequency_blocks: int = 2,
    prb_size: int = 12,
    n_prb_fragments: int = 3,
    puncture_fraction: float = 0.15,
    puncture_base_family: str = "pilot_plus_data",
    full_grid_role: ResourceElementRole = ResourceElementRole.PILOT,
    active_symbol_indices: tuple[int, ...] | None = None,
) -> ResourceGrid:
    """Dispatch to one of the supported allocation-family builders."""

    if allocation_family == "full_grid":
        return build_full_grid(n_subcarriers, n_symbols, role=full_grid_role)
    if allocation_family == "comb_pilot":
        return build_comb_pilot_grid(
            n_subcarriers,
            n_symbols,
            pilot_subcarrier_period=pilot_subcarrier_period,
            pilot_symbol_period=pilot_symbol_period,
            pilot_subcarrier_offset=pilot_subcarrier_offset,
            pilot_symbol_offset=pilot_symbol_offset,
        )
    if allocation_family == "block_pilot":
        return build_block_pilot_grid(
            n_subcarriers,
            n_symbols,
            block_width_subcarriers=block_width_subcarriers,
            block_symbol_span=block_symbol_span,
            n_frequency_blocks=n_frequency_blocks,
        )
    if allocation_family == "fragmented_prb":
        return build_fragmented_prb_grid(
            n_subcarriers,
            n_symbols,
            prb_size=prb_size,
            n_prb_fragments=n_prb_fragments,
            pilot_subcarrier_period=pilot_subcarrier_period,
            pilot_symbol_period=pilot_symbol_period,
            active_symbol_indices=active_symbol_indices,
        )
    if allocation_family == "pilot_plus_data":
        return build_pilot_plus_data_grid(
            n_subcarriers,
            n_symbols,
            pilot_subcarrier_period=pilot_subcarrier_period,
            pilot_symbol_period=pilot_symbol_period,
        )
    if allocation_family == "punctured_grid":
        return build_punctured_grid(
            n_subcarriers,
            n_symbols,
            puncture_fraction=puncture_fraction,
            puncture_base_family=puncture_base_family,
            pilot_subcarrier_period=pilot_subcarrier_period,
            pilot_symbol_period=pilot_symbol_period,
            prb_size=prb_size,
            n_prb_fragments=n_prb_fragments,
        )
    supported = ", ".join(ALLOCATION_FAMILIES)
    raise ValueError(f"allocation_family must be one of {supported}")

