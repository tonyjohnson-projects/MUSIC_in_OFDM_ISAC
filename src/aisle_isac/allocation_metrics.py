"""Summary metrics for communications-scheduled OFDM resource grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisle_isac.resource_grid import ResourceGrid


@dataclass(frozen=True)
class AllocationSummary:
    """Compact summary of one resource-grid allocation."""

    occupied_fraction: float
    pilot_fraction: float
    contiguous_bandwidth_span_subcarriers: int
    contiguous_bandwidth_span_fraction: float
    slow_time_span_symbols: int
    slow_time_span_fraction: float
    fragmentation_index: float


def occupied_fraction(resource_grid: ResourceGrid) -> float:
    return float(np.mean(resource_grid.occupied_mask))


def pilot_fraction(resource_grid: ResourceGrid) -> float:
    occupied_count = int(np.count_nonzero(resource_grid.occupied_mask))
    if occupied_count == 0:
        return 0.0
    return float(np.count_nonzero(resource_grid.pilot_mask) / occupied_count)


def contiguous_bandwidth_span_subcarriers(resource_grid: ResourceGrid) -> int:
    active_subcarriers = np.flatnonzero(np.any(resource_grid.occupied_mask, axis=1))
    if active_subcarriers.size == 0:
        return 0
    return int(active_subcarriers[-1] - active_subcarriers[0] + 1)


def slow_time_span_symbols(resource_grid: ResourceGrid) -> int:
    active_symbols = np.flatnonzero(np.any(resource_grid.occupied_mask, axis=0))
    if active_symbols.size == 0:
        return 0
    return int(active_symbols[-1] - active_symbols[0] + 1)


def fragmentation_index(resource_grid: ResourceGrid) -> float:
    """Return a simple occupancy-edge density over frequency and slow time."""

    occupied = resource_grid.occupied_mask.astype(np.int8)
    frequency_edges = np.abs(np.diff(occupied, axis=0))
    symbol_edges = np.abs(np.diff(occupied, axis=1))
    frequency_term = float(np.mean(frequency_edges)) if frequency_edges.size else 0.0
    symbol_term = float(np.mean(symbol_edges)) if symbol_edges.size else 0.0
    return 0.5 * (frequency_term + symbol_term)


def summarize_allocation(resource_grid: ResourceGrid) -> AllocationSummary:
    bandwidth_span = contiguous_bandwidth_span_subcarriers(resource_grid)
    slow_time_span = slow_time_span_symbols(resource_grid)
    return AllocationSummary(
        occupied_fraction=occupied_fraction(resource_grid),
        pilot_fraction=pilot_fraction(resource_grid),
        contiguous_bandwidth_span_subcarriers=bandwidth_span,
        contiguous_bandwidth_span_fraction=bandwidth_span / resource_grid.n_subcarriers,
        slow_time_span_symbols=slow_time_span,
        slow_time_span_fraction=slow_time_span / resource_grid.n_symbols,
        fragmentation_index=fragmentation_index(resource_grid),
    )

