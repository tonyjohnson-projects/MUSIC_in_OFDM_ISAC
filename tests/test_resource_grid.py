"""Tests for communications-scheduled OFDM resource-grid primitives."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.allocation_metrics import summarize_allocation
from aisle_isac.masked_observation import extract_known_symbol_cube, simulate_masked_observation
from aisle_isac.modulation import generate_symbol_map
from aisle_isac.resource_grid import build_resource_grid
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import nominal_trial_parameters


class ResourceGridTest(unittest.TestCase):
    """Verify the new irregular-grid OFDM building blocks."""

    def test_supported_allocation_families_have_expected_roles(self) -> None:
        full_grid = build_resource_grid("full_grid", 48, 8)
        self.assertEqual(full_grid.role_counts()["pilot"], 48 * 8)
        self.assertEqual(full_grid.role_counts()["data"], 0)

        comb_grid = build_resource_grid("comb_pilot", 48, 8, pilot_subcarrier_period=4, pilot_symbol_period=2)
        self.assertGreater(comb_grid.role_counts()["pilot"], 0)
        self.assertEqual(comb_grid.role_counts()["data"], 0)
        self.assertGreater(comb_grid.role_counts()["muted"], 0)

        fragmented_grid = build_resource_grid(
            "fragmented_prb",
            48,
            8,
            prb_size=12,
            n_prb_fragments=2,
            pilot_subcarrier_period=4,
            pilot_symbol_period=2,
        )
        self.assertGreater(fragmented_grid.role_counts()["pilot"], 0)
        self.assertGreater(fragmented_grid.role_counts()["data"], 0)
        self.assertGreater(fragmented_grid.role_counts()["muted"], 0)

        punctured_grid = build_resource_grid(
            "punctured_grid",
            48,
            8,
            puncture_fraction=0.2,
        )
        self.assertGreater(punctured_grid.role_counts()["punctured"], 0)

    def test_fragmentation_metric_separates_dense_and_sparse_allocations(self) -> None:
        full_summary = summarize_allocation(build_resource_grid("full_grid", 48, 8))
        fragmented_summary = summarize_allocation(
            build_resource_grid("fragmented_prb", 48, 8, prb_size=12, n_prb_fragments=2)
        )
        comb_summary = summarize_allocation(
            build_resource_grid("comb_pilot", 48, 8, pilot_subcarrier_period=4, pilot_symbol_period=2)
        )

        self.assertEqual(full_summary.fragmentation_index, 0.0)
        self.assertGreater(fragmented_summary.fragmentation_index, full_summary.fragmentation_index)
        self.assertGreater(comb_summary.fragmentation_index, fragmented_summary.fragmentation_index)

    def test_known_symbol_mode_marks_every_occupied_re_as_known(self) -> None:
        resource_grid = build_resource_grid("pilot_plus_data", 48, 8, pilot_subcarrier_period=4, pilot_symbol_period=2)
        known_symbols = generate_symbol_map(
            resource_grid,
            rng=np.random.default_rng(1),
            modulation_scheme="qpsk",
            knowledge_mode="known_symbols",
        )

        self.assertTrue(np.array_equal(known_symbols.known_symbol_mask, resource_grid.occupied_mask))
        self.assertTrue(np.allclose(known_symbols.symbols[resource_grid.pilot_mask], 1.0 + 0.0j))
        self.assertTrue(np.any(known_symbols.symbols[resource_grid.data_mask] != 0.0))

    def test_pilot_only_mode_marks_only_pilots_as_known(self) -> None:
        resource_grid = build_resource_grid("pilot_plus_data", 48, 8, pilot_subcarrier_period=4, pilot_symbol_period=2)
        symbol_map = generate_symbol_map(
            resource_grid,
            rng=np.random.default_rng(2),
            modulation_scheme="qpsk",
            knowledge_mode="pilot_only",
        )

        self.assertTrue(np.array_equal(symbol_map.known_symbol_mask, resource_grid.pilot_mask))
        self.assertTrue(np.allclose(symbol_map.symbols[resource_grid.pilot_mask], 1.0 + 0.0j))
        self.assertTrue(np.any(symbol_map.symbols[resource_grid.data_mask] != 0.0))

    def test_masked_observation_zero_fills_unoccupied_re_and_recovers_known_symbols(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        resource_grid = build_resource_grid(
            "fragmented_prb",
            cfg.n_subcarriers,
            cfg.burst_profile.n_snapshots,
            prb_size=12,
            n_prb_fragments=2,
            pilot_subcarrier_period=4,
            pilot_symbol_period=4,
        )
        masked_observation = simulate_masked_observation(
            cfg,
            nominal_trial_parameters(cfg),
            resource_grid,
            rng=np.random.default_rng(7),
            modulation_scheme="qpsk",
            knowledge_mode="known_symbols",
        )
        known_cube = extract_known_symbol_cube(masked_observation)

        self.assertEqual(masked_observation.measurement_cube.shape[0], cfg.effective_horizontal_positions_m.size)
        self.assertTrue(np.allclose(masked_observation.measurement_cube[:, ~resource_grid.occupied_mask], 0.0))
        self.assertTrue(np.allclose(known_cube[:, ~masked_observation.known_symbol_mask], 0.0))
        self.assertTrue(
            np.allclose(
                known_cube[:, resource_grid.occupied_mask],
                masked_observation.snapshot.radar_cube[:, resource_grid.occupied_mask],
            )
        )
