"""Tests for the masked FFT baseline on irregular OFDM grids."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.estimators import _doppler_music_covariance, _range_music_covariance, azimuth_steering_matrix, build_fft_cube, extract_candidates_from_fft
from aisle_isac.estimators_fft_masked import build_masked_fft_cube, prepare_masked_frontend
from aisle_isac.masked_observation import extract_known_symbol_cube, simulate_masked_observation
from aisle_isac.resource_grid import build_resource_grid
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import nominal_trial_parameters


class MaskedFftBaselineTest(unittest.TestCase):
    """Verify masked FFT normalization and legacy equivalence."""

    def test_full_grid_zero_fill_matches_legacy_fft_frontend(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        resource_grid = build_resource_grid("full_grid", cfg.n_subcarriers, cfg.burst_profile.n_snapshots)
        masked_observation = simulate_masked_observation(
            cfg,
            nominal_trial_parameters(cfg),
            resource_grid,
            rng=np.random.default_rng(0),
            knowledge_mode="known_symbols",
        )

        legacy_fft = build_fft_cube(cfg, masked_observation.snapshot.horizontal_cube)
        masked_fft = build_masked_fft_cube(cfg, masked_observation, embedding_mode="zero_fill")

        self.assertTrue(np.allclose(masked_fft.power_cube, legacy_fft.power_cube))
        self.assertTrue(np.allclose(masked_fft.azimuth_axis_deg, legacy_fft.azimuth_axis_deg))
        self.assertTrue(np.allclose(masked_fft.range_axis_m, legacy_fft.range_axis_m))
        self.assertTrue(np.allclose(masked_fft.velocity_axis_mps, legacy_fft.velocity_axis_mps))
        self.assertEqual(masked_fft.embedding_mode, "zero_fill")
        self.assertAlmostEqual(masked_fft.known_fraction, 1.0)
        self.assertAlmostEqual(masked_fft.normalization_gain, 1.0)

        legacy_candidates = extract_candidates_from_fft(cfg, legacy_fft, cfg.runtime_profile.coarse_candidate_count)
        masked_candidates = extract_candidates_from_fft(cfg, masked_fft, cfg.runtime_profile.coarse_candidate_count)
        self.assertEqual(legacy_candidates, masked_candidates)

    def test_sparse_grid_weighted_mode_applies_expected_support_normalization(self) -> None:
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
            rng=np.random.default_rng(3),
            knowledge_mode="known_symbols",
        )

        zero_fill_fft = build_masked_fft_cube(cfg, masked_observation, embedding_mode="zero_fill")
        weighted_fft = build_masked_fft_cube(cfg, masked_observation, embedding_mode="weighted")
        expected_gain = weighted_fft.full_support_energy / weighted_fft.support_energy

        self.assertLess(weighted_fft.known_fraction, 1.0)
        self.assertLess(weighted_fft.support_energy, weighted_fft.full_support_energy)
        self.assertGreater(weighted_fft.normalization_gain, 1.0)
        self.assertAlmostEqual(weighted_fft.normalization_gain, expected_gain)
        self.assertTrue(np.allclose(weighted_fft.power_cube, zero_fill_fft.power_cube * expected_gain))
        self.assertGreater(float(np.max(weighted_fft.power_cube)), float(np.max(zero_fill_fft.power_cube)))

    def test_prepare_masked_frontend_returns_sparse_grid_candidates(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        resource_grid = build_resource_grid(
            "comb_pilot",
            cfg.n_subcarriers,
            cfg.burst_profile.n_snapshots,
            pilot_subcarrier_period=4,
            pilot_symbol_period=2,
        )
        masked_observation = simulate_masked_observation(
            cfg,
            nominal_trial_parameters(cfg),
            resource_grid,
            rng=np.random.default_rng(5),
            knowledge_mode="known_symbols",
        )
        frontend = prepare_masked_frontend(cfg, masked_observation, embedding_mode="weighted")

        self.assertEqual(frontend.fft_cube.embedding_mode, "weighted")
        self.assertGreater(len(frontend.coarse_candidates), 0)
        self.assertGreater(frontend.frontend_runtime_s, 0.0)

    def test_range_fbss_engages_for_contiguous_block_support(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        resource_grid = build_resource_grid(
            "block_pilot",
            cfg.n_subcarriers,
            cfg.burst_profile.n_snapshots,
            block_width_subcarriers=48,
            block_symbol_span=cfg.burst_profile.n_snapshots,
            n_frequency_blocks=1,
        )
        masked_observation = simulate_masked_observation(
            cfg,
            nominal_trial_parameters(cfg),
            resource_grid,
            rng=np.random.default_rng(11),
            knowledge_mode="known_symbols",
        )
        known_cube = extract_known_symbol_cube(masked_observation)
        azimuth_weights = azimuth_steering_matrix(
            cfg.effective_horizontal_positions_m,
            np.array([0.0]),
            cfg.wavelength_m,
        )[:, 0]
        azimuth_weights /= np.sqrt(max(1, azimuth_weights.size))
        beamformed = np.einsum("h,hft->ft", azimuth_weights.conj(), known_cube, optimize=True)

        range_cov, range_frequencies_hz, applied = _range_music_covariance(
            cfg,
            beamformed,
            masked_observation.known_symbol_mask,
        )

        self.assertTrue(applied)
        self.assertLess(range_cov.shape[0], cfg.frequencies_hz.size)
        self.assertEqual(range_cov.shape[0], range_frequencies_hz.size)

    def test_doppler_fbss_engages_for_contiguous_symbol_support(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        resource_grid = build_resource_grid(
            "fragmented_prb",
            cfg.n_subcarriers,
            cfg.burst_profile.n_snapshots,
            prb_size=12,
            n_prb_fragments=4,
            pilot_subcarrier_period=4,
            pilot_symbol_period=4,
            active_symbol_indices=tuple(range(cfg.burst_profile.n_snapshots // 2)),
        )
        masked_observation = simulate_masked_observation(
            cfg,
            nominal_trial_parameters(cfg),
            resource_grid,
            rng=np.random.default_rng(13),
            knowledge_mode="known_symbols",
        )
        known_cube = extract_known_symbol_cube(masked_observation)
        azimuth_weights = azimuth_steering_matrix(
            cfg.effective_horizontal_positions_m,
            np.array([0.0]),
            cfg.wavelength_m,
        )[:, 0]
        azimuth_weights /= np.sqrt(max(1, azimuth_weights.size))
        beamformed = np.einsum("h,hft->ft", azimuth_weights.conj(), known_cube, optimize=True)
        range_weights = np.ones(beamformed.shape[0], dtype=np.complex128)
        doppler_signal = (beamformed * range_weights.conj()[:, np.newaxis]).T

        doppler_cov, doppler_times_s, applied = _doppler_music_covariance(
            cfg,
            doppler_signal,
            masked_observation.known_symbol_mask,
        )

        self.assertTrue(applied)
        self.assertLess(doppler_cov.shape[0], cfg.snapshot_times_s.size)
        self.assertEqual(doppler_cov.shape[0], doppler_times_s.size)
