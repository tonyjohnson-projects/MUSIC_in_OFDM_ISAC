"""Tests for source generation, jitter, and reproducibility."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.channel_models import simulate_radar_cube
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import nominal_trial_parameters


class ChannelModelTest(unittest.TestCase):
    """Verify the upgraded stochastic source and jitter model."""

    def test_scene_coherence_ranking_matches_scene_intent(self) -> None:
        means: dict[str, float] = {}
        for scene_name in ("open_aisle", "intersection", "rack_aisle"):
            cfg = build_study_config("fr1", scene_name, "quick", suite="headline", trial_count_override=1)
            params = nominal_trial_parameters(cfg)
            values = []
            for seed in range(12):
                snapshot = simulate_radar_cube(cfg, params, np.random.default_rng(seed))
                values.append(snapshot.scenario.source_model.configured_target_coherence)
            means[scene_name] = float(np.mean(values))

        self.assertLess(means["open_aisle"], means["intersection"])
        self.assertLess(means["intersection"], means["rack_aisle"])

    def test_trial_jitter_stays_within_configured_bounds(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        params = nominal_trial_parameters(cfg)
        snapshot = simulate_radar_cube(cfg, params, np.random.default_rng(7))
        jitter = snapshot.scenario.trial_jitter

        self.assertLessEqual(abs(jitter.center_range_offset_m), cfg.center_range_jitter_m + 1.0e-12)
        self.assertLessEqual(abs(jitter.range_separation_offset_m), cfg.range_separation_jitter_cells * cfg.range_resolution_m + 1.0e-12)
        self.assertLessEqual(abs(jitter.velocity_separation_offset_mps), cfg.velocity_separation_jitter_cells * cfg.velocity_resolution_mps + 1.0e-12)
        self.assertLessEqual(abs(jitter.angle_separation_offset_deg), cfg.angle_separation_jitter_cells * cfg.azimuth_resolution_deg + 1.0e-12)
        self.assertTrue(all(abs(offset) <= cfg.target_amplitude_jitter_db + 1.0e-12 for offset in jitter.target_amplitude_offsets_db))
        self.assertTrue(all(abs(offset) <= cfg.nuisance_gain_jitter_db + 1.0e-12 for offset in jitter.nuisance_gain_offsets_db))

    def test_seeded_cube_generation_is_reproducible(self) -> None:
        cfg = build_study_config("fr1", "intersection", "quick", suite="headline", trial_count_override=1)
        params = nominal_trial_parameters(cfg)
        snapshot_a = simulate_radar_cube(cfg, params, np.random.default_rng(123))
        snapshot_b = simulate_radar_cube(cfg, params, np.random.default_rng(123))

        self.assertEqual(snapshot_a.scenario.trial_parameters, snapshot_b.scenario.trial_parameters)
        self.assertEqual(snapshot_a.scenario.trial_jitter, snapshot_b.scenario.trial_jitter)
        self.assertEqual(snapshot_a.scenario.source_model, snapshot_b.scenario.source_model)
        self.assertTrue(np.allclose(snapshot_a.radar_cube, snapshot_b.radar_cube))
        self.assertTrue(np.allclose(snapshot_a.target_only_cube, snapshot_b.target_only_cube))
