"""System tests for the communications-scheduled study runner."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.scheduled_study import (
    PUBLIC_SWEEP_NAMES,
    SUBMISSION_SWEEP_NAMES,
    _build_sweep_point_specs,
    nominal_trial_parameters,
    run_communications_study,
    simulate_communications_trial,
)
from aisle_isac.scenarios import build_study_config


class CommunicationsScheduledStudyTest(unittest.TestCase):
    """Verify the new allocation-driven study path."""

    def test_public_sweep_names_match_new_plan(self) -> None:
        self.assertEqual(
            PUBLIC_SWEEP_NAMES,
            (
                "allocation_family",
                "occupied_fraction",
                "fragmentation",
                "range_separation",
                "velocity_separation",
                "angle_separation",
            ),
        )
        self.assertIn("allocation_family", SUBMISSION_SWEEP_NAMES)

    def test_occupied_fraction_sweep_is_monotone(self) -> None:
        specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", "occupied_fraction")
        occupied = [spec.occupied_fraction for spec in specs]
        self.assertEqual(occupied, sorted(occupied))
        self.assertGreater(occupied[-1], occupied[0])

    def test_nominal_trial_runs_masked_fft_and_superres_estimators(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        trial = simulate_communications_trial(
            cfg,
            nominal_trial_parameters(cfg),
            "fragmented_prb",
            "Fragmented PRB",
            "known_symbols",
            "qpsk",
            {
                "prb_size": 12,
                "n_prb_fragments": 4,
                "pilot_subcarrier_period": 4,
                "pilot_symbol_period": 4,
            },
            np.random.default_rng(11),
        )
        self.assertIn("fft_masked", trial.estimates)
        self.assertIn("music_masked", trial.estimates)
        self.assertGreater(trial.allocation_summary.occupied_fraction, 0.0)
        self.assertGreaterEqual(trial.estimates["fft_masked"].reported_target_count, 0)
        self.assertGreaterEqual(trial.estimates["music_masked"].reported_target_count, 0)

    def test_allocation_sweeps_keep_geometry_fixed(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        nominal = nominal_trial_parameters(cfg)
        for sweep_name in ("allocation_family", "occupied_fraction", "fragmentation"):
            specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", sweep_name)
            for spec in specs:
                self.assertEqual(spec.center_range_m, nominal.center_range_m)
                self.assertEqual(spec.range_separation_m, nominal.range_separation_m)
                self.assertEqual(spec.velocity_separation_mps, nominal.velocity_separation_mps)
                self.assertEqual(spec.angle_separation_deg, nominal.angle_separation_deg)

    def test_nominal_masked_music_detects_both_targets_on_fragmented_prb(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        trial = simulate_communications_trial(
            cfg,
            nominal_trial_parameters(cfg),
            "fragmented_prb",
            "Fragmented Scheduled PRB",
            "known_symbols",
            "qpsk",
            {
                "prb_size": 12,
                "n_prb_fragments": 4,
                "pilot_subcarrier_period": 4,
                "pilot_symbol_period": 4,
            },
            np.random.default_rng(0),
        )
        self.assertTrue(trial.metrics["music_masked"].joint_detection_success)
        self.assertEqual(trial.metrics["music_masked"].matched_target_count, cfg.expected_target_count)

    def test_study_runner_produces_requested_new_sweeps(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        study = run_communications_study(
            cfg,
            show_progress=False,
            max_workers=1,
            suite="headline",
            sweep_names=("allocation_family", "occupied_fraction"),
        )
        self.assertEqual([sweep.sweep_name for sweep in study.sweeps], ["allocation_family", "occupied_fraction"])
        self.assertEqual(study.nominal_point.allocation_family, "fragmented_prb")
