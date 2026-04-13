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


NOMINAL_PRB_KWARGS = {
    "prb_size": 12,
    "n_prb_fragments": 4,
    "pilot_subcarrier_period": 4,
    "pilot_symbol_period": 4,
}


class CommunicationsScheduledStudyTest(unittest.TestCase):
    """Verify the allocation-driven evidence path."""

    def test_public_sweep_names_match_evidence_plan(self) -> None:
        self.assertEqual(
            PUBLIC_SWEEP_NAMES,
            (
                "allocation_family",
                "occupied_fraction",
                "fragmentation",
                "bandwidth_span",
                "slow_time_span",
                "range_separation",
                "velocity_separation",
                "angle_separation",
                "nuisance_gain_offset",
            ),
        )
        self.assertEqual(SUBMISSION_SWEEP_NAMES, PUBLIC_SWEEP_NAMES)

    def test_support_stress_sweeps_are_monotone(self) -> None:
        for sweep_name, accessor in (
            ("occupied_fraction", lambda spec: spec.occupied_fraction),
            ("bandwidth_span", lambda spec: spec.bandwidth_span_fraction),
            ("slow_time_span", lambda spec: spec.slow_time_span_fraction),
        ):
            specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", sweep_name)
            values = [accessor(spec) for spec in specs]
            self.assertEqual(values, sorted(values))
            self.assertGreater(values[-1], values[0])

    def test_axis_sweeps_hold_the_other_axes_in_isolation(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)

        range_specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", "range_separation")
        self.assertTrue(all(spec.velocity_separation_mps >= 2.0 * cfg.velocity_resolution_mps for spec in range_specs))
        self.assertTrue(all(spec.angle_separation_deg >= 2.0 * cfg.azimuth_resolution_deg for spec in range_specs))

        velocity_specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", "velocity_separation")
        self.assertTrue(all(spec.range_separation_m >= 2.0 * cfg.range_resolution_m for spec in velocity_specs))
        self.assertTrue(all(spec.angle_separation_deg >= 2.0 * cfg.azimuth_resolution_deg for spec in velocity_specs))

        angle_specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 1, "headline", "angle_separation")
        self.assertTrue(all(spec.range_separation_m >= 2.0 * cfg.range_resolution_m for spec in angle_specs))
        self.assertTrue(all(spec.velocity_separation_mps >= 2.0 * cfg.velocity_resolution_mps for spec in angle_specs))

    def test_nominal_masked_music_hits_positive_scenes(self) -> None:
        for scene_name in ("open_aisle", "intersection"):
            cfg = build_study_config("fr1", scene_name, "quick", suite="headline", trial_count_override=1)
            trial = simulate_communications_trial(
                cfg,
                nominal_trial_parameters(cfg),
                "fragmented_prb",
                "Fragmented Scheduled PRB",
                "known_symbols",
                "qpsk",
                NOMINAL_PRB_KWARGS,
                np.random.default_rng(0),
            )
            music_metrics = trial.metrics["music_masked"]
            self.assertTrue(music_metrics.joint_detection_success)
            self.assertTrue(music_metrics.joint_resolution_success)
            self.assertTrue(music_metrics.range_resolution_success)
            self.assertTrue(music_metrics.velocity_resolution_success)
            self.assertTrue(music_metrics.angle_resolution_success)

    def test_rack_aisle_nominal_trial_runs_and_reports_music_model_order(self) -> None:
        cfg = build_study_config("fr1", "rack_aisle", "quick", suite="headline", trial_count_override=1)
        trial = simulate_communications_trial(
            cfg,
            nominal_trial_parameters(cfg),
            "fragmented_prb",
            "Fragmented Scheduled PRB",
            "known_symbols",
            "qpsk",
            NOMINAL_PRB_KWARGS,
            np.random.default_rng(0),
            include_fbss_ablation=True,
        )
        self.assertIn("music_masked", trial.estimates)
        self.assertIsNotNone(trial.estimates["music_masked"].estimated_model_order)
        self.assertEqual(trial.estimates["music_masked"].reported_target_count, 2)
        self.assertIsNotNone(trial.fbss_ablation_estimates)
        self.assertIsNotNone(trial.fbss_ablation_metrics)
        self.assertIn("fbss_spatial_range_doppler", trial.fbss_ablation_estimates)

    def test_study_runner_produces_requested_new_sweeps(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        study = run_communications_study(
            cfg,
            show_progress=False,
            suite="headline",
            sweep_names=("bandwidth_span", "slow_time_span"),
        )
        self.assertEqual([sweep.sweep_name for sweep in study.sweeps], ["bandwidth_span", "slow_time_span"])
        self.assertEqual(study.nominal_point.allocation_family, "fragmented_prb")
        self.assertEqual(study.pilot_only_nominal_point.knowledge_mode, "pilot_only")
        self.assertEqual(study.evidence_profile_name, "music_waveform_limited_v2")

    def test_support_sensitive_points_carry_fbss_ablation_summaries(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        study = run_communications_study(
            cfg,
            show_progress=False,
            suite="headline",
            sweep_names=("bandwidth_span", "slow_time_span"),
        )
        self.assertIsNotNone(study.nominal_point.fbss_ablation_summaries)
        self.assertIn("fbss_spatial_range_doppler", study.nominal_point.fbss_ablation_summaries)
        for sweep in study.sweeps:
            self.assertTrue(all(point.fbss_ablation_summaries is not None for point in sweep.points))

    def test_nominal_points_export_trial_level_rows(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        study = run_communications_study(
            cfg,
            show_progress=False,
            max_workers=1,
            suite="headline",
            sweep_names=("bandwidth_span",),
        )
        self.assertGreater(len(study.nominal_point.trial_rows), 0)
        self.assertGreater(len(study.pilot_only_nominal_point.trial_rows), 0)
        self.assertIn("trial_spawn_key", study.nominal_point.trial_rows[0])
        self.assertIn("truth_targets", study.nominal_point.trial_rows[0])
        self.assertIn("detections", study.nominal_point.trial_rows[0])
