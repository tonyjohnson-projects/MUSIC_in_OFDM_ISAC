"""System tests for the private-5G angle-range-Doppler simulator."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.channel_models import TrialParameters, build_truth_targets, calibrated_noise_variance, path_amplitude, simulate_radar_cube
from aisle_isac.estimators import config_search_bounds, extract_candidates_from_fft, prepare_frontend, run_estimators, validate_targets_within_search_bounds
from aisle_isac.metrics import evaluate_trial
from aisle_isac.ofdm import azimuth_steering_matrix, doppler_steering_matrix, range_steering_matrix
from aisle_isac.reporting import write_all_outputs
from aisle_isac.scenarios import build_study_config
from aisle_isac.study import PUBLIC_SWEEP_NAMES, SUBMISSION_SWEEP_NAMES, _build_sweep_point_specs, nominal_trial_parameters, run_study, simulate_trial


REPO_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _synthetic_radar_cube(
    cfg,
    targets: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Build a noiseless radar cube from simple steering products."""

    cube = np.zeros(
        (cfg.effective_horizontal_positions_m.size, cfg.n_subcarriers, cfg.burst_profile.n_snapshots),
        dtype=np.complex128,
    )
    for range_m, velocity_mps, azimuth_deg, amplitude in targets:
        azimuth_weights = azimuth_steering_matrix(
            cfg.effective_horizontal_positions_m,
            np.array([azimuth_deg]),
            cfg.wavelength_m,
        )[:, 0]
        range_weights = range_steering_matrix(cfg.frequencies_hz, np.array([range_m]))[:, 0]
        doppler_weights = doppler_steering_matrix(cfg.snapshot_times_s, np.array([velocity_mps]), cfg.wavelength_m)[:, 0]
        cube += (
            amplitude
            * azimuth_weights[:, np.newaxis, np.newaxis]
            * range_weights[np.newaxis, :, np.newaxis]
            * doppler_weights[np.newaxis, np.newaxis, :]
        )
    return cube


class IsacSimulatorTest(unittest.TestCase):
    """Verify the refactored simulator surfaces and outputs."""

    def test_configuration_uses_physical_ofdm_grid(self) -> None:
        for anchor in ("fr1", "fr2"):
            cfg = build_study_config(anchor, "open_aisle", "quick")
            physical_spacing = np.diff(cfg.physical_frequencies_hz)
            self.assertTrue(np.allclose(physical_spacing, cfg.anchor.subcarrier_spacing_hz))
            self.assertTrue(np.all(np.isin(cfg.frequencies_hz, cfg.physical_frequencies_hz)))
            self.assertGreater(cfg.sampled_occupied_bandwidth_hz, 0.9 * cfg.anchor.occupied_bandwidth_hz)

    def test_every_sweep_point_truth_lies_inside_search_domain(self) -> None:
        for sweep_name in PUBLIC_SWEEP_NAMES:
            specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 8, "headline", sweep_name)
            for spec in specs:
                cfg = build_study_config(
                    spec.anchor_name,
                    spec.scene_name,
                    spec.profile_name,
                    spec.burst_profile_name,
                    spec.rx_columns,
                    spec.suite,
                    trial_count_override=spec.trial_count,
                )
                params = TrialParameters(
                    center_range_m=spec.center_range_m,
                    range_separation_m=spec.range_separation_m,
                    velocity_separation_mps=spec.velocity_separation_mps,
                    angle_separation_deg=spec.angle_separation_deg,
                )
                validate_targets_within_search_bounds(build_truth_targets(cfg, params), config_search_bounds(cfg))

    def test_noise_calibration_is_fixed_across_absolute_range_sweep(self) -> None:
        specs = _build_sweep_point_specs("fr1", "open_aisle", "quick", 8, "headline", "absolute_range")
        noise_values = []
        for spec in specs:
            cfg = build_study_config(spec.anchor_name, spec.scene_name, spec.profile_name, spec.burst_profile_name, spec.rx_columns, spec.suite)
            params = TrialParameters(
                center_range_m=spec.center_range_m,
                range_separation_m=spec.range_separation_m,
                velocity_separation_mps=spec.velocity_separation_mps,
                angle_separation_deg=spec.angle_separation_deg,
            )
            snapshot = simulate_radar_cube(cfg, params, np.random.default_rng(0))
            noise_values.append(snapshot.noise_variance)
        self.assertTrue(np.allclose(noise_values, noise_values[0]))

    def test_path_amplitude_decreases_with_range(self) -> None:
        wl = 0.0857  # FR1 wavelength
        self.assertGreater(path_amplitude(10.0, 0.0, wl), path_amplitude(20.0, 0.0, wl))
        self.assertGreater(path_amplitude(20.0, 0.0, wl), path_amplitude(30.0, 0.0, wl))
        # shorter wavelength (FR2) gives lower amplitude at same range
        wl_fr2 = 0.0107
        self.assertGreater(path_amplitude(20.0, 0.0, wl), path_amplitude(20.0, 0.0, wl_fr2))

    def test_rmse_over_crb_changes_with_actual_target_snr(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        near_target = build_truth_targets(
            cfg,
            TrialParameters(center_range_m=12.0, range_separation_m=0.0, velocity_separation_mps=0.0, angle_separation_deg=0.0),
        )[0]
        far_target = build_truth_targets(
            cfg,
            TrialParameters(center_range_m=30.0, range_separation_m=0.0, velocity_separation_mps=0.0, angle_separation_deg=0.0),
        )[0]
        near_metrics = evaluate_trial(
            cfg,
            truth_targets=(near_target,),
            detections=(
                type("DetectionStub", (), {
                    "range_m": near_target.range_m + 0.1 * cfg.range_resolution_m,
                    "velocity_mps": near_target.velocity_mps,
                    "azimuth_deg": near_target.azimuth_deg,
                })(),
            ),
            estimated_model_order=1,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        far_metrics = evaluate_trial(
            cfg,
            truth_targets=(far_target,),
            detections=(
                type("DetectionStub", (), {
                    "range_m": far_target.range_m + 0.1 * cfg.range_resolution_m,
                    "velocity_mps": far_target.velocity_mps,
                    "azimuth_deg": far_target.azimuth_deg,
                })(),
            ),
            estimated_model_order=1,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        self.assertIsNotNone(near_metrics.conditional_rmse_over_crb)
        self.assertIsNotNone(far_metrics.conditional_rmse_over_crb)
        self.assertGreater(near_metrics.unconditional_rmse_over_crb, far_metrics.unconditional_rmse_over_crb)
        self.assertGreater(near_metrics.conditional_rmse_over_crb, far_metrics.conditional_rmse_over_crb)

    def test_unconditional_metrics_penalize_misses(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        truth_targets = build_truth_targets(
            cfg,
            TrialParameters(center_range_m=20.0, range_separation_m=0.0, velocity_separation_mps=0.0, angle_separation_deg=0.0),
        )
        target = truth_targets[0]
        hit_metrics = evaluate_trial(
            cfg,
            truth_targets=(target,),
            detections=(
                type("DetectionStub", (), {
                    "range_m": target.range_m,
                    "velocity_mps": target.velocity_mps,
                    "azimuth_deg": target.azimuth_deg,
                })(),
            ),
            estimated_model_order=1,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        miss_metrics = evaluate_trial(
            cfg,
            truth_targets=(target,),
            detections=tuple(),
            estimated_model_order=0,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        self.assertEqual(hit_metrics.scene_cost, 0.0)
        self.assertIsNotNone(hit_metrics.conditional_joint_assignment_rmse)
        self.assertIsNone(miss_metrics.conditional_joint_assignment_rmse)
        self.assertGreater(miss_metrics.unconditional_joint_assignment_rmse, hit_metrics.unconditional_joint_assignment_rmse)
        self.assertGreater(miss_metrics.scene_cost, hit_metrics.scene_cost)

    def test_unconditional_metrics_cap_out_of_gate_assignments_at_miss_penalty(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        target = build_truth_targets(
            cfg,
            TrialParameters(center_range_m=20.0, range_separation_m=0.0, velocity_separation_mps=0.0, angle_separation_deg=0.0),
        )[0]
        miss_metrics = evaluate_trial(
            cfg,
            truth_targets=(target,),
            detections=tuple(),
            estimated_model_order=0,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        stray_detection_metrics = evaluate_trial(
            cfg,
            truth_targets=(target,),
            detections=(
                type("DetectionStub", (), {
                    "range_m": target.range_m + 100.0,
                    "velocity_mps": target.velocity_mps + 25.0,
                    "azimuth_deg": target.azimuth_deg + 120.0,
                })(),
            ),
            estimated_model_order=1,
            noise_variance=calibrated_noise_variance(cfg),
            frontend_runtime_s=0.0,
            incremental_runtime_s=0.0,
            total_runtime_s=0.0,
        )
        self.assertEqual(
            stray_detection_metrics.unconditional_range_rmse_m,
            miss_metrics.unconditional_range_rmse_m,
        )
        self.assertEqual(
            stray_detection_metrics.unconditional_joint_assignment_rmse,
            miss_metrics.unconditional_joint_assignment_rmse,
        )
        self.assertGreater(stray_detection_metrics.scene_cost, miss_metrics.scene_cost)

    def test_nominal_fr1_open_aisle_targets_fit_doppler_domain(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        targets = build_truth_targets(cfg, nominal_trial_parameters(cfg))
        validate_targets_within_search_bounds(targets, config_search_bounds(cfg))

    def test_candidate_backfill_returns_expected_target_count(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        cube = _synthetic_radar_cube(cfg, [(20.0, -2.0, -10.0, 1.0), (24.0, 2.5, 12.0, 0.9)])
        frontend = prepare_frontend(cfg, cube)
        detections = extract_candidates_from_fft(cfg, frontend.fft_cube, cfg.runtime_profile.coarse_candidate_count)
        self.assertGreaterEqual(len(detections), cfg.expected_target_count)

    def test_single_target_fft_and_music_lock_near_truth(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        cube = _synthetic_radar_cube(cfg, [(20.0, 1.0, 8.0, 1.0)])
        estimates = run_estimators(cfg, cube)
        best_fft = estimates["fft"].detections[0]
        best_music = estimates["music"].detections[0]
        self.assertLess(abs(best_fft.range_m - 20.0), 2.0 * cfg.range_resolution_m)
        self.assertLess(abs(best_music.range_m - 20.0), 1.5 * cfg.range_resolution_m)
        self.assertLess(abs(best_music.azimuth_deg - 8.0), 2.0 * cfg.azimuth_resolution_deg)

    def test_fbss_beats_plain_music_in_coherent_rack_trial_bundle(self) -> None:
        cfg = build_study_config("fr1", "rack_aisle", "quick")
        params = TrialParameters(
            center_range_m=cfg.scene_class.nominal_range_m,
            range_separation_m=0.8 * cfg.range_resolution_m,
            velocity_separation_mps=0.8 * cfg.velocity_resolution_mps,
            angle_separation_deg=0.8 * cfg.azimuth_resolution_deg,
        )
        music_successes = 0
        fbss_successes = 0
        for seed in range(4):
            trial = simulate_trial(cfg, params, np.random.default_rng(seed))
            music_successes += int(trial.metrics["music"].joint_resolution_success)
            fbss_successes += int(trial.metrics["fbss"].joint_resolution_success)
        self.assertGreaterEqual(fbss_successes, music_successes)

    def test_nominal_full_search_music_detects_both_targets(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick")
        trial = simulate_trial(cfg, nominal_trial_parameters(cfg), np.random.default_rng(0))
        metrics = trial.metrics["music_full"]
        self.assertTrue(metrics.joint_detection_success)
        self.assertEqual(metrics.matched_target_count, cfg.expected_target_count)

    def test_report_outputs_use_distinct_nominal_csv_schemas(self) -> None:
        studies = [
            run_study(build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1), max_workers=1),
            run_study(build_study_config("fr2", "rack_aisle", "quick", suite="headline", trial_count_override=1), max_workers=1),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            write_all_outputs(studies, output_root, clean_outputs=True)
            scene_path = output_root / "data" / "scene_comparison.csv"
            fr_path = output_root / "data" / "fr1_vs_fr2.csv"
            crb_path = output_root / "data" / "crb_gap.csv"
            scene_text = scene_path.read_text(encoding="utf-8")
            fr_text = fr_path.read_text(encoding="utf-8")
            crb_text = crb_path.read_text(encoding="utf-8")
            self.assertNotEqual(scene_text, fr_text)
            self.assertNotEqual(scene_text, crb_text)
            self.assertNotEqual(fr_text, crb_text)

            with scene_path.open(newline="", encoding="utf-8") as handle:
                scene_header = next(csv.reader(handle))
            with fr_path.open(newline="", encoding="utf-8") as handle:
                fr_header = next(csv.reader(handle))
            with crb_path.open(newline="", encoding="utf-8") as handle:
                crb_header = next(csv.reader(handle))
            self.assertIn("total_runtime_s", scene_header)
            self.assertNotIn("runtime_s", scene_header)
            self.assertIn("unconditional_joint_assignment_rmse", fr_header)
            self.assertIn("conditional_joint_assignment_rmse", fr_header)
            self.assertIn("unconditional_rmse_over_crb", fr_header)
            self.assertIn("conditional_rmse_over_crb", fr_header)
            self.assertNotEqual(scene_header, fr_header)
            self.assertNotEqual(scene_header, crb_header)

    def test_selected_sweep_outputs_respect_requested_artifact_set(self) -> None:
        study = run_study(
            build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1),
            max_workers=1,
            sweep_names=SUBMISSION_SWEEP_NAMES,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            write_all_outputs(
                [study],
                output_root,
                clean_outputs=True,
                sweep_names=SUBMISSION_SWEEP_NAMES,
                include_scene_comparison=False,
                include_fr1_vs_fr2=False,
                include_crb_gap=False,
                include_representative_cube_slices=False,
            )
            for sweep_name in SUBMISSION_SWEEP_NAMES:
                self.assertTrue((output_root / "data" / f"{sweep_name}.csv").exists())
                self.assertTrue((output_root / "figures" / f"{sweep_name}.png").exists())
            self.assertEqual(
                sorted(path.name for path in (output_root / "data").iterdir()),
                sorted(f"{sweep_name}.csv" for sweep_name in SUBMISSION_SWEEP_NAMES),
            )
            self.assertEqual(
                sorted(path.name for path in (output_root / "figures").iterdir()),
                sorted(f"{sweep_name}.png" for sweep_name in SUBMISSION_SWEEP_NAMES),
            )

    def test_cli_smoke_fr1_open_aisle_without_uv(self) -> None:
        result = subprocess.run(
            [
                str(REPO_PYTHON),
                "run_study.py",
                "--profile",
                "quick",
                "--suite",
                "headline",
                "--anchor",
                "fr1",
                "--scene-class",
                "open_aisle",
                "--jobs",
                "1",
                "--trials",
                "1",
                "--clean-outputs",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        self.assertIn("Private-5G Angle-Range-Doppler Study", result.stdout)
        self.assertTrue((REPO_ROOT / "results" / "quick" / "data" / "range_separation.csv").exists())
        self.assertTrue((REPO_ROOT / "results" / "quick" / "data" / "absolute_range.csv").exists())
        self.assertTrue((REPO_ROOT / "results" / "quick" / "figures" / "scene_comparison.png").exists())
        self.assertFalse((REPO_ROOT / "results" / "quick" / "data" / "fr1_vs_fr2.csv").exists())

    def test_submission_bundle_smoke(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(REPO_PYTHON),
                "JOBS": "1",
                "TRIALS": "1",
                "CLEAN_OUTPUTS": "1",
            }
        )
        subprocess.run(
            ["bash", "scripts/build_submission_bundle.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        submission_root = REPO_ROOT / "results" / "submission"
        for filename in (
            "range_separation.csv",
            "velocity_separation.csv",
            "angle_separation.csv",
            "burst_profile.csv",
            "aperture.csv",
            "scene_comparison.csv",
            "crb_gap.csv",
            "build_manifest.txt",
        ):
            self.assertTrue((submission_root / "data" / filename).exists() if filename.endswith(".csv") else (submission_root / filename).exists())
        expected_figures = (
            "range_separation.png",
            "velocity_separation.png",
            "angle_separation.png",
            "burst_profile.png",
            "aperture.png",
            "scene_comparison.png",
            "crb_gap.png",
            "representative_cube_slices.png",
        )
        for filename in expected_figures:
            self.assertTrue((submission_root / "figures" / filename).exists())
        self.assertEqual(
            sorted(path.name for path in (submission_root / "data").iterdir()),
            sorted(
                (
                    "range_separation.csv",
                    "velocity_separation.csv",
                    "angle_separation.csv",
                    "burst_profile.csv",
                    "aperture.csv",
                    "scene_comparison.csv",
                    "crb_gap.csv",
                )
            ),
        )
        self.assertEqual(
            sorted(path.name for path in (submission_root / "figures").iterdir()),
            sorted(expected_figures),
        )
        for filename in (
            "absolute_range.csv",
            "fr1_vs_fr2.csv",
        ):
            self.assertFalse((submission_root / "data" / filename).exists())
        for filename in (
            "absolute_range.png",
            "fr1_vs_fr2.png",
        ):
            self.assertFalse((submission_root / "figures" / filename).exists())


if __name__ == "__main__":
    unittest.main()
