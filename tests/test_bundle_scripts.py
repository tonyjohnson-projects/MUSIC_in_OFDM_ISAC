"""Integration tests for build-script guards and artifact schema."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class BundleScriptsTest(unittest.TestCase):
    """Verify the cleaned bundle scripts and output schema."""

    def test_submission_bundle_rejects_smoke_trials_without_override(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(REPO_ROOT / ".venv" / "bin" / "python"),
                "TRIALS": "1",
                "CLEAN_OUTPUTS": "1",
                "SCENE_CLASS": "open_aisle",
            }
        )
        result = subprocess.run(
            ["bash", "scripts/build_submission_bundle.sh"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ALLOW_SMOKE_SUBMISSION=1", result.stderr)

    def test_quick_bundle_manifest_and_schema_are_current(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(REPO_ROOT / ".venv" / "bin" / "python"),
                "PROFILE": "quick",
                "ANCHOR": "fr1",
                "SCENE_CLASS": "open_aisle",
                "TRIALS": "1",
                "CLEAN_OUTPUTS": "1",
            }
        )
        subprocess.run(
            ["bash", "scripts/build_results_bundle.sh"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        manifest_text = (REPO_ROOT / "results" / "quick" / "build_manifest.txt").read_text(encoding="utf-8")
        self.assertIn("schema_version=2.0", manifest_text)
        self.assertIn("estimator_set=fft_masked,music_masked", manifest_text)
        self.assertIn("fbss_ablation_set=fbss_spatial_only,fbss_spatial_range,fbss_spatial_doppler,fbss_spatial_range_doppler", manifest_text)
        self.assertIn("knowledge_modes=known_symbols,pilot_only", manifest_text)
        self.assertIn("git_commit=", manifest_text)
        self.assertIn("git_commit_short=", manifest_text)
        self.assertIn("bandwidth_span", manifest_text)
        self.assertIn("slow_time_span", manifest_text)
        self.assertNotIn("pilot_fraction", manifest_text)

        nominal_path = REPO_ROOT / "results" / "quick" / "data" / "nominal_summary.csv"
        with nominal_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertGreater(len(rows), 0)
        self.assertIn("range_resolution_probability", rows[0])
        self.assertIn("velocity_resolution_probability", rows[0])
        self.assertIn("angle_resolution_probability", rows[0])
        self.assertIn("joint_resolution_probability_ci95_lower", rows[0])
        self.assertIn("joint_resolution_probability_ci95_upper", rows[0])
        self.assertNotIn("scene_cost", rows[0])
        self.assertEqual(sorted({row["method"] for row in rows}), ["fft_masked", "music_masked"])

        data_dir = REPO_ROOT / "results" / "quick" / "data"
        self.assertTrue((data_dir / "all_sweep_results.csv").exists())
        self.assertTrue((data_dir / "trial_level_results.csv").exists())
        self.assertTrue((data_dir / "usefulness_windows.csv").exists())
        self.assertTrue((data_dir / "pilot_only_nominal_summary.csv").exists())
        self.assertTrue((data_dir / "fbss_ablation_results.csv").exists())
        self.assertTrue((data_dir / "representative_resource_mask.csv").exists())
        self.assertTrue((data_dir / "representative_scene_geometry.csv").exists())
        self.assertTrue((data_dir / "representative_range_doppler.csv").exists())
        self.assertTrue((data_dir / "representative_music_spectra.csv").exists())
        self.assertTrue((data_dir / "representative_fbss_ablation_spectra.csv").exists())
        self.assertFalse((data_dir / "scene_comparison.csv").exists())

        with (data_dir / "trial_level_results.csv").open("r", encoding="utf-8", newline="") as handle:
            trial_rows = list(csv.DictReader(handle))
        self.assertGreater(len(trial_rows), 0)
        self.assertIn("trial_index", trial_rows[0])
        self.assertIn("trial_spawn_key", trial_rows[0])
        self.assertIn("truth_targets", trial_rows[0])
        self.assertIn("detections", trial_rows[0])
        self.assertIn("music_stage_azimuth_candidate_count", trial_rows[0])
        self.assertIn("music_stage_coarse_candidates", trial_rows[0])

        with (data_dir / "pilot_only_nominal_summary.csv").open("r", encoding="utf-8", newline="") as handle:
            pilot_rows = list(csv.DictReader(handle))
        self.assertGreater(len(pilot_rows), 0)
        self.assertEqual(sorted({row["knowledge_mode"] for row in pilot_rows}), ["pilot_only"])

        with (data_dir / "representative_scene_geometry.csv").open("r", encoding="utf-8", newline="") as handle:
            geometry_rows = list(csv.DictReader(handle))
        self.assertGreater(len(geometry_rows), 0)
        self.assertIn("entity_kind", geometry_rows[0])
        self.assertIn("x_m", geometry_rows[0])
        self.assertIn("y_m", geometry_rows[0])

        with (data_dir / "representative_music_spectra.csv").open("r", encoding="utf-8", newline="") as handle:
            spectrum_rows = list(csv.DictReader(handle))
        self.assertGreater(len(spectrum_rows), 0)
        self.assertEqual(sorted({row["dimension"] for row in spectrum_rows}), ["azimuth", "doppler", "range"])

        with (data_dir / "fbss_ablation_results.csv").open("r", encoding="utf-8", newline="") as handle:
            ablation_rows = list(csv.DictReader(handle))
        self.assertGreater(len(ablation_rows), 0)
        self.assertEqual(
            sorted({row["method"] for row in ablation_rows}),
            [
                "fbss_spatial_doppler",
                "fbss_spatial_only",
                "fbss_spatial_range",
                "fbss_spatial_range_doppler",
            ],
        )

        with (data_dir / "representative_fbss_ablation_spectra.csv").open("r", encoding="utf-8", newline="") as handle:
            ablation_spectrum_rows = list(csv.DictReader(handle))
        self.assertGreater(len(ablation_spectrum_rows), 0)
        self.assertEqual(sorted({row["dimension"] for row in ablation_spectrum_rows}), ["azimuth", "doppler", "range"])

        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "scripts/plot_results_from_csv.py",
                "--input-root",
                str(REPO_ROOT / "results" / "quick"),
                "--clean-output",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        figure_dir = REPO_ROOT / "results" / "quick" / "figures_from_csv"
        expected_story_figures = {
            "story_nominal_verdict_from_csv.png",
            "story_intersection_resolution_from_csv.png",
            "story_regime_map_from_csv.png",
            "story_coherence_overlap_from_csv.png",
            "story_pilot_only_collapse_from_csv.png",
            "story_trial_delta_from_csv.png",
        }
        actual_story_figures = {path.name for path in figure_dir.glob("*.png")}
        self.assertEqual(actual_story_figures, expected_story_figures)
        self.assertFalse((figure_dir / "sweep_bandwidth_span_from_csv.png").exists())

    def test_run_study_supports_fast_iteration_flags_and_fixed_music_order(self) -> None:
        output_dir = REPO_ROOT / "results" / "test_fast_iter"
        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "run_study.py",
                "--profile",
                "quick",
                "--anchor",
                "fr1",
                "--scene-class",
                "open_aisle",
                "--trials",
                "1",
                "--jobs",
                "1",
                "--sweeps",
                "bandwidth_span",
                "--skip-pilot-only",
                "--skip-representative",
                "--disable-fbss-ablation",
                "--music-fixed-order",
                "2",
                "--output-dir",
                "results/test_fast_iter",
                "--clean-outputs",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        data_dir = output_dir / "data"
        self.assertTrue((data_dir / "bandwidth_span.csv").exists())
        self.assertTrue((data_dir / "nominal_summary.csv").exists())
        self.assertTrue((data_dir / "trial_level_results.csv").exists())
        self.assertFalse((data_dir / "allocation_family.csv").exists())
        self.assertFalse((data_dir / "pilot_only_nominal_summary.csv").exists())
        self.assertFalse((data_dir / "fbss_ablation_results.csv").exists())
        self.assertFalse((data_dir / "representative_resource_mask.csv").exists())

        with (data_dir / "nominal_summary.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        music_rows = [row for row in rows if row["method"] == "music_masked"]
        self.assertEqual(len(music_rows), 1)
        self.assertEqual(music_rows[0]["mean_estimated_model_order"], "2.000000")

        with (data_dir / "trial_level_results.csv").open("r", encoding="utf-8", newline="") as handle:
            trial_rows = list(csv.DictReader(handle))
        self.assertGreater(len(trial_rows), 0)
        self.assertEqual(sorted({row["sweep_name"] for row in trial_rows}), ["bandwidth_span", "nominal"])
        self.assertEqual(sorted({row["music_model_order_mode"] for row in trial_rows}), ["fixed"])
