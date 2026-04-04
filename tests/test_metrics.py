"""Tests for the upgraded thesis-facing metrics."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.channel_models import TargetState
from aisle_isac.config import StudyConfig
from aisle_isac.estimators import Detection
from aisle_isac.metrics import evaluate_trial, summarize_method_metrics
from aisle_isac.scenarios import build_study_config


class MetricsTest(unittest.TestCase):
    """Verify per-axis and model-order metric semantics."""

    def test_per_axis_resolution_flags_split_cleanly(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        truth_targets = (
            TargetState("A", "amr", 10.0, 1.0, 0.0, 0.0, 1.0),
            TargetState("B", "amr", 12.0, 3.0, 3.0, 0.0, 1.0),
        )
        detections = (
            Detection(10.05, 1.05, 0.05, 1.0),
            Detection(12.05, 5.1, 3.05, 0.8),
        )

        metrics = evaluate_trial(
            cfg,
            truth_targets=truth_targets,
            detections=detections,
            reported_target_count=2,
            estimated_model_order=4,
            frontend_runtime_s=0.1,
            incremental_runtime_s=0.2,
            total_runtime_s=0.3,
        )

        self.assertTrue(metrics.joint_detection_success)
        self.assertFalse(metrics.joint_resolution_success)
        self.assertTrue(metrics.range_resolution_success)
        self.assertFalse(metrics.velocity_resolution_success)
        self.assertTrue(metrics.angle_resolution_success)
        self.assertEqual(metrics.estimated_model_order, 4)

    def test_summary_keeps_reported_count_and_model_order_separate(self) -> None:
        cfg = build_study_config("fr1", "open_aisle", "quick", suite="headline", trial_count_override=1)
        truth_targets = (
            TargetState("A", "amr", 10.0, 1.0, 0.0, 0.0, 1.0),
            TargetState("B", "amr", 12.0, 3.0, 3.0, 0.0, 1.0),
        )
        detections = (
            Detection(10.0, 1.0, 0.0, 1.0),
            Detection(12.0, 3.0, 3.0, 0.9),
        )
        trial_metrics = [
            evaluate_trial(
                cfg,
                truth_targets=truth_targets,
                detections=detections,
                reported_target_count=2,
                estimated_model_order=4,
                frontend_runtime_s=0.1,
                incremental_runtime_s=0.2,
                total_runtime_s=0.3,
            ),
            evaluate_trial(
                cfg,
                truth_targets=truth_targets,
                detections=detections,
                reported_target_count=2,
                estimated_model_order=2,
                frontend_runtime_s=0.1,
                incremental_runtime_s=0.2,
                total_runtime_s=0.3,
            ),
        ]

        summary = summarize_method_metrics(trial_metrics, expected_target_count=2)
        self.assertEqual(summary.reported_target_count_accuracy, 1.0)
        self.assertAlmostEqual(summary.mean_estimated_model_order, 3.0)
        self.assertAlmostEqual(summary.estimated_model_order_accuracy, 0.5)
