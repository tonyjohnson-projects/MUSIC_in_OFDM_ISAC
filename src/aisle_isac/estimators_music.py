"""Masked MUSIC estimators for communications-limited OFDM ISAC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aisle_isac.config import StudyConfig
from aisle_isac.estimators import FrontendArtifacts, MethodEstimate, _run_full_search_music
from aisle_isac.masked_observation import MaskedObservation, extract_known_symbol_cube


METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT Baseline",
    "music_masked": "Masked Full-Search MUSIC",
}


def _run_masked_fft_estimator(
    cfg: StudyConfig,
    frontend: FrontendArtifacts,
) -> MethodEstimate:
    detections = frontend.coarse_candidates[: max(1, cfg.expected_target_count)]
    return MethodEstimate(
        label=METHOD_LABELS["fft_masked"],
        detections=detections,
        reported_target_count=len(detections),
        frontend_runtime_s=frontend.frontend_runtime_s,
        incremental_runtime_s=0.0,
        total_runtime_s=frontend.frontend_runtime_s,
    )


def _run_masked_music_estimator(
    cfg: StudyConfig,
    known_cube: np.ndarray,
    frontend: FrontendArtifacts,
) -> MethodEstimate:
    estimate = _run_full_search_music(
        cfg,
        known_cube,
        frontend.search_bounds,
        frontend.frontend_runtime_s,
        use_fbss=True,
    )
    return replace(estimate, label=f"{METHOD_LABELS['music_masked']} + FBSS")


def run_masked_estimators(
    cfg: StudyConfig,
    masked_observation: MaskedObservation,
    frontend: FrontendArtifacts,
) -> dict[str, MethodEstimate]:
    """Run the active thesis comparison: masked FFT versus masked MUSIC."""

    known_cube = extract_known_symbol_cube(masked_observation)
    return {
        "fft_masked": _run_masked_fft_estimator(cfg, frontend),
        "music_masked": _run_masked_music_estimator(cfg, known_cube, frontend),
    }
