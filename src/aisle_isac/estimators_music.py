"""Masked MUSIC estimators for communications-limited OFDM ISAC."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from aisle_isac.config import StudyConfig
from aisle_isac.estimators import (
    FrontendArtifacts,
    MethodEstimate,
    _range_search_upper_m,
    _run_full_search_music,
    refine_detection_set_local,
)
from aisle_isac.masked_observation import MaskedObservation, extract_known_symbol_cube


METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT + Local Refinement",
    "music_masked": "Masked Staged MUSIC + FBSS",
}
FBSS_ABLATION_ORDER = (
    "fbss_spatial_only",
    "fbss_spatial_range",
    "fbss_spatial_doppler",
    "fbss_spatial_range_doppler",
)
FBSS_ABLATION_LABELS = {
    "fbss_spatial_only": "Spatial FBSS Only",
    "fbss_spatial_range": "Spatial + Range FBSS",
    "fbss_spatial_doppler": "Spatial + Doppler FBSS",
    "fbss_spatial_range_doppler": "Spatial + Range + Doppler FBSS",
}
FBSS_ABLATION_FLAGS = {
    "fbss_spatial_only": (False, False),
    "fbss_spatial_range": (True, False),
    "fbss_spatial_doppler": (False, True),
    "fbss_spatial_range_doppler": (True, True),
}


def _run_masked_fft_estimator(
    cfg: StudyConfig,
    frontend: FrontendArtifacts,
    known_cube: np.ndarray,
) -> MethodEstimate:
    start_time = time.perf_counter()
    detections = refine_detection_set_local(
        cfg,
        known_cube,
        frontend.coarse_candidates,
        frontend.search_bounds,
        candidate_pool_size=cfg.runtime_profile.coarse_candidate_count,
        range_upper_bound_m=_range_search_upper_m(cfg, frontend.search_bounds),
    )
    incremental_runtime_s = time.perf_counter() - start_time
    return MethodEstimate(
        label=METHOD_LABELS["fft_masked"],
        detections=detections,
        reported_target_count=len(detections),
        estimated_model_order=None,
        frontend_runtime_s=frontend.frontend_runtime_s,
        incremental_runtime_s=incremental_runtime_s,
        total_runtime_s=frontend.frontend_runtime_s + incremental_runtime_s,
    )


def _run_masked_music_variant(
    cfg: StudyConfig,
    known_cube: np.ndarray,
    known_mask: np.ndarray,
    frontend: FrontendArtifacts,
    *,
    variant_name: str,
) -> MethodEstimate:
    range_fbss_enabled, doppler_fbss_enabled = FBSS_ABLATION_FLAGS[variant_name]
    variant_cfg = replace(
        cfg,
        music_range_fbss_fraction=cfg.music_range_fbss_fraction if range_fbss_enabled else 0.0,
        music_doppler_fbss_fraction=cfg.music_doppler_fbss_fraction if doppler_fbss_enabled else 0.0,
    )
    estimate = _run_full_search_music(
        variant_cfg,
        known_cube,
        frontend.search_bounds,
        frontend.frontend_runtime_s,
        use_fbss=True,
        known_mask=known_mask,
    )
    return replace(estimate, label=FBSS_ABLATION_LABELS[variant_name])


def _run_music_fbss_ablation_estimators(
    cfg: StudyConfig,
    known_cube: np.ndarray,
    known_mask: np.ndarray,
    frontend: FrontendArtifacts,
) -> dict[str, MethodEstimate]:
    return {
        variant_name: _run_masked_music_variant(
            cfg,
            known_cube,
            known_mask,
            frontend,
            variant_name=variant_name,
        )
        for variant_name in FBSS_ABLATION_ORDER
    }


def run_masked_estimators_with_fbss_ablation(
    cfg: StudyConfig,
    masked_observation: MaskedObservation,
    frontend: FrontendArtifacts,
) -> tuple[dict[str, MethodEstimate], dict[str, MethodEstimate]]:
    known_cube = extract_known_symbol_cube(masked_observation)
    fbss_ablation_estimates = _run_music_fbss_ablation_estimators(
        cfg,
        known_cube,
        masked_observation.known_symbol_mask,
        frontend,
    )
    headline_music = replace(
        fbss_ablation_estimates["fbss_spatial_range_doppler"],
        label=f"{METHOD_LABELS['music_masked']} + FBSS",
    )
    estimates = {
        "fft_masked": _run_masked_fft_estimator(cfg, frontend, known_cube),
        "music_masked": replace(headline_music, label=METHOD_LABELS["music_masked"]),
    }
    return estimates, fbss_ablation_estimates


def run_masked_estimators(
    cfg: StudyConfig,
    masked_observation: MaskedObservation,
    frontend: FrontendArtifacts,
) -> dict[str, MethodEstimate]:
    """Run the active thesis comparison: masked FFT versus masked MUSIC."""

    known_cube = extract_known_symbol_cube(masked_observation)
    headline_music = _run_masked_music_variant(
        cfg,
        known_cube,
        masked_observation.known_symbol_mask,
        frontend,
        variant_name="fbss_spatial_range_doppler",
    )
    return {
        "fft_masked": _run_masked_fft_estimator(cfg, frontend, known_cube),
        "music_masked": replace(headline_music, label=METHOD_LABELS["music_masked"]),
    }
