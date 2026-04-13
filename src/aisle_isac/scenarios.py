"""Scenario factories for the communications-limited MUSIC study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aisle_isac.config import (
    ArrayGeometry,
    BurstProfile,
    OutputConfig,
    RuntimeProfile,
    ScattererTemplate,
    SceneClass,
    StudyConfig,
    TargetClass,
    WaveformAnchor,
)


def build_runtime_profile(profile_name: str) -> RuntimeProfile:
    """Return one of the supported runtime profiles."""

    if profile_name == "quick":
        return RuntimeProfile(
            name="quick",
            n_trials=8,
            n_simulated_subcarriers=96,
            fft_range_oversample=4,
            fft_doppler_oversample=4,
            fft_angle_oversample=4,
            music_grid_points=61,
            coarse_candidate_count=4,
        )
    if profile_name == "submission":
        return RuntimeProfile(
            name="submission",
            n_trials=64,
            n_simulated_subcarriers=96,
            fft_range_oversample=6,
            fft_doppler_oversample=6,
            fft_angle_oversample=6,
            music_grid_points=81,
            coarse_candidate_count=4,
        )
    raise ValueError("profile_name must be either 'quick' or 'submission'")


def build_waveform_anchor(anchor_name: str) -> WaveformAnchor:
    """Build one of the two public waveform anchors."""

    if anchor_name == "fr1":
        return WaveformAnchor(
            name="fr1",
            label="FR1",
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100.0e6,
            subcarrier_spacing_hz=30.0e3,
        )
    if anchor_name == "fr2":
        return WaveformAnchor(
            name="fr2",
            label="FR2",
            carrier_frequency_hz=28.0e9,
            bandwidth_hz=400.0e6,
            subcarrier_spacing_hz=120.0e3,
        )
    raise ValueError("anchor_name must be one of 'fr1' or 'fr2'")


def build_burst_profile(profile_name: str) -> BurstProfile:
    """Return the requested NR-like CPI template."""

    if profile_name == "short_cpi":
        return BurstProfile(name="short_cpi", label="Short CPI", n_snapshots=8)
    if profile_name == "balanced_cpi":
        return BurstProfile(name="balanced_cpi", label="Balanced CPI", n_snapshots=16)
    if profile_name == "long_cpi":
        return BurstProfile(name="long_cpi", label="Long CPI", n_snapshots=32)
    raise ValueError("profile_name must be one of 'short_cpi', 'balanced_cpi', or 'long_cpi'")


def build_array_geometry(rx_columns: int = 8) -> ArrayGeometry:
    """Return the study's horizontal TDM-MIMO base-station array.

    TX spacing is set to N_rx * d_rx (= rx_columns * 0.5 lambda) so that the
    virtual array is a filled ULA with 2 * rx_columns unique elements at
    half-wavelength spacing -- the standard design for maximum virtual aperture.
    """

    return ArrayGeometry(
        name=f"linear_2x{rx_columns}",
        n_tx=2,
        n_rx_rows=1,
        n_rx_cols=rx_columns,
        tx_spacing_lambda=rx_columns * 0.5,
    )


def build_target_catalog() -> dict[str, TargetClass]:
    """Return the fixed industrial mover classes."""

    return {
        "amr": TargetClass(
            name="amr",
            label="AMR",
            rcs_db=0.0,
            height_m=0.55,
            nominal_speed_mps=1.2,
        ),
        "forklift": TargetClass(
            name="forklift",
            label="Forklift",
            rcs_db=4.5,
            height_m=1.80,
            nominal_speed_mps=2.0,
        ),
    }


def build_scene_class(scene_name: str) -> SceneClass:
    """Build one of the supported indoor industrial scene classes."""

    if scene_name == "open_aisle":
        return SceneClass(
            name=scene_name,
            label="Open Aisle",
            target_pair=("amr", "amr"),
            nominal_range_m=24.0,
            nominal_azimuth_center_deg=0.0,
            nominal_center_velocity_mps=1.2,
            nominal_snr_db=20.0,
            default_range_separation_cells=1.25,
            default_velocity_separation_cells=1.35,
            default_angle_separation_cells=1.25,
            second_target_power_offset_db=-1.0,
            base_station_height_m=4.0,
            target_coherence=0.15,
            center_range_jitter_m=0.20,
            range_separation_jitter_cells=0.08,
            velocity_separation_jitter_cells=0.08,
            angle_separation_jitter_cells=0.08,
            target_amplitude_jitter_db=0.40,
            nuisance_gain_jitter_db=0.75,
            static_clutter=(
                ScattererTemplate("support_beam", -8.0, -18.0, -20.0),
                ScattererTemplate("column", 11.0, 14.0, -23.0),
            ),
            multipath=(
                ScattererTemplate("floor_bounce", 1.5, 0.0, -16.0, coherent_with_target_index=0),
            ),
        )

    if scene_name == "rack_aisle":
        return SceneClass(
            name=scene_name,
            label="Rack Aisle",
            target_pair=("amr", "amr"),
            nominal_range_m=22.0,
            nominal_azimuth_center_deg=2.0,
            nominal_center_velocity_mps=1.0,
            nominal_snr_db=16.0,
            default_range_separation_cells=0.85,
            default_velocity_separation_cells=0.90,
            default_angle_separation_cells=0.90,
            second_target_power_offset_db=-3.0,
            base_station_height_m=4.5,
            target_coherence=0.65,
            center_range_jitter_m=0.25,
            range_separation_jitter_cells=0.10,
            velocity_separation_jitter_cells=0.10,
            angle_separation_jitter_cells=0.10,
            target_amplitude_jitter_db=0.50,
            nuisance_gain_jitter_db=1.00,
            static_clutter=(
                ScattererTemplate("left_rack", -7.5, -24.0, -12.0),
                ScattererTemplate("right_rack", -7.0, 23.0, -12.5),
                ScattererTemplate("far_endcap", 9.0, 3.0, -18.0),
            ),
            multipath=(
                ScattererTemplate("left_wall_bounce", 1.0, -11.0, -8.0, coherent_with_target_index=0),
                ScattererTemplate("right_wall_bounce", 1.6, 10.0, -9.0, coherent_with_target_index=1),
            ),
        )

    if scene_name == "intersection":
        return SceneClass(
            name=scene_name,
            label="Intersection",
            target_pair=("amr", "forklift"),
            nominal_range_m=18.0,
            nominal_azimuth_center_deg=10.0,
            nominal_center_velocity_mps=0.5,
            nominal_snr_db=18.0,
            default_range_separation_cells=1.00,
            default_velocity_separation_cells=1.25,
            default_angle_separation_cells=1.55,
            second_target_power_offset_db=1.5,
            base_station_height_m=4.5,
            target_coherence=0.30,
            center_range_jitter_m=0.20,
            range_separation_jitter_cells=0.08,
            velocity_separation_jitter_cells=0.10,
            angle_separation_jitter_cells=0.08,
            target_amplitude_jitter_db=0.45,
            nuisance_gain_jitter_db=0.90,
            static_clutter=(
                ScattererTemplate("cross_aisle_wall", 4.0, -28.0, -15.0),
                ScattererTemplate("forklift_bay", 7.5, 24.0, -17.0),
                ScattererTemplate("storage_corner", 12.0, 36.0, -21.0),
            ),
            multipath=(
                ScattererTemplate("floor_return", 1.5, 0.0, -13.5, coherent_with_target_index=1),
            ),
        )

    raise ValueError("scene_name must be one of 'open_aisle', 'rack_aisle', or 'intersection'")


def build_study_config(
    anchor_name: str,
    scene_name: str,
    profile_name: str,
    burst_profile_name: str = "balanced_cpi",
    rx_columns: int = 8,
    suite: str = "headline",
    trial_count_override: int | None = None,
    music_model_order_mode: str = "mdl",
    music_fixed_model_order: int | None = None,
    enable_fbss_ablation: bool = True,
    global_nuisance_gain_offset_db: float = 0.0,
    skip_local_refinement: bool = False,
) -> StudyConfig:
    """Resolve one complete study configuration."""

    if music_model_order_mode not in {"mdl", "fixed", "expected"}:
        raise ValueError("music_model_order_mode must be 'mdl', 'fixed', or 'expected'")
    if music_fixed_model_order is not None and music_fixed_model_order < 1:
        raise ValueError("music_fixed_model_order must be at least 1 when provided")
    if music_model_order_mode == "fixed" and music_fixed_model_order is None:
        raise ValueError("music_fixed_model_order is required when music_model_order_mode='fixed'")

    runtime_profile = build_runtime_profile(profile_name)
    if trial_count_override is not None:
        runtime_profile = replace(runtime_profile, n_trials=trial_count_override)
    scene_class = build_scene_class(scene_name)
    return StudyConfig(
        anchor=build_waveform_anchor(anchor_name),
        burst_profile=build_burst_profile(burst_profile_name),
        array_geometry=build_array_geometry(rx_columns=rx_columns),
        scene_class=scene_class,
        runtime_profile=runtime_profile,
        output_config=OutputConfig(root_dir=Path("results") / profile_name),
        sweep_suite=suite,
        evidence_profile_name="music_waveform_limited_v2",
        target_coherence=scene_class.target_coherence,
        center_range_jitter_m=scene_class.center_range_jitter_m,
        range_separation_jitter_cells=scene_class.range_separation_jitter_cells,
        velocity_separation_jitter_cells=scene_class.velocity_separation_jitter_cells,
        angle_separation_jitter_cells=scene_class.angle_separation_jitter_cells,
        target_amplitude_jitter_db=scene_class.target_amplitude_jitter_db,
        nuisance_gain_jitter_db=scene_class.nuisance_gain_jitter_db,
        music_model_order_mode=music_model_order_mode,
        music_fixed_model_order=music_fixed_model_order,
        enable_fbss_ablation=enable_fbss_ablation,
        global_nuisance_gain_offset_db=global_nuisance_gain_offset_db,
        skip_local_refinement=skip_local_refinement,
    )
