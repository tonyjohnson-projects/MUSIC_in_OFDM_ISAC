"""Package exports for the active communications-limited MUSIC study."""

from aisle_isac.allocation_metrics import summarize_allocation
from aisle_isac.estimators_fft_masked import build_masked_fft_cube, build_masked_fft_cube_from_cube, prepare_masked_frontend
from aisle_isac.estimators_music import METHOD_LABELS, METHOD_ORDER, run_masked_estimators
from aisle_isac.masked_observation import extract_known_symbol_cube, simulate_masked_observation
from aisle_isac.resource_grid import ResourceElementRole, ResourceGrid, build_resource_grid
from aisle_isac.scheduled_study import nominal_trial_parameters, run_communications_study, simulate_communications_trial
from aisle_isac.scenarios import build_study_config

__all__ = [
    "METHOD_LABELS",
    "METHOD_ORDER",
    "ResourceElementRole",
    "ResourceGrid",
    "build_masked_fft_cube",
    "build_masked_fft_cube_from_cube",
    "build_resource_grid",
    "build_study_config",
    "extract_known_symbol_cube",
    "nominal_trial_parameters",
    "prepare_masked_frontend",
    "run_communications_study",
    "run_masked_estimators",
    "simulate_communications_trial",
    "simulate_masked_observation",
    "summarize_allocation",
]
