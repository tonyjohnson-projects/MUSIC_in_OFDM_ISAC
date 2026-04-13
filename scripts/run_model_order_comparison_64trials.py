"""Run 64-trial nominal comparison: MDL vs expected-order MUSIC vs FFT."""
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import csv
from aisle_isac.scenarios import build_study_config
from aisle_isac.scheduled_study import run_communications_study

OUTPUT = REPO / "results" / "analysis" / "model_order_nominal_64trials.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SCENES = ("open_aisle", "intersection", "rack_aisle")
MODES = [
    ("mdl", None),
    ("expected", None),
]

rows = []
for scene in SCENES:
    for mode, fixed_order in MODES:
        cfg = build_study_config(
            "fr1", scene, "submission",
            trial_count_override=64,
            music_model_order_mode=mode,
            music_fixed_model_order=fixed_order,
        )
        study = run_communications_study(
            cfg, show_progress=True,
            suite="headline",
            sweep_names=("nominal",),
            include_pilot_only=False,
            include_representative=False,
        )
        nom = study.nominal_point
        fft = nom.method_summaries["fft_masked"]
        mus = nom.method_summaries["music_masked"]
        rows.append({
            "scene": scene,
            "music_model_order_mode": mode,
            "fft_joint_pres": f"{fft.joint_resolution_probability:.6f}",
            "music_joint_pres": f"{mus.joint_resolution_probability:.6f}",
            "fft_joint_pdet": f"{fft.joint_detection_probability:.6f}",
            "music_joint_pdet": f"{mus.joint_detection_probability:.6f}",
            "music_mean_order": f"{mus.mean_estimated_model_order:.1f}" if mus.mean_estimated_model_order is not None else "",
        })
        print(f"  {scene}/{mode}: FFT Pres={fft.joint_resolution_probability:.3f}, MUSIC Pres={mus.joint_resolution_probability:.3f}")
        # Free the full study result before the next iteration.
        del study, nom, fft, mus, cfg
        gc.collect()

with open(OUTPUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nWritten to {OUTPUT}")
