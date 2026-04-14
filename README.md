# MUSIC in Waveform-Limited OFDM ISAC

MSEE final project in communications and signal processing.

This project started from a 1-D range-only MUSIC study that showed clear super-resolution upside under controlled conditions. This repo is the follow-on: a regime-boundary study of what survives when MUSIC operates in a waveform-limited OFDM ISAC setting with fragmented support, structured clutter, coherent multipath, and a fair FFT baseline.

## Thesis Framing

The thesis claim is not "MUSIC is generically better than FFT." It is:

**MUSIC has real super-resolution capability in realistic OFDM ISAC, but its value is regime-dependent. The project identifies two failure mechanisms---model-order overestimation under masked covariance (correctable) and nuisance-dominated spatial subspace capture (a hard boundary)---and maps the regimes where the earlier 1-D upside survives into 3-D staged operation.**

## Key Results (64-Trial FR1 Submission Bundle)

### Nominal Joint-Resolution Probability

| Scene | FFT Baseline | MDL MUSIC | Expected-Order MUSIC |
|-------|-------------|-----------|---------------------|
| Intersection | 0.156 | 0.703 | 0.844 |
| Open Aisle | 0.719 | 0.563 | 0.719 |
| Rack Aisle | 0.031 | 0.000 | 0.000 |

### What the Evidence Shows

1. **MDL path:** MUSIC wins strongly only in `intersection`, loses to FFT in `open_aisle`, and collapses in `rack_aisle`.
2. **Model-order diagnostic:** Expected-order MUSIC (K=2) recovers `open_aisle` to FFT parity and strengthens `intersection`, proving the super-resolution machinery is intact when subspace dimensioning is correct.
3. **Rack-aisle failure mechanism:** Fixing model order does not help. Stage diagnostics show nuisance-aligned azimuth candidates near the left-rack clutter at -24° capturing the spatial search in every trial, with 39% of final detections landing on a nuisance branch.
4. **Nuisance-strength sweep:** `intersection` stays robust across -6 to +6 dB clutter variation; `open_aisle` degrades sharply as nuisance power rises; `rack_aisle` remains at zero throughout.

## Project Scope

- Two-target indoor industrial scenes (open aisle, intersection, rack aisle)
- Monostatic FR1 OFDM sensing from a private-5G-style waveform anchor
- Communications-limited resource grids with fragmented occupancy
- Known-symbol sensing as the main path
- A fair comparison between masked FFT + local refinement and masked staged MUSIC + FBSS

## Methods

Headline comparison:

1. **`fft_masked`** — Weighted masked angle-range-Doppler FFT with local matched-filter refinement.
2. **`music_masked`** — Staged masked azimuth → range → Doppler MUSIC with spatial FBSS, support-aware range/Doppler FBSS where valid, and local matched-filter refinement.

Focused diagnostics:

- FBSS ablation variants
- `pilot_only` nominal diagnostic
- Expected-order MUSIC (`--music-model-order expected`)
- Fixed-order MUSIC (`--music-fixed-order`)
- Nuisance-gain-offset sweep (`nuisance_gain_offset`)
- Per-trial azimuth-stage diagnostics in `stage_diagnostics.csv`

## Repository Layout

- [src/aisle_isac](src/aisle_isac) — estimator, channel, resource-grid, and reporting code
- [run_study.py](run_study.py) — main CLI entrypoint
- [scripts/plot_results_from_csv.py](scripts/plot_results_from_csv.py) — generate story figures from saved CSVs
- [scripts/generate_1d_motivation_figure.py](scripts/generate_1d_motivation_figure.py) — 1-D range-only motivation figure
- [scripts/run_model_order_comparison_64trials.py](scripts/run_model_order_comparison_64trials.py) — nominal 64-trial MDL vs expected-order comparison
- [scripts/run_staged_submission.py](scripts/run_staged_submission.py) — staged long-run helper for follow-on experiments
- [report/current_assessment.tex](report/current_assessment.tex) — report source
- [results/submission](results/submission) — main submission output (64-trial FR1 bundle)
- [results/analysis](results/analysis) — model-order comparison tables
- [results/submission_nuisance](results/submission_nuisance) — 64-trial nuisance-strength sweep (MDL)
- [results/submission_expected_order](results/submission_expected_order) — 64-trial nuisance-strength sweep (expected K=2)

## Running

### Full submission bundle rebuild

```bash
bash scripts/build_submission_bundle.sh
PYTHONPATH=src .venv/bin/python scripts/plot_results_from_csv.py --input-root results/submission --clean-output
```

### Follow-on analyses

```bash
PYTHONPATH=src .venv/bin/python scripts/run_model_order_comparison_64trials.py
PYTHONPATH=src .venv/bin/python scripts/run_staged_submission.py
```

### Targeted debug run

```bash
PYTHONPATH=src .venv/bin/python run_study.py \
  --profile submission \
  --anchor fr1 \
  --scene-class intersection \
  --trials 8 \
  --sweeps bandwidth_span \
  --skip-pilot-only \
  --skip-representative \
  --disable-fbss-ablation \
  --output-dir results/debug_intersection \
  --clean-outputs
```

### Build the report

```bash
cd report
latexmk -pdf -outdir=build current_assessment.tex
```

## Limitations

- Staged MUSIC, not true joint 3-D MUSIC
- Fixed two-target study with three scene templates
- Scene classes are composite regimes, not clean one-factor sweeps
- FR1-only for the final submission snapshot
- MDL model-order estimation fails systematically under masked covariance
- Simulation-only: no hardware, channel estimation error, or communication-link feedback
- Known-symbol sensing; `pilot_only` collapses for both methods
