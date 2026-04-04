# MUSIC for Waveform-Limited OFDM ISAC

This repository is scoped as a single-student MSEE final project.

The repo now contains the full FR1 submission-grade evidence path, including:

- a fairer `fft_masked` baseline with local matched-filter refinement
- the staged `music_masked` pipeline with spatial/range/Doppler FBSS options
- per-trial CSV export
- a `pilot_only` diagnostic
- a completed 64-trial FR1 submission bundle under [results/submission](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/results/submission)

## Final Defensible Claim

The original broad thesis is **not** supported after the remaining gaps were closed.

The strongest claim the saved evidence supports is:

**Against a fair masked FFT + local-refinement baseline, MUSIC is not generically superior for waveform-limited OFDM ISAC. Its gains are strongly regime-specific: it wins clearly in the `intersection` nominal scene and in selected support/separation sweeps, loses in `open_aisle`, collapses in `rack_aisle`, and provides no useful nominal result under `pilot_only` sensing.**

That is now the main value of the project: a regime map and failure-boundary study, not a broad positive proof of MUSIC.

## Research Question

The project does not ask whether MUSIC works on an ideal radar waveform.

It asks whether range, angle, and Doppler super-resolution from a communications waveform remains useful when sensing is limited by:

- fragmented OFDM support
- reduced occupied resource fraction
- reduced effective bandwidth span
- reduced effective slow-time span
- structured clutter and coherent nuisance paths

## Active Estimator Path

The headline comparison is intentionally narrow:

1. `fft_masked`
   Masked angle-range-Doppler FFT with local matched-filter refinement on the same de-embedded observation.
2. `music_masked`
   Masked staged azimuth/range/Doppler MUSIC with spatial FBSS, support-aware range/Doppler FBSS where valid, and matched-filter refinement.

Focused diagnostics:

- `fbss_spatial_only`
- `fbss_spatial_range`
- `fbss_spatial_doppler`
- `fbss_spatial_range_doppler`
- `pilot_only` nominal diagnostic on the same fragmented PRB grid

## Study Design

### Scenes

- `open_aisle`
  Cleaner aisle geometry.
- `intersection`
  Mixed geometry where MUSIC is currently strongest.
- `rack_aisle`
  Clutter/coherence stress case and explicit failure boundary.

### Public Sweeps

- `allocation_family`
- `occupied_fraction`
- `fragmentation`
- `bandwidth_span`
- `slow_time_span`
- `range_separation`
- `velocity_separation`
- `angle_separation`

The separation sweeps are axis-isolated:

- range sweep keeps angle and Doppler comfortably separated
- velocity sweep keeps range and angle comfortably separated
- angle sweep keeps range and Doppler comfortably separated

### Signal / Scene Model

- snapshot-varying stochastic target source processes
- explicit target coherence control
- coherent nuisance paths tied to the appropriate target process
- bounded trial-to-trial jitter in geometry and path strengths

## Submission Snapshot

The current thesis-grade snapshot is the saved FR1 submission run from April 3, 2026:

- profile: `submission`
- trials per sweep point: `64`
- scenes: `open_aisle`, `intersection`, `rack_aisle`
- anchor: `fr1`

Nominal joint-resolution results:

- `open_aisle`: FFT `0.719` vs MUSIC `0.563`
- `intersection`: FFT `0.156` vs MUSIC `0.703`
- `rack_aisle`: FFT `0.031` vs MUSIC `0.000`

Nominal interpretation:

- `open_aisle`: the fairer FFT baseline now beats MUSIC
- `intersection`: MUSIC is clearly better
- `rack_aisle`: both methods are poor and MUSIC fully collapses

`pilot_only` nominal diagnostic:

- all three scenes: FFT `0.000`, MUSIC `0.000`

So the repo no longer supports the simpler story that MUSIC is broadly useful whenever the scene is favorable. The stronger and more honest result is that MUSIC helps in some specific regimes, but the gain is fragile enough that a better baseline and reduced transmitter knowledge can erase it completely.

Sweep-level usefulness windows (`music_minus_fft >= 0.10` on the active headline metric):

- `open_aisle`: `1 / 37`
- `intersection`: `20 / 37`
- `rack_aisle`: `12 / 37`

The rack-aisle usefulness points are concentrated in easier separation sweeps, not the nominal cluttered point. This is why the project is best argued as a regime map, not as a blanket success result.

## What Is Now Closed

The repo has closed the major defensibility gaps that were still actionable in code:

- the full 64-trial submission bundle exists and is reproducible
- stale submission artifacts are gone and the bundle scripts enforce the current schema
- the baseline is fairer because `fft_masked` now gets local refinement
- every run writes `trial_level_results.csv`
- the study includes a `pilot_only` diagnostic
- the reporting and plotting pipeline now exposes positive cases, negative cases, and ablations from saved CSVs

Important saved artifacts:

- headline submission data: [results/submission/data](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/results/submission/data)
- headline submission plots: [results/submission/figures_from_csv](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/results/submission/figures_from_csv)
- assessment report source: [report/current_assessment.tex](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/report/current_assessment.tex)
- assessment report PDF: [report/build/current_assessment.pdf](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/report/build/current_assessment.pdf)

## Remaining Limits

These are now limitations of scope, not unclosed repo gaps:

- the active MUSIC path is staged, not true joint 3D MUSIC
- the main study still assumes known symbols; `pilot_only` is only a diagnostic, not a complete unknown-data receiver
- the scene model is still a fixed two-target study
- the project is a simulation study, not a hardware or link-level implementation

## Running

Quick FR1 sweep across all scenes:

```bash
PYTHONPATH=src .venv/bin/python run_study.py --profile quick --anchor fr1 --scene-class all --clean-outputs
PYTHONPATH=src .venv/bin/python scripts/plot_results_from_csv.py --input-root results/quick --clean-output
```

Submission-grade FR1 bundle:

```bash
bash scripts/build_submission_bundle.sh
PYTHONPATH=src .venv/bin/python scripts/plot_results_from_csv.py --input-root results/submission --clean-output
```

The submission bundle enforces the default 64-trial floor.
If you need a smoke submission bundle for development, set `ALLOW_SMOKE_SUBMISSION=1` explicitly.

## Artifact Outputs

Every `run_study.py` execution writes CSV artifacts under `results/<profile>/data`, including:

- per-sweep CSV summaries
- `all_sweep_results.csv`
- `trial_level_results.csv`
- `nominal_summary.csv`
- `pilot_only_nominal_summary.csv`
- representative mask / geometry / spectrum CSVs

The CSV-driven figure script writes:

- nominal and sweep comparisons
- usefulness and failure overviews
- representative resource-mask and geometry figures
- representative range-Doppler and MUSIC-spectrum figures
- FBSS ablation figures
- pilot-only comparison figures
- trial-level joint-error distributions

To compile the current assessment report:

```bash
cd report
latexmk -pdf -outdir=build current_assessment.tex
```
