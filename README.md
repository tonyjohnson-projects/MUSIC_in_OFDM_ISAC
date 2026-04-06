# MUSIC for Waveform-Limited OFDM ISAC

This repository is scoped as a single-student MSEE final project.
The current thesis framing is a regime-boundary study against a fair masked FFT baseline, not a broad positive proof that MUSIC is generically better.

The repo currently contains the full FR1 submission-grade evidence path, including:

- a fairer `fft_masked` baseline with local matched-filter refinement
- the staged `music_masked` pipeline with spatial/range/Doppler FBSS options
- per-trial CSV export
- a `pilot_only` diagnostic
- a completed and audited 64-trial FR1 submission bundle under [results/submission](results/submission)

## Current Status

As of April 4, 2026, the project status is:

- the broad positive thesis has narrowed; MUSIC is not generically better than a fair masked FFT baseline
- paired nominal trial analysis now supports the main nominal verdicts statistically: open-aisle favors FFT, intersection favors MUSIC, and rack-aisle is mostly shared failure
- the saved `submission` bundle has been checked for internal consistency, and no aggregation or deterministic replay bug was found in the audited runs
- the CSV plotting path has been trimmed to a small story-first figure set under [results/submission/figures_from_csv](results/submission/figures_from_csv)
- the older, denser CSV figure set was archived under [results/submission/figures_from_csv_archive_20260404](results/submission/figures_from_csv_archive_20260404)
- the nominal runtime summary shows that MUSIC adds about `0.20-0.24 s` per nominal sweep point over FFT, or about `8-10%` total runtime in the current FR1 bundle
- the current local test status is `25 passed` via `PYTHONPATH=src .venv/bin/python -m pytest -q`

## Final Defensible Claim

The original broad thesis is not supported after the remaining actionable gaps were closed.

The strongest claim the saved evidence supports is:

**Against a fair masked FFT + local-refinement baseline, MUSIC is not generically superior for waveform-limited OFDM ISAC. Its gains are strongly regime-specific: it wins clearly in the `intersection` nominal scene and in selected support/separation sweeps, loses in `open_aisle`, collapses in the `rack_aisle` nominal point, and provides no useful nominal result under `pilot_only` sensing. The paired nominal trial logs support the `open_aisle` loss and `intersection` win statistically, while `rack_aisle` is best interpreted as a shared-failure region rather than a meaningful nominal win for either method.**

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

The current thesis-grade snapshot is the saved FR1 submission bundle recorded in [results/submission/build_manifest.txt](results/submission/build_manifest.txt) on April 4, 2026:

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
- `rack_aisle`: both methods are poor and MUSIC fully collapses at the nominal point

Paired nominal significance from saved trial logs:

- `open_aisle`: McNemar exact `p≈0.041`, RMSE sign-test `p≈7.7e-5`
- `intersection`: McNemar exact `p≈5.8e-11`, RMSE sign-test `p≈2.6e-4`
- `rack_aisle`: McNemar exact `p=0.5`, RMSE sign-test `p≈0.79`

`pilot_only` nominal diagnostic:

- all three scenes: FFT `0.000`, MUSIC `0.000`

Sweep-level usefulness windows (`music_minus_fft >= 0.10` on the active headline metric):

- `open_aisle`: `1 / 37`
- `intersection`: `20 / 37`
- `rack_aisle`: `12 / 37`

The rack-aisle usefulness points are concentrated in easier separation sweeps, not the nominal cluttered point. This is why the project is best argued as a regime map, not as a blanket success result.

Nominal runtime summary:

- `open_aisle`: FFT `2.445 s`, MUSIC `2.647 s` (`+0.202 s`, `+8.3%`)
- `intersection`: FFT `2.431 s`, MUSIC `2.630 s` (`+0.200 s`, `+8.2%`)
- `rack_aisle`: FFT `2.424 s`, MUSIC `2.661 s` (`+0.238 s`, `+9.8%`)

So in this bundle the practical objection to MUSIC is mainly conditional usefulness, not a dramatic runtime penalty.

## Audit Status

The latest repo audit did not find evidence that the disappointing submission outcome came from a corrupted bundle or a summary bug.

In the current saved bundle:

- `trial_level_results.csv` reconstructs the exported summary CSVs
- sampled saved trials replay deterministically aside from runtime fields
- the submission figure set has been reduced to five story figures and visually re-audited for readability

The interpretation issue that remains is methodological, not a broken run: MUSIC helps in some regimes, but that gain is not broad enough to support the original thesis.

## What Is Now Closed

The repo has closed the major defensibility gaps that were still actionable in code:

- the full 64-trial submission bundle exists and is reproducible from the current tree
- the baseline is fairer because `fft_masked` now gets local refinement
- every run writes `trial_level_results.csv`
- the study includes a `pilot_only` diagnostic
- the reporting pipeline now exposes positive cases, negative cases, and ablations from saved CSVs
- the CSV figure output has been trimmed to a concise story-first set instead of a large appendix-style dump

Important saved artifacts:

- headline submission data: [results/submission/data](results/submission/data)
- story-first submission plots: [results/submission/figures_from_csv](results/submission/figures_from_csv)
- archived CSV figure set: [results/submission/figures_from_csv_archive_20260404](results/submission/figures_from_csv_archive_20260404)
- legacy run-study figures: [results/submission/figures](results/submission/figures)
- assessment report source: [report/current_assessment.tex](report/current_assessment.tex)
- assessment report PDF: [report/build/current_assessment.pdf](report/build/current_assessment.pdf)

## Remaining Limits

These are now limitations of scope, not unclosed repo gaps:

- the active MUSIC path is staged, not true joint 3D MUSIC
- the main study still assumes known symbols; `pilot_only` is only a diagnostic, not a complete unknown-data receiver
- the scene model is still a fixed two-target study
- the scene classes are composite regimes rather than a clean one-factor coherence sweep
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

Refresh story figures from existing CSVs only, without rerunning Monte Carlo:

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_results_from_csv.py --input-root results/submission --clean-output
```

The submission bundle enforces the default 64-trial floor.
If you need a smoke submission bundle for development, set `ALLOW_SMOKE_SUBMISSION=1` explicitly.

## Artifact Outputs

Every `run_study.py` execution writes:

- CSV artifacts under `results/<profile>/data`
- legacy study figures under `results/<profile>/figures`

The main saved CSV artifacts include:

- per-sweep CSV summaries
- `all_sweep_results.csv`
- `trial_level_results.csv`
- `nominal_summary.csv`
- `pilot_only_nominal_summary.csv`
- `usefulness_windows.csv`
- `fbss_ablation_results.csv`
- representative mask / geometry / spectrum CSVs

The CSV-driven story-figure script reads saved CSVs only and writes six figures under `results/<profile>/figures_from_csv`:

- `story_nominal_verdict_from_csv.png`
- `story_intersection_resolution_from_csv.png`
- `story_regime_map_from_csv.png`
- `story_coherence_overlap_from_csv.png`
- `story_pilot_only_collapse_from_csv.png`
- `story_trial_delta_from_csv.png`

For the current submission bundle, the previous exhaustive CSV figure set has been archived under `results/submission/figures_from_csv_archive_20260404`.

To compile the current assessment report:

```bash
cd report
latexmk -pdf -outdir=build current_assessment.tex
```
