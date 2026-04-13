# MUSIC in Waveform-Limited OFDM ISAC

This repository is a single-student MSEE final project in communications and signal processing.

The project started from a 1-D range-only MUSIC study that showed clear upside under controlled conditions. This repo is the follow-on study: what survives when MUSIC is moved into a waveform-limited integrated sensing and communications (ISAC) setting with a communications-style OFDM resource grid, masked support, clutter, multipath, and a fair FFT baseline.

## Project Scope

The project is intentionally narrow:

- two-target indoor industrial scenes
- monostatic FR1 OFDM sensing
- communications-limited resource grids with fragmented occupancy
- known-symbol sensing as the main path
- `pilot_only` as a diagnostic, not a complete unknown-data receiver
- a fair comparison between masked FFT and masked staged MUSIC

The right thesis framing is not "MUSIC is generically better than FFT."

It is closer to:

**How much super-resolution benefit from MUSIC survives in realistic, waveform-limited OFDM ISAC, what adaptations are needed, and where does the method break down?**

## Current Position

The canonical saved evidence is still the FR1 submission bundle under [results/submission](results/submission).

That bundle supports a regime-boundary conclusion:

- MUSIC is not generically superior to a fair masked FFT + local-refinement baseline.
- MUSIC helps strongly in some regimes, especially the nominal `intersection` scene.
- MUSIC loses in nominal `open_aisle` in the saved submission bundle.
- MUSIC collapses in nominal `rack_aisle`.
- `pilot_only` collapses for both methods under the current receiver path.

Since that submission snapshot, the repo has added faster iteration tools and new diagnostics. The most important current exploratory finding is:

- MDL-based model-order estimation is materially hurting staged MUSIC in `open_aisle` and `intersection`.
- For nominal low-trial confirmation runs at submission-density search settings, forcing `K=2` improves MUSIC in those scenes.
- `rack_aisle` still fails even with fixed `K=2`, which points to an earlier azimuth/clutter failure rather than a pure model-order problem.

Those exploratory comparisons are saved under [results/analysis](results/analysis). They are useful for steering the project, but they are not yet a replacement for the thesis-grade submission bundle.

## Methods

Headline comparison:

1. `fft_masked`
   Weighted masked angle-range-Doppler FFT with local matched-filter refinement.
2. `music_masked`
   Staged masked azimuth -> range -> Doppler MUSIC with spatial FBSS, support-aware range/Doppler FBSS where valid, and local matched-filter refinement.

Focused diagnostics:

- FBSS ablation variants
- `pilot_only` nominal diagnostic
- fixed-order MUSIC via `--music-fixed-order`
- staged candidate logging in the trial-level CSVs

## Repository Layout

- [src/aisle_isac](src/aisle_isac): estimator, channel, resource-grid, and reporting code
- [run_study.py](run_study.py): main CLI entrypoint
- [report/current_assessment.tex](report/current_assessment.tex): current report source
- [results/submission](results/submission): canonical saved submission bundle
- [results/analysis](results/analysis): exploratory analyses used to steer the next phase

## Canonical Submission Snapshot

Saved bundle:

- anchor: `fr1`
- profile: `submission`
- scenes: `open_aisle`, `intersection`, `rack_aisle`
- trials per sweep point: `64`
- public sweeps: `allocation_family`, `occupied_fraction`, `fragmentation`, `bandwidth_span`, `slow_time_span`, `range_separation`, `velocity_separation`, `angle_separation`

Nominal `Pjoint` from the saved submission bundle:

- `open_aisle`: FFT `0.719`, MUSIC `0.563`
- `intersection`: FFT `0.156`, MUSIC `0.703`
- `rack_aisle`: FFT `0.031`, MUSIC `0.000`

Paired nominal interpretation:

- `open_aisle`: FFT advantage is statistically supported
- `intersection`: MUSIC advantage is statistically supported
- `rack_aisle`: shared-failure region, not a meaningful nominal win for either method

## Fast Iteration Workflow

The full submission bundle still takes hours. The repo now supports much faster targeted runs.

Important CLI options in [run_study.py](run_study.py):

- `--sweeps`: run only selected sweep families
- `--skip-pilot-only`: skip the pilot-only nominal diagnostic
- `--skip-representative`: skip representative-trial artifacts
- `--disable-fbss-ablation`: skip FBSS ablation runs
- `--music-fixed-order`: force MUSIC to use a fixed model order
- `--output-dir`: write debug runs outside the canonical results tree

Recommended iteration style:

- use submission-density search settings with low trial counts when debugging estimator behavior
- avoid overwriting `results/submission`
- treat `quick` as a smoke profile, not as the main scientific debug mode

Example: targeted debug run on one scene and one sweep

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

Example: fixed-order MUSIC diagnostic

```bash
PYTHONPATH=src .venv/bin/python run_study.py \
  --profile submission \
  --anchor fr1 \
  --scene-class open_aisle \
  --trials 8 \
  --sweeps bandwidth_span \
  --skip-pilot-only \
  --skip-representative \
  --disable-fbss-ablation \
  --music-fixed-order 2 \
  --output-dir results/debug_open_k2 \
  --clean-outputs
```

## Rebuilding the Submission Bundle

```bash
bash scripts/build_submission_bundle.sh
PYTHONPATH=src .venv/bin/python scripts/plot_results_from_csv.py --input-root results/submission --clean-output
```

## Report

The current report source is [report/current_assessment.tex](report/current_assessment.tex).

Build it locally with:

```bash
cd report
latexmk -pdf -outdir=build current_assessment.tex
```

Build artifacts are not treated as canonical repository content.

## Current Limits

- staged MUSIC, not true joint 3-D MUSIC
- fixed two-target study
- scene classes are composite regimes, not clean one-factor sweeps
- known-symbol sensing is still the main path
- exploratory model-order and stage-diagnostic findings are not yet rolled into a revised thesis-grade report

## Near-Term Direction

The next technically justified steps are:

1. treat known-target-count or capped-order MUSIC as a formal study variant for this fixed two-mover project
2. use the new stage diagnostics to explain why `rack_aisle` fails even when model order is fixed
3. add a controlled factor study so the final report is not only a regime map, but also has at least one cleaner causal result
