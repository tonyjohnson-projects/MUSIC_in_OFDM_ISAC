# MUSIC for Communications-Limited OFDM ISAC

This repository is scoped as a single-student MSEE final project.

The thesis is:

**MUSIC can provide range-angle-Doppler super-resolution on communications-limited OFDM radar/ISAC waveforms, but its benefit depends strongly on how much frequency-time support the communications scheduler leaves available.**

## Real-Life Problem

The project is anchored in an indoor private-5G style deployment:

- a ceiling-mounted monostatic base-station array
- warehouse and factory scenes with AMRs and forklifts
- OFDM waveforms whose sensing support is constrained by communications scheduling
- fragmented frequency allocations, sparse reference structure, and puncturing

The question is not whether MUSIC works on an ideal radar waveform.
The question is whether it still works when sensing has to live inside a communications resource grid.

## Active Study

The runnable study compares:

1. `fft_masked`
   - a masked angle-range-Doppler FFT baseline
2. `music_masked`
   - masked full-search MUSIC with FBSS on the spatial covariance

The active study sweeps are:

- allocation family
- occupied resource fraction
- fragmentation
- range separation
- velocity separation
- angle separation

The nominal waveform-limited case is a fragmented PRB allocation on an FR1-like carrier in stylized indoor industrial scenes.

## What This Repo Supports

The current codebase supports these thesis claims:

- MUSIC can be evaluated as the primary super-resolution tool on communications-limited OFDM grids.
- The comparison against FFT is done on the same masked observations.
- Range, azimuth, and Doppler are all estimated jointly in the study.
- The scenes are tied to plausible indoor industrial deployments rather than abstract point-target-only benchmarks.

The repo does **not** claim:

- hardware validation
- 3GPP link realism
- unknown-data sensing on arbitrary downlink payloads
- deployment-ready automotive or defense radar performance

## Scope

### In Scope

- monostatic MIMO-OFDM ISAC
- azimuth-only angle estimation
- two-target range-angle-Doppler resolution analysis
- communications-scheduled resource grids
- masked FFT versus masked MUSIC comparison
- stylized indoor industrial scenarios

### Out of Scope

- tracking across multiple CPIs
- elevation or full 3D geometry
- hardware measurements
- wide multi-target scenes
- joint comms decoding and sensing receiver design
- NOMP or sparse Bayesian estimators as the main thesis path

## Code Map

- [run_study.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/run_study.py)
  Active CLI entrypoint.
- [src/aisle_isac/scheduled_study.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/scheduled_study.py)
  Sweep construction and Monte Carlo study runner.
- [src/aisle_isac/estimators_music.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/estimators_music.py)
  Active estimator comparison: masked FFT versus masked MUSIC.
- [src/aisle_isac/estimators.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/estimators.py)
  Shared FFT, covariance, and MUSIC utilities.
- [src/aisle_isac/masked_observation.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/masked_observation.py)
  Communications-limited observation synthesis.
- [src/aisle_isac/resource_grid.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/resource_grid.py)
  Frequency-time allocation families.
- [src/aisle_isac/scenarios.py](/Users/tonyjohnson/Documents/Documents_Mac/Tony UCSB 5th Year/Projects/MUSIC_in_OFDM_ISAC/src/aisle_isac/scenarios.py)
  FR1/FR2 anchors and indoor scene definitions.

## Running

Quick smoke run:

```bash
PYTHONPATH=src .venv/bin/python run_study.py --profile quick --scene-class open_aisle --anchor fr1
```

Targeted study slice:

```bash
PYTHONPATH=src .venv/bin/python run_study.py --profile quick --scene-class rack_aisle --anchor fr1 --jobs 1
```

Tests for the active thesis path:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_resource_grid.py tests/test_masked_fft.py tests/test_scheduled_study.py
```

## Evidence Standard

For the thesis to stay defensible, the repo should be used to argue comparative insight:

- when MUSIC still resolves two close targets under masked communications support
- how fast the FFT baseline degrades as support becomes sparse or fragmented
- which real-life allocation patterns are most damaging to super-resolution

The project should not be used to argue absolute field performance.
