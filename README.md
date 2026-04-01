# Waveform-Constrained Angle-Range-Doppler Super-Resolution Sensing with Private-5G MIMO-OFDM

This project is a radar-first, private-5G-inspired simulation study of angle-range-Doppler sensing in stylized indoor industrial scenes. The study uses two-target scenes built from AMR and forklift target classes: `open_aisle` and `rack_aisle` are `AMR/AMR`, while `intersection` is `AMR/Forklift`. Communications enters the study only through waveform constraints: carrier frequency, bandwidth, subcarrier spacing, burst profile, CPI length, and horizontal array aperture.

The estimator family is:

- `Angle-Range-Doppler FFT`
- `FFT-Seeded Staged Azimuth MUSIC`
- `FFT-Seeded Azimuth MUSIC + FBSS`
- `Full-Search MUSIC`

The goal is to determine when subspace-based super-resolution provides practical gains in range, Doppler, and azimuth estimation under private-5G-inspired waveform and environment constraints.

## Overview

The simulator uses an NR-like MIMO-OFDM sensing architecture with two waveform anchors:

- `FR1`: `3.5 GHz`, `100 MHz`, `30 kHz` SCS
- `FR2`: `28 GHz`, `400 MHz`, `120 kHz` SCS

The code derives a full physical OFDM grid from the anchor numerology, then samples `96` simulated tones from that grid for runtime control. Range steering, FFT axes, and CRB calculations all use this physical-tone model rather than a synthetic uniformly spread frequency grid.

The sensing data cube is azimuth-facing and is indexed by:

- horizontal virtual antenna index
- simulated subcarrier index
- slow-time snapshot index

This is a fixed two-target, angle-range-Doppler study. It does not estimate elevation, and it should be read as a stylized case study rather than as a broad industrial performance claim.

## Signal and Scenario Model

The base station uses a horizontal TDM-MIMO aperture with `2` transmitters and `N` horizontal receivers. Receiver spacing is `0.5 lambda`, and transmitter spacing is `N * 0.5 lambda`, so the effective horizontal virtual array is a filled `2N`-element ULA at half-wavelength spacing. Public aperture sweeps vary `N` over `8`, `12`, `16`, and `24`.

Targets are modeled as industrial movers:

- `AMR`
- `Forklift`

Every trial contains exactly two movers. Supported scene classes are:

- `open_aisle`: `AMR/AMR`
- `rack_aisle`: `AMR/AMR`
- `intersection`: `AMR/Forklift`

Each scene defines:

- target pair
- nominal range, azimuth, and radial velocity
- target coherence level
- second-target power offset
- static clutter and multipath templates
- nominal scene SNR used only for baseline receiver-noise calibration
- base-station height metadata

Propagation uses an azimuth-only monostatic proxy with received amplitude proportional to `lambda * sqrt(RCS) / R^2`. Receiver noise is calibrated once per anchor and scene at the nominal balanced-CPI, `8`-column aperture, `96`-tone baseline point and then held fixed across sweeps so the absolute-range study remains meaningful.

## Estimator Stack

The FFT front-end is shared across all methods:

1. embed simulated tones onto the physical OFDM grid
2. form an angle-range-Doppler FFT cube
3. estimate a noise floor from the FFT power cube
4. extract local maxima with thresholding, candidate backfill, and normalized-cell NMS

The FFT-seeded MUSIC variants are staged refiners:

1. estimate global model order with MDL on the spatial covariance
2. align each FFT candidate by its coarse range and Doppler
3. estimate azimuth with covariance-based MUSIC
4. beamform on the refined azimuth
5. refine range with MUSIC on a bounded range grid
6. refine Doppler with MUSIC on a bounded slow-time grid
7. merge duplicate hypotheses with normalized-cell NMS

`FFT-Seeded Azimuth MUSIC + FBSS` applies forward-backward spatial smoothing during the azimuth refinement stage to improve robustness under coherent rack-aisle returns.

`Full-Search MUSIC` is the standalone super-resolution baseline. It performs a global azimuth MUSIC search, constrains range search to the sparse-tone unambiguous interval, and then locally refines the final angle-range-Doppler hypotheses with matched-filter coordinate descent.

## Metrics

The study reports radar-only outputs:

- `joint_detection_probability`
- `joint_resolution_probability`
- `scene_cost`
- `unconditional_range_rmse_m`
- `unconditional_velocity_rmse_mps`
- `unconditional_angle_rmse_deg`
- `unconditional_joint_assignment_rmse`
- `unconditional_rmse_over_crb`
- `conditional_range_rmse_m`
- `conditional_velocity_rmse_mps`
- `conditional_angle_rmse_deg`
- `conditional_joint_assignment_rmse`
- `conditional_rmse_over_crb`
- `false_alarm_probability`
- `miss_probability`
- `model_order_accuracy`
- `frontend_runtime_s`
- `incremental_runtime_s`
- `total_runtime_s`

Joint detection requires both movers to be assigned within one nominal resolution cell in range, velocity, and azimuth. Joint resolution uses the stricter `0.35`-cell tolerance used by the evaluation code. Unconditional metrics penalize misses and out-of-gate assignments with one-cell errors, while conditional metrics average only over detection-gated assignments. `false_alarm_probability` and `miss_probability` are per-trial event rates, and `model_order_accuracy` checks whether the estimator returned the expected two-target model order.

## Runtime Profiles

- `quick`: `8` trials, `96` simulated tones, `4x` FFT oversampling, `61`-point MUSIC grids, smoke-test profile
- `submission`: `64` trials, `96` simulated tones, `6x` FFT oversampling, `81`-point MUSIC grids, curated evidence profile for the final write-up

`run_study.py` also accepts `--trials` to override the Monte Carlo count without editing source files. `--suite headline` uses the coarser public sweep point sets, while `--suite full` uses denser point sets for the same sweep families.

## Repository Organization

- `run_study.py`
  CLI entrypoint for generic study runs
- `src/`
  waveform definitions, signal synthesis, estimators, metrics, reporting, and study orchestration
- `tests/`
  system tests for estimator behavior, metrics, outputs, and CLI/bundle smoke coverage
- `results/`
  generated CSVs and figures for `quick` and `submission` runs
- `scripts/`
  repeatable build utilities for exploratory and submission bundles

## Running the Study

Use the project virtual environment directly:

```bash
./.venv/bin/python run_study.py --profile quick --suite headline --anchor fr1 --scene-class open_aisle --jobs 1 --clean-outputs
```

To override the trial count:

```bash
./.venv/bin/python run_study.py --profile quick --suite headline --anchor fr1 --scene-class open_aisle --jobs 1 --trials 1 --clean-outputs
```

CLI choices are:

- `--anchor {fr1, fr2, all}`
- `--scene-class {open_aisle, rack_aisle, intersection, all}`
- `--profile {quick, submission}`
- `--suite {headline, full}`

A generic CLI run writes per-sweep CSVs and figures for `range_separation`, `velocity_separation`, `angle_separation`, `absolute_range`, `burst_profile`, and `aperture`, plus `scene_comparison`, `crb_gap`, and `representative_cube_slices`. `fr1_vs_fr2` is written only when both anchors are included in the run.

For the generic results-bundle wrapper:

```bash
bash scripts/build_results_bundle.sh
```

That script defaults to `PROFILE=submission`, `SUITE=headline`, `ANCHOR=all`, and `SCENE_CLASS=all`. Its post-run artifact checks expect the full exploratory artifact set, including `fr1_vs_fr2`, so it is intended for runs that include both anchors.

To run the curated submission bundle:

```bash
bash scripts/build_submission_bundle.sh
```

The submission bundle is fixed to:

- `FR1`
- `open_aisle` and `rack_aisle`
- `range_separation`, `velocity_separation`, `angle_separation`, `burst_profile`, and `aperture`
- `scene_comparison`, `crb_gap`, and one representative cube-slice figure

It intentionally omits `absolute_range` and `fr1_vs_fr2`. `FR2`, `intersection`, `absolute_range`, and the denser `full` sweep suite remain available through the generic CLI as exploratory extensions, but they are not part of the default submission evidence path.

## Limitations

- The scene model is hand-crafted and deployment-motivated, not measurement-backed.
- The clutter and multipath templates support illustrative comparisons, not broad industrial claims.
- Every trial contains exactly two targets; this is not a general multi-target tracking framework.
- The simulator is azimuth-only. Scene base-station height is stored in configuration metadata but does not enter the current channel or estimator pipeline.
- The FFT-seeded MUSIC methods are refiners; only `Full-Search MUSIC` acts as a standalone global super-resolution baseline.
- The main submission narrative is FR1-only. FR2 and the intersection scene are retained as extensions rather than core evidence.
- The repository currently focuses on code and generated results only; presentation and manuscript artifacts have been removed.

## Tests

The canonical test command is:

```bash
./.venv/bin/python -m unittest discover -s tests -q
```

The tests verify physical OFDM-grid consistency, search-domain validity, fixed noise calibration, wavelength/range-dependent path loss, unconditional versus conditional metric behavior, candidate backfill, single-target estimator sanity, FBSS robustness in coherent rack-aisle trials, nominal `Full-Search MUSIC` detection, CSV-schema distinctness, selectable artifact sets, generic CLI execution without `uv`, and curated submission-bundle generation.

## References

- [ETSI TS 38.211: NR Physical Channels and Modulation](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.05.00_60/ts_138211v180500p.pdf)
- [Kai Wu, Jian Andrew Zhang, Xiaojing Huang, and Yingjie Jay Guo, "Joint Communications and Sensing Employing Multi- or Single-Carrier OFDM Communication Signals: A Tutorial on Sensing Methods, Recent Progress and a Novel Design"](https://www.mdpi.com/1424-8220/22/4/1613)
- [Chaoyue Zhang, Zhiwen Zhou, Huizhi Wang, and Yong Zeng, "Integrated Super-Resolution Sensing and Communication with 5G NR Waveform: Signal Processing with Uneven CPs and Experiments"](https://arxiv.org/abs/2305.05142)
- [Zichao Xiao, Rang Liu, Ming Li, Qian Liu, and A. Lee Swindlehurst, "A Novel Joint Angle-Range-Velocity Estimation Method for MIMO-OFDM ISAC Systems"](https://par.nsf.gov/servlets/purl/10643382)
- [Musa Furkan Keskin, Mohammad Mahdi Mojahedian, Jesus O. Lacruz, Carina Marcus, Olof Eriksson, Andrea Giorgetti, Joerg Widmer, and Henk Wymeersch, "Fundamental Trade-Offs in Monostatic ISAC: A Holistic Investigation Towards 6G"](https://arxiv.org/abs/2401.18011)
