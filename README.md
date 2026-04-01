# Waveform-Constrained Angle-Range-Doppler Super-Resolution Sensing with Private-5G MIMO-OFDM

This project is a radar-first, private-5G-inspired simulation study of angle-range-Doppler sensing in stylized indoor industrial scenes. The sensing task centers on closely spaced AMRs and forklifts moving through open aisles, rack aisles, and intersections. Communications enters the study only through waveform constraints: carrier frequency, bandwidth, subcarrier spacing, burst profile, CPI length, and horizontal array aperture.

The estimator family is:

- `Angle-Range-Doppler FFT`
- `FFT-Seeded Staged Azimuth MUSIC`
- `FFT-Seeded Azimuth MUSIC + FBSS`

The goal is to determine when FFT-seeded azimuth-domain subspace refinement provides practical gains in range, Doppler, and azimuth estimation under private-5G-inspired waveform and environment constraints.

## Overview

The simulator uses an NR-like MIMO-OFDM sensing architecture with two waveform anchors:

- `FR1`: `3.5 GHz`, `100 MHz`, `30 kHz` SCS
- `FR2`: `28 GHz`, `400 MHz`, `120 kHz` SCS

The code derives a full physical OFDM grid from the anchor numerology, then selects a reduced set of simulated tones from that grid for runtime control. Range steering, FFT axes, and CRB calculations all use this physical-tone model rather than a synthetic uniformly spread frequency grid.

The sensing data cube is azimuth-facing and is indexed by:

- horizontal virtual antenna index
- simulated subcarrier index
- slow-time snapshot index

This is an angle-range-Doppler study. It does not estimate elevation, and it should be read as a stylized case study rather than as a broad industrial performance claim.

## Signal and Scenario Model

The base station uses a horizontal TDM-MIMO aperture with `2` transmitters and `N` horizontal receivers at half-wavelength spacing. Public aperture sweeps vary `N` over `8`, `12`, `16`, and `24`.

Targets are modeled as industrial movers:

- `AMR`
- `Forklift`

Supported scene classes are:

- `open_aisle`
- `rack_aisle`
- `intersection`

Each scene defines:

- nominal range, azimuth, and radial velocity
- target coherence level
- clutter and multipath templates
- nominal scene SNR used only for baseline receiver-noise calibration

Propagation uses an azimuth-only monostatic proxy with received amplitude proportional to `sqrt(RCS) / R^2`. Receiver noise is calibrated once per anchor and scene at the nominal balanced-CPI, `8`-column aperture point and then held fixed across sweeps so the absolute-range study remains meaningful.

## Estimator Stack

The FFT front-end is shared across all methods:

1. embed simulated tones onto the physical OFDM grid
2. form an angle-range-Doppler FFT cube
3. estimate a noise floor from the FFT power cube
4. extract local maxima with thresholding and candidate backfill

The MUSIC variants are not standalone detectors in this repository. They are staged FFT-seeded refiners:

1. estimate azimuth with covariance-based MUSIC
2. beamform on the refined azimuth
3. refine range on the selected tone grid
4. refine Doppler on the slow-time grid
5. merge duplicate hypotheses with normalized-cell NMS

`FFT-Seeded Azimuth MUSIC + FBSS` applies forward-backward spatial smoothing during the azimuth refinement stage to improve robustness under coherent rack-aisle returns.

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

Joint detection requires both movers to be assigned within one nominal resolution cell in range, velocity, and azimuth. Joint resolution uses the stricter `0.35`-cell tolerance used by the evaluation code. The exported tables now report both unconditional metrics and conditional-on-success metrics so weak methods do not appear cleaner simply because they fail more often.

## Runtime Profiles

- `quick`: `8` trials, reduced grid density, smoke-test profile
- `paper`: `128` trials, full evidence profile

`run_study.py` also accepts `--trials` to override the Monte Carlo count without editing source files.

## Repository Organization

- `src/`
  waveform definitions, signal synthesis, estimators, metrics, reporting, and study orchestration
- `results/`
  generated CSVs and figures for `quick` and `paper` runs
- `scripts/`
  repeatable build utilities for regenerating result bundles and manifests

## Running the Study

Use the project virtual environment directly:

```bash
./.venv/bin/python run_study.py --profile quick --suite headline --anchor fr1 --scene-class open_aisle --jobs 1 --clean-outputs
```

To override the trial count:

```bash
./.venv/bin/python run_study.py --profile quick --suite headline --anchor fr1 --scene-class open_aisle --jobs 1 --trials 1 --clean-outputs
```

To run the full evidence-oriented bundle:

```bash
./.venv/bin/python run_study.py --profile paper --suite headline --anchor all --scene-class all --jobs 4 --clean-outputs
```

For a repeatable results-only build with manifest generation:

```bash
bash scripts/build_results_bundle.sh
```

## Limitations

- The scene model is hand-crafted and deployment-motivated, not measurement-backed.
- The clutter and multipath templates support illustrative comparisons, not broad industrial claims.
- The MUSIC methods are FFT-seeded refiners in this codebase, not independent global detectors.
- The repository currently focuses on code and generated results only; presentation and manuscript artifacts have been removed.

## Tests

The canonical test command is:

```bash
./.venv/bin/python -m unittest discover -s tests -q
```

The tests verify physical OFDM-grid consistency, fixed noise calibration, search-domain validity, candidate backfill behavior, corrected runtime fields, CSV-schema distinctness, and CLI execution without `uv`.

## References

- [ETSI TS 38.211: NR Physical Channels and Modulation](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.05.00_60/ts_138211v180500p.pdf)
- [Kai Wu, Jian Andrew Zhang, Xiaojing Huang, and Yingjie Jay Guo, "Joint Communications and Sensing Employing Multi- or Single-Carrier OFDM Communication Signals: A Tutorial on Sensing Methods, Recent Progress and a Novel Design"](https://www.mdpi.com/1424-8220/22/4/1613)
- [Chaoyue Zhang, Zhiwen Zhou, Huizhi Wang, and Yong Zeng, "Integrated Super-Resolution Sensing and Communication with 5G NR Waveform: Signal Processing with Uneven CPs and Experiments"](https://arxiv.org/abs/2305.05142)
- [Zichao Xiao, Rang Liu, Ming Li, Qian Liu, and A. Lee Swindlehurst, "A Novel Joint Angle-Range-Velocity Estimation Method for MIMO-OFDM ISAC Systems"](https://par.nsf.gov/servlets/purl/10643382)
- [Musa Furkan Keskin, Mohammad Mahdi Mojahedian, Jesus O. Lacruz, Carina Marcus, Olof Eriksson, Andrea Giorgetti, Joerg Widmer, and Henk Wymeersch, "Fundamental Trade-Offs in Monostatic ISAC: A Holistic Investigation Towards 6G"](https://arxiv.org/abs/2401.18011)
