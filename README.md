# MUSIC in OFDM ISAC

This repository is a simulation study of two-target resolution in communications-limited OFDM integrated sensing and communications (ISAC). The code constructs a scene, synthesizes a noisy TDM-MIMO OFDM radar cube, applies realistic communications scheduling masks, runs a masked FFT baseline and a staged masked MUSIC pipeline, evaluates both methods against truth, and writes the resulting tables and figures used by the rest of the project.

The core implementation lives in `src/aisle_isac`. The top-level executable is `run_study.py`. The `figure_scripts`, `presentation`, and `report` directories sit downstream of the main study and consume either the same modules or the generated result bundles.

## What the code is actually doing

At the highest level, each Monte Carlo trial does this:

1. Pick one waveform anchor, scene template, burst length, array aperture, and sweep point.
2. Realize two moving targets plus scene-dependent nuisance scatterers.
3. Synthesize a complex-valued radar cube over horizontal virtual array index, subcarrier index, and OFDM snapshot index.
4. Apply a communications resource grid and transmit symbols to create a masked observation.
5. Recover the subset of sensing samples whose transmit symbols are known.
6. Build a masked FFT front-end from that irregular support.
7. Run two estimators:
   - `fft_masked`: masked FFT + local matched-filter refinement
   - `music_masked`: staged MUSIC with spatial/range/Doppler FBSS when support allows it
8. Match estimated detections to ground truth with a Hungarian assignment.
9. Aggregate detection, resolution, RMSE, false-alarm, miss, and runtime statistics across many trials and sweep points.
10. Write CSV summaries and diagnostic figures.

Everything else in the repository exists to support, test, or visualize that loop.

## The main data objects

Most of the repository revolves around one tensor:

- `radar_cube[h, n, m]`

where:

- $h$ is horizontal virtual-array channel index
- $n$ is simulated OFDM subcarrier index
- $m$ is OFDM snapshot / slow-time index

The important derived tensors are:

- `target_only_cube[h, n, m]`: same cube without nuisance or noise
- `measurement_cube[h, n, m]`: after resource-grid masking and transmit-symbol multiplication
- `known_cube[h, n, m]`: after de-embedding known symbols and zero-filling unknown resource elements
- `fft_cube.power_cube[a, r, d]`: coarse power cube on azimuth, range, and Doppler axes

The full code path is easiest to understand by following execution order.

## End-to-end execution order

### 1. `run_study.py`

`run_study.py` is the top-level CLI.

It:

- parses command-line arguments
- chooses anchors and scene classes
- builds a `StudyConfig` with `build_study_config(...)`
- calls `run_communications_study(...)`
- passes the returned study bundle to `write_all_outputs(...)`

The CLI is intentionally thin. Almost all substantive logic is in the package modules.

### 2. `src/aisle_isac/scenarios.py`

`build_study_config(...)` resolves one complete configuration object by composing:

- a waveform anchor
- a burst profile
- a TDM-MIMO array geometry
- a scene template
- a runtime profile
- estimator configuration
- output configuration

This is where the simulation gets its public knobs:

- anchor: `fr1` or `fr2`
- scene: `open_aisle`, `rack_aisle`, or `intersection`
- runtime profile: `quick` or `submission`
- burst profile: `short_cpi`, `balanced_cpi`, or `long_cpi`
- array aperture: `2 x N_rx` with default `N_rx = 8`
- MUSIC model-order mode: `mdl`, `eigengap`, `fixed`, or `expected`
- whether FBSS ablations are enabled

### 3. `src/aisle_isac/scheduled_study.py`

`run_communications_study(...)` is the study harness.

It:

- builds sweep-point specifications for the requested sweep families
- evaluates each sweep point over many Monte Carlo trials
- always evaluates a nominal point
- optionally evaluates a pilot-only nominal diagnostic
- optionally synthesizes one deterministic representative nominal trial

Internally the important functions are:

- `_build_sweep_point_specs(...)`: defines the exact parameter values for each sweep
- `_evaluate_point_task(...)`: runs all trials for one sweep point
- `simulate_communications_trial(...)`: runs one full trial from synthesis through metrics

If `jobs > 1`, sweep points are parallelized across worker processes with `ProcessPoolExecutor`. Each point gets its own reproducible seed sequence.

## Configuration and geometry

The main config dataclasses live in `src/aisle_isac/config.py`.

### Waveform anchor

`WaveformAnchor` defines:

- carrier frequency $f_c$
- occupied bandwidth $B$
- subcarrier spacing $\Delta f$

From those, the code derives:

$$
\lambda = \frac{c}{f_c}
$$

$$
N_{\text{phys}} = \mathrm{round}\!\left(\frac{B}{\Delta f}\right)
$$

$$
T_{\text{slot}} = 10^{-3}\frac{15\,\text{kHz}}{\Delta f}
$$

$$
\Delta r_{\text{anchor}} = \frac{c}{2 B_{\text{occupied}}}
$$

The public anchors are:

| Anchor | Carrier | Bandwidth | SCS |
|---|---:|---:|---:|
| `fr1` | 3.5 GHz | 100 MHz | 30 kHz |
| `fr2` | 28 GHz | 400 MHz | 120 kHz |

### Runtime profile

`RuntimeProfile` controls Monte Carlo size and estimator grid density.

| Profile | Trials | Simulated subcarriers | FFT oversampling | MUSIC grid points |
|---|---:|---:|---:|---:|
| `quick` | 8 | 96 | 4x in range/Doppler/angle | 61 |
| `submission` | 64 | 96 | 6x in range/Doppler/angle | 81 |

The simulation never uses every physical subcarrier. Instead it keeps `n_simulated_subcarriers` tones spread across the full physical band. `StudyConfig.simulated_subcarrier_indices` is the evenly spaced embedding map from the simulated grid into the physical grid.

### Burst profile

`BurstProfile` only contains `n_snapshots`, but that single number sets the coherent processing interval:

$$
T_{\text{CPI}} = N_{\text{snap}} T_{\text{slot}}
$$

The public burst presets are:

| Burst profile | Snapshots |
|---|---:|
| `short_cpi` | 8 |
| `balanced_cpi` | 16 |
| `long_cpi` | 32 |

### Array geometry

`ArrayGeometry` defines a monostatic horizontal TDM-MIMO array. The default builder uses:

- `n_tx = 2`
- `n_rx_cols = 8`
- receiver spacing $d_{\text{rx}} = 0.5 \lambda$
- transmitter spacing $d_{\text{tx}} = N_{\text{rx}} \cdot 0.5 \lambda$

That transmitter spacing is chosen so the virtual array becomes a filled ULA with `2 * n_rx_cols` unique positions at half-wavelength spacing.

For each wavelength, the code builds:

- physical RX positions
- physical TX positions
- virtual TDM-MIMO positions
- unique horizontal positions used for azimuth estimation

The approximate azimuth resolution proxy stored in `StudyConfig.azimuth_resolution_deg` is:

$$
\Delta \theta_{\text{res}} \approx \mathrm{deg}\!\left(0.886 \frac{\lambda}{D}\right)
$$

where $D$ is the effective horizontal aperture.

### Derived study resolutions

The study-level resolution proxies are:

$$
\Delta r_{\text{res}} = \frac{c}{2 B_{\text{sampled}}}
$$

$$
\Delta v_{\text{res}} = \frac{\lambda}{2 T_{\text{CPI}}}
$$

$$
\Delta \theta_{\text{res}} \approx \mathrm{deg}\!\left(0.886 \frac{\lambda}{D}\right)
$$

`B_sampled` is the span of the retained simulated tones, not simply `96 * Delta f`.

## Scene generation

Scene factories live in `src/aisle_isac/scenarios.py`.

### Target catalog

The target catalog contains two mover classes:

| Target | Label | RCS offset | Height | Nominal speed |
|---|---|---:|---:|---:|
| `amr` | AMR | 0.0 dB | 0.55 m | 1.2 m/s |
| `forklift` | Forklift | 4.5 dB | 1.80 m | 2.0 m/s |

### Scene templates

Each `SceneClass` defines:

- which two target classes appear
- nominal center range, azimuth center, and center velocity
- nominal SNR used for noise calibration
- default target separations, expressed in resolution cells
- target coherence
- bounded trial-to-trial jitter
- static clutter templates
- coherent or incoherent multipath templates

The public scenes are:

| Scene | Target pair | Nominal range | Nominal azimuth center | Nominal center velocity | Nominal SNR |
|---|---|---:|---:|---:|---:|
| `open_aisle` | AMR, AMR | 24.0 m | 0.0 deg | 1.2 m/s | 20 dB |
| `rack_aisle` | AMR, AMR | 22.0 m | 2.0 deg | 1.0 m/s | 16 dB |
| `intersection` | AMR, Forklift | 18.0 m | 10.0 deg | 0.5 m/s | 18 dB |

Their default separations are measured in the study's own resolution units:

$$
\Delta r = c_r \Delta r_{\text{res}}, \quad
\Delta v = c_v \Delta v_{\text{res}}, \quad
\Delta \theta = c_\theta \Delta \theta_{\text{res}}
$$

where the coefficients $(c_r, c_v, c_\theta)$ come from the scene template.

### Exact public scene definitions

The scene templates are fully hard-coded in `build_scene_class(...)`. The important public values are:

- `open_aisle`
  - default separations: `1.25` range cells, `1.35` velocity cells, `1.25` angle cells
  - second-target power offset: `-1.0 dB`
  - target coherence: `0.15`
  - jitter limits: `0.20 m` center-range, `0.08` range cells, `0.08` velocity cells, `0.08` angle cells
  - amplitude jitter: `0.40 dB`
  - nuisance gain jitter: `0.75 dB`
  - static clutter templates:
    - `support_beam`: `(-8.0 m, -18.0 deg, -20.0 dB)`
    - `column`: `(11.0 m, 14.0 deg, -23.0 dB)`
  - multipath templates:
    - `floor_bounce`: `(1.5 m, 0.0 deg, -16.0 dB)`, coherent with target `0`
- `rack_aisle`
  - default separations: `0.85` range cells, `0.90` velocity cells, `0.90` angle cells
  - second-target power offset: `-3.0 dB`
  - target coherence: `0.65`
  - jitter limits: `0.25 m` center-range, `0.10` range cells, `0.10` velocity cells, `0.10` angle cells
  - amplitude jitter: `0.50 dB`
  - nuisance gain jitter: `1.00 dB`
  - static clutter templates:
    - `left_rack`: `(-7.5 m, -24.0 deg, -12.0 dB)`
    - `right_rack`: `(-7.0 m, 23.0 deg, -12.5 dB)`
    - `far_endcap`: `(9.0 m, 3.0 deg, -18.0 dB)`
  - multipath templates:
    - `left_wall_bounce`: `(1.0 m, -11.0 deg, -8.0 dB)`, coherent with target `0`
    - `right_wall_bounce`: `(1.6 m, 10.0 deg, -9.0 dB)`, coherent with target `1`
- `intersection`
  - default separations: `1.00` range cells, `1.25` velocity cells, `1.55` angle cells
  - second-target power offset: `+1.5 dB`
  - target coherence: `0.30`
  - jitter limits: `0.20 m` center-range, `0.08` range cells, `0.10` velocity cells, `0.08` angle cells
  - amplitude jitter: `0.45 dB`
  - nuisance gain jitter: `0.90 dB`
  - static clutter templates:
    - `cross_aisle_wall`: `(4.0 m, -28.0 deg, -15.0 dB)`
    - `forklift_bay`: `(7.5 m, 24.0 deg, -17.0 dB)`
    - `storage_corner`: `(12.0 m, 36.0 deg, -21.0 dB)`
  - multipath templates:
    - `floor_return`: `(1.5 m, 0.0 deg, -13.5 dB)`, coherent with target `1`

For each clutter or multipath tuple above, the fields are `(range offset, azimuth offset, gain)` relative to the scene center, with velocity fixed by the template and coherence indicated separately when present.

### Truth target placement

For each trial, the two movers are placed symmetrically around the scene center:

$$
r_1 = r_c - \frac{\Delta r}{2}, \qquad r_2 = r_c + \frac{\Delta r}{2}
$$

$$
v_1 = v_c - \frac{\Delta v}{2}, \qquad v_2 = v_c + \frac{\Delta v}{2}
$$

$$
\theta_1 = \theta_c - \frac{\Delta \theta}{2}, \qquad \theta_2 = \theta_c + \frac{\Delta \theta}{2}
$$

The second target can also receive a scene-specific power offset, for example to make the pair intentionally unbalanced.

### Trial jitter

Before synthesis, `_sample_trial_jitter(...)` draws bounded random offsets for:

- center range
- range separation
- velocity separation
- angle separation
- each target amplitude
- each nuisance gain

Then `_realize_trial_parameters(...)` applies those offsets while enforcing minimum positive separations.

The intent is to avoid a brittle "same exact geometry every trial" simulation while still keeping each sweep point centered on a controlled nominal regime.

## Signal synthesis

Signal synthesis is implemented in `src/aisle_isac/channel_models.py`.

### Source sequences and target coherence

The targets are not synthesized as deterministic constant-amplitude tones in slow time. Instead, each target gets a complex Gaussian source sequence with explicit temporal correlation and controlled inter-target coherence.

First, `_complex_gaussian_sequence(...)` generates a normalized AR(1)-like slow-time process:

$$
x[m] = \rho_t x[m-1] + \sqrt{1 - \rho_t^2}\, w[m]
$$

where $w[m]$ is complex Gaussian innovation and $\rho_t$ is `cfg.source_temporal_correlation`.

Then `_target_source_sequences(...)` creates two target source sequences as:

$$
s_k[m] = \sqrt{\rho_c}\, c[m] + \sqrt{1 - \rho_c}\, u_k[m]
$$

where:

- $c[m]$ is a shared common sequence
- $u_k[m]$ is an independent sequence for target $k$
- $\rho_c$ is the configured target coherence from the scene

The function also computes the empirical coherence actually realized in that trial.

### Path amplitude model

Each target or nuisance path amplitude uses:

$$
\alpha(R) = 10^{A_{\text{dB}}/20} \frac{\lambda}{R^2}
$$

This is implemented by `path_amplitude(...)`. It is not a full electromagnetic environment simulator; it is a controlled radar-equation-style amplitude law that gives physically reasonable range decay and scene-level power differences.

### Steering vectors

The code uses three separable steering terms:

Azimuth steering over horizontal position $x_h$:

$$
a_h(\theta) = \exp\!\left(-j 2 \pi \frac{x_h \sin(\theta)}{\lambda}\right)
$$

Range steering over baseband subcarrier frequency $f_n$:

$$
b_n(R) = \exp\!\left(-j 2 \pi f_n \frac{2R}{c}\right)
$$

Doppler steering over snapshot time $t_m$:

$$
d_m(v) = \exp\!\left(j 2 \pi t_m \frac{2v}{\lambda}\right)
$$

These are implemented by:

- `_azimuth_steering(...)`
- `_range_steering(...)`
- `_doppler_steering(...)`

Equivalent matrix-building helpers for estimator use live in `src/aisle_isac/ofdm.py`.

### Target-only radar cube

Given the truth targets, the clean target cube is synthesized as:

$$
Y_{\text{target}}[h,n,m]
=
\sum_{k=1}^{K}
\alpha_k
a_h(\theta_k)
b_n(R_k)
d_m(v_k)
s_k[m]
$$

where $K = 2$ in the current study.

In code this is the loop inside `simulate_radar_cube(...)` that sums contributions from every truth target.

### Nuisance scatterers

Each scene also contains nuisance templates:

- static clutter reflectors
- coherent multipath tied to one of the two targets
- possibly independent nuisance returns if `coherent_with_target_index` is `None`

`_build_nuisance(...)` converts each template into a realized `TargetState` with actual range, azimuth, velocity, gain, and path amplitude.

The nuisance cube is:

$$
Y_{\text{nuis}}[h,n,m]
=
\sum_{\ell=1}^{L}
\beta_\ell
a_h(\theta_\ell)
b_n(R_\ell)
d_m(v_\ell)
u_\ell[m]
$$

where $u_\ell[m]$ is either:

- the source sequence of one target, for coherent multipath, or
- a fresh correlated complex Gaussian sequence, for independent nuisance

### Noise calibration

Noise power is not chosen arbitrarily on every trial. The function `calibrated_noise_variance(cfg)` computes one fixed variance per anchor/scene by:

1. constructing a baseline configuration with:
   - `balanced_cpi`
   - a `2 x 8` array
   - `96` simulated subcarriers
2. synthesizing the nominal target-only cube
3. measuring its average signal power
4. solving for the receiver noise variance that yields the scene's nominal SNR

So if the nominal scene SNR is `18 dB`, the code sets:

$$
\sigma_w^2 = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}
$$

That calibrated variance is then reused across trials so that sweep results are comparable.

### Final noisy radar cube

The final cube returned by `simulate_radar_cube(...)` is:

$$
Y = Y_{\text{target}} + Y_{\text{nuis}} + W
$$

with circular complex Gaussian noise:

$$
W[h,n,m] \sim \mathcal{CN}(0, \sigma_w^2)
$$

The returned `CubeSnapshot` contains:

- the complete realized `ScenarioState`
- the noisy `radar_cube`
- the clean `target_only_cube`
- the scalar `noise_variance`

## Communications masking and symbol embedding

This stage is handled by:

- `src/aisle_isac/resource_grid.py`
- `src/aisle_isac/modulation.py`
- `src/aisle_isac/masked_observation.py`

### Resource grids

The sensing waveform is constrained by a communications schedule over subcarrier and OFDM-symbol index. Every resource element is labeled as one of:

- `MUTED`
- `PILOT`
- `DATA`
- `PUNCTURED`

The available grid families are:

| Family | Description |
|---|---|
| `full_grid` | Every resource element is occupied |
| `comb_pilot` | Sparse comb pilots in frequency and time |
| `block_pilot` | Contiguous pilot blocks |
| `fragmented_prb` | Communication-style scheduled PRB fragments with pilot embedding |
| `pilot_plus_data` | Fully occupied grid with pilots embedded in data |
| `punctured_grid` | Occupied grid with deterministic holes removed |

For sensing, the code uses:

$$
M[n,m] =
\begin{cases}
1, & \text{if the RE is occupied and available to sensing} \\
0, & \text{otherwise}
\end{cases}
$$

implemented by `ResourceGrid.available_sensing_mask`.

### Symbol generation

`generate_symbol_map(...)` assigns actual communication symbols to the occupied resource elements:

- pilots are a fixed known symbol, default `1 + 0j`
- data REs draw symbols from either QPSK or 16-QAM

The study mainly uses `qpsk`.

### Knowledge modes

There are two estimator knowledge modes:

- `known_symbols`: all occupied symbols are known to the sensing receiver
- `pilot_only`: only pilot symbols are treated as known

That difference only changes the de-embedding mask, not the physical masked observation itself.

### Masked observation model

`apply_resource_grid(...)` creates the measured cube:

$$
Z[h,n,m] = Y[h,n,m]\, M[n,m]\, X[n,m]
$$

where:

- $Y[h,n,m]$ is the synthesized radar cube
- $M[n,m]$ is the occupancy mask
- $X[n,m]$ is the transmitted communication symbol

`simulate_masked_observation(...)` packages the result into a `MaskedObservation`.

### Known-symbol recovery

The estimators do not work directly on `measurement_cube`. They first recover a "known-symbol cube" with `extract_known_symbol_cube(...)`:

$$
\tilde{Y}[h,n,m] =
\begin{cases}
\dfrac{Z[h,n,m]}{X[n,m]}, & (n,m) \in \Omega_{\text{known}} \\
0, & (n,m) \notin \Omega_{\text{known}}
\end{cases}
$$

This is the main bridge from communications scheduling into the sensing estimators. Unknown symbols are not guessed or reconstructed; they are simply zero-filled.

## Allocation summary metrics

`src/aisle_isac/allocation_metrics.py` computes simple descriptors of each resource grid:

- occupied fraction
- pilot fraction
- contiguous bandwidth span
- slow-time span
- fragmentation index

The fragmentation index is a simple edge-density metric over the binary occupancy mask:

$$
\text{fragmentation}
=
\frac{1}{2}
\left(
\mathrm{mean}\, |\Delta_f M|
+
\mathrm{mean}\, |\Delta_t M|
\right)
$$

This is not meant to be a communications-theory fragmentation metric. It is a compact scalar summary used to sort and label study points.

## FFT axes and ambiguity handling

`src/aisle_isac/ofdm.py` builds the steering matrices and FFT axes used throughout the study.

### Range axis

The range FFT axis is:

$$
r[q] = q \frac{c}{2 B_{\text{occupied}} O_r}
$$

where $O_r$ is the range FFT oversampling factor.

### Doppler axis

The Doppler FFT axis comes from the oversampled FFT frequency grid:

$$
v[p] = f_D[p] \frac{\lambda}{2}
$$

### Azimuth axis

The azimuth FFT axis is produced from the spatial FFT sine grid and mapped through `arcsin`, with a sign convention chosen to match the steering-vector definition used everywhere else.

### Sparse unambiguous range

Because the study only keeps a sparse subset of physical subcarriers, the first unambiguous range interval is limited by the largest gap between sampled frequencies:

$$
R_{\text{unamb}}
=
\frac{c}{2 \max \Delta f_{\text{gap}}}
$$

`sparse_unambiguous_range_m(cfg)` computes this interval, and FFT candidate extraction is restricted to it. This matters because the simulated tones are spread across a wide physical band rather than forming a dense contiguous low-rate subband.

## Masked FFT front-end

The masked FFT front-end is implemented in `src/aisle_isac/estimators_fft_masked.py`.

### Frequency embedding

The known-symbol cube only has `n_subcarriers = 96` simulated tones. Before taking FFTs, the code embeds that sparse cube back into a physical frequency grid of length `anchor.physical_subcarrier_count`, placing the simulated tones at `cfg.simulated_subcarrier_indices` and filling every unsampled tone with zero.

### Windowing

The front-end applies Hann windows in:

- azimuth / channel dimension
- frequency dimension
- slow-time dimension

### 3-D FFT order

The coarse spectrum is formed by:

1. spatial FFT over channel index
2. inverse FFT over frequency to convert delay to range
3. FFT over slow time to convert phase progression to Doppler

Conceptually:

$$
P_{\text{FFT}}(\theta, R, v)
=
\left|
\mathcal{F}_{m}
\left[
\mathcal{F}^{-1}_{n}
\left[
\mathcal{F}_{h}
\left\{
\tilde{Y}[h,n,m]
\right\}
\right]
\right]
\right|^2
$$

The actual implementation oversamples each axis according to the runtime profile and keeps only positive range bins inside the sparse unambiguous interval.

### Weighted masked normalization

The active study uses `embedding_mode="weighted"`.

The code computes:

- `support_energy`: total window-squared energy over known resource elements
- `full_support_energy`: total window-squared energy over the fully supported grid

and scales the raw FFT power by:

$$
G = \frac{E_{\text{full}}}{E_{\text{support}}}
$$

so the front-end is not artificially penalized simply because fewer REs were known.

### Coarse candidate extraction

`extract_candidates_from_fft(...)` turns the FFT power cube into a small list of coarse detections.

It does the following:

1. Find all 3-D local maxima using a `3 x 3 x 3` maximum filter.
2. Discard maxima outside the first unambiguous range interval.
3. Sort surviving maxima by FFT power.
4. Estimate a noise floor as the median power in the allowed cube.
5. Keep maxima above:

$$
\tau = \text{median power} \times \texttt{detector\_threshold\_scale}
$$

6. If too few maxima survive thresholding, backfill from the strongest remaining maxima.
7. Apply NMS.

Here NMS means **non-maximum suppression**. Two detections are considered duplicates if their normalized distance in range, velocity, and azimuth cells is smaller than `cfg.detection_nms_radius_cells`.

The output of this stage is `FrontendArtifacts`, which contains:

- the masked FFT power cube
- the coarse candidate list
- the estimator search bounds
- measured front-end runtime

## Local matched-filter refinement

Both headline estimators use the same local refinement code from `src/aisle_isac/estimators.py`.

`_refine_detection_local(...)` performs a short coordinate-descent search around a coarse detection:

1. search azimuth on a small local grid
2. search range on a small local grid
3. search velocity on a small local grid
4. repeat for two passes

At each point the score is the magnitude of a separable matched filter:

$$
\left|
\mathbf{a}(\theta)^{H}
\, Y \,
\mathbf{b}(R)^{*}
\mathbf{d}(v)^{*}
\right|
$$

`refine_detection_set_local(...)` applies this to a small candidate pool, then runs NMS again and keeps at most the expected number of targets.

This refinement step is what converts a coarse FFT or staged MUSIC hypothesis into the final reported detection coordinates.

## The two headline estimators

The repository contains a few historical and shared estimator utilities, but the actual thesis comparison is wired in `src/aisle_isac/estimators_music.py`.

The only headline methods are:

- `fft_masked`
- `music_masked`

### `fft_masked`

`_run_masked_fft_estimator(...)`:

- takes the masked FFT front-end candidates
- optionally runs local matched-filter refinement on the known-symbol cube
- reports the top surviving detections

If `cfg.skip_local_refinement` is enabled, this method becomes a pure coarse FFT detector so the study can isolate front-end quality.

### `music_masked`

`music_masked` is not a joint 3-D MUSIC search. It is a staged subspace pipeline implemented by `_run_full_search_music(...)` in `src/aisle_isac/estimators.py` and called through `_run_masked_music_variant(...)`.

The staged logic is:

1. Build a global spatial covariance from the known-symbol cube.
2. Estimate model order.
3. Run a dense full azimuth MUSIC search over the entire allowed azimuth interval.
4. Extract several azimuth peaks, not just one.
5. For each azimuth candidate:
   - beamform to that azimuth
   - build a range covariance
   - run range MUSIC over a dense range grid
   - keep several range peaks
6. For each `(azimuth, range)` candidate:
   - build a Doppler covariance
   - run Doppler MUSIC over a dense velocity grid
7. Optionally refine each candidate locally with the same matched-filter coordinate descent used by the FFT path.
8. Merge duplicates with NMS and report the strongest surviving detections.

So the MUSIC path still uses a search decomposition:

$$
\theta \rightarrow R \rightarrow v
$$

It does **not** solve a joint three-parameter MUSIC optimization.

## Covariances, model order, and MUSIC pseudospectra

### Sample covariance

The sample covariance helper is:

$$
\mathbf{R} = \frac{1}{N} \mathbf{X} \mathbf{X}^{H}
$$

where columns of $\mathbf{X}$ are snapshots.

### Model-order estimation

`_estimate_music_model_order(...)` supports four modes:

- `expected`: use the known target count from the config
- `fixed`: use `music_fixed_model_order`
- `eigengap`: choose the leading eigen-gap
- `mdl`: classical MDL criterion

For `mdl`, the implementation scans candidate source counts and minimizes the standard MDL objective built from the arithmetic and geometric means of the trailing eigenvalues.

### MUSIC pseudospectrum

Given covariance $\mathbf{R}$, estimated target count $\hat{K}$, and steering vector $\mathbf{a}(\xi)$, the code evaluates:

$$
P_{\text{MUSIC}}(\xi)
=
\frac{1}{
\mathbf{a}(\xi)^H
\mathbf{E}_n
\mathbf{E}_n^H
\mathbf{a}(\xi)
}
$$

where $\mathbf{E}_n$ is the noise-subspace basis from the eigendecomposition of $\mathbf{R}$.

The same core function `music_pseudospectrum(...)` is reused for:

- azimuth MUSIC
- range MUSIC
- Doppler MUSIC

The only thing that changes is the steering matrix and the covariance used for that stage.

## FBSS: where and when it is applied

FBSS stands for **forward-backward spatial smoothing**.

The base FBSS routine is `fbss_covariance(...)`. It averages overlapping subarray covariances and then averages the forward covariance with its reversed conjugate form.

### Spatial FBSS

The headline MUSIC path always uses spatial FBSS in azimuth. The subarray length is:

$$
L_{\text{FBSS}} = \mathrm{round}(f_{\text{FBSS}} N_{\text{chan}})
$$

clipped into a valid range, where `f_FBSS = cfg.fbss_fraction`.

### Range and Doppler FBSS

Range and Doppler FBSS are only used when the known-symbol mask provides enough contiguous common support.

The logic is conservative:

- identify active symbols or active frequencies
- find the longest contiguous run where support is common
- require that run to be dense enough relative to all common support
- require at least two stable snapshots in the complementary dimension
- otherwise fall back to the ordinary covariance

This matters because the masked observation is irregular. The code does not blindly apply smoothing where the support geometry would make the covariance meaningless.

### FBSS ablation variants

For diagnostics, the repository can also run four MUSIC variants:

- `fbss_spatial_only`
- `fbss_spatial_range`
- `fbss_spatial_doppler`
- `fbss_spatial_range_doppler`

These are evaluated on the nominal point plus the `bandwidth_span` and `slow_time_span` sweeps when FBSS ablation is enabled.

## Trial scoring and metrics

All scoring logic lives in `src/aisle_isac/metrics.py`.

### Assignment

Each method returns a set of detections:

$$
\hat{\mathcal{T}} = \{(\hat{R}_j, \hat{v}_j, \hat{\theta}_j)\}
$$

These detections are matched to truth targets with a Hungarian assignment on the cost:

$$
c_{ij}
=
\frac{1}{3}
\left(
\epsilon_{r,ij}^2
+
\epsilon_{v,ij}^2
+
\epsilon_{\theta,ij}^2
\right)
$$

where:

$$
\epsilon_{r,ij} = \frac{|\hat{R}_j - R_i|}{\Delta r_{\text{res}}}
$$

$$
\epsilon_{v,ij} = \frac{|\hat{v}_j - v_i|}{\Delta v_{\text{res}}}
$$

$$
\epsilon_{\theta,ij} = \frac{|\hat{\theta}_j - \theta_i|}{\Delta \theta_{\text{res}}}
$$

### Detection gate

An assignment counts as a matched detection only if:

$$
\epsilon_r \le 1,\quad \epsilon_v \le 1,\quad \epsilon_\theta \le 1
$$

So a method can be "close" in one dimension and still fail the joint detection gate if it is more than one resolution cell away in another dimension.

### Resolution gate

Joint resolution requires a stricter threshold:

$$
\epsilon_r \le \gamma,\quad \epsilon_v \le \gamma,\quad \epsilon_\theta \le \gamma
$$

with:

$$
\gamma = \texttt{cfg.resolution\_cell\_fraction} = 0.35
$$

The code also records per-axis resolution success:

- range resolution success
- velocity resolution success
- angle resolution success

### RMSE definitions

The metrics module reports two RMSE styles:

- conditional RMSE: only over detections that passed the detection gate
- unconditional RMSE: every missed target is penalized with one full resolution cell in each dimension

That unconditional convention ensures a method cannot look artificially good just by refusing to report a difficult target.

### False alarms and misses

After gated matching:

- unmatched detections count as false alarms
- unmatched truth targets count as misses

Both event probabilities and raw counts are tracked.

### Point summaries

`summarize_method_metrics(...)` aggregates per-trial results into `MethodPointSummary`, which contains:

- detection and resolution probabilities
- unconditional and conditional RMSEs
- false-alarm and miss probabilities
- reported target-count accuracy
- mean estimated model order and model-order accuracy when relevant
- front-end, incremental, and total runtime

## Sweep design

Sweep generation is encoded directly in `src/aisle_isac/scheduled_study.py`.

The nominal operating point uses:

- allocation family: `fragmented_prb`
- knowledge mode: `known_symbols`
- modulation: `qpsk`
- `prb_size = 12`
- `n_prb_fragments = 4`
- pilot periods of `4` in frequency and time

The public sweep families are:

| Sweep | What changes |
|---|---|
| `allocation_family` | Swaps among full-grid, comb, block, fragmented, and punctured layouts |
| `occupied_fraction` | Changes how much of the grid is filled |
| `fragmentation` | Changes the occupancy pattern structure |
| `bandwidth_span` | Changes contiguous occupied frequency span |
| `slow_time_span` | Changes how many OFDM snapshots contain occupied REs |
| `range_separation` | Changes target range separation |
| `velocity_separation` | Changes target Doppler separation |
| `angle_separation` | Changes target azimuth separation |
| `nuisance_gain_offset` | Globally shifts nuisance power |

For the axis-isolation sweeps (`range_separation`, `velocity_separation`, `angle_separation`), the two untouched separations are pushed to `2.4` resolution cells so the chosen axis is the intended bottleneck.

### Exact sweep grids

The study has two sweep suites:

- `headline`: the smaller public grid used for most fast comparisons
- `full`: a denser grid used when more detail is needed

The exact parameter grids are:

- `allocation_family`
  - `full_grid`: all REs occupied as pilots
  - `comb_pilot`: pilot periods `(4 subcarriers, 2 symbols)`
  - `block_pilot`: one `48`-subcarrier block across `16` symbols
  - `fragmented_prb`: `4` PRB fragments of size `12` with pilot periods `(4, 4)`
  - `punctured_grid`: `15%` punctures applied to the fragmented PRB grid
- `occupied_fraction`
  - `headline`: fragment counts `1, 2, 4, 6, 8`
  - `full`: fragment counts `1, 2, 3, 4, 5, 6, 8`
- `fragmentation`
  - low: block pilot
  - medium: fragmented PRB
  - punctured: punctured fragmented PRB
  - high: comb pilot with periods `(2, 1)`
- `bandwidth_span`
  - `headline`: fractions `0.25, 0.50, 0.75, 1.00`
  - `full`: fractions `0.20, 0.35, 0.50, 0.70, 0.85, 1.00`
- `slow_time_span`
  - `headline`: fractions `0.25, 0.50, 0.75, 1.00`
  - `full`: fractions `0.20, 0.35, 0.50, 0.70, 0.85, 1.00`
  - these are converted to the nearest valid active-symbol count for the current burst length
- `range_separation`
  - `headline`: multipliers `0.60, 0.85, 1.00, 1.20, 1.50` times `cfg.range_resolution_m`
  - `full`: multipliers `0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80`
- `velocity_separation`
  - `headline`: multipliers `0.50, 0.75, 1.00, 1.25, 1.50` times `cfg.velocity_resolution_mps`
  - `full`: multipliers `0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 1.80`
- `angle_separation`
  - `headline`: multipliers `0.60, 0.85, 1.00, 1.20, 1.50` times `cfg.azimuth_resolution_deg`
  - `full`: multipliers `0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 1.80`
- `nuisance_gain_offset`
  - `headline`: `-6, -3, 0, +3, +6 dB`
  - `full`: `-9, -6, -3, 0, +3, +6, +9 dB`

## Reproducibility and seeding

The study is intentionally deterministic.

- `StudyConfig.rng_seed` is fixed at `20_260_331`
- each sweep point builds a `SeedSequence` from:
  - the root seed
  - point index
  - occupied fraction
  - fragmentation index
  - bandwidth span fraction
  - slow-time span fraction
- each trial then receives a spawned child seed

That design makes repeated runs stable while still changing the random draws when the allocation geometry changes.

The representative trial is also deterministic: it is the nominal point run once with `np.random.default_rng(study_cfg.rng_seed)`. It is not cherry-picked for success or failure.

## Reporting and outputs

`src/aisle_isac/scheduled_reporting.py` converts study results into CSVs and figures.

### Core CSV outputs

The main output directory contains a `data/` folder with files such as:

- one CSV per individual sweep family
- `all_sweep_results.csv`: one row per sweep point and method summary
- `trial_level_results.csv`: one row per trial and method
- `nominal_summary.csv`: nominal-point summary
- `pilot_only_nominal_summary.csv`: optional pilot-only diagnostic
- `runtime_summary.csv`: nominal runtimes
- `failure_modes.csv`: sweep points where the headline metric is not essentially perfect
- `usefulness_windows.csv`: points where MUSIC beats FFT by at least `0.10`
- `stage_diagnostics.csv`: azimuth-stage diagnostics for nominal MUSIC trials
- `fbss_ablation_results.csv`: FBSS ablation summaries

If multiple scenes or anchors are run together, the reporting layer also writes:

- `scene_comparison.csv`
- `anchor_comparison.csv`

### Representative-artifact CSV outputs

If representative artifacts are enabled, the code also writes:

- `representative_resource_mask.csv`
- `representative_scene_geometry.csv`
- `representative_range_doppler.csv`
- `representative_music_spectra.csv`
- `representative_fbss_ablation_spectra.csv`

These are particularly useful for figure-building because they preserve the data behind the nominal example trial.

### Figures written by the study harness

The `figures/` output directory includes:

- one sweep figure per requested sweep
- `runtime_summary.png`
- `representative_resource_mask.png`
- `representative_spectrum.png`

Probability summaries in the CSV outputs also include Wilson 95% confidence intervals.

## What each source file is for

### Core package

| File | Role |
|---|---|
| `src/aisle_isac/config.py` | Dataclasses and derived waveform/array/study properties |
| `src/aisle_isac/scenarios.py` | Public factories for anchors, scenes, runtime profiles, and full study configs |
| `src/aisle_isac/channel_models.py` | Truth generation, source processes, nuisance realization, noise calibration, cube synthesis |
| `src/aisle_isac/resource_grid.py` | Resource-grid families and occupancy masks |
| `src/aisle_isac/modulation.py` | QPSK/16-QAM symbol generation and knowledge masks |
| `src/aisle_isac/masked_observation.py` | Symbol embedding, masking, and known-symbol recovery |
| `src/aisle_isac/allocation_metrics.py` | Occupancy and fragmentation summaries for resource grids |
| `src/aisle_isac/ofdm.py` | Steering matrices, FFT axes, and sparse-range ambiguity helpers |
| `src/aisle_isac/estimators.py` | Shared detection objects, covariance math, NMS, local refinement, staged MUSIC internals |
| `src/aisle_isac/estimators_fft_masked.py` | Masked FFT front-end construction and normalization |
| `src/aisle_isac/estimators_music.py` | Headline estimator wiring: masked FFT vs masked MUSIC, plus FBSS ablations |
| `src/aisle_isac/metrics.py` | Hungarian assignment, gating, RMSEs, and point summaries |
| `src/aisle_isac/scheduled_study.py` | Sweep construction, Monte Carlo loop, trial orchestration, parallel evaluation |
| `src/aisle_isac/scheduled_reporting.py` | CSV export and diagnostic plotting |

### Entrypoints and downstream consumers

| File or directory | Role |
|---|---|
| `run_study.py` | Main CLI entrypoint for the study |
| `figure_scripts/` | Figure-specific scripts used for slides and report graphics |
| `presentation/` | Slide-deck generation code |
| `report/` | Assessment writeup sources |
| `scripts/` | Convenience scripts for larger runs and result bundles |
| `tests/` | Unit and integration tests for the core pipeline |

## How one nominal trial moves through the code

If you want the concrete "MATLAB-style" execution flow for one trial, it is:

1. `run_study.py:main()`
2. `scenarios.build_study_config(...)`
3. `scheduled_study.run_communications_study(...)`
4. `scheduled_study._nominal_point_spec(...)`
5. `scheduled_study._evaluate_point_task(...)`
6. `scheduled_study.simulate_communications_trial(...)`
7. `resource_grid.build_resource_grid(...)`
8. `masked_observation.simulate_masked_observation(...)`
9. `channel_models.simulate_radar_cube(...)`
10. `modulation.generate_symbol_map(...)`
11. `masked_observation.apply_resource_grid(...)`
12. `masked_observation.extract_known_symbol_cube(...)`
13. `estimators_fft_masked.prepare_masked_frontend(...)`
14. `estimators_music.run_masked_estimators(...)`
15. `metrics.evaluate_trial(...)`
16. `metrics.summarize_method_metrics(...)`
17. `scheduled_reporting.write_all_outputs(...)`

That sequence is the backbone of the repository.

## Running the study

The repository is set up for `uv`. A minimal run is:

```bash
uv run python run_study.py --profile quick --anchor fr1 --scene-class intersection
```

A more explicit single-scene debug run is:

```bash
uv run python run_study.py \
  --profile submission \
  --anchor fr1 \
  --scene-class intersection \
  --suite headline \
  --trials 8 \
  --sweeps bandwidth_span \
  --output-dir results/debug_intersection \
  --clean-outputs
```

To run the tests:

```bash
uv run pytest
```

## Important implementation boundaries

There are a few design choices worth stating plainly because they shape how the results should be interpreted:

- The MUSIC estimator is staged, not joint 3-D MUSIC.
- The study always assumes two expected targets in the scene, even when model order is estimated differently.
- Unknown communication symbols are zero-filled, not reconstructed.
- Noise is calibrated once per anchor/scene baseline, not re-fit at every sweep point.
- The representative trial is deterministic and nominal, not cherry-picked.
- The FFT and MUSIC paths share the same local matched-filter refinement machinery, so the main comparison is about front-end candidate quality and subspace robustness rather than about giving one method a privileged back-end.

If you want to understand the repository quickly, start with:

1. `run_study.py`
2. `src/aisle_isac/scheduled_study.py`
3. `src/aisle_isac/channel_models.py`
4. `src/aisle_isac/masked_observation.py`
5. `src/aisle_isac/estimators_fft_masked.py`
6. `src/aisle_isac/estimators_music.py`
7. `src/aisle_isac/metrics.py`

That sequence follows the real data flow from configuration to final reported metrics.
