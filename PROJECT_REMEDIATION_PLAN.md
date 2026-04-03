# Project Remediation Plan

## Goal

Bring the project from "clean simulation code with weak scientific guarantees" to "defensible single-student MSEE final project with honest scope, reproducible evidence, and technically credible results."

This plan is scoped for one student. The priority is not to make the simulator perfect or industry-grade. The priority is to fix the issues that currently weaken the project's core claims.

## Priority Order

1. Fix scientific validity bugs first.
2. Fix estimator behavior that breaks the main study narrative.
3. Tighten metrics so reported numbers mean what they claim to mean.
4. Rebuild evidence artifacts from real runs.
5. Clean up documentation and stale outputs so the repo tells one consistent story.

## Problem 1: Target Coherence Is Not Actually Modeled

### Current issue

The current source generation makes each target source constant across slow-time snapshots. In practice this forces effective coherence to magnitude 1 across targets, regardless of the configured scene coherence value.

### Fix plan

1. Replace the current target-source generator with a proper stochastic model:
   - Generate a common complex Gaussian slow-time source sequence.
   - Generate independent complex Gaussian slow-time source sequences for each target.
   - Form each target source as:
     - `s_k = rho * s_common + sqrt(1 - rho^2) * s_independent_k`
   - Normalize source power after mixing.
2. Keep Doppler phase separate from the random source process.
3. Decide and document what `target_coherence` means:
   - recommended: magnitude of normalized cross-correlation between target source processes
4. Confirm that nuisance paths marked as coherent with a target inherit the correct target source process.

### Tests to add

1. Empirical coherence test:
   - For each scene, generate many source realizations.
   - Verify measured coherence is close to the configured value within tolerance.
2. Snapshot diversity test:
   - Verify target source vectors are not constant across all snapshots unless coherence is explicitly 1.
3. Regression test for FBSS setup:
   - Confirm `rack_aisle` produces stronger coherence than `open_aisle`.

### Acceptance criteria

1. Empirical coherence matches scene configuration.
2. FBSS advantage appears only where coherence justifies it.
3. README language about coherence is now true.

## Problem 2: Staged MUSIC Fails Because Range Ambiguity Is Not Handled Consistently

### Current issue

`Full-Search MUSIC` restricts range search to the sparse-tone unambiguous interval, but the FFT-seeded staged methods do not. That allows aliased FFT seeds to survive into refinement, which can break joint detection at the nominal point.

### Fix plan

1. Centralize sparse-tone ambiguity logic in one utility:
   - compute maximum frequency gap
   - compute first unambiguous range interval
   - expose it through config or estimator utilities
2. Apply the same ambiguity limit to all range searches:
   - FFT interpretation
   - staged MUSIC range grids
   - local coordinate-descent refinement
3. Decide how aliased FFT candidates should be treated:
   - recommended: reject or wrap candidates into the first unambiguous interval before refinement
4. Revisit FFT candidate extraction:
   - reduce duplicate alias candidates
   - prevent the same target from appearing at one physical range and one ambiguous range
5. Re-run nominal cases and tune only after the ambiguity handling is physically correct.

### Tests to add

1. Nominal `FR1/open_aisle` two-target regression:
   - staged MUSIC must detect both targets
2. Aliasing regression:
   - construct a case where FFT produces an aliased seed
   - verify staged MUSIC resolves within the intended unambiguous interval
3. Candidate uniqueness test:
   - ensure two returned detections are not just ambiguity copies of the same target

### Acceptance criteria

1. Staged MUSIC no longer fails on the nominal `open_aisle` case.
2. Returned range estimates stay in the intended ambiguity interval.
3. The performance gap between staged MUSIC and full-search MUSIC becomes interpretable instead of pathological.

## Problem 3: Model-Order Accuracy Is Misdefined

### Current issue

The reported `model_order_accuracy` is not meaningful because:

1. FFT reports model order as number of coarse candidates.
2. MUSIC methods report MDL order.
3. `Full-Search MUSIC` still searches using the expected target count even when reported order differs.

This mixes different concepts in one metric.

### Fix plan

1. Define separate quantities:
   - `estimated_model_order_mdl`
   - `reported_detection_count`
   - `expected_target_count`
2. Decide whether `model_order_accuracy` should mean:
   - MDL source-count accuracy, or
   - correct number of final reported targets
   - recommended: keep both as separate metrics
3. Update estimator outputs so every method reports the same semantics.
4. Update CSV schema and README metric definitions accordingly.

### Tests to add

1. Unit test for model-order metric semantics.
2. Regression test where a method returns 2 final detections but MDL says 4:
   - verify the CSV exposes both values instead of hiding the discrepancy

### Acceptance criteria

1. Every model-order field has one clear meaning.
2. A reviewer can understand why a method detected two targets but still misestimated subspace order.

## Problem 4: Submission Artifacts Are Smoke-Test Outputs, Not Submission Evidence

### Current issue

The checked-in `results/submission` bundle was generated with `trials=1`, which makes it unsuitable as final evidence.

### Fix plan

1. Stop treating checked-in artifacts as authoritative unless the manifest proves they are full runs.
2. Add a hard guard in bundle generation:
   - fail or emit a warning if `submission` profile is run with too few trials
   - recommended threshold: require default submission trials unless an explicit dev override is set
3. Separate developer smoke outputs from final evidence:
   - `results/quick/` for smoke tests
   - `results/submission/` only for real final runs
   - optional: `results/dev/` for temporary debugging
4. Rebuild submission artifacts after code fixes using real trial counts.
5. Keep manifest files and make them part of the review checklist.

### Tests to add

1. Script test:
   - submission bundle should fail or clearly mark itself non-final when run with `TRIALS=1`
2. Manifest validation test:
   - check that `submission` artifacts record expected profile and trial count

### Acceptance criteria

1. No checked-in submission artifact can be mistaken for a smoke run.
2. Final plots and CSVs are traceable to real Monte Carlo runs.

## Problem 5: Tests Do Not Protect the Main Scientific Claims

### Current issue

The current test suite mainly checks execution, schema, and single-target sanity. It does not protect the critical two-target study claims.

### Fix plan

1. Add science-facing regression tests for nominal two-target cases:
   - `open_aisle`
   - `rack_aisle`
2. Add coherence-validation tests.
3. Add ambiguity-handling tests.
4. Add metric-semantics tests for:
   - model order
   - conditional vs unconditional metrics
   - false alarms and misses
5. Keep smoke tests, but clearly separate them from scientific regression tests.

### Suggested test tiers

1. Fast unit tests:
   - source generation
   - ambiguity utility
   - metric semantics
2. Medium integration tests:
   - single nominal trials for each estimator
3. Slow evidence tests:
   - a few Monte Carlo bundles for pre-release verification

### Acceptance criteria

1. A broken coherence model fails tests.
2. A broken staged MUSIC nominal case fails tests.
3. A fake submission bundle fails tests.

## Problem 6: Repo Evidence Is Internally Inconsistent

### Current issue

The repo contains stale `results/paper` artifacts that reference scenarios and study structures that do not match the current codebase. That makes the repository look partially unreproducible.

### Fix plan

1. Decide which of these is the project's authoritative evidence path:
   - current `results/submission`
   - current `results/quick`
   - legacy `results/paper`
2. Remove or archive stale result trees that cannot be reproduced from current code.
3. If legacy artifacts must remain, label them clearly as archival and not generated by the current simulator.
4. Update README so it only describes artifacts that the current code can produce.

### Acceptance criteria

1. Every checked-in result tree corresponds to the current code or is explicitly marked archival.
2. README and on-disk artifacts no longer contradict each other.

## Problem 7: The Study Is Too Deterministic To Support Strong Claims

### Current issue

The project is already honest about being stylized, but the scene generation is still very fixed:

1. target geometry is mostly deterministic at each sweep point
2. clutter templates are hand-crafted
3. nuisance structure is not measurement-backed

That is acceptable for a final project, but only if the claims are calibrated.

### Fix plan

1. Add modest trial-to-trial scene randomization:
   - small range offsets
   - small azimuth offsets
   - small velocity offsets
   - nuisance gain jitter
2. Keep the randomization bounded and documented.
3. Reframe the study as robustness under stylized perturbations, not as deployment prediction.
4. Avoid adding too much complexity; one student should prefer a clean perturbation model over an ambitious scene engine.

### Tests to add

1. Sanity test that randomized scenes remain within search bounds.
2. Reproducibility test that seeded runs are repeatable.

### Acceptance criteria

1. Monte Carlo covers more than just noise and random phase.
2. Claims remain modest and defensible.

## Problem 8: Metrics and Story Need To Be Reframed For a Final Project

### Current issue

Some outputs look more precise than the underlying model justifies. The project should emphasize comparative insight over absolute realism.

### Fix plan

1. Keep the strongest metrics:
   - joint detection probability
   - joint resolution probability
   - false alarm probability
   - miss probability
   - runtime
2. Keep RMSE and CRB-gap metrics, but document their limits carefully.
3. Move weak or confusing metrics to secondary status if they are not central.
4. In the write-up, frame conclusions as:
   - under this stylized indoor two-target model
   - under these waveform constraints
   - under this estimator implementation

### Acceptance criteria

1. A faculty reviewer can tell exactly what was shown and what was not shown.
2. The conclusions do not overreach the simulator.

## Implementation Sequence

### Phase 1: Scientific correctness

1. Fix source coherence generation.
2. Add coherence tests.
3. Fix shared ambiguity handling.
4. Add nominal staged-MUSIC regression tests.

### Phase 2: Metrics and estimator semantics

1. Redefine model-order reporting.
2. Update CSV schemas.
3. Update README metric descriptions.

### Phase 3: Evidence rebuild

1. Delete or archive stale non-reproducible artifacts.
2. Regenerate quick outputs for smoke testing.
3. Regenerate submission outputs with real trial counts.
4. Check manifests into the repo only for real evidence runs.

### Phase 4: Final packaging

1. Tighten README limitations section.
2. Add a short reproducibility section:
   - environment
   - command lines
   - expected artifact tree
3. Prepare a concise "what this project proves / does not prove" section for the final report or slides.

## Concrete Deliverables

1. Fixed source-generation implementation.
2. Shared ambiguity utility used by all estimators.
3. Updated metrics schema with clear model-order semantics.
4. Expanded test suite covering coherence, ambiguity, and nominal two-target cases.
5. Rebuilt `results/submission` with real trials.
6. Cleaned README and result directories.

## Minimum Viable Recovery

If time is limited, do these first:

1. Fix coherence modeling.
2. Fix staged MUSIC ambiguity handling.
3. Replace misleading model-order metric or split it into two metrics.
4. Rebuild submission artifacts with real trial counts.
5. Update README to match the true final scope.

That set is enough to materially improve the technical credibility of the final project.

