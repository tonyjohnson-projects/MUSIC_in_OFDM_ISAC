# Planned Fixes Before Defense

Based on holistic assessment of the project against the goal of investigating MUSIC as a super-resolution tool in ISAC.

## 1. Run the nuisance-strength sweep at submission density [RUNNING]

**Status:** Running in background. MDL and expected-order variants both in flight.

```bash
PYTHONPATH=src .venv/bin/python run_study.py \
  --profile submission --anchor fr1 --scene-class all --trials 64 \
  --sweeps nuisance_gain_offset \
  --skip-pilot-only --skip-representative --disable-fbss-ablation \
  --output-dir results/submission_nuisance --clean-outputs

PYTHONPATH=src .venv/bin/python run_study.py \
  --profile submission --anchor fr1 --scene-class all --trials 64 \
  --sweeps nuisance_gain_offset \
  --skip-pilot-only --skip-representative --disable-fbss-ablation \
  --music-model-order expected \
  --output-dir results/submission_expected_order --clean-outputs
```

## 2. Run expected-order MUSIC nominal at 64 trials [RUNNING]

**Status:** Running in background. Will produce `results/analysis/model_order_nominal_64trials.csv`.

**After it completes:** Update report Table 5 with the 64-trial numbers.

## 3. Add a 1-D predecessor figure to the report [DONE]

- Created `scripts/generate_1d_motivation_figure.py`
- Generated `results/figures/motivation_1d_range.png` (FFT 0%, MUSIC 100% at 0.70-cell separation)
- Added Figure 1 and subsection "Motivating 1-D Result" to report

## 4. Rename "realistic" to "waveform-limited" [DONE]

- Report title, abstract, conclusion updated
- README title updated
- PDF title metadata updated

## 5. Add --skip-local-refinement mode [DONE]

- Added `skip_local_refinement` to `StudyConfig`
- Wired through estimators (both FFT and MUSIC paths)
- Added `--skip-local-refinement` CLI flag
- All 26 tests pass

**Still needed:** Run a nominal comparison with/without refinement and add discussion to report.

```bash
# Without refinement
PYTHONPATH=src .venv/bin/python run_study.py \
  --profile submission --anchor fr1 --scene-class all --trials 64 \
  --sweeps nuisance_gain_offset \
  --skip-pilot-only --skip-representative --disable-fbss-ablation \
  --skip-local-refinement \
  --output-dir results/debug_no_refinement --clean-outputs
```

## 6. Mention the CRB in the discussion [DONE]

- Added subsection "Relation to Fundamental Bounds" in Discussion
- Referenced Stoica & Nehorai 1989
- Discussed why CRB computation is non-trivial for masked model
- Added bibliography entry
