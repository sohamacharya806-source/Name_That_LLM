# Name That LLM notebook workflow

This directory contains three notebooks that are part of the v12 workflow and
one separate upstream fork:

| File | Purpose | Use |
|---|---|---|
| `ntl_bottleneck_diagnostics_cpu.ipynb` | CPU feature/readout diagnostics, prompt-grouped validation, extra-pool sampling, and an optional unseen-set export | Run first to compare the structural SVC4 member and create `structural_member_probs.csv`. |
| `ntl_lora_finetune_honest_validation.ipynb` | Honest prompt-grouped validation for the LoRA classifier, with optional labelled extras and unseen evaluation | Run after diagnostics when a GPU is available and a LoRA member is needed. |
| `ntl-0662-augmented-eval-and-safety.ipynb` | Full ensemble inference/evaluation harness | Run last. Set `OUR_MEMBER_PROBS` to the diagnostics CSV or an id-aligned CSV/Parquet probability file. |
| `ntl-fork664-m25.ipynb` | **Actual submission notebook**: frozen multi-member inference and submission writing | Use this for the final `submission.parquet`. It loads pre-trained LoRA24 adapters; it does not fine-tune LoRA. |

The requested `ntl_pool_cleaning_coverage_report.ipynb` is not present in this
directory, so no claims are made about changes to that notebook.

## Required v12 input schema

The preferred labelled pool is the competition `train.csv` at
`/kaggle/input/competitions/name-that-llm/train.csv`. Pool/extra files may be
CSV, Parquet, or JSONL and should contain:

* `id`, `text`, `model`
* `source_dataset`
* `prompt_canon`
* `panel_labels`, `prompt_evidence`, `match_weight`
* `style_score`, `flag_kb_proxy`, `n_sources`, `max_cos_train`

`source_dataset` is mapped to `source`. For `prompt_canon`, shared `p...`
identifiers remain unchanged; source-only `s...` identifiers are made
source-specific; empty values become missing. `prompt_group` is intentionally
ignored. `style_score` is retained for reporting but is not used as a filter.

For generated labelled files, accepted text columns are `text`, `passage`,
`generation`, or `response`; accepted label columns are `model`, `label`,
`llm`, or `generator`. Unrecognised labels are reported and dropped, so check
the printed count before trusting a result.

## Kaggle execution order

1. Attach the competition dataset and the required model/feature datasets.
2. Open `ntl_bottleneck_diagnostics_cpu.ipynb`. Leave `extra_paths=[]` for a
   baseline. Add labelled pool files to `CFG['extra_paths']` only when they
   are not derived from validation rows. Configure
   `extra_cap_per_label`, `extra_cap_source_share`, and optionally
   `extra_sample_by_match_weight=True`.
3. Inspect prompt-grouped balanced accuracy and per-class recall. The exported
   `structural_member_probs.csv` is id-aligned and contains `id` plus one
   probability column per model label.
4. Open `ntl_lora_finetune_honest_validation.ipynb` if the LoRA member is
   required. Put only training-safe extras in `CFG['extra_paths']`; use
   `max_train_rows_per_epoch` to control cost. Use `CFG['seeds']` for repeated
   runs and compare balanced accuracy, recall, and seed spread.
5. If you are using the actual submission workflow, open
   `ntl-fork664-m25.ipynb`. Its Section 6 run cell writes
   `/kaggle/working/submission.parquet`. It loads pre-trained adapters and
   frozen readouts; it does not train LoRA.
6. Only use `ntl-0662-augmented-eval-and-safety.ipynb` for the separate
   augmented ensemble workflow. It accepts `OUR_MEMBER_PROBS`, but it is not
   the notebook currently used for your submission.
7. Submit only the final `submission.parquet` produced by
   `ntl-fork664-m25.ipynb`. Keep intermediate probability files separate.

## What is automated versus what you must test

The notebooks can prepare data, enforce prompt-grouped folds, train/evaluate
the configured readouts, report balanced accuracy and per-class recall, and
write probability files. They cannot determine the universally correct number
of transformer layers or guarantee leaderboard accuracy without your actual
Kaggle data, attached checkpoints, GPU hardware, and an unseen evaluation
set.

Layer selection is therefore an experiment you must run, not a value to
assume. The current LoRA presets are:

| Backbone | Total layers | Default trainable/readout layers |
|---|---:|---:|
| `qwen3_4b` | 36 | 24 |
| `qwen3_8b` | 36 | 24 |
| `qwen3_14b` | 40 | 27 |

These are starting points inherited by the notebook, not proof that they are
optimal. More layers increase memory, runtime, and overfitting risk. Fewer
layers can miss identity signals. Compare them only with the same folds,
training-row cap, epoch budget, seed list, and evaluation set.

### Recommended step-by-step experiment

Do not begin with a large private pool or a full layer sweep. Use this order:

1. **Data audit:** verify labels, row counts, source counts, prompt IDs,
   missing values, duplicate/near-duplicate rates, and class balance.
2. **Prompt analysis:** run the diagnostics notebook with no extras. Compare
   topic-aware and topic-blinded views. If topic-blinded performance collapses,
   the model may be learning prompt/content rather than generator identity.
3. **Structural baseline:** record prompt-grouped balanced accuracy and
   per-class recall for the no-extra baseline. This is the control result.
4. **Sampling experiment:** add one pool source at a time. Compare natural,
   capped, source-capped, and match-weighted sampling. Keep the same folds and
   seeds. Reject a plan that improves plain accuracy but harms balanced
   accuracy or minority-class recall.
5. **LoRA baseline:** use `ntl_lora_finetune_honest_validation.ipynb` with
   `baseline_24`, `extra_paths=[]`. This is the training/validation notebook;
   save validation metrics, training/validation gap, runtime, and memory use.
6. **Fine-tuning layer sweep:** in the LoRA notebook, test
   `baseline_12`, `baseline_24`, then `baseline_all`. Do not change backbone,
   data, learning rate, and layer count simultaneously.
7. **Submission inference sweep:** in `ntl-fork664-m25.ipynb`, `keep` is the
   truncated backbone depth and `layers` is the frozen readout-layer list.
   These are not equivalent to fine-tuning layers. Test changes only through
   Section 8's labelled `RUN_SIM=True` path before changing the submission
   run cell.
8. **Backbone sweep:** only after choosing a reasonable fine-tuning depth,
   compare the available pre-trained submission members. Keep the selected
   data plan and evaluation protocol fixed.
9. **Seed check:** rerun the best one or two configurations with
   `CFG['seeds'] = [42, 43]` (or more seeds if affordable). Report mean and
   spread. A single high score is not enough if the seed spread is large.
10. **Private-data ablation:** compare competition-only, each private source
   separately, and the combined pool. Keep only sources that improve the
   prompt-grouped metrics without leakage warnings.
11. **Unseen-set check:** use genuinely new prompts and run `unseen_path`.
    Treat this as the final model-selection check. Never use competition
    `test.csv` labels, if unavailable, to tune layers or weights.
12. **Final submission:** freeze the selected configuration, run Section 6 of
    `ntl-fork664-m25.ipynb`, verify `/kaggle/working/submission.parquet`, and
    submit it. Do not continue
    tuning after inspecting the competition submission results unless you
    create a new held-out evaluation split.

For every experiment, keep a small table with: configuration, data plan,
number of rows, number of prompts, layers kept, epochs, seed, balanced
accuracy, per-class recall, unseen score, runtime, and GPU memory. This is the
only reliable way to decide whether additional layers or private data help.

### Practical stopping rules

Stop increasing layers when validation balanced accuracy and unseen
performance stop improving, the train/validation gap grows, or runtime/memory
becomes disproportionate. Stop adding private data when the gain appears only
on prompt-overlapping validation, only on the dominant class, or disappears
across seeds. Prefer the smallest configuration that is competitive and
stable.

Both diagnostic notebooks use synthetic data when the competition
`train.csv` is unavailable. This checks cell wiring and plotting only; it is
not evidence of model quality. In this environment the required notebook
execution packages and competition datasets were unavailable, so execution
results could not be independently reproduced here.

## Private dataset workflow

Private data is useful, but it must be treated as training-only data unless
its provenance and split are known.

1. Store private files outside the submission directory. Keep a manifest with
   filename, source, collection date, model label, prompt identifier policy,
   and whether the text may overlap competition data.
2. Normalise columns to the v12 schema. Assign stable IDs, preserve the
   original source in `source_dataset`, and use a unique prefix for
   source-only prompt IDs. Do not invent shared `p...` IDs.
3. Remove exact and near duplicates against competition train/test before
   adding private rows. If `max_cos_train` or another leakage indicator is
   available, retain it for auditing and exclude suspicious rows from the
   validation comparison.
4. Keep private prompts disjoint from validation prompts. Never construct
   private training rows by paraphrasing or copying a validation/test passage.
5. Add private labelled files through `CFG['extra_paths']`. Let the notebook
   apply per-label and per-source caps; do not concatenate a large private
   source without measuring its label/source dominance.
6. Run two conditions: competition-only and competition-plus-private. Report
   balanced accuracy, per-class recall, prompt-grouped folds, source counts,
   and effective sample size when match-weight sampling is enabled.
7. For generation planning, allow only evidence-backed rows (`own_text` or
   `own_bank` where that field is available). Keep generated text in a
   separate unseen file and score it through `unseen_path`; do not merge it
   into validation training.
8. Export an id-aligned probability file and verify that IDs are unique,
   probabilities are finite/non-negative, each row sums to approximately one,
   and all expected label columns are present before setting `OUR_MEMBER_PROBS`.
9. Keep private data and checkpoints private. Do not commit credentials,
   signed URLs, raw private passages, or private probability files to a
   repository.

## Minimal configuration examples

```python
# diagnostics
CFG['extra_paths'] = ['/kaggle/input/my-private-pool/private.parquet']
CFG['extra_cap_per_label'] = 2500
CFG['extra_cap_source_share'] = 0.30
CFG['extra_sample_by_match_weight'] = True
CFG['unseen_path'] = '/kaggle/input/my-unseen/unseen.csv'
```

```python
# LoRA
CFG['extra_paths'] = ['/kaggle/input/my-private-pool/private.parquet']
CFG['max_train_rows_per_epoch'] = 12000
CFG['seeds'] = [42, 43]
CFG['unseen_path'] = '/kaggle/input/my-unseen/unseen.csv'
```

```python
# ensemble
OUR_MEMBER_PROBS = '/kaggle/working/structural_member_probs.csv'
```

## Change/verification note

The diagnostics and LoRA notebooks were updated for v12 column propagation,
per-label/source caps, optional `match_weight` sampling, balanced-accuracy
reporting, and bounded training controls. The ensemble already accepted
id-aligned CSV/Parquet `OUR_MEMBER_PROBS`, so it was not changed. All four
notebook files parse as valid JSON. Full smoke execution requires the notebook
runtime dependencies and datasets listed above.

## Where to find and run the LoRA layer experiments

Open `ntl_lora_finetune_honest_validation.ipynb` and search for **EXPERIMENT
CONTROL PANEL**. It is immediately below the `CFG`/`PRESETS` cell. This is now
the only place you need to change layer count:

```python
EXPERIMENT_NAME = 'baseline_12'  # or baseline_24, baseline_all
select_experiment(EXPERIMENT_NAME)
```

The notebook currently defines these ready-to-run controls:

| Name | Backbone | Layers used | Purpose |
|---|---|---:|---|
| `baseline_12` | Qwen3 4B | 12 | Cheap layer-reduction test |
| `baseline_24` | Qwen3 4B | 24 | Existing/default preset |
| `baseline_all` | Qwen3 4B | 36 | Full-depth comparison |
| `backbone_8b_24` | Qwen3 8B | 24 | Larger-backbone comparison |
| `backbone_14b_27` | Qwen3 14B | 27 | Largest preset; most expensive |

The loader now reads `CFG['layers_kept']`, validates it against the selected
backbone, sets `cfg.num_hidden_layers`, and truncates `layer_types` when the
model provides that field. Thus changing `layers_kept` is a real architecture
change, not just a label in the report.

### Exact step-by-step run procedure

Run these steps in Kaggle, recording the printed metrics after each run:

1. **Audit:** attach competition `train.csv`; inspect row/label/source/prompt
   counts and duplicate warnings.
2. **Prompt analysis:** run the CPU diagnostics notebook with `extra_paths=[]`.
   Record topic-aware versus topic-blinded balanced accuracy.
3. **Structural baseline:** keep the diagnostics defaults and save its
   balanced accuracy, per-class recall, and `structural_member_probs.csv`.
4. **Pool sampling:** add one extra source at a time in diagnostics; compare
   natural, capped, source-capped, and match-weighted settings.
5. **LoRA baseline:** in the LoRA notebook choose `baseline_24`, keep
   `extra_paths=[]`, and run the validation cells.
6. **Layer sweep:** run `baseline_12`, then `baseline_24`, then
   `baseline_all`. Restart/re-run the training cells for each selection; do
   not merely change the printed result. Keep folds, epochs, seeds, and data
   fixed.
7. **Backbone sweep:** after selecting the best layer depth, run
   `backbone_8b_24` and, if GPU memory/time permits, `backbone_14b_27`.
8. **Seed check:** set `CFG['seeds'] = [42, 43]` and repeat the best one or
   two configurations. Compare mean and spread, not one lucky run.
9. **Private-data ablation:** run competition-only, each private source alone,
   then the combined private pool. Keep only leakage-safe improvements.
10. **Unseen check:** set `CFG['unseen_path']` to a genuinely new labelled
    prompt set and evaluate without adding it to training.
12. **Final submission:** freeze the selected configuration, run Section 6 of
    `ntl-fork664-m25.ipynb`, verify `/kaggle/working/submission.parquet`, and
    submit it.

## Source-held-out ensemble validation (LOSO)

The submission notebook now includes Section 9, **Source-held-out ensemble
validation (LOSO)**. This is separate from the final submission run and from
the old Section 7 post-run diagnostics.

Section 7 is label-free test-distribution analysis. It cannot prove accuracy.
Section 9 is the accuracy check: for each source, it holds out every row from
that source, fits the inference readouts on all remaining official sources,
runs the same `infer.py` ensemble on the held-out source, and aggregates the
predictions as out-of-fold results. It requires `source_dataset` or `source`
in `train.csv`. Missing source values are grouped under the explicit
`__missing__` bucket.

Run LOSO **before** making the final submission if you will use its results to
choose weights, members, layer settings, or data sources. Running it after a
submission is still useful as an audit, but it cannot improve or retroactively
validate the already-submitted file.

The submission run is now protected by `RUN_SUBMISSION` in Section 6:

1. Set `RUN_SUBMISSION = False`.
2. Run the notebook through the setup cells and then run Section 9 LOSO.
3. Inspect the LOSO metrics and confusion matrices.
4. Choose/freeze the strategy.
5. Rerun Section 6 with `RUN_SUBMISSION = True`.

This prevents an accidental final submission from being created before the
source-held-out experiment is complete.

LOSO is not automatically a completely unseen-model test. It prevents the
official held-out rows from being used by the per-run readout fitting and
removes exact held-out-text matches from external files. However, the
pre-trained adapter/checkpoint weights may have been trained on overlapping
sources or prompts before this notebook starts. A strict unseen-model claim
requires training a fresh adapter separately inside each source-held-out fold.

In the new Section 9 cell:

```python
RUN_LOSO = True
LOSO_SOURCES = None       # all sources; missing values use "__missing__"
LOSO_MAX_SOURCES = None   # optional quick subset
```

Run this only in an interactive GPU session. It invokes the ensemble once per
source and can be expensive. The scored competition rerun keeps it disabled.
External `--extra` and `--pool` files are source-filtered when they contain a
source column; files without one are reported as not strictly filterable.

Outputs are written to `/kaggle/working/loso_ensemble/`:

* `loso_summary.csv`: rows, accuracy, and balanced accuracy per held-out source.
* `oof_predictions.parquet`: every held-out row with true label, prediction,
  and source.
* `per_class_recall.csv`: precision, recall, and F1 for the aggregate OOF run.
* `confusion_<source>.csv/.png`: confusion matrix for each held-out source.
* `confusion_all_sources.csv/.png`: aggregate LOSO confusion matrix.

Use the aggregate balanced accuracy and per-class recall—not the real test
post-run diagnostics—to decide whether a strategy generalises. A source with
very few rows or missing classes should be interpreted separately; report its
row count and class coverage alongside its score.

## What data the submission ensemble actually uses

The actual submission notebook is not only `train.csv`; the inputs are used by
different members:

| Input | Used by |
|---|---|
| Competition `train.csv` | All supervised readouts, Gaussian heads, TF-IDF, SVC, and validation splits |
| Competition `test.csv` | Final inference rows only; it has no labels |
| `nlm_matched_clean.parquet` (`EXTRA`) | TF-IDF/distortion training and, when `--svc_w > 0`, SVC training |
| `nlm_pool_v5.parquet` (`POOL`) | SVC training only when `--svc_w > 0` |
| Pre-trained adapter folders | Frozen transformer members; not trained by this notebook |
| `comp_logit.npy` | Complementary frozen stack when `--comp_w > 0` |
| ATLAS bundle | ATLAS member when `--atlas_w > 0` |

The current Section 6 command passes:

```text
--extra <EXTRA> --extra_w 0.5 --pool <POOL> --svc_w 1.5
```

Therefore `EXTRA` and `POOL` are not unused. `EXTRA` affects the TF-IDF
member and SVC; `POOL` affects SVC because `svc_w=1.5`. If a member weight is
zero, that member's data is intentionally skipped. Check the notebook output
for `matched-clean extra rows` and `svc4 fit rows` to verify actual usage.

For LOSO, source-filtered copies of `EXTRA` and `POOL` are created when those
files contain `source_dataset` or `source`. If an external file lacks a source
column, it cannot be strictly excluded from the held-out source and the
notebook reports a warning.

For each run, save a row containing `experiment_name`, `backbone`,
`layers_kept`, `extra_paths`, `epochs`, `seed`, validation balanced accuracy,
per-class recall, unseen balanced accuracy, runtime, and GPU memory. The
notebook cannot select the winning layer without these real runs because the
competition data and checkpoints are not available in this workspace.

Git is now initialized in this workspace. The current baseline commit is
`30b7c0d`. To inspect future edits:

```bash
git diff --stat
git diff -- ntl-fork664-m25.ipynb
git log --oneline --max-count=5
```

The in-place fine-tuning change is visible by searching the LoRA notebook for
`layers_kept`, `EXPERIMENTS`, and `spec = dict(PRESETS[CFG['backbone']])`.
The submission-notebook clarification is visible by searching
`ntl-fork664-m25.ipynb` for `Submission notebook: what the layer settings mean`
and `Frozen readout layers`.
