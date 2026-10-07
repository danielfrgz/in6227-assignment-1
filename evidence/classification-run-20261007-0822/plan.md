# Plan — classification-run-20261007-0822

Format: **measured value → action → reason → course reference.** All values from `profile.json`.

## Data decisions confirmed at Checkpoint A
- Target `label` (2 classes: no 23,645 / yes 7,464; 3 rows missing target) → drop the 3 unlabelled training rows and the 1 unlabelled test row → they cannot be trained on or scored (G1).
- Test file `test.csv`: 13,334 rows, same columns in same order, `label` present → held-out test set, untouched until final evaluation, **not deduplicated** → respects the provider's split; test data must stay unseen (L4 p3).
- `unknown` in `personal_interest` (1,767 rows, 5.68%) and `species` (1,760, 5.66%) → **kept as a real level** (user decision) → may be an informative answer; converting would push both columns over 5% missing and impute over recorded information. Stated in the report as a deliberate choice, not a default.
- Shared zeros: `performance_score` & `stability_index` zero on the same 265 rows (Jaccard 1.0; class counts no 265 / yes 0); `load_ratio` & `activity_duration` on the same 38 rows (Jaccard 1.0; no 32 / yes 6); zero outside the Tukey fences of the remaining values in 3 of 4 columns (`stability_index`, `load_ratio`, `activity_duration`) → **kept as real** (user decision) → consistent with a not-recorded code but the data cannot distinguish it from genuine zeros; imputing would replace recorded values on an unverifiable assumption. Limitation: if these are codes, the models learn a recording artifact and conclusions would change.
- Maximum pile-ups: `aptitude_score` = 510 (35 rows, frequency spike), `performance_score` = 1010 (7 rows, no spike) → **kept as real** → a spike at the maximum is consistent with a cap but not proof. Same limitation; class counts of these rows will be recorded in `metrics.json`.

## G0 Split
- Test file confirmed; train/test max class-share difference 0.002 (< 0.05) → use `test.csv` as the test set; tune by CV on `train.csv` only; no shift limitation needed (L4 p3, L4 p81).
- `rows_duplicating_a_training_row` = 0; training exact duplicates = 0 → no deduplication needed (L2 p29).

## G2 Missing values
- Every feature's missing share ≤ 0.00023 (max `brew_preference`, 7 rows) — all > 0 and < 5% → impute only, no indicators: median for numeric (L3 p10), most frequent for categorical (L3 p11), fitted inside the pipeline on training folds only (L4 p3).

## G3 Outliers
- IQR-fence shares: `activity_duration` 0.172, `load_ratio` 0.081, `stability_index` 0.036, `index_weight` 0.033, `aptitude_score` 0.020, `performance_score` 0.005, `composite_rank` 0.004 → keep all values, report counts → no task knowledge to call them noise (L2 p32, p60); trees insensitive (L4 p96 Q5).
- `negative_count` = 0 in every numeric column → no impossible values to convert (L2 p8–10).
- Neither chosen model is scale-sensitive (tree; NB on bins) → no robust scaling, no log transform (max |skew| 1.705 `load_ratio` < 2 anyway) (L3 p13, L2 p57).

## G4 IDs and leakage
- `id_like` false for every column; max distinct ratio 0.785 (`performance_score`, a float) → nothing dropped (L2 p6, p51).
- One-rule screen max 0.591 (`aptitude_score`) ≤ 0.95 → no suspected leak.
- Semantics unknown → keep all columns; the report states that the semantic (timing) leakage check could not be performed.

## G5 Imbalance and validation
- Minority share 0.240 ≥ 0.20 → no class weighting, no resampling; stratify every fold (L2 p39, L4 p81).
- Training rows 31,109 (> 1,000) → 5-fold stratified CV, shuffled, seed 42 (L4 p81; folds are a setting, not a hyperparameter, L5 p10).
- Smallest class 7,464 rows (≥ 10) → no fold reduction.

## G6 Models
- Baseline: majority class (`no`) (L5 p16, L4 p74).
- Model 1: decision tree (CART, Gini) (L4 p44, p58–59). Interpretability judged after fitting, from depth/leaves (L4 p45).
- Model 2: categorical share 0.533 (8 of 15 usable features categorical/boolean) ≥ 0.50 → **Naive Bayes** (CategoricalNB; numeric features in equal-frequency bins) → first matching G6 row; high-bias probabilistic model with independence assumption (L5 p36) vs the tree's low-bias, high-variance partitions (L5 p48). (For reference: one-hot width 64 < 200; variance pilot shallow − full = −0.002 < 0.02, so the RF row would not fire either.)
- Variance pilot (balanced accuracy): full tree CV 0.695 with train 1.000; depth-5 tree CV 0.693 with train 0.702 → the full tree memorises training data (train–CV gap 0.305) without gaining CV score; motivates pruning in the tree grid (L4 p52, p58–59).

## G7 Encoding
- Tree: categoricals as ordinal codes (arbitrary order); numeric as is.
- NB: rare levels (< 1% of training-fold rows) grouped into `__rare__` (profile: `region` 17 rare levels, 4.8% of rows; `geological_era` 2, 1.8%; `mineral_type` 2, 0.6%; `weather_pattern` 1, 0.08%); unseen test levels map to `__rare__`; numeric → equal-frequency bins (L2 p56, L5 p38).
- All categoricals treated as **nominal**. Only `geological_era` has values with an apparent real-world order (Cambrian < Devonian < Permian < Triassic < Jurassic), but semantics are unknown, so the order is not assumed → **to confirm at Checkpoint B** (L2 p8).

## G8 Feature selection
- Near-constant: max top-value share 0.897 (`region`) < 0.99 → none dropped (L2 p51).
- `correlated_pairs` empty (no |r| > 0.95) → nothing dropped; exact duplicate columns checked in `train.py` (L2 p51, p59).
- No PCA (L2 p46–48); no engineered features (no skew ≥ 2, no datetime columns) (L1 p54).
- Tree feature importances reported as embedded selection (L2 p52).
- Finding to report: all 15 features kept.

## G9 Tuning
- Decision tree grid: `max_depth` {3, 5, 8, None} × `min_samples_leaf` {1, 5, 20} × `ccp_alpha` {0, 0.001} = 24 configs.
- Naive Bayes grid: `alpha` {0.1, 1} × bins {5, 10} = 4 configs.
- Exhaustive grid search, 5-fold stratified CV, scored by macro-F1. Stopping: tree pre-pruning (`max_depth`, `min_samples_leaf`, L4 p58) and post-pruning (`ccp_alpha`, L4 p59); search stops when the grid is exhausted; train vs CV gap > 0.10 flagged (L4 p50–54).
- Seed 42 for CV shuffling and models. Runtime estimate: well under 2 minutes.

## G10 Metrics
- Accuracy: baseline 0.760 > random 0.500 → **rejected** (L4 p74). Macro-F1: baseline 0.432 < random 0.464 → **accepted** → **primary metric = macro-F1** (L4 p77).
- Also reported: accuracy (with baseline's), confusion matrices (L4 p72), per-class precision/recall (L4 p77), CV mean ± sd, AUPRC for `yes` with baseline = prevalence 0.240 (L4 p84), ROC-AUC (L4 p83).
- Comparison: Wilson 95% CI for each model's test accuracy (L4 p87–88); paired bootstrap (1,000 resamples, seed 42) of the macro-F1 difference; recommendation per the G10 rule (Occam only if the CI contains 0, using fitted model complexity: tree leaves vs NB conditional-probability count) (L4 p81, p91, p56).
- Figures: confusion matrices side by side; PR curves (binary, accuracy rejected) (L4 p84).
