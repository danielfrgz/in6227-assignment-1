# Plan — classification-run-20261006-2339

Format: **measured value → action → reason (reference)**. All values from `profile.json` unless noted.

## G0 Split
- test.csv confirmed held-out, labels present (no 10,165 / yes 3,168 / empty 1), same columns in same order → use as test set, untouched until final evaluation; tune on train.csv only → provider's split, test data must stay unseen (L4 p3).
- max class-share difference train vs test 0.002 ≤ 0.05 → no shift limitation needed.

## G1 Target
- `label`, binary (no 23,645 / yes 7,464), rows_missing_target 3 (train), 1 (test) → drop those rows → cannot be trained on or scored. Training rows after drop: 31,109.

## Phase 1 decisions carried over
- exact duplicate rows 0 (train), 0 (test) → nothing to drop.
- `Unknown` in personal_interest (1,767) and species (1,760) → kept as an informative category, not missing (user decision, Checkpoint A).
- Zeros (performance_score & stability_index = 0 on the same 265 rows, all `no`; load_ratio & activity_duration = 0 on the same 38 rows) and caps (aptitude_score 510 ×35, performance_score 1010 ×7) → kept as real values (user decision); reported as a limitation: if they are not-recorded codes, the conclusions could change.

## G2 Missing values
- Max missing share of any feature 0.000225 (brew_preference) < 0.05 → impute only, no indicator columns → too few to carry a reliable signal ⚠.
- Numeric → median (L3 p10); categorical/boolean → most frequent (L3 p11). Fitted inside each pipeline on training folds only ⚠ (L4 p3).

## G3 Outliers
- IQR outlier share: activity_duration 0.172, load_ratio 0.081, stability_index 0.036, index_weight 0.033, aptitude_score 0.020, performance_score 0.005, composite_rank 0.004 → keep all; report counts → no task knowledge to call them noise (L2 p32, p60).
- negative_count 0 for all numeric features → no impossible values to convert.
- Robust scaling rule not triggered: no scale-sensitive model is chosen (see G6).
- |skewness| max 1.705 (load_ratio) < 2 → no log transform.

## G4 IDs and leakage
- Max distinct ratio of a non-float column 0.0007 (region) ≤ 0.95 → no ID-like columns.
- One-rule screen max 0.591 (aptitude_score) ≤ 0.95 → no suspected leak.
- Semantics unknown (Phase 1) → keep all columns; report states that the semantic leakage check could not be performed.

## G5 Imbalance and validation
- minority share 0.240 ≥ 0.20 → no class weighting, no resampling; stratify every CV fold (L2 p39, L4 p81).
- training rows 31,109 > 1,000 → 5-fold stratified CV, shuffle, seed 42 (L4 p81).
- smallest class 7,464 rows ≥ 10 → no fold warning.

## G6 Models
- Baseline: majority class (`no`) → default rule (L5 p16); shows accuracy's limitation (L4 p74).
- Model 1: decision tree (CART, Gini) → scale-free, mixed types, interpretable (L4 p44, p58–59).
- Model 2: categorical_share 0.533 ≥ 0.50 → **Naive Bayes** (CategoricalNB, numeric features in equal-frequency bins) → high-bias probabilistic model vs the tree's low-bias partitions (L5 p36, p48). Variance pilot shallow − full = −0.002 < 0.02 → the random-forest row would not have fired either.

## G7 Encoding
- Tree: categorical → ordinal codes (`handle_unknown` → −1); numeric as is.
- NB: categorical → rare levels (< 1% of training rows, and unseen test levels) grouped into `__rare__`, then ordinal codes; numeric → equal-frequency bins (`KBinsDiscretizer`, quantile; bins tuned). Rare-level counts per profile: region 17 levels (0.048 of rows), geological_era 2 (0.018), mineral_type 2 (0.006), weather_pattern 1 (0.001).
- Ordinal mapping: all categorical features treated as **nominal** (confirmed at Checkpoint B). geological_era has a real chronological order (Cambrian < Devonian < Permian < Triassic < Jurassic), but with semantics unknown it cannot be verified as meaningful for this target, and a false ordering would create arbitrary split boundaries → nominal; codes in alphabetical order for the tree. Reported as a judgement call.

## G8 Features
- top_value_share max 0.897 (region) < 0.99 → no near-constant drops.
- correlated_pairs: none with |r| > 0.95 → no drops (would apply to SVM only anyway).
- No datetime or text columns → no engineering. No PCA (L2 p46–48).
- Net result: all 15 features kept → a finding to state in the report.
- Tree feature importances reported as embedded selection (L2 p52).

## G9 Tuning
- Tree grid: max_depth {3, 5, 8, None} × min_samples_leaf {1, 5, 20} × ccp_alpha {0, 0.001} = 24 configs.
- NB grid: alpha {0.1, 1} × bins {5, 10} = 4 configs.
- GridSearchCV, 5-fold stratified, scored by **macro-F1** (G10 override); exhaustive search. Report train vs CV score; gap > 0.10 is flagged.
- Runtime estimate: a few minutes at most (28 configs × 5 folds = 140 fits on 31k rows of trees / NB; well under the 15-minute budget).

## G10 Metrics
- minority share 0.240 ≥ 0.20 → default would be accuracy. **OVERRIDE (user, Checkpoint B):** primary metric **macro-F1** for tuning and headline comparison → majority baseline accuracy is 0.760 (majority_share), so tuning on accuracy selects configurations that favour the majority class; macro-F1 weights both classes equally (L4 p74).
- Also: accuracy shown in the results table directly beside the baseline accuracy; confusion matrices (L4 p72), per-class precision/recall (L4 p77), ROC-AUC (L4 p83), **AUPRC** (added at Checkpoint B; PR preferred over ROC under imbalance, L4 p84), CV mean ± sd, baseline row.
- Test comparison: Wilson 95% CI per model's accuracy (L4 p87–88); paired bootstrap (1,000 resamples, seed 42) of the macro-F1 difference tree − NB; if the interval contains 0 → not significant → recommend the simpler model (L4 p56).
- Figures: confusion matrices side by side; **PR curves** (replacing ROC, user change at Checkpoint B, L4 p84).
