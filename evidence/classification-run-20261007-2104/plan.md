# Plan — classification-run-20261007-2104

Format: **measured value → action → reason → course reference**. All values from `profile.json`.

## Phase 1 decisions (confirmed at Checkpoint A)
- Target `label`: 2 classes (no 23,645 / yes 7,464), last column, generic name → target → (G1). 3 rows with missing `label` → dropped (cannot be trained or scored).
- Exact duplicate rows in train.csv = 0; test rows duplicating a training row = 0 → nothing to drop (L2 p29).
- `Unknown` in `personal_interest` (5.7% of rows) and `species` (5.7%) → kept as a real category (user decision) → may be an informative level (L2 p30); stated as a limitation.
- Co-occurring zeros: `performance_score` = `stability_index` = 0 on 265 rows (Jaccard 1.0; class counts no 265 / yes 0); `load_ratio` = `activity_duration` = 0 on 38 rows (Jaccard 1.0; no 32 / yes 6) → kept as real values (user decision) → stated as a limitation, because they may be "not recorded" codes. `aptitude_score` max 510 on 35 rows (frequency spike, possible cap) → kept.
- Column semantics → **unknown** (names are real words but form no coherent domain; appear disguised).

## G0 Split
- test.csv confirmed, same 16 columns in same order, 13,334 rows → used as the test set, untouched until final evaluation; tuning by CV on train.csv only → test data must stay unseen (L4 p3).
- Max class-share difference train vs test = 0.002 (< 0.05) → no distribution-shift limitation needed.

## G2 Missing values
- Every feature has missing share < 5% (max 0.022%, `brew_preference`) → impute only, no indicator columns → too few to carry signal ⚠.
- Numeric → median (L3 p10); categorical/boolean → most frequent (L3 p11). Imputers are fitted inside each pipeline on training folds only ⚠ (L4 p3).

## G3 Outliers
- IQR-fence shares: `activity_duration` 17.2%, `load_ratio` 8.1%, `stability_index` 3.6%, `index_weight` 3.3%, `aptitude_score` 2.0%, `performance_score` 0.5%, `composite_rank` 0.4% → keep all values, report counts → without task knowledge removal cannot be justified (L2 p32, p60); trees are insensitive (L4 p96 Q5).
- No negative values in any numeric column → no impossible values (L2 p8–10).
- Outlier share > 5% for two features, but no scale-sensitive model is chosen (Model 2 = Naive Bayes, which bins numerics) → no scaling needed.
- Largest |skewness| = 1.705 (`load_ratio`) < 2 → no log transform.

## G4 IDs and leakage
- No column with distinct ratio > 0.95 → no ID-like drops (L2 p6, p51).
- One-rule screen max balanced accuracy = 0.591 (`aptitude_score`) < 0.95 → no suspected leak ⚠.
- Semantics unknown → all columns kept; the report states that the semantic leakage check (is the value known before the outcome?) could not be performed.

## G5 Imbalance and validation
- Minority share 0.240 (≥ 0.20) → no class weighting, no resampling ⚠ (L4 p75 not needed).
- Training rows 31,109 (> 1,000) → 5-fold stratified CV, shuffled, seed 42 (L4 p81; folds are a setting, not a hyperparameter, L5 p10). Smallest class 7,464 rows → no fold warning.

## G6 Models
- Baseline: majority class (`no`) (L5 p16; L4 p74).
- Model 1: decision tree, CART/Gini (L4). Interpretability judged after fitting, from depth and leaf count (L4 p45).
- Model 2: categorical share 0.533 (8 of 15 features categorical/boolean) ≥ 0.50 → **categorical Naive Bayes** (numeric features equal-frequency binned, L2 p56, L5 p38) → high-bias probabilistic model with independence assumption (L5 p36) vs the tree's low-bias, high-variance partitions (L5 p48). First matching G6 row; for reference, variance pilot shallow − full = −0.002 (< 0.02, the RF trigger would not fire), and rows 31,109 < 50,000.

## G7 Encoding (per model)
- Tree: categorical/boolean → ordinal codes (unknown test levels → −1); numeric as is, no scaling.
- Naive Bayes: levels with < 1% of rows grouped into `__rare__` (affects `geological_era` 2 levels / 1.8% of rows, `weather_pattern` 1 / 0.08%, `region` 17 / 4.8%, `mineral_type` 2 / 0.6%); then ordinal codes; unseen test levels → `__rare__`. Numeric → equal-frequency bins (count tuned).
- No attribute has an order evident from its values → all categorical treated as nominal (L2 p8).

## G8 Feature selection / engineering
- No feature with top value share ≥ 0.99 (max 0.897, `region`) → no near-constant drops (L2 p51).
- No exact duplicate columns; no numeric pairs with |r| > 0.95 (`correlated_pairs` empty) → nothing dropped (L2 p51, p59).
- No PCA (L2 p46–48). No datetime/text columns; no skew trigger → no engineered features. All 15 features used — this is a finding.
- Tree feature importances reported as embedded selection (L2 p52).

## G9 Tuning
- Decision tree: `max_depth` ∈ {3, 5, 8, None} × `min_samples_leaf` ∈ {1, 5, 20} × `ccp_alpha` ∈ {0, 0.001} = 24 configs.
- Naive Bayes: `alpha` ∈ {0.1, 1} × bins ∈ {5, 10} = 4 configs.
- Exhaustive grid search, 5-fold stratified CV, scored by macro-F1; seed 42.
- Stopping: tree pre-pruning (`max_depth`, `min_samples_leaf`, L4 p58) and post-pruning (`ccp_alpha`, L4 p59); NB has closed-form counting (no iterative stopping); search stops when the grid is exhausted. Train vs CV gap reported, > 0.10 flagged (L4 p50–54).
- Runtime estimate: < 2 minutes total.

## G10 Metrics
- Baseline accuracy 0.760 > random 0.500 → accuracy rejected as primary (L4 p74). Baseline macro-F1 0.432 < random 0.464 → **macro-F1 primary** ⚠.
- Reported also: accuracy (with baseline), confusion matrices (L4 p72), per-class precision/recall (L4 p77), CV mean ± sd, AUPRC for `yes` (baseline = prevalence 0.240, L4 p84), ROC-AUC (L4 p83).
- Comparison: Wilson 95% CI for each model's accuracy (L4 p87–88); paired bootstrap (1,000 resamples, seed 42) of the macro-F1 difference (L4 p81, p91). CI excludes 0 → better model recommended; contains 0 → simpler model by fitted complexity (L4 p56).
- Figures: confusion matrices side by side; PR curves (binary, accuracy rejected).
