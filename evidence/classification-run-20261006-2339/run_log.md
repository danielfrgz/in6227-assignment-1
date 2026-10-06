# Run log — classification-run-20261006-2339

## Environment
- Model: claude-opus-5-5 (Opus 5.5)
- Interface: Claude Code 2.1.291
- Skill: tabular-classification-report v0.1.0 (repo URL not recorded in skill)
- Python 3.12.14; pandas 3.0.6; numpy 2.5.3; scikit-learn 1.9.1; matplotlib 3.11.2; reportlab 5.0.1; pypdf 6.19.0
- Seed: 42
- Input: ~/VSProjects/in6227-assignment-1/data/raw/train.csv (read-only, not modified)
- Header fields: read from ./report_config.yaml (name, matric, assignment, variant, repo_url all present)

## Phase 1
- `profile_raw.json`: delimiter ",", encoding utf-8-sig, 31,112 rows × 16 columns, not sampled.
- Exact duplicate rows: 0 → nothing to drop.
- Empty strings converted to missing (1–7 per column). Flagged word token `unknown`: personal_interest 1,767, species 1,760 (not converted, pending Checkpoint A).
- Profiler sentinel candidates: none. Observed by inspection of top values: value 0 appears 265× in both performance_score and stability_index, 38× in both load_ratio and activity_duration; max values 510 (aptitude_score, 35×) and 1010 (performance_score, 7×) repeat.
- Target candidates: label (score 4), brew_preference (1), geological_era (1).
- Column names are real words but do not describe a coherent domain (e.g. geological_era, brew_preference, species side by side) → **semantics unknown**.
- Sibling file test.csv has an identical header → asked whether it is the held-out test set.

## Checkpoint A
User answers (paraphrased closely):
1. "Yes, label." → target = `label`.
2. "Does test.csv contain the label column? Confirm before treating it as held-out. If it has no labels, use a stratified split from train.csv and say so." → Checked: test.csv has 13,334 rows × 16 columns; label values no 10,165 / yes 3,168 / empty 1. Labels present → test.csv used as held-out test set (G0); the 1 empty-label row is dropped (G1).
3. "Keep Unknown as a category, and state in the report that it's treated as informative rather than missing." → `Unknown` in personal_interest and species kept as a level; no --missing-token passed. Report must state this.
4. "Keep zeros and caps as real values, but record this as an explicit limitation: the 265 all-zero rows are all no, and I can't determine from the data whether that's signal or a not-recorded code. The report must state that the conclusion would change if those are missing-value codes." → No --sentinel passed. Report Limitations must carry this statement.

## Phase 2
- profile.json written (target label, test file). Key values: minority_share 0.240, categorical_share 0.533, one-rule max 0.591 (aptitude_score), variance pilot shallow − full −0.002, correlated_pairs none, test max class-share difference 0.002.

## Phase 3
- Decisions written to plan.md (copied below in final form at Phase 6).

## Checkpoint B
User answers (closely paraphrased / quoted):
1. "Confirmed — keep geological_era nominal. The chronological order is real, but with semantics unknown I can't verify it's meaningful for this target, and imposing a false ordering creates arbitrary split boundaries. Note it in the report as a judgement call." → all categoricals nominal; report notes it as a judgement call.
2. "Override: tune on macro-F1, not accuracy … the baseline already scores 0.76, so tuning on accuracy selects configurations that favour the majority class. Log it as an explicit override of the ≥ 0.20 rule." → **OVERRIDE of G10 default**: primary metric macro-F1 (tuning, bootstrap, headline) although minority share 0.240 ≥ 0.20.
3. "Add AUPRC to the reported metrics. L4 p84 prefers PR over ROC under imbalance" → AUPRC (average precision, positive class `yes`) added.
4. "Replace the ROC curves figure with PR curves" → figure 2 = PR curves; confusion matrices kept.
5. "State the majority-class baseline's accuracy (≈0.76) directly beside both models' accuracy in the results table" → results table includes accuracy for baseline, tree and NB in one column.
6. "Everything else approved as planned."

## Phase 4
- train.py run 1: succeeded (wall time ~24 s). No failed attempts.
- train.py edited and re-run once to add `train_zero_value_rows` (training-data-only diagnostic) to metrics.json, so that the 265/38 zero-row counts quoted in the report come from metrics.json rather than an ad-hoc check. Test metrics and bootstrap verified identical to run 1 (deterministic, seed 42). The final train.py is the version that produced metrics.json.
- Result summary: NB best (alpha 0.1, 10 bins) test macro-F1 0.756 vs tree (depth 8, min_samples_leaf 20, ccp_alpha 0) 0.745; paired bootstrap tree − NB CI [−0.020, −0.002] excludes 0. Tree has higher test accuracy (0.831 vs 0.799) — metric choice decides the ranking.

## Phase 5
- report.md written from profile.json / metrics.json. Two corrections made by the agent before first build: figure 1 caption (wrongly described a PR panel) and an unverified claim that models "learn 0 → no" replaced by a factual statement.
- report.pdf built: 2 pages, limit 2: OK. No cuts needed.

## Checkpoint C
User corrections (closely paraphrased / quoted):
1. Interpretability: "The chosen tree has 173 leaves at depth 8, so it is not interpretable in the sense L4 p45 means … state plainly that at this size the tree forfeits that advantage." → Agent checked: report did not claim tree interpretability anywhere. Added one sentence to Section 5 Findings as requested.
2. Stopping criteria: "Make section 3 say explicitly that max_depth=8, min_samples_leaf=20 and ccp_alpha=0 are the pre-pruning stopping criteria (L4 p44, p58) … Say the same for NB's grid where applicable." → Section 3 rewritten to name max_depth = 8 and min_samples_leaf = 20 as pre-pruning stopping criteria (L4 p44, p58). **Deviation from the literal request, flagged to user:** ccp_alpha is post-pruning (L4 p59), not pre-pruning; the report states ccp_alpha = 0 means no post-pruning was applied. NB: report states it has no stopping criterion (single counting pass); alpha and bins are smoothing/discretisation settings.
3. Dropped rows: "Confirm section 1 states the 3 training rows and 1 test row dropped for a missing target." → Already present; no change.
4. "Confirm, without changing anything: does the report state the tree's recall on yes (0.529) against NB's (0.796)?" → Confirmed present in Section 5; no change.
- report.pdf rebuilt: 2 pages, limit 2: OK.

## Phase 6 — decisions with evidence (final plan.md)
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

## Overrides of skill defaults
- G10 primary metric: macro-F1 instead of accuracy (minority share 0.240 ≥ 0.20 would select accuracy). Reason (user, Checkpoint B): majority baseline accuracy 0.760, so tuning on accuracy favours the majority class.
- G10 figure: PR curves instead of ROC curves for a binary, balanced (≥ 0.20) target; AUPRC added. Reason (user, Checkpoint B): L4 p84.

## Deviations, warnings, failed attempts
- No failed training attempts.
- train.py re-run once (training-data diagnostic added); test results identical to run 1.
- Sentinel/zero values kept as real by user decision; profiler flagged no sentinel candidates, co-occurring zeros were found by the agent through a one-off check in Phase 1 and later reproduced inside train.py (`train_zero_value_rows`).
- Agent corrections to its own draft before Checkpoint C: figure caption, one unverified claim (see Phase 5).
- Checkpoint C request 2 not applied literally for ccp_alpha (post-pruning, not pre-pruning), as noted above.
