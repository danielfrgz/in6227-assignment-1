# Run log — classification-run-20261007-0822

## Environment
- Model: Claude Opus 5.5, model ID `claude-opus-5-5`
- Interface: Claude Code **2.1.291** (running session, confirmed by the user at Checkpoint A). `claude --version` reported 2.1.292 = installed binary with an update not yet active; not used.
- Skill: tabular-classification-report, `metadata.version` **1.0.0**; git tag **v1.0.0**, commit `ae423da81a6b1059ee0aec34f0946ab5a85e0a72` (HEAD of `main`, = `origin/main`) in `~/VSProjects/in6227-assignment-1/repo` (the installed skill directory `~/.claude/skills/tabular-classification-report` is a symlink to `repo/tabular-classification-report`). `git status --porcelain` was empty at Checkpoint C, so the tagged files are the ones that ran. Skill files were not modified during the run.
- Python 3.12.14; pandas 3.0.6; numpy 2.5.3; scikit-learn 1.9.1; matplotlib 3.11.2; reportlab 5.0.1; pypdf 6.19.0
- Seed: 42 (split not needed — test file provided; CV shuffling, models, bootstrap)
- Report header fields read from `report_config.yaml` (name, matric, assignment, variant, repo); all present, none asked or invented.

## Inputs
- Training data: ~/VSProjects/in6227-assignment-1/data/raw/train.csv (read-only, not modified)
- Test data: ~/VSProjects/in6227-assignment-1/data/raw/test.csv (read-only, not modified; confirmed as held-out test set at Checkpoint A)

## Phase 1 (profile_raw.json)
- 31,112 rows × 16 columns, delimiter ",", encoding utf-8-sig, not sampled.
- Exact duplicate rows: 0.
- Empty strings converted to missing in every column (1–7 per column).
- Flagged word token `unknown`: personal_interest 1,767 (0.057), species 1,760 (0.057) — not converted, put to the user.
- Sentinel / boundary pile-ups: 0 in performance_score & stability_index (265 rows, co-occurring, Jaccard 1.0); 0 in load_ratio & activity_duration (38 rows, co-occurring, Jaccard 1.0); max 510 in aptitude_score (35 rows, frequency spike); max 1010 in performance_score (7 rows, no spike).
- Target candidates: label (score 4), brew_preference (1), geological_era (1).
- Column semantics: names are real words but do not describe a coherent domain → "semantics unknown".

## Checkpoint A — user answers (closely paraphrased / quoted)
1. Target: "Yes, target is label." → target = `label`.
2. Test file: "Yes, test.csv is the held-out test set — confirm it contains the label column first … If it does, keep it untouched until final evaluation and don't deduplicate it." → Checked: test.csv has 13,334 rows, `label` present (no 10,165 / yes 3,168 / 1 missing). Used as test set; not deduplicated; only the 1 unlabelled row dropped (cannot be scored).
3. `unknown`: keep as a real level in `personal_interest` and `species` ("may be an informative answer rather than an absence, and converting it would push both columns over 5% missing and then impute over information that was actually recorded"). Report states this is a deliberate choice, not a missing-value default.
4. Shared zeros: keep as real values. **User correction:** my Checkpoint A wording "looks like not recorded" was an inference, not a measurement, and must not be presented as a finding. Report states only the measured pattern (265 rows / 38 rows, Jaccard 1.0; zero outside the Tukey fences of the remaining values in 3 of 4 columns), that it is consistent with a not-recorded code but indistinguishable from genuine zeros, and in Limitations that if these are codes the models learn a recording artifact and the conclusions would change.
5. Maximum pile-ups (510, 1010): keep as real ("a frequency spike at the maximum is consistent with a cap but not proof of one"); mentioned in the same limitation.
6. Interface version: user stated the running session is 2.1.291, not 2.1.292 → report uses 2.1.291.

## Phase 2 (profile.json)
- Run with `--target label --test test.csv`; no `--missing-token` or `--sentinel` (nothing confirmed as a code).
- New measurement surfaced at Checkpoint B: the 265 shared-zero rows are all class `no` (no 265 / yes 0); the 38-row pair is no 32 / yes 6.

## Phase 3 decisions (copied from plan.md; measured value → action → reason → reference)
- G1: `label`, 3 train + 1 test rows missing target → dropped (cannot be trained on or scored).
- G0: test file confirmed; max class-share difference 0.002 (< 0.05) → test.csv is the test set, tuning by CV on train.csv only; no shift limitation (L4 p3, p81). Train duplicates 0, test rows duplicating training 0 → no deduplication (L2 p29).
- G2: every feature's missing share < 5% (max 7 rows, `brew_preference`) → impute only (median numeric, L3 p10; most frequent categorical, L3 p11), fitted inside training folds (L4 p3). `unknown` kept as a level (user decision).
- G3: IQR-fence shares activity_duration 0.172, load_ratio 0.081, stability_index 0.036, index_weight 0.033, aptitude_score 0.020, performance_score 0.005, composite_rank 0.004 → keep, report counts (L2 p32, p60; L4 p96). No negatives → no impossible values (L2 p8–10). No scale-sensitive model; max |skew| 1.705 < 2 → no robust scaling, no log (L3 p13, L2 p57).
- G4: no ID-like column; one-rule screen max 0.591 (aptitude_score) ≤ 0.95 → no leak flagged. Semantics unknown → semantic leakage check not possible; stated in report.
- G5: minority share 0.240 ≥ 0.20 → no class weighting; 31,109 training rows → 5-fold stratified CV (L2 p39, L4 p81, L5 p10). User recorded the no-weighting outcome as a known form inconsistency and left it.
- G6: baseline = majority (L5 p16); Model 1 = CART tree (L4 p44, p58–59); Model 2 = Naive Bayes because categorical share 0.533 ≥ 0.50 (first matching row) (L5 p36, p48). Variance pilot (balanced accuracy): full tree CV 0.695 / train 1.000; depth-5 CV 0.693 / train 0.702; shallow − full −0.002.
- G7: tree → ordinal codes, numeric as is; NB → rare levels (< 1%) to `__rare__`, equal-frequency bins (L2 p56, L5 p38). All categoricals nominal; `geological_era` kept nominal by user decision at Checkpoint B (judgement call, L2 p8).
- G8: near-constant none (max 0.897 region); correlated_pairs empty; no exact duplicate columns → all 15 features kept; no PCA, no engineering (L2 p46–51, L1 p54).
- G9: tree 24 configs, NB 4 configs; exhaustive grid, 5-fold stratified CV, macro-F1; three-layer stopping criteria (L4 p50–59, L5 p8–10).
- **G10 metric selection — by rule, not by override:** accuracy baseline 0.760 > random 0.500 → `majority_beats_random: true` → accuracy rejected (L4 p74); macro-F1 baseline 0.432 < random 0.464 → accepted → primary metric = macro-F1 (L4 p77). No default was overridden anywhere in Phase 3.

## Checkpoint B — user answers
- "Approved." Plan accepted without changes to models, grids, metric or CV.
- Ordinal question: keep `geological_era` nominal ("The chronological order is real, but with column semantics unknown I can't verify it's meaningful for this target, and imposing a false ordering creates arbitrary split boundaries."). Report records it as a judgement call.
- User asked to confirm before training: (1) `train.py` records fitted model complexity in `metrics.json` (R-12) → confirmed: tree depth + leaf count; NB number of estimated conditional probabilities (+ class priors separately). (2) Recommendation rule is the R-12 version → confirmed: bootstrap CI excludes 0 → primary metric decides; contains 0 → Occam's razor with simplicity by fitted complexity only, never grid size, search effort or fit time.
- Report must note that macro-F1 was selected by the skill's baseline-relative rule (macro-F1 0.432 vs 0.464; accuracy 0.760 vs 0.500), not chosen by hand → done in report §4.
- User note: G5's no-class-weighting at minority share 0.240 is expected; recorded by the user as a known form inconsistency. No change.

## Phase 4 (train.py → metrics.json)
- `train.py` ran successfully on the first attempt (~24 s wall clock); no `train_attempt<N>.py`.
- Implementation notes: CSV read with only empty strings as missing (matches the profiler; `unknown` kept). NB rare grouping uses code 0 for `__rare__` so unseen/rare test levels always fall in the fitted range; tree maps unseen test levels to −1. Measured: no unseen test levels in any categorical column. Numeric bins: equal-frequency (`quantile`, `averaged_inverted_cdf`); 10 bins kept in every column.
- Exact duplicate feature columns: none.
- Wilson check reproduces the slide table: N=100, acc=0.8 → [0.711, 0.867].
- Chosen configurations: tree max_depth 8, min_samples_leaf 20, ccp_alpha 0; NB alpha 0.1, 10 bins. Train − CV macro-F1: tree 0.014, NB 0.000 (both < 0.10).
- **Recommendation decision (computed in train.py, `metrics.json → recommendation`):** paired bootstrap CI of macro-F1 (tree − NB) = [−0.020, −0.002], `ci_excludes_zero: true` → primary metric decides → **Naive Bayes**; Occam's razor not applied. Fitted complexity recorded for the record (`complexity_compared`): tree 173 leaves (depth 8); NB 262 conditional probabilities + 2 priors. Complexity played no part in the decision because the CI excludes 0.
- Figures: `figures/confusion_matrices.png`, `figures/pr_curves.png` (PR curves because binary and accuracy rejected).

## Phase 5 (report)
- First build: 3 pages (one line over). Cut prose only (shortened sentences in §1, §3, §5); no figure, table or font change. Second build: 2 pages. Fixed a punctuation typo ("NB),;") and rebuilt: 2 pages.
- Layout: the §4 heading falls at the foot of page 1 with its content on page 2.

## Checkpoint C — user answers
- "Draft approved." User verified independently before approving: recomputing accuracy, per-class precision/recall and macro-F1 for both models from the page-2 confusion matrices reproduces every reported figure (tree macro-F1 0.7454, NB 0.7563); baseline AUPRC equals test prevalence 3,168/13,333 = 0.2376.
- Optional fix requested: keep the §4 heading with its following paragraph if `build_report.py` can do so without prose cuts; otherwise leave it ("content beats typography here").
  - `build_report.py` has no keep-with-next on headings. Tested on a scratch copy only (heading style `keepWithNext=1`; skill file untouched): result 3 pages, over the limit. Per the user's instruction, no prose was cut and the orphaned heading is left as is. Final `report.pdf` is unchanged from the approved draft (2 pages).
- User asked for the audit log to be completed, and for confirmation of what it records plus the exact skill version and git tag (see Environment).

## Deviations, warnings, failed attempts
- No training failures. One report build over the page limit (fixed by prose cuts). One rejected layout fix (keep-with-next → 3 pages).
- Baseline CV macro-F1 sd is 1.6e-05; shown as 0.000 at 3 decimals in the report.
- User archives the transcript themselves.
