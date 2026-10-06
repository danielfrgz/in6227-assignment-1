---
name: tabular-classification-report
description: >-
  End-to-end classification on any tabular dataset file. Profiles the data, chooses
  and justifies preprocessing and at least two classifiers, evaluates them against a
  trivial baseline, and generates a PDF report of at most two pages plus an audit log.
  Use whenever the user gives a path to a CSV or similar file and wants to predict a
  categorical label, compare classifiers, or produce a data-mining report, even if
  they don't name the target column or mention this skill.
compatibility: Python 3.10+ with pandas, numpy, scikit-learn, matplotlib, reportlab, pypdf.
metadata:
  version: "1.0.0"
---

# Tabular classification report

## Why this skill works the way it does

The report is graded on reasoning, not accuracy. A reader must be able to trust every
decision and challenge it, so:

- **Scripts measure, you decide.** `scripts/` only measures data and renders documents.
  Every choice (what to clean, which models, which metric) is made by you, following
  the rules in Phase 3, and every choice cites a value measured on *this* dataset.
- **Every rule has a default and a reason.** Thresholds below are defaults whose
  rationale does not depend on any particular dataset. You may override one, but only
  with a stated reason, and the override goes into `run_log.md`.
- **Simple and justified beats complex and unexplained.** Prefer the simpler model when
  two perform *indistinguishably* (Occam's razor, L4 p56; see G10 for when that applies).
- **The human stays in the loop.** Stop at Checkpoints A, B and C and wait for the user.
- **No invented numbers.** Every number in the report must come from `profile.json` or
  `metrics.json` produced by this run. If a value is missing, say so; never estimate it.

Course references are written `L<lecture> p<page>` (IN6227 Lectures 1–5, PDF page numbers).
A rule marked ⚠ has no direct course slide; its rationale is given instead.

## Scope

**Supported:** one table in a delimited text file (CSV, TSV, or other delimiter that
pandas can sniff), one row per object, one categorical target with 2 or more classes
(binary or multi-class). Mixed numeric and categorical features. Optionally a second
file with the same columns to use as the test set.

**Out of scope, and what happens:**

| Input | Action |
|---|---|
| Continuous target (regression, L1 p29) | Stop at Checkpoint A, explain, suggest a regression workflow |
| Target with one class, or a class with < 2 rows | Stop, explain |
| Excel, Parquet, JSON, databases | Ask the user to export to CSV; do not guess a converter |
| Multiple related tables | Ask which single table to use; do not join |
| Free-text columns (long strings, mostly unique) | Drop with a stated reason; text mining is out of scope |
| Images, time series forecasting, multi-label targets | Stop, explain |

## Inputs

1. **Dataset path.** From the invocation arguments or the user's message. If missing,
   ask. Never modify the input file.
2. **Optional test path.** If the user gives a second file, or a file in the same folder
   has the same column names, *ask* whether it is a held-out test set. If confirmed, it
   is the test set (G0).
3. **Report header fields:** full name, matric number, assignment line, variant line,
   repo URL. Read `report_config.yaml` in the working directory if present (simple
   `key: value` lines: `name`, `matric`, `assignment`, `variant`, `repo`; `repo_url` is
   accepted as an alias for `repo`).
   Otherwise ask at Checkpoint A. Never invent them; a placeholder such as
   `<matric number>` is allowed only if the user asks for one. The repo URL goes into
   the report header's `repo:` field (the assignment requires the link in the report);
   if none is available, ask, and never leave the field silently empty.
4. **Output folder:** create `./classification-run-<YYYYMMDD-HHMM>/` in the working
   directory. If it exists, append `-2`, `-3`, … Never reuse or overwrite a run folder.

## Environment check

Before Phase 1, run:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, reportlab, pypdf; print('ok')"
```

If an import fails, report which package is missing and point to `requirements.txt`.
**Do not install anything yourself.** Then record, for the report metadata:

- **Model name and version:** your own model identifier (e.g. the exact model ID you
  run as). If you are not certain of it, ask the user.
- **Interface and version:** for Claude Code, run `claude --version`, but treat the
  result as a proposal only: it reports the *installed* binary, which can differ from the
  version of the running session (e.g. when an update was installed after the session
  started). **Ask the user to confirm** the version shown by their running session at
  Checkpoint A. Other interfaces: ask.
- **Skill version:** `metadata.version` above (header field `skill:`); the repo URL goes
  in the separate header field `repo:`.

Write these to `run_log.md` immediately.

## Workflow

`<skill_dir>` below is the directory containing this SKILL.md. `<run>` is the output folder.

### Phase 1: Load and sanity-check

Run the profiler without a target first, to see the whole table:

```bash
python <skill_dir>/scripts/profile_data.py <data_path> --out <run>/profile_raw.json
```

Read `profile_raw.json` (not the raw rows) and establish:

1. **Format:** delimiter, encoding, header row, row and column counts, as detected.
2. **Duplicates:** exact duplicate rows (count). Default: drop them **from the training
   data only**, because a duplicate counts the same object twice and, without a separate
   test file, can sit on both sides of the split (L2 p29): deduplicate the whole file
   *before* splitting it. A provided test file is **never deduplicated or filtered**
   (except rows with no target, which cannot be scored): removing test rows changes the
   population the scores describe. Report how many test rows duplicate a training row
   (`test_file.rows_duplicating_a_training_row`) as a limitation instead. Keep training
   duplicates if the profile suggests repeated rows are legitimate events (no ID column
   and few columns); say which you chose.
3. **Disguised missing values (G2):** the profiler counts a *generic* token list
   (empty string, `?`, `NA`, `N/A`, `null`, `None`, `-`, `unknown`, case-insensitive,
   whitespace stripped) and flags numeric *sentinel candidates* (`-1`, `0`, `999`,
   `9999`, …) whose frequency spikes far above neighbouring values. Empty strings and
   `NA`-style tokens are converted to missing. Word tokens (`unknown`, `none`, `-`) and
   sentinels are **never converted automatically**; list them for Checkpoint A, because
   a level such as `unknown` or a value such as 0 can be a real, informative value.
   Also list the profiler's **boundary pile-ups** (a column's minimum or maximum repeated
   unusually often: a possible cap, floor or censoring code) and **co-occurring zeros**
   (numeric columns that are zero on the same rows: possibly "not recorded"). Ask the
   user whether each is a real value or a missing-value code. If kept as real, the
   report must state it as a limitation, with the class counts of the affected rows.
4. **Target candidates (G1):** rank columns by: few distinct values (2 to about 20),
   position (last column is a common convention), a name matching a generic list
   (`label`, `target`, `class`, `y`, `outcome`), no missing values. Show the top 3 with
   that evidence.
5. **Column semantics:** do the column names describe real-world quantities? If they look
   synthetic, coded or anonymised, record **"semantics unknown"**. This changes G4.
6. **Test file:** if a test path is confirmed, check it has the same columns and
   compare its class distribution and missing shares with the main file.

**Stop rules:** if the confirmed target is continuous (numeric with many distinct values),
has a single class, or no column could plausibly be a target → stop and explain.

### Checkpoint A: confirm data understanding (wait for the user)

Present, briefly: shape; the target candidates with evidence; duplicates; disguised-
missing counts; sentinel candidates; ID-like columns; whether semantics are known; test
file status; missing header fields; model and interface identifiers if unsure.
Ask the user to confirm the target, the sentinel handling and the test file. **Wait.**
Log every answer and every change the user makes in `run_log.md`.

### Phase 2: Profile

Re-run with the confirmed target (and test file, if any):

```bash
python <skill_dir>/scripts/profile_data.py <data_path> --target <target> \
    [--test <test_path>] [--missing-token <token>] [--sentinel <feature_a>=<value>] \
    --out <run>/profile.json
```

`profile.json` contains per column: inferred type (numeric / categorical / boolean /
datetime / text / ID-like), missing share (true + disguised), distinct count, top
values, rare-level share, near-constant share, numeric summary (mean, median, std,
skewness, quartiles, IQR outlier count and share). Plus:

- class counts and minority share; usable features by type; categorical share;
  one-hot width estimate;
- **one-rule screen:** per feature, 3-fold CV balanced accuracy of a one-rule classifier
  (each level, or each of 10 equal-frequency bins, predicts its majority class; L5 p13,
  L2 p56). Used for leakage (G4), not for feature ranking: 0.5 means the feature alone
  never changes the majority prediction;
- **variance pilot:** 5-fold CV balanced accuracy of a fully grown tree vs a depth-5 tree
  (G6), on at most 20,000 training rows;
- **baseline metric scores:** the majority baseline's and a random guesser's expected
  accuracy and macro-F1 (and AUPRC for binary) on this class distribution (G10);
- co-occurring zeros with the class counts of the shared rows;
- highly correlated numeric pairs; for a test file, its schema check, class distribution
  and missing shares next to the training file's.

Use `--missing-token` and `--sentinel` only for values the user confirmed at Checkpoint A.

If the file is very large (over about 1 million rows), the profiler samples rows with a
fixed seed and records that it did; say so in the report.

### Phase 3: Decide (the core of the skill)

Apply the groups **in this order**. Data cleaning comes first, then model choice, then
the preprocessing each model needs (L2 p60). For every decision, write into
`<run>/plan.md` one line: **measured value → action → reason → course reference**.

Use placeholders like `<feature_a>` only in this file; in `plan.md` and the report use the
real column names of the dataset being analysed.

#### G0. Split

| Condition | Action | Reason |
|---|---|---|
| Test file confirmed | Use it as the test set, untouched until final evaluation. Tune on the main file only | Respects the provider's split; the test set must stay unseen (L4 p3) |
| No test file | Stratified 80/20 split, fixed seed, **before any fitting** | Stratified keeps class proportions (L2 p39, L4 p81). ⚠ 80/20 rather than the slide's 2/3–1/3 (L4 p81): tuning uses CV on the training part, so the test set only has to report the final score |
| Test class distribution differs from training by more than 5 percentage points for any class | Keep the split; report the shift as a limitation | The comparison is still fair (both models see the same test set) but scores may not transfer |

#### G1. Target — decided at Checkpoint A. Rows with a missing target are dropped (they cannot be used for training or scoring).

#### G2. Missing values

Per feature, by missing share (true + disguised):

| Missing share | Action | Reason |
|---|---|---|
| > 50% | Drop the column | Most values would be invented by imputation ⚠ |
| 5–50% | Impute + add a missing-indicator column | The fact of missingness can carry signal ("not applicable", L2 p30) |
| > 0 and < 5% | Impute only | Too few to carry a reliable signal ⚠ |

Imputation: **median** for numeric (robust to skew and outliers, L3 p10); **most frequent**
for categorical (mode, L3 p11), or a separate `missing` level if missingness is above 5%.
Imputers are fitted **inside the pipeline on training folds only** ⚠ — otherwise test
information leaks into training, contradicting "test data is unseen" (L4 p3).

#### G3. Outliers

For each numeric feature, report the count and share outside Tukey's fences
(Q1 − 1.5·IQR, Q3 + 1.5·IQR) ⚠ (the course box plot, L3 p22, shows whiskers at the
10th/90th percentiles; that would flag 20% of every column by construction, so the
IQR fence is used for counting).

| Condition | Action | Reason |
|---|---|---|
| Default | Keep the values; report counts | Removing outliers is only right if they are noise *for this task* (L2 p32, p60); without task knowledge it can't be justified. Trees are insensitive to them (L4 p96 Q5) |
| Value impossible for the attribute (e.g. negative for a ratio-scale count or duration, L2 p8–10) | Treat as missing (G2) | It is an error, not a rare value |
| Outlier share > 5% **and** a scale-sensitive model is chosen | Use robust scaling for that model (median/IQR) instead of z-score | z-scores are pulled by the outliers they should detect (L3 p13) |

Skewness above about 2 in absolute value on a positive feature: a log transform is
*allowed* for scale-sensitive models (L2 p57), not required. State the choice.

#### G4. IDs and leakage

| Condition | Action | Reason |
|---|---|---|
| Non-float column with distinct values / rows > 0.95 | Drop as ID-like | An identifier carries no information about the class (L2 p6, p51) ⚠ threshold |
| One-rule screen balanced accuracy > 0.95 | **Suspected leak.** Ask: is this value known *before* the outcome? If yes, keep and say why; if no or unclear, drop | A feature that alone predicts the class almost perfectly is usually recorded after, or derived from, the outcome ⚠ |
| Semantics unknown (from Phase 1) | Keep columns unless the screen fires; **state in the report that the semantic leakage check could not be performed** | You cannot judge timing from names that mean nothing; say so rather than guess |

#### G5. Class imbalance and validation scheme

Minority share = smallest class count / total rows (training part).

| Condition | Action | Reason |
|---|---|---|
| Always | Stratify the split and every CV fold | Keeps class proportions in each part (L2 p39, L4 p81) |
| Minority share < 20% | `class_weight="balanced"` for models that support it. No resampling by default. (The primary metric is chosen separately, by G10) | Weighting applies the cost-matrix idea (L4 p75) during training without adding or removing rows ⚠ |
| Training rows < about 1,000 | 10-fold stratified CV | Small data → high variance estimates; more folds use more data per fit (L4 p53, p80) |
| Otherwise | 5-fold stratified CV | Adequate estimate at lower cost (L4 p81). Folds are an experimental setting, not a hyperparameter (L5 p10) |
| Any class with < 10 training rows | Warn; reduce folds so each fold holds at least one of each class | Stratified CV cannot place a class in every fold otherwise |

#### G6. Model choice

Always fit three: a **majority-class baseline** (the default rule, L5 p16; shows how
misleading accuracy can be, L4 p74) and **two models with contrasting inductive biases**.
Only course-taught models are used.

**Model 1: decision tree (CART, Gini)** (L4). Scale-free, handles mixed types,
insensitive to outliers, and its complexity is controlled through stopping criteria the
report must explain (L4 p44, p58–59). It is *potentially* interpretable: the course says
"easy to interpret **for small-sized trees**" (L4 p45), and the fitted size is only known
after tuning. So `plan.md` must not call it interpretable; the report judges
interpretability from the fitted depth and leaf count in `metrics.json`. Its limitation —
axis-parallel boundaries (L4 p66, p69) — is what Model 2 contrasts with.

**Model 2**, chosen by measured properties (first matching row wins):

| Measured condition (after G2–G4 drops) | Model 2 | Contrast with the tree |
|---|---|---|
| Categorical features ≥ 50% of features, **or** one-hot width would exceed 200 | **Naive Bayes** (categorical NB; numeric features discretised into equal-frequency bins, L2 p56, L5 p38) | High-bias probabilistic model with an independence assumption (L5 p36) vs the tree's low-bias, high-variance partitions (L5 p48). Robust to irrelevant attributes (L5 p42) |
| Variance pilot: shallow-tree CV − full-tree CV ≥ 0.02 | **Random forest** | The full tree is fitting noise (high variance, L4 p52). The forest reduces variance by averaging bootstrap trees with random feature subsets (L5 p52, p56); the tuned tree reduces it by pruning (L4 p58–59). Two answers to the same problem, which the report compares ⚠ margin |
| Training rows > 50,000 | **Random forest** | As above; also SVM training scales poorly with rows ⚠ |
| Otherwise (mostly numeric, moderate size) | **SVM** (linear and RBF kernels in the grid) | Max-margin, oblique or curved boundaries (L5 p57–71) vs axis-parallel splits |

kNN is not used by default: after one-hot encoding the feature space is usually wide,
where distances lose meaning (L2 p44, L5 p11). The user may request it at Checkpoint B.

#### G7. Encoding and scaling (per model, L2 p59)

Each model gets its own pipeline:

| Model | Categorical | Numeric |
|---|---|---|
| Decision tree / random forest | Ordinal codes (trees split on them; no scaling needed) | As is |
| SVM | Group levels with < 1% of rows into one rare-level bucket (a name that cannot clash with existing levels, e.g. `__rare__`), then one-hot | Standardise (z-score); robust scaling if G3 says so (L2 p57, p64) |
| Naive Bayes | Group rare levels as above; ordinal codes for categorical NB | Equal-frequency bins (number of bins tuned) |

An attribute is treated as **ordinal** (encoded in order) only if the order is evident from
its values (e.g. `<level_low>` < `<level_mid>` < `<level_high>`); otherwise nominal (L2 p8).
Confirm any ordinal mapping at Checkpoint B. Unknown categories in the test set map to
the rare-level bucket or are ignored; they never raise an error. ⚠ The 1% rare-level and 200-column
thresholds bound the one-hot width (curse of dimensionality, L2 p44).

#### G8. Feature selection and engineering

Conservative by default, because every removal must be justified and, without domain
knowledge, constructing features is guesswork (L1 p54):

- Drop: ID-like (G4); near-constant (one value in ≥ 99% of rows — almost no information,
  L2 p51); exact duplicate columns (redundant, L2 p51).
- Numeric pairs with |r| > 0.95: drop one of each pair **for SVM only** (multicollinearity,
  L2 p59); trees are unaffected.
- No PCA by default (costs interpretability, L2 p46–48).
- Report the tree's feature importances as *embedded* selection (L2 p52).
- Engineering only with a measured trigger (e.g. a log transform for strong skew, G3;
  date columns split into year / month / weekday). State the trigger.

If almost nothing is removed or created, say why in the report: that is a finding.

#### G9. Tuning and stopping criteria

Grid search with the CV scheme of G5, scored by the primary metric of G10, at most about
30 configurations per model (L5 p8–9). Default grids:

| Model | Grid |
|---|---|
| Decision tree | `max_depth` ∈ {3, 5, 8, None}; `min_samples_leaf` ∈ {1, 5, 20}; `ccp_alpha` ∈ {0, 0.001} |
| SVM | kernel ∈ {linear, rbf}; `C` ∈ {0.1, 1, 10}; `gamma` ∈ {scale, 0.1} for rbf |
| Random forest | `n_estimators` ∈ {200}; `max_features` ∈ {sqrt, 0.5}; `min_samples_leaf` ∈ {1, 5} |
| Naive Bayes | smoothing `alpha` ∈ {0.1, 1}; bins ∈ {5, 10} |

Report stopping criteria in three layers:
1. **Model-internal:** tree pre-pruning (`max_depth`, `min_samples_leaf`, L4 p58) and
   post-pruning (`ccp_alpha`, L4 p59); SVM solver tolerance and iteration limit ⚠;
   random forest stops at `n_estimators` trees.
2. **Search-level:** the grid is exhaustive; the search stops when it is exhausted.
3. **Overfitting check:** report training score vs CV score for the chosen configuration
   (L4 p50–54). A gap above 0.10 is called out in Findings.

Use one seed (default `42`) for the split, CV shuffling and models; report it as an
experimental setting, not a hyperparameter (L5 p10). Runtime budget: if one model's
search would exceed about 15 minutes, shrink the grid or use 3 folds, and say so.

#### G10. Metrics and comparison

**Primary metric (used for tuning, the bootstrap and the headline comparison) is chosen
relative to the trivial baseline, not by a class-share cutoff.** A primary metric must
not reward a classifier for exploiting the class distribution, which is exactly why
accuracy misleads on L4 p74. `profile.json → baseline_metric_scores` gives, for each
candidate metric, the expected score of the majority-class baseline and of a uniform
random guesser on this class distribution:

| Rule | Reason |
|---|---|
| A candidate metric is **rejected** if the majority baseline beats the random guesser on it (`majority_beats_random: true`) | The metric then rewards always predicting the majority class: a model can look good without separating the classes (L4 p74) |
| Among the accepted candidates, choose **accuracy** if accepted (only when classes are balanced), otherwise **macro-F1** | Accuracy is the most familiar metric (L4 p73); macro-F1 is the F-measure (L4 p77) averaged over classes so no class can be hidden ⚠ (macro averaging is not on a slide) |

Name both baseline scores in the report when justifying the choice (e.g. "baseline
accuracy <a> > chance <b> → accuracy rejected; baseline macro-F1 <c> < chance <d> →
macro-F1"). Accuracy is still **reported** in the results table, next to the
baseline's accuracy, to show its limitation.

Always report: confusion matrices (L4 p72), per-class precision and recall (L4 p77),
CV mean ± sd of the primary metric, and the baseline row. Binary targets: also AUPRC for
the minority class, with its baseline (= prevalence) (L4 p84), and ROC-AUC (L4 p83).
⚠ Macro averaging weights every class equally, so a minority class cannot be hidden by
the majority.

**Comparison on the single test set:**
1. **95% CI for each model's accuracy** using the course interval (L4 p87–88; Wilson
   form, which reproduces the slide's table: N=100, acc=0.8 → [0.711, 0.867]).
2. **Paired bootstrap of the difference:** resample test rows with replacement 1,000
   times (fixed seed), compute the primary metric for both models on the *same*
   resample, report the 2.5th–97.5th percentile of the difference. If it contains 0,
   the difference is **not significant** (L4 p81 bootstrap; L4 p91 reasoning). Paired
   resampling is used because both models are tested on the same rows, which breaks the
   independence assumption of L4 p89–90. Limitation: this captures test-sample variance,
   not retraining variance — the CV sd gives that view.
3. **Recommendation rule.**
   - CI of the difference **excludes 0** → the primary metric decides; recommend the
     better model. Do not invoke Occam's razor to support or overturn this.
   - CI **contains 0** → the models are not distinguishable on this test set; only then
     apply Occam's razor (L4 p56) and recommend the simpler model.
   - "Simpler" means **model complexity** from `metrics.json` (e.g. a tree's leaf count vs
     Naive Bayes's number of probability estimates; support vectors; number of trees).
     Never use grid size, search effort, training time or the absence of pruning as
     evidence of simplicity: those describe the search, not the model.

### Checkpoint B: approve the plan (wait for the user)

Show `plan.md`: every decision with its measured trigger, the two models and why, the
grids, the metric, the runtime estimate. Ask the user to approve or change it. **Wait.**
Log the answer and any changes.

### Phase 4: Train and evaluate

Write `<run>/train.py` — the exact code that runs, saved before running it. It must:

1. Load the data and apply the Phase 1 decisions (confirmed tokens/sentinels,
   target-missing rows) to every file. Deduplicate **training data only**: the whole
   file before the G0 split when there is no test file, or the training file alone when
   there is one; a provided test file keeps all its labelled rows. Then split per G0
   **before any fitting**.
2. Build one sklearn `Pipeline` per model (G2, G3, G7, G8 steps inside it), so every
   transform is fitted on training folds only.
3. Run the G9 grid search with stratified CV and record per-model: best parameters,
   CV mean and sd, training score, fit time, and the **fitted model's complexity**
   (tree: depth and leaf count; random forest: trees and mean leaves; SVM: support
   vectors; Naive Bayes: number of estimated conditional probabilities). Phase 5 needs
   this for any interpretability or simplicity claim.
4. Refit the best configuration on the full training part; **evaluate on the test set
   once**.
5. Compute everything in G10 and write `<run>/metrics.json`: dataset shapes after each
   step, class counts, chosen parameters, CV results, test metrics per model (baseline
   included), confusion matrices, CI and bootstrap results, seeds, library versions.
6. Save at most two figures to `<run>/figures/`: confusion matrices side by side (always),
   and either PR curves (binary, accuracy rejected by G10; L4 p84), ROC curves (binary,
   accuracy accepted; L4 p83) or tree
   feature importances (multi-class).

If training fails, fix the code, keep the failed version as `train_attempt<N>.py`, and log it.

### Phase 5: Write the report

Fill `<skill_dir>/assets/report_template.md` into `<run>/report.md`. Rules:

- **Every number** is copied from `profile.json` or `metrics.json`. No rounding beyond
  3 decimals; no number that is not in those files.
- **Every justification names the measured value** that triggered it, and the course
  reference in parentheses, e.g. "baseline accuracy <a> > chance <b> → macro-F1 (L4 p74)".
- Space budget (of 2 pages): exploration & cleaning ~25%, features ~10%, training ~20%,
  evaluation ~25%, findings ~20%. At most two figures, one results table.
- State limitations honestly: semantics unknown, sampling, distribution shift, what the
  bootstrap does not capture.

Build and check:

```bash
python <skill_dir>/scripts/build_report.py <run>/report.md --out <run>/report.pdf
python <skill_dir>/scripts/check_pages.py <run>/report.pdf --max-pages 2
```

If over two pages: cut prose first (never figures or the results table first), never
shrink fonts or spacing, then rebuild. Repeat until it fits.

### Checkpoint C: review the draft (wait for the user)

Give the path of `report.pdf`, its page count, and a list of the headline numbers with
the JSON key each came from, so the user can spot-check. Ask for corrections. **Wait.**
Apply corrections by editing `report.md` and rebuilding — never edit the PDF.

### Phase 6: Audit log

Complete `<run>/run_log.md`:

- Environment: model identifier, interface and version, skill version, library versions, seed.
- Every Phase 3 decision with its evidence (copy from `plan.md`), and every override
  of a default with its reason.
- Every user intervention at Checkpoints A, B and C, verbatim or closely paraphrased,
  and what changed as a result.
- Deviations, warnings, failed attempts.

The log is evidence only. Do not write the user's reflection or evaluate their oversight.

## Guardrails

- Never modify or overwrite the input files. Never reuse a run folder.
- Never install packages. Never invent numbers, header fields or model identifiers.
- Never touch the test set before final evaluation (no profiling-driven decisions from
  test labels; the side-by-side class distribution is for reporting only).
- Datetime columns: extract year / month / weekday, or drop with a reason.
  Text columns: drop with a reason.
- Many classes (more than about 10): macro-averaged metrics, and a confusion matrix
  figure only if it stays legible; otherwise report per-class recall in the table.
- If a rule does not fit the data, override it with a stated reason rather than force it.

## Output contract

Inside `<run>/`:

| File | Content |
|---|---|
| `profile_raw.json`, `profile.json` | Measurements before and after target confirmation |
| `plan.md` | Every Phase 3 decision: measured value → action → reason → reference |
| `train.py` | Exact code executed (plus `train_attempt<N>.py` if any failed) |
| `metrics.json` | All numbers the report uses |
| `figures/` | At most two PNGs |
| `report.md`, `report.pdf` | The report (≤ 2 pages) |
| `run_log.md` | Audit log |
