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
<!-- Readers must be able to trust and challenge every decision, so each one cites
     evidence measured on THIS dataset. A simple, justified model beats an
     unexplained complex one. -->

## Scope
<!-- Supported inputs and tasks. What is out of scope, and what the skill does when
     it meets it. -->

## Inputs
<!-- Dataset path: from the invocation arguments or the user's message; ask if missing.
     Header fields: report_config.yaml if present, otherwise ask.
     Output: ./classification-run-<YYYYMMDD-HHMM>/, never reused. -->

## Environment check
<!-- Verify imports; report what's missing instead of installing.
     Record the model identifier and harness version; ask if unknown. -->

## Workflow

### Phase 1: Load and sanity-check
<!-- Format, delimiter, encoding. Duplicates. Disguised missing values (generic token
     list + profiling, never one dataset's quirks). Target: propose candidates with
     evidence, confirm with the user. Stop if the target is continuous or single-class. -->

### Checkpoint A: confirm data understanding (wait for the user)

### Phase 2: Profile
<!-- scripts/profile_data.py → profile.json: types, missing %, cardinality, ID-likeness,
     near-constant columns, class distribution, outliers, single-feature leakage screen. -->

### Phase 3: Decide (the core of the skill)
<!-- Table: measured condition → action → justification template → course reference.
     Thresholds are defaults with a rationale independent of any one dataset; override
     only with a stated reason, and log it. Groups:
     - missing values
     - outliers and scaling
     - categorical encoding by cardinality
     - ID and leakage removal (statistical screen + semantic check: is this value
       known before the outcome?)
     - class imbalance (stratification, class weights, metric choice)
     - dataset size (cross-validation vs hold-out)
     - dimensionality (feature selection)
     - model choice (two models with contrasting inductive biases + a majority-class baseline)
     - tuning (search space, CV, stopping criteria) -->

### Checkpoint B: approve the plan (wait for the user)

### Phase 4: Train and evaluate
<!-- Split first. Fit all preprocessing inside a pipeline on training folds only.
     Tune with stratified CV on the training set; touch the test set once. Fixed seeds.
     Save the exact code executed (train.py) and metrics.json. -->

### Phase 5: Write the report
<!-- Fill assets/report_template.md. Every number from metrics.json or profile.json.
     Every justification names the measured value that triggered it. Space budget per
     section; at most two figures. Build, then check_pages.py. If over two pages, cut
     prose before figures and rebuild. -->

### Checkpoint C: review the draft (wait for the user)

### Phase 6: Audit log
<!-- run_log.md: every decision with its evidence, every user intervention at A/B/C,
     deviations and warnings. Evidence only; the human writes the reflection. -->

## Guardrails
<!-- Never modify the input file. Never invent numbers. State limitations.
     Large data: profile a sample and say so. Text and date columns: handle, or drop
     with a stated reason. Many classes: macro-averaged metrics. Respect a runtime budget. -->

## Output contract
<!-- profile.json, plan.md, train.py, metrics.json, figures/, report.md, report.pdf, run_log.md -->