---
title: <descriptive title: task and dataset in plain words>
name: <full name>
matric: <matric number>
assignment: <assignment line>
variant: <variant line>
model: <model name and version>
interface: <interface and version>
skill: <repo URL @ tag or version>
---
<!--
Instructions for the agent (comments are removed when the PDF is built):
- Every number comes from profile.json or metrics.json of this run. Never round
  beyond 3 decimals, never estimate.
- Every justification names the measured value that triggered it, with the course
  reference in parentheses, e.g. "(L4 p84)".
- Budget of 2 pages: section 1 ~25%, 2 ~10%, 3 ~20%, 4 ~25%, 5 ~20%.
  At most two figures and one results table. Never shrink fonts to fit; cut prose.
- Use the real column names of the dataset here, in the run's report.md.
-->

## 1. Data exploration and cleaning
<!-- Shape (rows, columns, feature types), target and class distribution, test file
     if any. Issues found with numbers: missing and disguised missing values, outliers
     (IQR counts), duplicates, ID-like columns, leakage screen, semantics known or not.
     Each cleaning step taken and why; what was deliberately NOT done and why. -->

## 2. Feature selection and engineering
<!-- What was dropped or created, with the measured trigger. If (almost) nothing,
     say why that is the justified choice. Encoding and scaling per model (L2 p59). -->

## 3. Model training
<!-- Baseline; Model 1 and Model 2 and why this pair (measured trigger, contrasting
     inductive biases). Validation scheme and why. Search space, number of
     configurations, chosen parameters. Stopping criteria in three layers
     (model-internal, search-level, overfitting check). Seed. -->

## 4. Evaluation and comparison
<!-- Metric choice and why (minority share). One results table: baseline and both
     models; CV mean ± sd, test metrics, accuracy 95% CI. Paired bootstrap CI of the
     difference and whether it is significant. One figure. -->

| Model | CV <primary> (mean ± sd) | Test accuracy [95% CI] | Test macro-F1 | <other metric> |
|---|---|---|---|---|
| Majority baseline | | | | |
| <model 1> | | | | |
| <model 2> | | | | |

![<caption>](figures/<figure>.png){width=80%}

## 5. Findings and discussion
<!-- What the comparison shows and what it does not. Recommendation (Occam's razor
     if not significant, L4 p56). Overfitting check. Limitations: semantics, sampling,
     distribution shift, what the bootstrap does not capture. What would change the
     conclusion. -->
