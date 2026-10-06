# Tabular Classification Report: Claude Code Skill

IN6227 Data Mining, Assignment 1, Variant 2.

`tabular-classification-report` is an agent skill for end-to-end classification on any
tabular dataset. Given a path to a CSV/TSV file, it profiles the data, has the AI
choose and justify the preprocessing and two contrasting classifiers, evaluates them
against a majority-class baseline, and generates a PDF report of at most two pages, plus
an audit log of every decision and every human intervention.

```
tabular-classification-report/
├── SKILL.md                    workflow, decision rules (G0–G10), checkpoints, guardrails
├── scripts/
│   ├── profile_data.py         measures the data → profile.json
│   ├── build_report.py         renders report.md → report.pdf (course template layout)
│   └── check_pages.py          fails if the PDF exceeds the page limit
└── assets/
    └── report_template.md      header fields and the five report sections
```

## Design

The assignment asks the skill to have *the AI* select models and preprocessing. So the
repository is split in two:

| Part | Role | Contains |
|---|---|---|
| `SKILL.md` | **Decides** | The workflow and the decision rules: for each decision group (split, missing values, outliers, IDs and leakage, imbalance, model choice, encoding, feature selection, tuning, metrics) a table of *measured condition → action → reason → course reference*. Every threshold is a default with a rationale that doesn't depend on any particular dataset; the agent may override one only with a stated reason, which is logged |
| `scripts/` | **Measures and renders** | No modelling decisions. `profile_data.py` turns the file into numbers (types, missing and disguised-missing values, outliers, caps and shared zeros, a one-rule leakage screen, a model-variance pilot, baseline metric scores, train/test comparison). `build_report.py` and `check_pages.py` only lay out and check the document |

The training code is not a fixed script. The agent writes `train.py` for each run,
following the Phase 4 contract in SKILL.md, and saves the exact code it executed in the
run folder, so every reported number can be traced back to it.

Three principles follow from the grading brief ("a well-justified simple model beats an
unexplained complex one"):

1. **Every decision cites evidence measured on the dataset at hand**, and every number in
   the report comes from `profile.json` or `metrics.json` of that run.
2. **Decisions map to course content.** Rules cite IN6227 lectures as `L<lecture> p<page>`.
   Where the course has no slide for something the pipeline needs (e.g. macro averaging,
   fitting preprocessing on training folds only), the rule is marked ⚠ and its rationale
   is stated instead.
3. **The human stays in the loop.** The skill stops at three checkpoints: A (data
   understanding: target, disguised missing values, caps), B (the plan: models,
   preprocessing, metric) and C (the draft report). Every intervention is logged.

## Skill generalization

- **Nothing dataset-specific in the skill folder.** No column names, target names,
  category values, file names or paths. Examples use placeholders (`<feature_a>`,
  `<target>`). Checked by a grep of the skill folder against the provided dataset's
  column names and category values before tagging.
- **The target is proposed, not assumed.** Candidates are ranked by measured evidence
  (number of classes, position, a generic list of conventional names, missingness) and
  confirmed by the user.
- **Disguised missing values come from a generic token list plus profiling**, not from
  one dataset's quirks. Ambiguous tokens (e.g. a level named "unknown") and numeric codes
  (0, −1, 999, repeated extremes) are flagged for the user, never converted silently.
- **Model choice follows measured properties.** There's no fixed pair. Model 1 is always a
  decision tree. Model 2 is Naive Bayes, a random forest or an SVM, depending on the
  categorical share, a variance pilot and the dataset size.
- **The primary metric is chosen relative to the trivial baseline.** A metric is rejected
  if the majority-class baseline beats random guessing on it. There's no class-share cutoff.
- **Scope is explicit.** SKILL.md lists what is out of scope (regression, multi-table,
  non-delimited formats, free text) and what the skill does when it meets it.

## Evidence

`evidence/` holds run outputs on the assignment's dataset, with placeholder
report headers (`plan.md`, `metrics.json`, `report.pdf`, `run_log.md`). The run logs show
the checkpoints in use, including a case where the human overrode a rule at Checkpoint B
and the rule was then changed in the skill (see the commit history).

## Installation and use

Requirements: Python 3.10+ (developed on 3.12) and the pinned packages in
`requirements.txt`. The skill checks its imports at start and reports anything missing;
it never installs packages itself.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# make the skill available to Claude Code (user-level skills folder)
ln -s "$(pwd)/tabular-classification-report" ~/.claude/skills/tabular-classification-report
```

Then, in a Claude Code session started in an empty working folder:

```
/tabular-classification-report /path/to/data.csv
```

Optionally, put a `report_config.yaml` in the working folder with the report header
fields (`name`, `matric`, `assignment`, `variant`, `repo_url`); otherwise the skill asks
for them. Each run writes to a new `classification-run-<YYYYMMDD-HHMM>/` folder:
`profile.json`, `plan.md`, `train.py`, `metrics.json`, `figures/`, `report.md`,
`report.pdf` and `run_log.md`.

The scripts can also be run on their own (`--help` documents each):

```bash
python tabular-classification-report/scripts/profile_data.py data.csv --out profile.json
python tabular-classification-report/scripts/build_report.py report.md --out report.pdf
python tabular-classification-report/scripts/check_pages.py report.pdf --max-pages 2
```

## AI Usage and Accountability

This skill was built in **Claude Code** with the aid of **Claude Opus 5.5**. 

Based on the assignment description and requirements, **I wrote a build brief myself** (i.e., a .md file setting out my own requirements and implementation plan for the assignment) and then iteratively wrote and extended the SKILL.md file and the individual scripts with the support of Claude Opus 5.5 (**not** Claude Code, i.e., no code was written/generated without my supervision; **all AI-generated code fragments and SKILL.md prompt refinements were requested and supervised by myself**). 

AI was also used to generate fragments of this file (e.g., repository structure, installation and use section), once again under my own request and supervision.

During the development of this skill, **all judgement calls were made by myself**. Determining the rule to adopt for each decision group, the layout of the report, and changes to be made after executing a trial run were all **decisions that I made**, although AI was used to seek advice at times. 

Rules such as the primary-metric rule, Occam’s-razor reasoning, claims about tree interpretability, test-set deduplication, and detection of capped and zero-coded values are a result of my own questioning and challenge of AI output.

I verified a test run’s numbers independently, outside of the skill, by recomputing metrics from the confusion matrices and checking the baseline scores analytically.

The reflection I wrote on oversight, critical evaluation and trustworthiness is part of the submitted report, but not part of this repository (i.e., this skill cannot generate such a reflection).


## License

No license is granted. The repository is public for assessment purposes only.