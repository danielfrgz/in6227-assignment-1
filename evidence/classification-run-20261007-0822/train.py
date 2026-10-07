"""Train and evaluate: majority baseline, decision tree (CART), categorical Naive Bayes.

Implements plan.md of classification-run-20261007-0822. Every transform is fitted
inside a Pipeline, so it only ever sees training folds. The test file is read once,
after model selection, and used once for the final evaluation.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RUN = Path(__file__).resolve().parent
TRAIN_PATH = Path("~/VSProjects/in6227-assignment-1/data/raw/train.csv").expanduser()
TEST_PATH = Path("~/VSProjects/in6227-assignment-1/data/raw/test.csv").expanduser()
TARGET = "label"
CLASSES = ["no", "yes"]
POSITIVE = "yes"  # minority class
SEED = 42
N_FOLDS = 5
N_BOOT = 1000
RARE_SHARE = 0.01
RARE = "__rare__"
Z = 1.96

profile = json.loads((RUN / "profile.json").read_text())
NUMERIC = [c for c, v in profile["columns"].items() if v["type"] == "numeric"]
CATEGORICAL = [c for c, v in profile["columns"].items()
               if v["type"] in ("categorical", "boolean") and c != TARGET]


def load(path):
    # Only empty strings become missing: that is the only token the profiler converted
    # and the only one confirmed at Checkpoint A ("unknown" is kept as a real level).
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c])
    return df


class RareGrouper(BaseEstimator, TransformerMixin):
    """Integer-code categorical columns; levels below `min_share` of the fitted rows,
    and levels unseen at fit time, share one code. That code is 0, so it always lies
    inside the range CategoricalNB was fitted on, even when no training row uses it."""

    def __init__(self, min_share=RARE_SHARE):
        self.min_share = min_share

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.levels_ = []
        for c in X.columns:
            share = X[c].value_counts(normalize=True)
            self.levels_.append([RARE] + sorted(share[share >= self.min_share].index))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=int)
        for j, c in enumerate(X.columns):
            codes = {lvl: i for i, lvl in enumerate(self.levels_[j])}
            out[:, j] = X[c].map(codes).fillna(0).astype(int).to_numpy()
        return out


def tree_pipeline():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            # nominal levels get arbitrary codes; unseen test levels -> -1, never an error
            ("codes", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", DecisionTreeClassifier(criterion="gini", random_state=SEED))])


def nb_pipeline():
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("bins", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile",
                                      quantile_method="averaged_inverted_cdf", subsample=None)),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("group", RareGrouper()),
        ]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", CategoricalNB())])


def wilson_ci(acc, n, z=Z):
    """Course interval for accuracy (L4 p87-88)."""
    centre = 2 * n * acc + z ** 2
    half = z * np.sqrt(z ** 2 + 4 * n * acc - 4 * n * acc ** 2)
    denom = 2 * (n + z ** 2)
    return [float((centre - half) / denom), float((centre + half) / denom)]


def class_counts(series):
    return {k: int(v) for k, v in series.value_counts().reindex(CLASSES, fill_value=0).items()}


metrics = {"seed": SEED, "steps": {}, "versions": {
    "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}}

# ---- 1. Load training data, apply Phase 1 decisions ---------------------------------
train = load(TRAIN_PATH)
metrics["steps"]["train_raw_rows"] = len(train)
train = train[train[TARGET].notna()].reset_index(drop=True)
metrics["steps"]["train_after_dropping_missing_target"] = len(train)
n_dup = int(train.duplicated().sum())
train = train.drop_duplicates().reset_index(drop=True)
metrics["steps"]["train_duplicates_dropped"] = n_dup
metrics["steps"]["train_final_rows"] = len(train)
metrics["steps"]["features_numeric"] = NUMERIC
metrics["steps"]["features_categorical"] = CATEGORICAL

feats = NUMERIC + CATEGORICAL
dup_cols = [[a, b] for i, a in enumerate(feats) for b in feats[i + 1:] if train[a].equals(train[b])]
metrics["steps"]["exact_duplicate_feature_columns"] = dup_cols

X, y = train[feats], train[TARGET]
metrics["train_class_counts"] = class_counts(y)

# Evidence for the limitations agreed at Checkpoint A (training rows only).
metrics["flagged_value_class_counts_train"] = {
    "performance_score_and_stability_index_both_0": class_counts(
        y[(X.performance_score == 0) & (X.stability_index == 0)]),
    "load_ratio_and_activity_duration_both_0": class_counts(
        y[(X.load_ratio == 0) & (X.activity_duration == 0)]),
    "aptitude_score_eq_510": class_counts(y[X.aptitude_score == 510]),
    "performance_score_eq_1010": class_counts(y[X.performance_score == 1010]),
    "personal_interest_eq_unknown": class_counts(y[X.personal_interest.str.lower() == "unknown"]),
    "species_eq_unknown": class_counts(y[X.species.str.lower() == "unknown"]),
}

# ---- 2-3. Grid search with stratified CV, primary metric macro-F1 ---------------------
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
grids = {
    "decision_tree": (tree_pipeline(), {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [1, 5, 20],
        "model__ccp_alpha": [0.0, 0.001],
    }),
    "naive_bayes": (nb_pipeline(), {
        "model__alpha": [0.1, 1.0],
        "pre__num__bins__n_bins": [5, 10],
    }),
}

metrics["cv"] = {"folds": N_FOLDS, "stratified": True, "shuffle": True, "scoring": "f1_macro"}
best = {}
base_cv = cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=cv,
                         scoring="f1_macro", return_train_score=True)
metrics["models"] = {"majority_baseline": {
    "cv_macro_f1_mean": float(base_cv["test_score"].mean()),
    "cv_macro_f1_sd": float(base_cv["test_score"].std()),
    "train_macro_f1_mean": float(base_cv["train_score"].mean())}}

for name, (pipe, grid) in grids.items():
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, refit=True,
                          return_train_score=True, n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    r, i = search.cv_results_, search.best_index_
    best[name] = search.best_estimator_
    m = {
        "n_configurations": len(r["params"]),
        "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        "cv_macro_f1_mean": float(r["mean_test_score"][i]),
        "cv_macro_f1_sd": float(r["std_test_score"][i]),
        "train_macro_f1_mean": float(r["mean_train_score"][i]),
        "train_minus_cv": float(r["mean_train_score"][i] - r["mean_test_score"][i]),
        "mean_fit_time_s": float(r["mean_fit_time"][i]),
        "refit_time_s": float(search.refit_time_),
        "search_time_s": float(elapsed),
        "all_configs": [{"params": {k.split("__")[-1]: v for k, v in p.items()},
                         "cv_mean": float(mu), "cv_sd": float(sd), "train_mean": float(tr)}
                        for p, mu, sd, tr in zip(r["params"], r["mean_test_score"],
                                                 r["std_test_score"], r["mean_train_score"])],
    }
    est = search.best_estimator_.named_steps["model"]
    if name == "decision_tree":
        m["complexity"] = {"depth": int(est.get_depth()), "leaves": int(est.get_n_leaves()),
                           "nodes": int(est.tree_.node_count)}
        imp = dict(zip(feats, est.feature_importances_))
        m["feature_importances"] = {k: float(v) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    else:
        levels = [int(c.shape[1]) for c in est.category_count_]
        m["complexity"] = {
            "levels_per_feature": dict(zip(feats, levels)),
            "total_levels": int(sum(levels)),
            "conditional_probabilities": int(sum(levels) * len(est.classes_)),
            "class_priors": int(len(est.classes_)),
        }
        m["numeric_bins_after_fit"] = dict(zip(NUMERIC, (int(b) for b in search.best_estimator_
                                                         .named_steps["pre"].named_transformers_["num"]
                                                         .named_steps["bins"].n_bins_)))
    metrics["models"][name] = m

# ---- 4. Final evaluation on the held-out test file, once -------------------------------
test = load(TEST_PATH)
metrics["steps"]["test_raw_rows"] = len(test)
test = test[test[TARGET].notna()].reset_index(drop=True)
metrics["steps"]["test_after_dropping_missing_target"] = len(test)  # never deduplicated
X_test, y_test = test[feats], test[TARGET]
metrics["test_class_counts"] = class_counts(y_test)
metrics["test_unseen_levels"] = {c: sorted(set(X_test[c].dropna()) - set(X[c].dropna()))
                                 for c in CATEGORICAL}

baseline = DummyClassifier(strategy="most_frequent").fit(X, y)
models = {"majority_baseline": baseline, **best}
n_test = len(y_test)
preds, scores = {}, {}
metrics["test"] = {}
for name, model in models.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(POSITIVE)]
    preds[name], scores[name] = pred, proba
    acc = accuracy_score(y_test, pred)
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=CLASSES, zero_division=0)
    metrics["test"][name] = {
        "accuracy": float(acc),
        "accuracy_ci95_wilson": wilson_ci(acc, n_test),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "per_class": {c: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                          "support": int(s[k])} for k, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred":
                             confusion_matrix(y_test, pred, labels=CLASSES).tolist()},
        "auprc_yes": float(average_precision_score(y_test == POSITIVE, proba)),
        "roc_auc": float(roc_auc_score(y_test == POSITIVE, proba)),
    }
metrics["test_n"] = n_test
metrics["auprc_baseline_prevalence"] = float((y_test == POSITIVE).mean())
metrics["wilson_check_N100_acc0.8"] = wilson_ci(0.8, 100)

# Paired bootstrap of the macro-F1 difference (tree - NB) on the same resampled rows.
rng = np.random.default_rng(SEED)
yt = y_test.to_numpy()
diffs = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, n_test, n_test)
    diffs[b] = (f1_score(yt[idx], preds["decision_tree"][idx], average="macro")
                - f1_score(yt[idx], preds["naive_bayes"][idx], average="macro"))
lo, hi = np.percentile(diffs, [2.5, 97.5])
observed = metrics["test"]["decision_tree"]["macro_f1"] - metrics["test"]["naive_bayes"]["macro_f1"]
excludes_zero = bool(lo > 0 or hi < 0)
metrics["bootstrap"] = {"metric": "macro_f1", "difference": "decision_tree - naive_bayes",
                        "resamples": N_BOOT, "seed": SEED, "observed_difference": float(observed),
                        "ci95": [float(lo), float(hi)], "mean": float(diffs.mean()),
                        "ci_excludes_zero": excludes_zero}

# Recommendation rule (G10 / R-12): the metric decides if the CI excludes 0; only
# otherwise Occam's razor, with simplicity judged by fitted model complexity.
tree_c = metrics["models"]["decision_tree"]["complexity"]["leaves"]
nb_c = metrics["models"]["naive_bayes"]["complexity"]["conditional_probabilities"]
if excludes_zero:
    winner = "decision_tree" if observed > 0 else "naive_bayes"
    reason = "bootstrap CI of the macro-F1 difference excludes 0: the primary metric decides"
else:
    winner = "decision_tree" if tree_c < nb_c else "naive_bayes"
    reason = ("bootstrap CI contains 0: models indistinguishable; Occam's razor on fitted "
              "complexity (tree leaves vs NB conditional probabilities)")
metrics["recommendation"] = {"model": winner, "reason": reason,
                             "complexity_compared": {"decision_tree_leaves": tree_c,
                                                     "naive_bayes_conditional_probabilities": nb_c}}

# ---- 6. Figures --------------------------------------------------------------------------
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"
LABELS = {"decision_tree": "Decision tree", "naive_bayes": "Naive Bayes"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig_dir = RUN / "figures"
fig_dir.mkdir(exist_ok=True)

ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#184f95"])
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
for ax, name in zip(axes, ["decision_tree", "naive_bayes"]):
    cm = np.array(metrics["test"][name]["confusion_matrix"]["rows_true_cols_pred"])
    ax.imshow(cm, cmap=ramp, vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() * 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(LABELS[name], color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "confusion_matrices.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.4, 2.8))
for name, colour in [("decision_tree", BLUE), ("naive_bayes", ORANGE)]:
    prec, rec, _ = precision_recall_curve(y_test == POSITIVE, scores[name])
    ax.plot(rec, prec, color=colour, lw=2,
            label=f"{LABELS[name]} (AUPRC {metrics['test'][name]['auprc_yes']:.3f})")
prev = metrics["auprc_baseline_prevalence"]
ax.axhline(prev, color=MUTED, lw=1, ls="--", label=f"Baseline = prevalence ({prev:.3f})")
ax.set_xlabel("Recall (yes)")
ax.set_ylabel("Precision (yes)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(color="#e6e5df", lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(fig_dir / "pr_curves.png", dpi=200)
plt.close(fig)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


(RUN / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_default))
print(json.dumps({k: metrics[k] for k in ("bootstrap", "recommendation")}, indent=2))
