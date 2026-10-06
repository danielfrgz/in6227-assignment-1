"""Train and evaluate: majority baseline, decision tree, categorical Naive Bayes.

Exact code executed for classification-run-20261006-2339. Decisions: see plan.md.
"""
import json
import os
import sys
import time

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
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.expanduser("~/VSProjects/in6227-assignment-1/data/raw/train.csv")
TEST = os.path.expanduser("~/VSProjects/in6227-assignment-1/data/raw/test.csv")
TARGET, POS, CLASSES = "label", "yes", ["no", "yes"]
NUMERIC = ["aptitude_score", "performance_score", "stability_index", "load_ratio",
           "index_weight", "activity_duration", "composite_rank"]
CATEGORICAL = ["geological_era", "weather_pattern", "region", "personal_interest",
               "brew_preference", "instrument", "species", "mineral_type"]
FEATURES = CATEGORICAL + NUMERIC


class RareOrdinal(BaseEstimator, TransformerMixin):
    """Nominal -> integer codes. Levels with < min_share of training rows, and levels
    unseen in training, share code 0 (the `__rare__` bucket); kept levels get 1..k in
    alphabetical order. Code 0 is always within range, so CategoricalNB never fails."""

    def __init__(self, min_share=0.01):
        self.min_share = min_share

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = []
        for c in X.columns:
            share = X[c].value_counts(normalize=True)
            kept = sorted(share[share >= self.min_share].index)
            self.maps_.append({v: i + 1 for i, v in enumerate(kept)})
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape, dtype=int)
        for j, c in enumerate(X.columns):
            out[:, j] = X[c].map(self.maps_[j]).fillna(0).astype(int).to_numpy()
        return out


def load(path, shapes, tag):
    d = pd.read_csv(path, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    shapes[f"{tag}_raw"] = list(d.shape)
    for c in CATEGORICAL + [TARGET]:
        d[c] = d[c].astype("string").str.strip().replace("", pd.NA).astype(object)
        d[c] = d[c].where(d[c].notna(), np.nan)
    d = d.drop_duplicates()
    shapes[f"{tag}_after_dedup"] = list(d.shape)
    d = d[d[TARGET].notna()].reset_index(drop=True)
    shapes[f"{tag}_after_target_drop"] = list(d.shape)
    return d[FEATURES], d[TARGET].to_numpy()


def tree_pipe():
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC)])
    return Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(criterion="gini",
                                                                  random_state=SEED))])


def nb_pipe():
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", RareOrdinal(min_share=0.01))]), CATEGORICAL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("bin", KBinsDiscretizer(encode="ordinal", strategy="quantile",
                                                   quantile_method="averaged_inverted_cdf",
                                                   subsample=None))]), NUMERIC)])
    return Pipeline([("pre", pre), ("clf", CategoricalNB())])


def wilson(acc, n, z=1.96):
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def test_metrics(y, pred, score):
    m = {"accuracy": accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro"),
         "precision": dict(zip(CLASSES, precision_score(y, pred, labels=CLASSES, average=None,
                                                        zero_division=0).tolist())),
         "recall": dict(zip(CLASSES, recall_score(y, pred, labels=CLASSES, average=None,
                                                  zero_division=0).tolist())),
         "roc_auc": roc_auc_score(y == POS, score),
         "auprc": average_precision_score(y == POS, score),
         "confusion_matrix": {"labels": CLASSES,
                              "matrix": confusion_matrix(y, pred, labels=CLASSES).tolist()}}
    m["accuracy_ci95_wilson"] = wilson(m["accuracy"], len(y))
    return m


def main():
    shapes = {}
    Xtr, ytr = load(TRAIN, shapes, "train")
    Xte, yte = load(TEST, shapes, "test")  # not used until final evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"macro_f1": "f1_macro", "accuracy": "accuracy"}

    models = {
        "decision_tree": (tree_pipe(), {"clf__max_depth": [3, 5, 8, None],
                                        "clf__min_samples_leaf": [1, 5, 20],
                                        "clf__ccp_alpha": [0.0, 0.001]}),
        "naive_bayes": (nb_pipe(), {"clf__alpha": [0.1, 1.0],
                                    "pre__num__bin__n_bins": [5, 10]}),
    }
    results = {}

    # Baseline: majority class, same CV scheme
    base = DummyClassifier(strategy="most_frequent")
    t0 = time.time()
    bcv = cross_validate(base, Xtr, ytr, cv=cv, scoring=scoring, return_train_score=True)
    base.fit(Xtr, ytr)
    results["baseline_majority"] = {
        "best_params": {"strategy": "most_frequent", "predicted_class": str(base.classes_[np.argmax(base.class_prior_)])},
        "n_configs": 1,
        "cv": {k: {"mean": float(np.mean(bcv[f"test_{k}"])), "sd": float(np.std(bcv[f"test_{k}"]))} for k in scoring},
        "train_score_macro_f1": float(np.mean(bcv["train_macro_f1"])),
        "search_time_s": time.time() - t0,
    }
    fitted = {"baseline_majority": base}

    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, scoring=scoring, refit="macro_f1", cv=cv,
                          return_train_score=True, n_jobs=-1)
        t0 = time.time()
        gs.fit(Xtr, ytr)
        elapsed = time.time() - t0
        i = gs.best_index_
        r = gs.cv_results_
        results[name] = {
            "best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
            "n_configs": len(r["params"]),
            "cv": {k: {"mean": float(r[f"mean_test_{k}"][i]), "sd": float(r[f"std_test_{k}"][i])} for k in scoring},
            "train_score_macro_f1": float(r["mean_train_macro_f1"][i]),
            "train_cv_gap_macro_f1": float(r["mean_train_macro_f1"][i] - r["mean_test_macro_f1"][i]),
            "mean_fit_time_s_best": float(r["mean_fit_time"][i]),
            "search_time_s": elapsed,
            "all_configs": [{"params": {k.split("__")[-1]: v for k, v in p.items()},
                             "cv_macro_f1": float(m), "cv_macro_f1_sd": float(s)}
                            for p, m, s in zip(r["params"], r["mean_test_macro_f1"], r["std_test_macro_f1"])],
        }
        fitted[name] = gs.best_estimator_  # refit on full training part by GridSearchCV

    tree = fitted["decision_tree"].named_steps["clf"]
    results["decision_tree"]["tree_depth"] = int(tree.get_depth())
    results["decision_tree"]["n_leaves"] = int(tree.get_n_leaves())
    imp = dict(zip(FEATURES, tree.feature_importances_.tolist()))
    results["decision_tree"]["feature_importances"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    # ---- Final evaluation on the test set: once ----
    test = {}
    preds, scores = {}, {}
    for name, est in fitted.items():
        pos_idx = list(est.classes_).index(POS)
        preds[name] = est.predict(Xte)
        scores[name] = est.predict_proba(Xte)[:, pos_idx]
        test[name] = test_metrics(yte, preds[name], scores[name])

    # Paired bootstrap of the primary-metric difference (tree - NB)
    rng = np.random.default_rng(SEED)
    n = len(yte)
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        diffs.append(f1_score(yte[idx], preds["decision_tree"][idx], average="macro")
                     - f1_score(yte[idx], preds["naive_bayes"][idx], average="macro"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot = {"metric": "macro_f1", "difference": "decision_tree - naive_bayes",
            "observed": test["decision_tree"]["macro_f1"] - test["naive_bayes"]["macro_f1"],
            "n_resamples": 1000, "seed": SEED, "ci95": [float(lo), float(hi)],
            "contains_zero": bool(lo <= 0 <= hi)}

    # ---- Figures ----
    figdir = os.path.join(HERE, "figures")
    os.makedirs(figdir, exist_ok=True)
    titles = {"decision_tree": "Decision tree", "naive_bayes": "Naive Bayes"}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, name in zip(axes, ["decision_tree", "naive_bayes"]):
        cm = np.array(test[name]["confusion_matrix"]["matrix"])
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{cm[a, b]:,}", ha="center", va="center",
                        color="white" if cm[a, b] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(titles[name])
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "confusion_matrices.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, col in [("decision_tree", "#2a6fdb"), ("naive_bayes", "#d9822b")]:
        p, r, _ = precision_recall_curve(yte == POS, scores[name])
        ax.plot(r, p, color=col, label=f"{titles[name]} (AUPRC {test[name]['auprc']:.3f})")
    prev = float(np.mean(yte == POS))
    ax.axhline(prev, color="grey", ls="--", label=f"Baseline (prevalence {prev:.3f})")
    ax.set_xlabel("Recall (yes)")
    ax.set_ylabel("Precision (yes)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "pr_curves.png"), dpi=200)
    plt.close(fig)

    def counts(y):
        u, c = np.unique(y, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    # Training-data diagnostic for the zero-value limitation (Checkpoint A); test not used
    zero_diag = {}
    for a, b in [("performance_score", "stability_index"), ("load_ratio", "activity_duration")]:
        both = ((Xtr[a] == 0) & (Xtr[b] == 0)).to_numpy()
        zero_diag[f"{a}=0 & {b}=0"] = {"rows": int(both.sum()), "label_counts": counts(ytr[both])}

    out = {
        "seed": SEED,
        "train_zero_value_rows": zero_diag,
        "primary_metric": "macro_f1 (override of accuracy default; Checkpoint B)",
        "cv_scheme": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "shapes": shapes,
        "class_counts": {"train": counts(ytr), "test": counts(yte)},
        "test_positive_prevalence": prev,
        "models": results,
        "test": test,
        "bootstrap": boot,
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote metrics.json")


if __name__ == "__main__":
    main()
