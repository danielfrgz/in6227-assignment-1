"""Train and evaluate: majority baseline, decision tree (CART), categorical Naive Bayes.

Implements plan.md (approved at Checkpoint B). Writes metrics.json and figures/.
"""
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RUN = Path(__file__).resolve().parent
TRAIN = Path("~/VSProjects/in6227-assignment-1/data/raw/train.csv").expanduser()
TEST = Path("~/VSProjects/in6227-assignment-1/data/raw/test.csv").expanduser()
TARGET, POS = "label", "yes"
SEED = 42
RARE = "__rare__"
NUM = ["aptitude_score", "performance_score", "stability_index", "load_ratio",
       "index_weight", "activity_duration", "composite_rank"]
CAT = ["geological_era", "weather_pattern", "region", "personal_interest",
       "brew_preference", "instrument", "species", "mineral_type"]
SCORING = "f1_macro"

shapes = {}


def load(path):
    # Only empty strings are missing (Checkpoint A: 'Unknown' and zeros kept as real values).
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.apply(lambda s: s.str.strip())
    df = df.replace("", np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c])
    return df


train = load(TRAIN)
test = load(TEST)
shapes["train_raw"] = list(train.shape)
shapes["test_raw"] = list(test.shape)

train = train.dropna(subset=[TARGET])
test = test.dropna(subset=[TARGET])
shapes["train_after_target_missing_drop"] = list(train.shape)
shapes["test_after_target_missing_drop"] = list(test.shape)

train = train.drop_duplicates()  # training data only; test file never deduplicated
shapes["train_after_dedup"] = list(train.shape)

X_tr, y_tr = train[NUM + CAT], (train[TARGET] == POS).astype(int).to_numpy()
X_te, y_te = test[NUM + CAT], (test[TARGET] == POS).astype(int).to_numpy()
CLASSES = ["no", "yes"]


# ---------- Model 1: decision tree pipeline ----------
tree_prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), CAT),
])
tree_pipe = Pipeline([("prep", tree_prep),
                      ("clf", DecisionTreeClassifier(criterion="gini", random_state=SEED))])
tree_grid = {"clf__max_depth": [3, 5, 8, None],
             "clf__min_samples_leaf": [1, 5, 20],
             "clf__ccp_alpha": [0.0, 0.001]}


# ---------- Model 2: categorical Naive Bayes on binned / rare-grouped features ----------
class BinnedCategoricalNB(ClassifierMixin, BaseEstimator):
    """Impute -> group rare levels (<1%) into __rare__ -> ordinal codes; numeric:
    impute median -> equal-frequency bins; then CategoricalNB. All fitted on the
    data passed to fit (training folds only). Every categorical feature reserves a
    __rare__ code, so unseen test levels never raise an error (they get the
    smoothed __rare__ probability)."""

    def __init__(self, alpha=1.0, n_bins=10, rare_threshold=0.01):
        self.alpha = alpha
        self.n_bins = n_bins
        self.rare_threshold = rare_threshold

    def _encode(self, X):
        num = self.num_imp_.transform(X[NUM])
        num = self.binner_.transform(num).astype(int)
        cat = self.cat_imp_.transform(X[CAT])
        cat_codes = np.empty(cat.shape, dtype=int)
        for j, c in enumerate(CAT):
            mapping = self.levels_[c]
            cat_codes[:, j] = [mapping.get(v, mapping[RARE]) for v in cat[:, j]]
        return np.hstack([num, cat_codes])

    def fit(self, X, y):
        self.num_imp_ = SimpleImputer(strategy="median").fit(X[NUM])
        self.binner_ = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal",
                                        strategy="quantile", quantile_method="averaged_inverted_cdf",
                                        subsample=None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # bins collapsed by ties are removed by sklearn
            self.binner_.fit(self.num_imp_.transform(X[NUM]))
        self.cat_imp_ = SimpleImputer(strategy="most_frequent").fit(X[CAT])
        cat = self.cat_imp_.transform(X[CAT])
        self.levels_ = {}
        for j, c in enumerate(CAT):
            vals, counts = np.unique(cat[:, j].astype(str), return_counts=True)
            keep = sorted(vals[counts / len(cat) >= self.rare_threshold])
            mapping = {v: i for i, v in enumerate(keep)}
            mapping[RARE] = len(keep)
            self.levels_[c] = mapping
        self.n_categories_ = np.array(list(self.binner_.n_bins_.astype(int)) +
                                      [len(self.levels_[c]) for c in CAT])
        self.nb_ = CategoricalNB(alpha=self.alpha, min_categories=self.n_categories_)
        self.nb_.fit(self._encode(X), y)
        self.classes_ = self.nb_.classes_
        return self

    def predict(self, X):
        return self.nb_.predict(self._encode(X))

    def predict_proba(self, X):
        return self.nb_.predict_proba(self._encode(X))


nb_grid = {"alpha": [0.1, 1.0], "n_bins": [5, 10]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def search(est, grid):
    gs = GridSearchCV(est, grid, scoring=SCORING, cv=cv, n_jobs=-1,
                      return_train_score=True, refit=True)
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    i = gs.best_index_
    res = gs.cv_results_
    return gs, {
        "n_configurations": len(res["params"]),
        "search_seconds": elapsed,
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "cv_primary_mean": float(res["mean_test_score"][i]),
        "cv_primary_sd": float(res["std_test_score"][i]),
        "cv_train_primary_mean": float(res["mean_train_score"][i]),
        "cv_train_minus_cv": float(res["mean_train_score"][i] - res["mean_test_score"][i]),
        "mean_fit_seconds_best": float(res["mean_fit_time"][i]),
        "all_configs": [
            {"params": {k.replace("clf__", ""): v for k, v in p.items()},
             "cv_mean": float(m), "cv_sd": float(s), "train_mean": float(t)}
            for p, m, s, t in zip(res["params"], res["mean_test_score"],
                                  res["std_test_score"], res["mean_train_score"])],
    }


tree_gs, tree_info = search(tree_pipe, tree_grid)
nb_gs, nb_info = search(BinnedCategoricalNB(), nb_grid)

# Baseline: same CV for reference
base = DummyClassifier(strategy="most_frequent")
base_cv = [f1_score(y_tr[te], clone(base).fit(X_tr.iloc[tr], y_tr[tr]).predict(X_tr.iloc[te]),
                    average="macro") for tr, te in cv.split(X_tr, y_tr)]
base.fit(X_tr, y_tr)
base_info = {"cv_primary_mean": float(np.mean(base_cv)), "cv_primary_sd": float(np.std(base_cv)),
             "predicts": CLASSES[int(base.predict(X_tr.iloc[:1])[0])]}

# ---------- Complexity of fitted (refit) models ----------
tree = tree_gs.best_estimator_.named_steps["clf"]
tree_info["complexity"] = {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
                           "nodes": int(tree.tree_.node_count)}
feat_names = NUM + CAT
imp = sorted(zip(feat_names, tree.feature_importances_), key=lambda t: -t[1])
tree_info["feature_importances"] = {k: float(v) for k, v in imp}
nb = nb_gs.best_estimator_
nb_info["complexity"] = {
    "categories_per_feature": dict(zip(feat_names, map(int, nb.n_categories_))),
    "total_categories": int(nb.n_categories_.sum()),
    "conditional_probabilities": int(nb.n_categories_.sum() * len(nb.classes_)),
    "class_priors": int(len(nb.classes_)),
}

# Training-set (full refit) score for the overfitting check
tree_info["train_primary_full_refit"] = float(f1_score(y_tr, tree_gs.predict(X_tr), average="macro"))
nb_info["train_primary_full_refit"] = float(f1_score(y_tr, nb_gs.predict(X_tr), average="macro"))


# ---------- Test evaluation (once) ----------
def wilson(acc, n, z=1.959964):
    centre = acc + z * z / (2 * n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return [float((centre - half) / d), float((centre + half) / d)]


assert np.allclose(wilson(0.8, 100), [0.711, 0.867], atol=1e-3)  # L4 p88 table check

N_TEST = len(y_te)
preds, probas, test_metrics = {}, {}, {}
for name, model in [("baseline", base), ("decision_tree", tree_gs), ("naive_bayes", nb_gs)]:
    p = model.predict(X_te)
    s = model.predict_proba(X_te)[:, 1]
    preds[name], probas[name] = p, s
    acc = accuracy_score(y_te, p)
    pr, rc, f1, sup = precision_recall_fscore_support(y_te, p, labels=[0, 1], zero_division=0)
    test_metrics[name] = {
        "accuracy": float(acc), "accuracy_ci95_wilson": wilson(acc, N_TEST),
        "macro_f1": float(f1_score(y_te, p, average="macro")),
        "per_class": {c: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]),
                          "support": int(sup[k])} for k, c in enumerate(CLASSES)},
        "auprc_yes": float(average_precision_score(y_te, s)),
        "roc_auc": float(roc_auc_score(y_te, s)),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred":
                             confusion_matrix(y_te, p, labels=[0, 1]).tolist()},
    }

# ---------- Paired bootstrap of the macro-F1 difference ----------
rng = np.random.default_rng(SEED)
B = 1000
diffs = np.empty(B)
for b in range(B):
    idx = rng.integers(0, N_TEST, N_TEST)
    diffs[b] = (f1_score(y_te[idx], preds["decision_tree"][idx], average="macro")
                - f1_score(y_te[idx], preds["naive_bayes"][idx], average="macro"))
lo, hi = np.percentile(diffs, [2.5, 97.5])
bootstrap = {"metric": "macro_f1", "difference": "decision_tree - naive_bayes",
             "resamples": B, "seed": SEED,
             "observed_difference": test_metrics["decision_tree"]["macro_f1"] - test_metrics["naive_bayes"]["macro_f1"],
             "ci95": [float(lo), float(hi)], "mean": float(diffs.mean()),
             "share_of_resamples_tree_better": float((diffs > 0).mean()),
             "contains_zero": bool(lo <= 0 <= hi)}

# ---------- Limitation counts on test: rows with the co-occurring zeros ----------
z1 = (X_te["performance_score"] == 0) & (X_te["stability_index"] == 0)
z2 = (X_te["load_ratio"] == 0) & (X_te["activity_duration"] == 0)
test_zero_rows = {
    "performance_score_and_stability_index_zero": {
        "rows": int(z1.sum()), "class_counts": test.loc[z1, TARGET].value_counts().to_dict()},
    "load_ratio_and_activity_duration_zero": {
        "rows": int(z2.sum()), "class_counts": test.loc[z2, TARGET].value_counts().to_dict()},
}

# ---------- Figures ----------
fig_dir = RUN / "figures"
fig_dir.mkdir(exist_ok=True)
INK, MUTED, BLUE, ORANGE = "#1a1a19", "#6b6a63", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
for ax, (name, title) in zip(axes, [("decision_tree", "Decision tree"), ("naive_bayes", "Naive Bayes")]):
    cm = np.array(test_metrics[name]["confusion_matrix"]["rows_true_cols_pred"])
    ax.imshow(cm, cmap=blues, vmin=0, vmax=cm.sum())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.45 * cm.sum() else INK, fontsize=10)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "confusion_matrices.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.7))
for name, title, col in [("decision_tree", "Decision tree", BLUE), ("naive_bayes", "Naive Bayes", ORANGE)]:
    pr, rc, _ = precision_recall_curve(y_te, probas[name])
    ax.plot(rc, pr, color=col, lw=2,
            label=f"{title} (AUPRC {test_metrics[name]['auprc_yes']:.3f})")
prev = float(y_te.mean())
ax.axhline(prev, color=MUTED, lw=1, ls="--", label=f"Baseline = prevalence {prev:.3f}")
ax.set_xlabel("Recall (yes)")
ax.set_ylabel("Precision (yes)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.45))
fig.tight_layout()
fig.savefig(fig_dir / "pr_curves.png", dpi=200)
plt.close(fig)

# ---------- metrics.json ----------
metrics = {
    "seed": SEED,
    "primary_metric": "macro_f1",
    "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
    "shapes": shapes,
    "class_counts": {"train": {c: int((y_tr == k).sum()) for k, c in enumerate(CLASSES)},
                     "test": {c: int((y_te == k).sum()) for k, c in enumerate(CLASSES)}},
    "test_prevalence_yes": prev,
    "features": {"numeric": NUM, "categorical": CAT, "n_features": len(NUM + CAT)},
    "models": {"baseline": base_info, "decision_tree": tree_info, "naive_bayes": nb_info},
    "test": test_metrics,
    "bootstrap": bootstrap,
    "test_zero_rows": test_zero_rows,
    "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__,
                 "matplotlib": matplotlib.__version__},
}
(RUN / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
print(json.dumps({k: metrics[k] for k in ["shapes", "bootstrap"]}, indent=1))
for m in ["decision_tree", "naive_bayes"]:
    print(m, {k: v for k, v in tree_info.items() if k != "all_configs"} if m == "decision_tree"
          else {k: v for k, v in nb_info.items() if k != "all_configs"})
print(json.dumps(test_metrics, indent=1))
