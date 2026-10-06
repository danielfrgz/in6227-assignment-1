#!/usr/bin/env python3
"""Measure a tabular dataset and write the measurements to JSON.

This script only *measures*. It makes no preprocessing or modelling decisions;
those are made by the agent following the rules in SKILL.md, using the numbers
written here as evidence.

Two modes:

* Without ``--target``: whole-table profile used in Phase 1 (format, duplicates,
  missing and disguised-missing values, column types, target candidates).
* With ``--target``: adds the class distribution, a one-rule leakage screen per
  feature, highly correlated numeric pairs, a model-variance pilot and, if
  ``--test`` is given, a train/test comparison.

Examples::

    python profile_data.py data.csv --out profile_raw.json
    python profile_data.py data.csv --target <target> --test holdout.csv \\
        --missing-token unknown --sentinel <feature_a>=-1 --out profile.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Converted to missing automatically: empty cells and NA-style markers.
AUTO_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "?"}
# Only flagged; converted only if the user confirms (``--missing-token``),
# because they are often legitimate category levels.
FLAG_MISSING_TOKENS = {"unknown", "none", "-", "missing"}
# Numeric values commonly used as "no value" codes. Flagged, never converted.
SENTINEL_CANDIDATES = [-1, 0, 9, 99, 999, 9999, -9, -99, -999, -9999]
# Generic names often given to a target column (ML conventions, no dataset).
TARGET_NAME_HINTS = {"label", "target", "class", "y", "outcome"}

ID_DISTINCT_RATIO = 0.95       # distinct / non-missing above this -> ID-like
TEXT_MEAN_LENGTH = 30          # mean string length above this (and mostly unique) -> text
TYPE_PARSE_SHARE = 0.95        # share of values that must parse as numeric / datetime
RARE_LEVEL_SHARE = 0.01        # levels below this share count as rare
CORRELATION_ABS = 0.95         # |r| above this -> highly correlated pair
TARGET_MAX_CLASSES = 20        # more distinct values than this -> unlikely target
MAX_ROWS_FULL = 1_000_000      # above this many rows, profile a sample
SCREEN_MAX_ROWS = 20_000       # sample size for the screen and pilot
PILOT_SHALLOW_DEPTH = 5        # reference depth for the variance pilot


def sniff_format(path: Path) -> tuple[str, str]:
    """Return (delimiter, encoding) detected from the start of the file."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=encoding, newline="") as handle:
                sample = handle.read(65536)
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return delimiter, encoding


def load_raw(path: Path, delimiter: str, encoding: str) -> pd.DataFrame:
    """Read every cell as a stripped string so missing markers can be counted."""
    frame = pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str,
                        keep_default_na=False, na_filter=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.apply(lambda col: col.str.strip())


def to_missing(frame: pd.DataFrame, extra_tokens: set[str],
               sentinels: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    """Replace missing markers with NaN and return per-column counts.

    Converts AUTO_MISSING_TOKENS, user-confirmed ``extra_tokens`` (all columns)
    and user-confirmed ``sentinels`` (per column). Flagged tokens are counted
    but left in place.
    """
    convert = AUTO_MISSING_TOKENS | {t.lower() for t in extra_tokens}
    counts = {}
    out = frame.copy()
    for col in frame.columns:
        lowered = frame[col].str.lower()
        auto = lowered.isin(convert)
        sent = frame[col].isin(sentinels.get(col, []))
        counts[col] = {
            "converted": {tok if tok else "<empty>": int((lowered == tok).sum())
                          for tok in sorted(convert) if (lowered == tok).any()},
            "flagged_tokens": {tok: int((lowered == tok).sum())
                               for tok in sorted(FLAG_MISSING_TOKENS - convert)
                               if (lowered == tok).any()},
            "confirmed_sentinels": int(sent.sum()),
        }
        out.loc[auto | sent, col] = np.nan
    return out, counts


def infer_type(series: pd.Series) -> tuple[str, pd.Series | None]:
    """Infer a column type from its non-missing string values.

    Returns the type name and, for numeric columns, the parsed numeric series.
    Types: empty, numeric, boolean, datetime, text, categorical.
    """
    values = series.dropna()
    if values.empty:
        return "empty", None
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= TYPE_PARSE_SHARE:
        parsed = pd.to_numeric(series, errors="coerce")
        if parsed.dropna().nunique() == 2:
            return "boolean", parsed
        return "numeric", parsed
    if values.nunique() == 2:
        return "boolean", None
    sample = values.sample(min(len(values), 500), random_state=0)
    dates = pd.to_datetime(sample, errors="coerce", format="mixed")
    if dates.notna().mean() >= TYPE_PARSE_SHARE:
        return "datetime", None
    if (values.str.len().mean() > TEXT_MEAN_LENGTH
            and values.nunique() / len(values) > 0.5):
        return "text", None
    return "categorical", None


def sentinel_flags(numeric: pd.Series) -> list[dict]:
    """Flag common sentinel values that look like codes rather than measurements.

    A value is flagged if it occurs in at least 1% of rows and either lies outside
    Tukey's fences of the remaining values or occurs at least 10 times as often as
    the median value frequency.
    """
    values = numeric.dropna()
    if values.empty:
        return []
    counts = values.value_counts()
    median_count = float(counts.median())
    flags = []
    for cand in SENTINEL_CANDIDATES:
        n = int(counts.get(cand, 0))
        if n == 0 or n / len(values) < 0.01:
            continue
        rest = values[values != cand]
        if rest.empty:
            continue
        q1, q3 = rest.quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = cand < q1 - 1.5 * iqr or cand > q3 + 1.5 * iqr
        spike = n >= 10 * median_count
        if outside or spike:
            flags.append({"value": cand, "count": n, "share": n / len(values),
                          "outside_fences_of_rest": bool(outside),
                          "frequency_spike": bool(spike)})
    return flags


def numeric_summary(numeric: pd.Series) -> dict:
    """Location, spread, skewness and Tukey-fence outlier counts."""
    values = numeric.dropna()
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((values < low) | (values > high)).sum())
    return {
        "mean": float(values.mean()), "median": float(median),
        "std": float(values.std()), "min": float(values.min()),
        "max": float(values.max()), "q1": float(q1), "q3": float(q3),
        "skewness": float(values.skew()) if len(values) > 2 else None,
        "iqr_outlier_count": outliers,
        "iqr_outlier_share": outliers / len(values),
        "negative_count": int((values < 0).sum()),
        "is_integer_valued": bool(np.all(np.mod(values, 1) == 0)),
    }


def profile_column(clean: pd.Series, missing_info: dict) -> dict:
    """All per-column measurements.

    ID-like: distinct values / non-missing values above ID_DISTINCT_RATIO, for any
    column except non-integer numerics (continuous measurements are naturally unique).
    """
    n = len(clean)
    col_type, numeric = infer_type(clean)
    values = clean.dropna()
    distinct = int(values.nunique())
    shares = values.value_counts(normalize=True)
    flagged = sum(missing_info["flagged_tokens"].values())
    continuous = numeric is not None and not np.all(np.mod(numeric.dropna(), 1) == 0)
    info = {
        "type": col_type,
        "missing_count": int(clean.isna().sum()),
        "missing_share": float(clean.isna().mean()),
        "missing_markers": missing_info,
        "flagged_token_share": flagged / n if n else 0.0,
        "distinct_count": distinct,
        "distinct_ratio": distinct / len(values) if len(values) else 0.0,
        "top_value_share": float(shares.iloc[0]) if distinct else None,
        "top_values": {str(k): int(v) for k, v in values.value_counts().head(5).items()},
        "id_like": bool(len(values) > 0 and not continuous
                        and distinct / len(values) > ID_DISTINCT_RATIO),
        "position": None,
    }
    if col_type in {"categorical", "boolean"} and distinct:
        rare = shares[shares < RARE_LEVEL_SHARE]
        info["rare_level_count"] = int(len(rare))
        info["rare_row_share"] = float(rare.sum())
        info["onehot_width_after_grouping"] = int((shares >= RARE_LEVEL_SHARE).sum()
                                                  + (1 if len(rare) else 0))
    if col_type == "text":
        info["mean_length"] = float(values.str.len().mean())
    if numeric is not None and col_type == "numeric":
        info["numeric"] = numeric_summary(numeric)
        info["sentinel_candidates"] = sentinel_flags(numeric)
    return info


def target_candidates(columns: dict, names: list[str]) -> list[dict]:
    """Rank columns that could be the target, with the evidence for each."""
    ranked = []
    last = names[-1]
    for name in names:
        info = columns[name]
        if info["type"] in {"empty", "text", "datetime"} or info["id_like"]:
            continue
        if not 2 <= info["distinct_count"] <= TARGET_MAX_CLASSES:
            continue
        evidence = {
            "distinct_count": info["distinct_count"],
            "is_last_column": name == last,
            "name_matches_generic_hint": name.lower() in TARGET_NAME_HINTS,
            "missing_share": info["missing_share"],
        }
        score = (2 * evidence["name_matches_generic_hint"] + evidence["is_last_column"]
                 + (info["missing_share"] == 0) + (info["distinct_count"] <= 10))
        ranked.append({"column": name, "score": int(score), "evidence": evidence})
    ranked.sort(key=lambda c: (-c["score"], c["evidence"]["distinct_count"]))
    return ranked[:5]


def stratified_sample(frame: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    """Return at most ``n`` rows, keeping class proportions."""
    if len(frame) <= n:
        return frame
    frac = n / len(frame)
    return (frame.groupby(target, group_keys=False)
            .sample(frac=frac, random_state=seed))


def one_rule_screen(frame: pd.DataFrame, target: str, features: list[str],
                    types: dict, seed: int) -> dict:
    """Cross-validated balanced accuracy of a one-rule classifier per feature.

    Numeric features are cut into 10 equal-frequency bins (fitted on the training
    fold); every level or bin predicts its majority class in the training fold;
    unseen levels and missing values predict the overall training majority.
    A single feature that scores near-perfectly is a leakage suspect.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    y = frame[target].to_numpy()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(frame, y))
    scores = {}
    for feat in features:
        fold_scores = []
        for train_idx, test_idx in folds:
            tr, te = frame.iloc[train_idx], frame.iloc[test_idx]
            if types[feat] == "numeric":
                x_tr = pd.to_numeric(tr[feat], errors="coerce")
                edges = np.unique(np.nanquantile(x_tr, np.linspace(0, 1, 11)))
                key_tr = pd.Series(np.digitize(x_tr, edges[1:-1]), index=tr.index).where(x_tr.notna(), -1)
                x_te = pd.to_numeric(te[feat], errors="coerce")
                key_te = pd.Series(np.digitize(x_te, edges[1:-1]), index=te.index).where(x_te.notna(), -1)
            else:
                key_tr, key_te = tr[feat].fillna("__missing__"), te[feat].fillna("__missing__")
            majority = tr[target].mode().iloc[0]
            rule = tr[target].groupby(key_tr).agg(lambda s: s.mode().iloc[0])
            pred = key_te.map(rule).fillna(majority)
            fold_scores.append(balanced_accuracy_score(te[target], pred))
        scores[feat] = float(np.mean(fold_scores))
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def encode_for_pilot(frame: pd.DataFrame, features: list[str], types: dict) -> np.ndarray:
    """Minimal encoding for the pilot trees: numeric median-imputed, others coded."""
    cols = []
    for feat in features:
        if types[feat] in {"numeric", "boolean"} and pd.to_numeric(frame[feat], errors="coerce").notna().any():
            x = pd.to_numeric(frame[feat], errors="coerce")
            cols.append(x.fillna(x.median()).to_numpy())
        else:
            cols.append(frame[feat].fillna("__missing__").astype("category").cat.codes.to_numpy())
    return np.column_stack(cols)


def variance_pilot(frame: pd.DataFrame, target: str, features: list[str],
                   types: dict, seed: int) -> dict:
    """Compare a fully grown tree with a shallow tree under stratified 5-fold CV.

    If the shallow tree scores clearly higher, the full tree is fitting noise
    (high variance). Scores are balanced accuracy so class imbalance does not
    hide the effect. Category codes here are arbitrary; the pilot is a coarse
    measurement, not a model.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.tree import DecisionTreeClassifier

    if not features:
        return {}
    x = encode_for_pilot(frame, features, types)
    y = frame[target].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    result = {}
    for name, depth in (("full_tree", None), ("shallow_tree", PILOT_SHALLOW_DEPTH)):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        out = cross_validate(tree, x, y, cv=cv, scoring="balanced_accuracy",
                             return_train_score=True)
        result[name] = {"max_depth": depth,
                        "cv_mean": float(out["test_score"].mean()),
                        "cv_std": float(out["test_score"].std()),
                        "train_mean": float(out["train_score"].mean())}
    result["shallow_minus_full_cv"] = (result["shallow_tree"]["cv_mean"]
                                       - result["full_tree"]["cv_mean"])
    result["rows_used"] = int(len(frame))
    return result


def correlated_pairs(frame: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    """Numeric feature pairs with |Pearson r| above CORRELATION_ABS."""
    if len(numeric_cols) < 2:
        return []
    data = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr()
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > CORRELATION_ABS:
                pairs.append({"a": a, "b": b, "r": float(r)})
    return pairs


def class_distribution(series: pd.Series) -> dict:
    counts = series.value_counts()
    return {"counts": {str(k): int(v) for k, v in counts.items()},
            "n_classes": int(len(counts)),
            "minority_share": float(counts.min() / counts.sum()),
            "majority_share": float(counts.max() / counts.sum())}


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def parse_sentinels(items: list[str]) -> dict[str, list[str]]:
    """Parse repeated ``COLUMN=VALUE`` arguments."""
    out: dict[str, list[str]] = {}
    for item in items:
        col, _, value = item.rpartition("=")
        if not col:
            sys.exit(f"--sentinel expects COLUMN=VALUE, got {item!r}")
        out.setdefault(col, []).append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="path to the dataset (CSV/TSV)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--target", help="confirmed target column (enables target mode)")
    parser.add_argument("--test", type=Path, help="optional held-out test file, same columns")
    parser.add_argument("--missing-token", action="append", default=[],
                        help="extra token to treat as missing in all columns (repeatable)")
    parser.add_argument("--sentinel", action="append", default=[],
                        help="COLUMN=VALUE confirmed as a missing-value code (repeatable)")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling and CV")
    args = parser.parse_args()

    delimiter, encoding = sniff_format(args.data)
    raw = load_raw(args.data, delimiter, encoding)
    report: dict = {"file": {"path": str(args.data), "delimiter": delimiter,
                             "encoding": encoding, "rows": len(raw),
                             "columns": raw.shape[1]},
                    "seed": args.seed}
    sampled = len(raw) > MAX_ROWS_FULL
    if sampled:
        raw = raw.sample(MAX_ROWS_FULL, random_state=args.seed)
    report["file"]["profiled_rows"] = len(raw)
    report["file"]["sampled"] = sampled

    report["duplicates"] = {"exact_duplicate_rows": int(raw.duplicated().sum())}
    sentinels = parse_sentinels(args.sentinel)
    clean, missing_info = to_missing(raw, set(args.missing_token), sentinels)

    names = list(clean.columns)
    columns = {}
    for pos, name in enumerate(names):
        columns[name] = profile_column(clean[name], missing_info[name])
        columns[name]["position"] = pos
    report["columns"] = columns
    report["type_counts"] = pd.Series([c["type"] for c in columns.values()]).value_counts().to_dict()

    if not args.target:
        report["target_candidates"] = target_candidates(columns, names)
    else:
        if args.target not in clean.columns:
            sys.exit(f"target column not found: {args.target!r}")
        labelled = clean[clean[args.target].notna()]
        report["target"] = {"column": args.target,
                            "rows_missing_target": int(clean[args.target].isna().sum()),
                            **class_distribution(labelled[args.target])}
        types = {k: v["type"] for k, v in columns.items()}
        features = [c for c in names if c != args.target
                    and types[c] not in {"empty", "text", "datetime"}
                    and not columns[c]["id_like"]]
        numeric_cols = [c for c in features if types[c] == "numeric"]
        categorical = [c for c in features if types[c] in {"categorical", "boolean"}]
        report["feature_summary"] = {
            "usable_features": len(features),
            "numeric": len(numeric_cols),
            "categorical_or_boolean": len(categorical),
            "categorical_share": len(categorical) / len(features) if features else 0.0,
            "onehot_width_estimate": int(len(numeric_cols) + sum(
                columns[c].get("onehot_width_after_grouping", 0) for c in categorical)),
            "excluded": {c: types[c] if types[c] in {"empty", "text", "datetime"} else "id_like"
                         for c in names if c != args.target and c not in features},
        }
        sample = stratified_sample(labelled, args.target, SCREEN_MAX_ROWS, args.seed)
        report["one_rule_screen"] = {"rows_used": len(sample), "metric": "balanced_accuracy",
                                     "cv_folds": 3,
                                     "scores": one_rule_screen(sample, args.target, features,
                                                               types, args.seed)}
        report["correlated_pairs"] = correlated_pairs(labelled, numeric_cols)
        report["variance_pilot"] = variance_pilot(sample, args.target, features, types, args.seed)

        if args.test:
            t_delim, t_enc = sniff_format(args.test)
            t_raw = load_raw(args.test, t_delim, t_enc)
            t_clean, _ = to_missing(t_raw, set(args.missing_token), sentinels)
            same = list(t_clean.columns) == names
            test_info = {"path": str(args.test), "rows": len(t_raw),
                         "same_columns_in_same_order": same,
                         "missing_columns": sorted(set(names) - set(t_clean.columns)),
                         "extra_columns": sorted(set(t_clean.columns) - set(names)),
                         "exact_duplicate_rows": int(t_raw.duplicated().sum())}
            if args.target in t_clean.columns:
                test_info["target"] = class_distribution(t_clean[args.target].dropna())
                train_share = labelled[args.target].value_counts(normalize=True)
                test_share = t_clean[args.target].value_counts(normalize=True)
                test_info["max_class_share_difference"] = float(
                    train_share.subtract(test_share, fill_value=0).abs().max())
            shared = [c for c in names if c in t_clean.columns]
            test_info["missing_share"] = {c: float(t_clean[c].isna().mean()) for c in shared}
            report["test_file"] = test_info

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=json_default))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
