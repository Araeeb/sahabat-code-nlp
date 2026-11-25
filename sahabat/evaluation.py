"""Honest evaluation helpers (Lab 6): metrics, slices, bootstrap CIs, retrieval."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true, y_pred, labels=None):
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def bootstrap_ci(y_true, y_pred, metric=accuracy_score, n=1000, seed=42, alpha=0.05):
    """Resample rows with replacement; return (point, low, high)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true[i], y_pred[i]))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(low), float(high)


def sliced_accuracy(df, slice_cols=("language", "dialect", "length"), seed=42):
    """Accuracy with a 95% bootstrap CI for every value of every slice column."""
    rows = []
    for col in slice_cols:
        for value, part in df.groupby(col):
            acc, low, high = bootstrap_ci(part["label"], part["pred"], seed=seed)
            rows.append({"slice": col, "value": value, "n": len(part),
                         "accuracy": round(acc, 2), "ci_low": round(low, 2), "ci_high": round(high, 2)})
    return pd.DataFrame(rows)


def recall_and_mrr(ranked_relevance, k=10):
    """ranked_relevance: one list of booleans per query, in ranked order."""
    recalls, rrs = [], []
    for rel in ranked_relevance:
        top = list(rel[:k])
        recalls.append(1.0 if any(top) else 0.0)
        rrs.append(1.0 / (top.index(True) + 1) if any(top) else 0.0)
    return float(np.mean(recalls)), float(np.mean(rrs))
