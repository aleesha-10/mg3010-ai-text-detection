"""Shared helpers: metrics, bootstrap CIs, McNemar test."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import config


def tpr_at_fpr(y, s, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def metrics(y, pred, score):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"),
        "tpr_at_5fpr": tpr_at_fpr(y, score, 0.05) if len(set(y)) > 1 else float("nan"),
    }


def bootstrap_ci(y, pred, score, metric="f1", n=config.N_BOOTSTRAP, seed=config.SEED):
    """Percentile bootstrap CI (n resamples of the test set)."""
    rng = np.random.default_rng(seed)
    y, pred, score = map(np.asarray, (y, pred, score))
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        yi, pi, si = y[i], pred[i], score[i]
        vals.append(f1_score(yi, pi, zero_division=0) if metric == "f1" else roc_auc_score(yi, si))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b):
    """McNemar chi-square with continuity correction. Returns (stat, p, b, c)."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb))
    n10 = int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stat), float(1 - chi2.cdf(stat, 1)), n01, n10
