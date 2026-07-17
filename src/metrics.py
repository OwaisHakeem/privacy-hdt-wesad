"""
Evaluation metrics.

Reported once, identically, for every condition in the study. The metric set
is deliberately fixed here rather than chosen per script: allowing each stage
to pick its own headline metric is how a privacy-utility comparison stops
being a comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: list[str],
) -> dict:
    """Full metric set for one evaluation.

    macro_f1 is the headline. Accuracy is retained but must not lead: with
    baseline at ~39% and amusement at ~12% of the cohort, a model that never
    predicts amusement still scores respectably on accuracy while failing at
    the task. Per-class F1 is reported alongside the macro average because
    the macro number alone hides *which* class collapsed — and under DP noise
    it is reliably the smallest class that goes first.
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(len(y_true)),
    }

    if y_proba is not None:
        try:
            if n_classes == 2:
                out["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # Only defined when every class is present in y_true; a fold
                # missing a class must yield null rather than a silent error.
                if len(np.unique(y_true)) == n_classes:
                    out["auc_ovr_macro"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    )
                else:
                    out["auc_ovr_macro"] = None
        except ValueError:
            out["auc"] = None

    return out


def aggregate(runs: list[dict], key: str) -> dict:
    """Mean and standard deviation of one metric across runs.

    Single-run numbers are not reportable. Every headline figure in the paper
    must carry a spread across seeds and folds, or a reviewer cannot tell a
    real effect from a lucky initialisation.
    """
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }
