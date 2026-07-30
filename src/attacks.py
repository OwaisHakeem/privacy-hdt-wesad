"""
Membership inference: scoring, aggregation to the user level, and ROC metrics.

WHAT THE ATTACK ASKS
--------------------
Was this example (or this person) part of the training set? An adversary who
can answer reliably has defeated the privacy guarantee in practice, whatever
the formal epsilon promises. The purpose of this stage is to place empirical
attack success beside the formal budget, so that the two can be compared
rather than assumed equal.

WHY A THRESHOLD ATTACK, NOT SHADOW MODELS
-----------------------------------------
The canonical strong attack (Shokri et al., 2017) trains many shadow models to
learn the member/non-member boundary. With fifteen subjects under LOSO there
is no way to train independent shadow models without reusing subjects across
them, which contaminates the very membership signal being measured. The
loss/confidence threshold attack (Yeom et al., 2018) needs no shadow models
and is the honest ceiling of what this data supports. A weak attack that
nonetheless succeeds is alarming; a weak attack that fails is precisely the
evidence that a formal budget overstates real risk. Either outcome is
informative.

WHY THESE METRICS
-----------------
Average-case accuracy is the discredited MIA metric: an attack that is right
on the easy majority but useless on anyone at genuine risk still scores well.
The field reports the ROC in full and, following Carlini et al. (2022), the
true-positive rate at low false-positive rate — the regime in which an
adversary confidently identifies a few members. Both are computed here; AUC
leads, TPR@1%FPR is the rigorous companion.

GRANULARITY
-----------
Window-level scores every window. It attacks a guarantee the user-level
mechanism did not formally make, but it is statistically stable (thousands of
points) and, if it fails, shows that user-level noise incidentally defeats the
easier attack too. User-level aggregates each subject's window scores into one
per-person score, matching the guarantee actually made; it is the principled
target but rests on very few points per fold and must be pooled across folds.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Per-example membership signals
# ---------------------------------------------------------------------


def cross_entropy_per_example(proba: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """-log p(true class). Members typically incur lower loss than non-members."""
    p = np.clip(proba[np.arange(len(y)), y], eps, 1.0)
    return -np.log(p)


def confidence_per_example(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p(true class). Higher for members. A monotone transform of the loss,
    retained because thresholding either gives an identical ROC and confidence
    is the more intuitive quantity to report."""
    return proba[np.arange(len(y)), y]


def negative_entropy_per_example(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Confidence signal that does not require the label: members tend to yield
    lower-entropy (more peaked) predictions. Useful as a label-free attacker."""
    p = np.clip(proba, eps, 1.0)
    return np.sum(p * np.log(p), axis=1)  # = -entropy


def calibrated_cross_entropy(
    proba: np.ndarray, y: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Cross-entropy calibrated by per-class expected difficulty.

    A raw threshold attack conflates two things: how well the model fits this
    example (membership signal) and how intrinsically easy this example is
    (nuisance). If amusement windows all have high loss regardless of
    membership, thresholding raw loss will call them non-members. Subtracting
    the per-class mean loss removes that bias.

    This is the Carlini et al. (2022) calibration principle without shadow
    models: instead of estimating per-example difficulty from a reference
    model population, we approximate it from the per-class mean on the same
    evaluation set. Weaker than true LiRA, but the strongest calibration
    that N = 15 supports without contamination.

    Returns the loss with the per-class mean subtracted. Lower is still more
    "member-like".
    """
    raw = cross_entropy_per_example(proba, y, eps)
    out = raw.copy()
    for c in np.unique(y):
        mask = y == c
        if mask.sum() > 0:
            out[mask] = raw[mask] - float(raw[mask].mean())
    return out


def within_subject_memorisation(
    train_ce: dict, attack_ce: dict
) -> dict:
    """Paired train-vs-attack comparison per member subject.

    For each member the model trained on their train slice and did not train
    on their attack slice. If the model memorised, the average loss on train
    windows is lower than on attack windows FOR THE SAME PERSON. Because the
    comparison is within-subject, generalisation-gap and physiology
    differences cancel out; only memorisation of specific windows can produce
    the gap. This is the correct user-level memorisation test for LOSO.

    Inputs are {subject_id: 1-D array of losses}. Returns per-subject means,
    the paired difference (train_mean - attack_mean), and a signed-rank test.
    """
    from scipy.stats import wilcoxon

    subjects = sorted(train_ce.keys() & attack_ce.keys())
    train_means = np.array([train_ce[s].mean() for s in subjects])
    attack_means = np.array([attack_ce[s].mean() for s in subjects])
    diffs = train_means - attack_means      # negative = memorised (lower on train)

    result = {
        "subjects": subjects,
        "train_mean_ce": train_means.tolist(),
        "attack_mean_ce": attack_means.tolist(),
        "paired_diff_mean": float(diffs.mean()),
        "paired_diff_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "n_memorised": int((diffs < 0).sum()),  # train loss lower than attack
        "n_subjects": len(subjects),
    }
    if len(diffs) >= 3 and np.any(diffs != 0):
        stat, p = wilcoxon(diffs, alternative="less")   # H1: memorisation
        result["wilcoxon_stat"] = float(stat)
        result["wilcoxon_p_one_sided"] = float(p)
    return result


# ---------------------------------------------------------------------
# ROC / AUC / TPR@FPR  (implemented directly to avoid a sklearn dependency
# in the hot path, and verified against sklearn in the self-test)
# ---------------------------------------------------------------------


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC for a membership score. Higher score => predicted member (label 1).

    Returns (fpr, tpr, thresholds), sorted by increasing fpr.
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]

    P = float(np.sum(labels == 1))
    N = float(np.sum(labels == 0))
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    # Keep one point per distinct threshold.
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, tps[distinct] / P]
    fpr = np.r_[0.0, fps[distinct] / N]
    thr = np.r_[np.inf, s[distinct]]
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Interpolated TPR at a fixed FPR — the low-FPR regime Carlini et al.
    argue is the one that matters for a privacy breach."""
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target_fpr, fpr, tpr))


def attack_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Full attack evaluation for one score vector.

    AUC is symmetric about 0.5: an AUC of 0.5 is chance, and a value below 0.5
    indicates the score is anti-correlated with membership (still a leak, in the
    opposite direction), so |AUC - 0.5| is the effect size.
    """
    fpr, tpr, _ = roc_curve(scores, labels)
    a = auc(fpr, tpr)
    return {
        "auc": a,
        "auc_advantage": abs(a - 0.5) * 2.0,   # 0 = chance, 1 = perfect
        "tpr_at_0.1pct_fpr": tpr_at_fpr(fpr, tpr, 0.001),
        "tpr_at_1pct_fpr": tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(fpr, tpr, 0.05),
        "n_members": int(np.sum(labels == 1)),
        "n_non_members": int(np.sum(labels == 0)),
    }


# ---------------------------------------------------------------------
# User-level aggregation
# ---------------------------------------------------------------------


def aggregate_to_user(
    window_scores: np.ndarray,
    window_subject: np.ndarray,
    subject_is_member: dict,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list]:
    """Collapse per-window scores into one score per subject.

    A user-level attack asks whether a PERSON was in training, matching the
    user-level guarantee. Each subject's window scores are combined into a
    single membership score; "mean" is the natural default, "median" is a
    robust alternative reported as a sensitivity check.
    """
    subjects = list(dict.fromkeys(window_subject.tolist()))
    scores, labels, ids = [], [], []
    for s in subjects:
        mask = window_subject == s
        vals = window_scores[mask]
        agg = np.median(vals) if method == "median" else np.mean(vals)
        scores.append(float(agg))
        labels.append(int(subject_is_member[s]))
        ids.append(s)
    return np.asarray(scores), np.asarray(labels), ids
