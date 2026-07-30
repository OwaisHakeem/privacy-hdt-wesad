"""
Load figure data directly from the experiment result JSON files, so figures
trace to source data rather than to hand-copied numbers.

Directory layout expected (relative to the repo root):
    results/loso/dp_fedavg/multiclass/eps{E}/summary.json      user-level utility
    results/loso/dp_sample/multiclass/eps{E}/summary.json      sample-level utility
    results/loso/mia/multiclass/eps{E}/summary.json            user-level MIA
    results/loso/mia_sample/multiclass/eps{E}/summary.json     sample-level MIA
    results/loso/dp_fedavg/binary/eps{E}/summary.json          binary utility

Each utility summary.json has:
    { "aggregate": { "macro_f1": {"mean":..,"std":..}, ... },
      "per_class_f1": { "amusement": {"mean":..,"std":..}, ... },
      "epsilon_target": E, "noise_multiplier_mean": .. }

Each MIA summary.json has (per the Stage D runner):
    { "window_level": { "cross_entropy": {"auc_mean":..,"auc_std":..}, ... },
      "utility_test_subject_macro_f1": {"mean":..} }

If a file is missing, the loader raises FileNotFoundError with the path, so a
missing result is loud rather than silently wrong. _style.py catches this and
falls back to its embedded values, printing a warning, so figures still render.
"""

from __future__ import annotations
import json
import os

# repo root = one level above paper_figures/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def _read(path: str) -> dict:
    full = os.path.join(_ROOT, path)
    if not os.path.exists(full):
        raise FileNotFoundError(full)
    with open(full) as fh:
        return json.load(fh)


def _eps_tag(e: float) -> str:
    return f"eps{e:g}"


def utility(mechanism: str, task: str = "multiclass") -> tuple[dict, dict]:
    """Return (macro_f1_mean, macro_f1_sd) dicts keyed by epsilon."""
    sub = {"user": "dp_fedavg", "sample": "dp_sample"}[mechanism]
    mean, sd = {}, {}
    for e in EPS:
        s = _read(f"results/loso/{sub}/{task}/{_eps_tag(e)}/summary.json")
        mean[e] = s["aggregate"]["macro_f1"]["mean"]
        sd[e] = s["aggregate"]["macro_f1"]["std"]
    return mean, sd


def per_class(mechanism: str, cls: str, task: str = "multiclass") -> dict:
    """Return {epsilon: F1 mean} for one class."""
    sub = {"user": "dp_fedavg", "sample": "dp_sample"}[mechanism]
    out = {}
    for e in EPS:
        s = _read(f"results/loso/{sub}/{task}/{_eps_tag(e)}/summary.json")
        out[e] = s["per_class_f1"][cls]["mean"]
    return out


def attack_auc(mechanism: str, task: str = "multiclass") -> tuple[dict, dict]:
    """Return (auc_mean, auc_sd) dicts keyed by epsilon, plus the 'inf' anchor."""
    sub = {"user": "mia", "sample": "mia_sample"}[mechanism]
    mean, sd = {}, {}
    for tag, key in [("epsinf", "inf")] + [(_eps_tag(e), e) for e in EPS]:
        s = _read(f"results/loso/{sub}/{task}/{tag}/summary.json")
        ce = s["window_level"]["cross_entropy"]
        mean[key] = ce["auc_mean"]
        sd[key] = ce["auc_std"]
    return mean, sd


def binary_utility() -> dict:
    """Return {epsilon: macro_f1 mean} for the binary task, user-level DP."""
    out = {}
    for e in EPS:
        s = _read(f"results/loso/dp_fedavg/binary/{_eps_tag(e)}/summary.json")
        out[e] = s["aggregate"]["macro_f1"]["mean"]
    return out
