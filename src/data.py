"""
Canonical data access for every stage.

This module is the single source of truth for four decisions that were
previously re-implemented per script, and drifted:

  1. which columns are features            (whitelist, fails closed)
  2. which windows belong to which split   (read from A2 manifests only)
  3. how features are standardised         (policy declared in splits.yaml)
  4. how class weights are derived         (train partition only)

No stage may make these decisions locally. Stage 02 imports from here too,
so the feature-selection rule exists exactly once in the codebase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def load_windows(cfg: dict) -> pd.DataFrame:
    """Load the stage-01 windowed feature table."""
    path = Path(cfg["input"]["windows_path"])
    if not path.exists():
        sys.exit(
            f"Input not found: {path}\n"
            "Run stage 01 first, or correct input.windows_path in the config."
        )

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".npz":
        blob = np.load(path, allow_pickle=True)
        X = blob["X"]
        df = pd.DataFrame(X, columns=[f"f{i:03d}" for i in range(X.shape[1])])
        df[cfg["input"]["subject_col"]] = blob["subject"]
        df[cfg["input"]["label_col"]] = blob["y"]
        return df
    sys.exit(f"Unsupported input format: {suffix}")


def validate_schema(df: pd.DataFrame, cfg: dict) -> None:
    icfg = cfg["input"]
    required = [icfg["subject_col"], icfg["label_col"], icfg["order_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"Missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)[:12]} ...\n"
            "Adjust the input.*_col entries in the config to match stage 01."
        )


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Select features by PREFIX WHITELIST, never by exclusion.

    Stage 01 emits numeric columns that are not features: window_id,
    start_sample_chest, end_sample_chest, start_sec, end_sec, window_seconds,
    original_wesad_label, and one *_label column per task. Selecting features
    as 'numeric and not on a blacklist' admits all of them the moment stage 01
    gains a column — which is precisely how the label reached the input matrix
    in the earlier centralised baseline. A whitelist fails closed: an
    unrecognised column is ignored rather than silently trained on.
    """
    icfg = cfg["input"]
    prefixes = tuple(icfg.get("feature_prefixes") or [])
    drop = set(icfg.get("drop_cols") or [])

    if not prefixes:
        sys.exit(
            "input.feature_prefixes is empty. Refusing to guess the feature set "
            "by exclusion — declare the prefixes explicitly (e.g. chest_, wrist_)."
        )

    cols = [
        c
        for c in df.columns
        if c.startswith(prefixes)
        and c not in drop
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    suspect = [c for c in cols if c.endswith(("_label", "_label_name", "_included"))]
    if suspect:
        sys.exit(f"Label-shaped columns matched the feature whitelist: {suspect}")

    if not cols:
        sys.exit(
            f"No columns matched prefixes {prefixes}. "
            f"Columns present: {list(df.columns)[:15]} ..."
        )
    return cols


def prepare_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Retain protocol labels 1-4 and attach both targets.

    Both tasks derive from one filtered table, so the binary result is a
    genuine ablation of the multiclass result rather than a parallel pipeline
    with its own (divergent) protocol.
    """
    label_col = cfg["input"]["label_col"]
    df = df[df[label_col].isin(cfg["task"]["keep_raw_labels"])].copy()

    mc_map = {int(k): int(v) for k, v in cfg["task"]["multiclass"]["label_map"].items()}
    df["y_multiclass"] = df[label_col].map(mc_map).astype(int)

    pos = set(cfg["task"]["binary"]["positive_labels"])
    df["y_binary"] = df[label_col].isin(pos).astype(int)
    return df.reset_index(drop=True)


def target_column(cfg: dict, task: str) -> str:
    if task not in {"multiclass", "binary"}:
        sys.exit(f"Unknown task: {task}")
    return "y_multiclass" if task == "multiclass" else "y_binary"


def class_names(cfg: dict, task: str) -> list[str]:
    return list(cfg["task"][task]["class_names"])


# ---------------------------------------------------------------------
# Split manifests (produced by stage A2 — never recomputed here)
# ---------------------------------------------------------------------


def load_fold(cfg: dict, fold_id: int) -> dict:
    path = Path(cfg["output"]["splits_dir"]) / "loso" / f"fold_{fold_id:02d}.json"
    if not path.exists():
        sys.exit(f"Fold manifest not found: {path}. Run stage 02 first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_within_subject(cfg: dict, subject: str) -> dict:
    path = Path(cfg["output"]["splits_dir"]) / "within_subject" / f"subject_{subject}.json"
    if not path.exists():
        sys.exit(f"Within-subject manifest not found: {path}. Run stage 02 first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_folds(cfg: dict) -> list[int]:
    d = Path(cfg["output"]["splits_dir"]) / "loso"
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("fold_*.json"))


def list_subjects(cfg: dict) -> list[str]:
    d = Path(cfg["output"]["splits_dir"]) / "within_subject"
    return sorted(p.stem.replace("subject_", "") for p in d.glob("subject_*.json"))


# ---------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------


def _standardise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # Guard constant features: a zero standard deviation would otherwise
    # produce inf/nan and silently poison the whole feature matrix.
    safe = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / safe


def normalise_per_subject(
    df: pd.DataFrame,
    feats: list[str],
    cfg: dict,
    calibration: str = "all",
) -> pd.DataFrame:
    """Standardise each subject using that subject's OWN statistics.

    Why this is the default:
      - It is federation-compatible. A global scaler would require pooling
        raw feature statistics at the server, which contradicts the threat
        model the paper is built on. Reporting a federated result computed
        with a centrally-fitted scaler is a common and quietly serious flaw.
      - It matches deployment. A new user's device holds its own recording
        and can standardise locally before inference.
      - It removes inter-subject offsets (resting heart rate, skin
        conductance level) that otherwise dominate the feature space and
        depress cross-subject generalisation.

    Honest caveat: for the LOSO test subject, statistics come from that
    subject's own held-out windows. No labels are used, so this is not
    label leakage, but it is mildly transductive and must be stated in the
    Methods. Setting calibration="baseline_only" restricts the statistics to
    the subject's baseline condition, which is the stricter variant and
    corresponds to a device calibrating on a resting period before use.
    """
    subj_col = cfg["input"]["subject_col"]
    label_col = cfg["input"]["label_col"]
    out = df.copy()

    for subj, sdf in df.groupby(subj_col):
        if calibration == "baseline_only":
            calib = sdf[sdf[label_col] == 1]
            if len(calib) < 2:
                calib = sdf  # fall back rather than divide by nothing
        else:
            calib = sdf

        mean = calib[feats].to_numpy().mean(axis=0)
        std = calib[feats].to_numpy().std(axis=0)
        out.loc[sdf.index, feats] = _standardise(sdf[feats].to_numpy(), mean, std)

    return out


def normalise_global_train(
    df: pd.DataFrame, feats: list[str], train_idx: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Fit one scaler on the pooled TRAINING partition only.

    Provided strictly as a centralised-baseline comparator. This violates the
    federated threat model — the server never sees raw client features — so it
    must not be used for any federated or DP condition.
    """
    out = df.copy()
    mean = df.loc[train_idx, feats].to_numpy().mean(axis=0)
    std = df.loc[train_idx, feats].to_numpy().std(axis=0)
    out[feats] = _standardise(df[feats].to_numpy(), mean, std)
    return out, {"mean": mean, "std": std}


def apply_normalisation(
    df: pd.DataFrame, feats: list[str], cfg: dict, train_idx: np.ndarray | None = None
) -> pd.DataFrame:
    """Dispatch on the policy declared in splits.yaml. No stage overrides it."""
    strategy = cfg["normalisation"]["strategy"]
    calibration = cfg["normalisation"].get("per_subject_calibration", "all")

    if strategy == "per_subject":
        return normalise_per_subject(df, feats, cfg, calibration=calibration)
    if strategy == "global_train":
        if train_idx is None:
            sys.exit("global_train normalisation requires the training indices.")
        out, _ = normalise_global_train(df, feats, train_idx)
        return out
    sys.exit(f"Unknown normalisation strategy: {strategy}")


# ---------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------


def make_xy(
    df: pd.DataFrame, indices: list[int] | np.ndarray, feats: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices)
    X = df.loc[idx, feats].to_numpy(dtype=np.float32)
    y = df.loc[idx, target].to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        sys.exit("Non-finite values in the feature matrix after normalisation.")
    return X, y


def class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights computed from the TRAINING partition only.

    WESAD's protocol yields an uneven cohort — amusement is roughly a third
    the size of baseline. Unweighted training lets a model score well on
    accuracy while effectively ignoring the smallest class, which is exactly
    the failure macro-F1 is meant to expose. Deriving the weights from any
    partition other than train would leak information about the evaluation
    distribution.
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = len(y_train) / (n_classes * counts)
    return w.astype(np.float32)
