#!/usr/bin/env python3
"""
=====================================================================
Stage A1 — Split generation
Project : privacy_hdt_wesad
Author  : (Owais Hakeem, Ulster University)
=====================================================================

WHAT THIS DOES
--------------
Produces leakage-free partitions of the windowed WESAD feature table and
writes them as index manifests. It does not train, normalise or model
anything; it only decides *which window belongs to which partition*, once,
so that every later stage inherits an identical protocol.

WHY IT EXISTS
-------------
In the earlier pipeline the held-out test set was used both for early
stopping and for final reporting. Model selection therefore consumed the
test set, and reported utility was optimistic. Because this project's
central claim is an *honest* privacy-utility trade-off, an optimistic
utility estimate would undermine the contribution specifically, not just
generically. This stage introduces a genuine three-way partition in which
the test data are never consulted before the final evaluation.

TWO PROTOCOLS
-------------
1. LOSO (primary): 12 train / 2 validation / 1 test subject per fold.
   Validation subjects are held out *entirely*, so the validation
   condition mirrors the test condition (an unseen person). Early stopping
   is then tuned for the problem actually being reported.

2. Within-subject (secondary): chronological 60/20/20 inside each
   condition block, with a discard buffer at each boundary.

OUTPUTS
-------
data/processed/splits/loso/fold_XX.json
data/processed/splits/within_subject/subject_XX.json
data/processed/splits/manifest.json
data/processed/splits/split_summary.csv

USAGE
-----
    python scripts/02_create_splits.py --config configs/splits.yaml
    python scripts/02_create_splits.py --config configs/splits.yaml --inspect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The data contract lives in src/data.py and is imported, never re-implemented.
# Duplicating feature selection or label preparation across stages is what let
# the previous pipeline's two experiment trees drift into contradicting each
# other; the rule must exist exactly once.
from src.data import feature_columns, prepare_labels, validate_schema  # noqa: E402

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hash the input table so a manifest can be traced to exact inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_windows(cfg: dict) -> pd.DataFrame:
    """Load the stage-01 table. Kept local: this stage is the only one that
    may read the raw table before splits exist."""
    path = Path(cfg["input"]["windows_path"])
    if not path.exists():
        sys.exit(
            f"[A1] Input not found: {path}\n"
            "     Run stage 01 first, or correct input.windows_path in the config."
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    sys.exit(f"[A1] Unsupported input format: {suffix}")


# ---------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------


def rotation_val_subjects(subjects: list[int], test_idx: int, k: int) -> list[int]:
    """Deterministic validation assignment.

    For test subject at index i, validation subjects are (i+1)...(i+k) mod N.
    This is preferred over a random draw because it is reproducible without
    an RNG and balanced by construction: across the N folds each subject
    serves as validation exactly k times, so no individual subject
    disproportionately shapes the stopping rule.
    """
    n = len(subjects)
    return [subjects[(test_idx + j) % n] for j in range(1, k + 1)]


def class_distribution(y: np.ndarray, n_classes: int) -> dict:
    counts = np.bincount(y, minlength=n_classes).tolist()
    total = int(sum(counts))
    return {
        "counts": counts,
        "total": total,
        "proportions": [round(c / total, 4) if total else 0.0 for c in counts],
    }


def build_loso(df: pd.DataFrame, cfg: dict) -> list[dict]:
    subj_col = cfg["input"]["subject_col"]
    subjects = [s for s in cfg["cohort"]["subjects"] if s in set(df[subj_col].unique())]
    absent = sorted(set(cfg["cohort"]["subjects"]) - set(subjects))
    if absent:
        print(f"[A1] WARNING: cohort subjects absent from the table: {absent}")

    k = cfg["protocol_loso"]["n_val_subjects"]
    n_classes = len(cfg["task"]["multiclass"]["class_names"])
    folds = []

    for i, test_subj in enumerate(subjects):
        val_subjs = rotation_val_subjects(subjects, i, k)
        train_subjs = [s for s in subjects if s != test_subj and s not in val_subjs]

        idx = {
            "train": df.index[df[subj_col].isin(train_subjs)].to_numpy(),
            "val": df.index[df[subj_col].isin(val_subjs)].to_numpy(),
            "test": df.index[df[subj_col] == test_subj].to_numpy(),
        }

        fold = {
            "fold_id": i,
            "protocol": "loso",
            "test_subjects": [str(test_subj)],
            "val_subjects": [str(s) for s in val_subjs],
            "train_subjects": [str(s) for s in train_subjs],
            "indices": {k_: v.tolist() for k_, v in idx.items()},
            "n_windows": {k_: int(len(v)) for k_, v in idx.items()},
            "class_distribution": {
                part: class_distribution(
                    df.loc[ids, "y_multiclass"].to_numpy(), n_classes
                )
                for part, ids in idx.items()
            },
            # Client identity for the federated stages: one subject = one client.
            "client_map": {str(s): "train" for s in train_subjs},
        }

        missing = [
            c
            for c, n in enumerate(fold["class_distribution"]["test"]["counts"])
            if n == 0
        ]
        if missing:
            names = cfg["task"]["multiclass"]["class_names"]
            fold["warnings"] = [
                f"test subject {test_subj} has no windows for class(es): "
                f"{[names[c] for c in missing]}"
            ]
            print(f"[A1] WARNING: {fold['warnings'][0]}")

        folds.append(fold)

    return folds


def build_within_subject(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Chronological split inside each condition block.

    Splitting randomly would be a methodological error here: consecutive
    10 s windows are strongly autocorrelated (and may overlap if stage 01
    used a stride shorter than the window), so random assignment places
    near-duplicates on both sides of the boundary and inflates utility.
    Splitting chronologically *within each condition block* preserves the
    temporal ordering while keeping all four classes present in every
    partition.
    """
    subj_col = cfg["input"]["subject_col"]
    label_col = cfg["input"]["label_col"]
    order_col = cfg["input"]["order_col"]
    pcfg = cfg["protocol_within_subject"]
    fr = pcfg["fractions"]
    buf = int(pcfg["buffer_windows"])
    n_classes = len(cfg["task"]["multiclass"]["class_names"])

    out = []
    for subj in sorted(df[subj_col].unique()):
        sdf = df[df[subj_col] == subj]
        parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}

        for _, block in sdf.groupby(label_col):
            block = block.sort_values(order_col)
            ids = block.index.to_numpy()
            n = len(ids)
            n_tr = int(n * fr["train"])
            n_va = int(n * fr["val"])

            # Buffers are discarded, not reassigned, so that no window
            # adjacent to a boundary appears in any partition.
            parts["train"].extend(ids[: max(0, n_tr - buf)].tolist())
            parts["val"].extend(ids[n_tr + buf : max(n_tr + buf, n_tr + n_va - buf)].tolist())
            parts["test"].extend(ids[n_tr + n_va + buf :].tolist())

        out.append(
            {
                "subject": str(subj),
                "protocol": "within_subject",
                "indices": parts,
                "n_windows": {k: len(v) for k, v in parts.items()},
                "class_distribution": {
                    part: class_distribution(
                        df.loc[ids, "y_multiclass"].to_numpy(), n_classes
                    )
                    for part, ids in parts.items()
                    if len(ids) > 0
                },
                "buffer_windows": buf,
            }
        )
    return out


# ---------------------------------------------------------------------
# Verification — assertions that must hold before anything downstream runs
# ---------------------------------------------------------------------


def verify(folds: list[dict], within: list[dict], df: pd.DataFrame, cfg: dict) -> None:
    subj_col = cfg["input"]["subject_col"]
    problems: list[str] = []

    for f in folds:
        tr, va, te = (set(f["indices"][p]) for p in ("train", "val", "test"))

        if tr & va or tr & te or va & te:
            problems.append(f"fold {f['fold_id']}: window indices overlap between partitions")

        roles = [set(f["train_subjects"]), set(f["val_subjects"]), set(f["test_subjects"])]
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            problems.append(f"fold {f['fold_id']}: a subject occupies more than one role")

        # The decisive check for this stage: no test-subject window may reach
        # the model-selection partition, in either direction.
        test_subj = f["test_subjects"][0]
        sel_ids = list(tr | va)
        if test_subj in set(df.loc[sel_ids, subj_col].unique()):
            problems.append(
                f"fold {f['fold_id']}: test subject {test_subj} appears in train/val"
            )

        if min(f["n_windows"].values()) == 0:
            problems.append(f"fold {f['fold_id']}: an empty partition")

    for w in within:
        tr, va, te = (set(w["indices"][p]) for p in ("train", "val", "test"))
        if tr & va or tr & te or va & te:
            problems.append(f"subject {w['subject']}: within-subject partitions overlap")

    if problems:
        print("\n[A1] VERIFICATION FAILED")
        for p in problems:
            print(f"      - {p}")
        sys.exit(1)

    print("[A1] Verification passed: partitions disjoint, roles exclusive, "
          "no test subject reachable during model selection.")


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def summarise(folds: list[dict], cfg: dict) -> pd.DataFrame:
    names = cfg["task"]["multiclass"]["class_names"]
    rows = []
    for f in folds:
        row = {
            "fold_id": f["fold_id"],
            "test_subject": f["test_subjects"][0],
            "val_subjects": "|".join(str(s) for s in f["val_subjects"]),
            "n_train_subjects": len(f["train_subjects"]),
            "n_train": f["n_windows"]["train"],
            "n_val": f["n_windows"]["val"],
            "n_test": f["n_windows"]["test"],
        }
        for c, nm in enumerate(names):
            row[f"test_prop_{nm}"] = f["class_distribution"]["test"]["proportions"][c]
        rows.append(row)
    return pd.DataFrame(rows)


def inspect(df: pd.DataFrame, cfg: dict) -> None:
    subj_col, label_col = cfg["input"]["subject_col"], cfg["input"]["label_col"]
    feats = feature_columns(df, cfg)
    print("\n[A1] INPUT INSPECTION")
    print(f"      rows            : {len(df)}")
    print(f"      feature columns : {len(feats)}")
    print(f"      subjects        : {sorted(df[subj_col].unique().tolist())}")
    print(f"      raw label counts: {df[label_col].value_counts().sort_index().to_dict()}")
    nan_cols = [c for c in feats if df[c].isna().any()]
    print(f"      features w/ NaN : {len(nan_cols)}"
          + (f" -> {nan_cols[:8]} ..." if nan_cols else ""))
    print("      windows/subject :")
    print(df.groupby(subj_col).size().to_string())


# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 — leakage-free split generation")
    ap.add_argument("--config", type=Path, default=Path("configs/splits.yaml"))
    ap.add_argument("--inspect", action="store_true",
                    help="Report on the input table and exit without writing splits.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[A1] Config: {args.config}")

    df = load_windows(cfg)
    validate_schema(df, cfg)

    if args.inspect:
        inspect(df, cfg)
        return

    df = prepare_labels(df, cfg).reset_index(drop=True)
    feats = feature_columns(df, cfg)
    print(f"[A1] Feature columns detected: {len(feats)}")

    folds = build_loso(df, cfg) if cfg["protocol_loso"]["enabled"] else []
    within = (
        build_within_subject(df, cfg)
        if cfg["protocol_within_subject"]["enabled"]
        else []
    )

    verify(folds, within, df, cfg)

    out_dir = Path(cfg["output"]["splits_dir"])
    (out_dir / "loso").mkdir(parents=True, exist_ok=True)
    (out_dir / "within_subject").mkdir(parents=True, exist_ok=True)

    for f in folds:
        with open(out_dir / "loso" / f"fold_{f['fold_id']:02d}.json", "w") as fh:
            json.dump(f, fh, indent=2)
    for w in within:
        with open(out_dir / "within_subject" / f"subject_{w['subject']}.json", "w") as fh:
            json.dump(w, fh, indent=2)

    in_path = Path(cfg["input"]["windows_path"])
    manifest = {
        "stage": "A1_create_splits",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "input_file": str(in_path),
        "input_sha256": sha256_of(in_path) if cfg["reproducibility"]["hash_input"] else None,
        "n_windows_total": int(len(df)),
        "n_features": len(feats),
        "feature_columns": feats,
        "n_loso_folds": len(folds),
        "n_within_subject": len(within),
        "notes": [
            "Splits are independent of the training seed; seeds vary model "
            "initialisation, client sampling and DP noise only.",
            "Normalisation policy is declared in the config and applied "
            "downstream, fitted on training data only.",
        ],
    }
    with open(out_dir / cfg["output"]["manifest_name"], "w") as fh:
        json.dump(manifest, fh, indent=2)

    summary = summarise(folds, cfg)
    summary.to_csv(out_dir / cfg["output"]["summary_name"], index=False)

    print(f"\n[A1] Wrote {len(folds)} LOSO folds and {len(within)} within-subject splits "
          f"to {out_dir}")
    print("\n[A1] Fold summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
