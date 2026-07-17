#!/usr/bin/env python3
"""
=====================================================================
Stage A3 - Local-only baseline
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS ESTABLISHES
---------------------
The no-collaboration floor. Each subject trains a model on their own data
alone; nothing is shared, so privacy is perfect by construction. Federation
is only worth its cost - and its privacy risk - if it beats this.

Together with stage 03 this brackets the study:

    local-only   : no sharing, perfect privacy, no collective knowledge
    FedAvg       : sharing model updates
    DP-FedAvg    : sharing noised model updates
    centralised  : sharing raw data, no privacy

WHY THERE IS NO LOSO VARIANT
----------------------------
Under LOSO the test subject contributes no training data, so no local model
for that subject can exist. Local-only is therefore definable only under the
within-subject protocol, and its numbers are comparable only with the
within-subject centralised run from stage 03 - never with the LOSO table.
Placing them together would compare across different test sets.

A NOTE ON THE EXPECTED RESULT
-----------------------------
The earlier pipeline reported local-only accuracy of ~0.995. That figure was
almost certainly an artefact of a random within-subject split: consecutive
10 s windows are strongly autocorrelated, so random assignment placed
near-duplicate windows on both sides of the boundary and the model was
largely recognising neighbours it had already seen. Stage A2 splits
chronologically with a discard buffer, so a substantially lower figure here
is the correct outcome, not a regression.

USAGE
-----
    python scripts/04_local_only_baseline.py
    python scripts/04_local_only_baseline.py --task binary --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import metrics as M
from src import models as Mod
from src import training as T
from src import utils as U


def run_subject(df, feats, target, names, split, cfg, seed, device, logger) -> dict | None:
    U.set_seed(seed)

    X_tr, y_tr = D.make_xy(df, split["indices"]["train"], feats, target)
    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)

    n_classes = len(names)

    # A single subject's partition may lack a class entirely - amusement is
    # only ~36 windows per subject before splitting. Report the shortfall
    # rather than letting a silently degenerate model into the average.
    present = len(np.unique(y_tr))
    if present < n_classes:
        missing = [names[c] for c in range(n_classes) if c not in set(y_tr.tolist())]
        logger.info(f"    WARNING: training partition lacks class(es): {missing}")

    cw = (
        D.class_weights(y_tr, n_classes)
        if cfg["training"]["class_weighted_loss"]
        else None
    )

    bs = int(cfg["training"]["batch_size"])
    train_loader = T.make_loader(X_tr, y_tr, bs, shuffle=True, seed=seed)
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False)
    test_loader = T.make_loader(X_te, y_te, bs, shuffle=False)

    model = Mod.build_model(cfg, input_dim=X_tr.shape[1], num_classes=n_classes).to(device)

    fit = T.train(
        model,
        train_loader,
        val_loader,
        cfg=cfg,
        class_weight=cw,
        device=device,
        score_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        logger=logger,
    )

    y_true, y_pred, y_proba = T.predict(model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    return {
        "seed": seed,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "classes_in_train": int(present),
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "best_val_macro_f1": fit["best_val_score"],
        "test": test_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 - local-only baseline")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--task", choices=["multiclass", "binary"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    cfg = U.load_configs(args.splits_config, args.model_config)
    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])

    logger = U.setup_logging(f"04_local_only_{task}", cfg["output"]["logs_dir"])
    logger.info(f"Stage A3 | local-only | within_subject | task={task} | device={device}")
    logger.info("LOSO is not applicable: a held-out subject has no local model.")

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)

    if cfg["normalisation"]["strategy"] != "per_subject":
        sys.exit("local-only requires per_subject normalisation by definition.")
    df = D.apply_normalisation(df, feats, cfg)

    subjects = args.subjects or D.list_subjects(cfg)
    out_dir = Path(cfg["output"]["results_dir"]) / "within_subject" / "local_only" / task
    runs = []

    for subj in subjects:
        split = D.load_within_subject(cfg, subj)
        for seed in seeds:
            logger.info(f"  {subj} | seed {seed}")
            r = run_subject(df, feats, target, names, split, cfg, seed, device, logger)
            if r is None:
                continue
            r["subject"] = subj
            r["protocol"] = "within_subject"
            r["task"] = task
            r["provenance"] = U.provenance("A3_local_only")
            U.write_json(out_dir / f"{subj}_seed{seed}.json", r)
            runs.append(r)
            logger.info(
                f"    -> test macro_f1 {r['test']['macro_f1']:.4f} "
                f"| acc {r['test']['accuracy']:.4f} (n_test={r['n_test']})"
            )

    flat_test = [r["test"] for r in runs]
    summary = {
        "provenance": U.provenance("A3_local_only_summary"),
        "protocol": "within_subject",
        "task": task,
        "n_runs": len(runs),
        "n_subjects": len(subjects),
        "seeds": seeds,
        "aggregate": {
            k: M.aggregate(flat_test, k)
            for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
        },
        "per_subject_macro_f1": {
            s: M.aggregate(
                [r["test"] for r in runs if r["subject"] == s], "macro_f1"
            )
            for s in subjects
        },
        "note": (
            "Local-only is defined only under the within-subject protocol. "
            "Compare with the within-subject centralised run from stage 03, "
            "never with the LOSO table."
        ),
    }
    U.write_json(out_dir / "summary.json", summary)

    logger.info("")
    logger.info(f"LOCAL-ONLY | within_subject | {task} | {len(runs)} runs")
    for k, v in summary["aggregate"].items():
        if v["mean"] is not None:
            logger.info(f"  {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")
    logger.info(f"  written to {out_dir}")


if __name__ == "__main__":
    main()
