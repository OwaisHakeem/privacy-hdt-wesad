#!/usr/bin/env python3
"""
=====================================================================
Stage A3 - Centralised baseline
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS ESTABLISHES
---------------------
The utility ceiling. This condition pools every training subject's data at
one location and trains a single model - the scenario in which no privacy is
preserved at all. Every federated and differentially private result later in
the study is expressed as a cost relative to this number.

Without it, a DP-FedAvg macro-F1 of, say, 0.72 is uninterpretable: it might
represent a heavy privacy penalty or a task that is simply hard. The
centralised baseline is what makes the privacy-utility trade-off a
measurement rather than an assertion.

PROTOCOLS
---------
loso            : train on 12 subjects, early-stop on 2 held-out subjects,
                  test on 1 unseen subject. 15 folds. PRIMARY.
within_subject  : pool every subject's chronological training portion, test
                  on their later portion. SECONDARY - and the only protocol
                  under which the local-only baseline (stage 04) is
                  definable, so it is the protocol in which the two are
                  comparable.

The two protocols answer different questions and their numbers must never be
placed in the same table.

USAGE
-----
    python scripts/03_centralised_baseline.py
    python scripts/03_centralised_baseline.py --protocol within_subject
    python scripts/03_centralised_baseline.py --task binary --seeds 1
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


def run_one(df, feats, target, names, split, cfg, seed, device, logger) -> dict:
    """One (split, seed) run: train, early-stop on val, evaluate test once."""
    U.set_seed(seed)

    X_tr, y_tr = D.make_xy(df, split["indices"]["train"], feats, target)
    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)

    n_classes = len(names)
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

    # The test set is touched exactly here, once, after training has ended.
    y_true, y_pred, y_proba = T.predict(model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    # Reported for context, never for selection.
    yv, pv, _ = T.predict(model, val_loader, device)
    val_metrics = M.compute_metrics(yv, pv, None, names)

    # Per-subject evaluation of the SAME pooled model.
    #
    # Why this exists: the local-only baseline (stage 04) trains one model per
    # subject and computes macro-F1 on that subject's own test windows, then
    # averages across subjects. Pooling every subject's windows into a single
    # macro-F1, as test_metrics above does, weights each WINDOW equally;
    # averaging per-subject scores weights each PERSON equally. Those are
    # different quantities, and comparing one against the other would not be a
    # comparison. This block reproduces the local-only aggregation exactly, so
    # that centralised and local-only differ only in the thing under study -
    # whether other people's data was used - and not in how the number was
    # computed.
    test_per_subject = None
    if split.get("test_by_subject"):
        test_per_subject = {}
        for subj, idx in split["test_by_subject"].items():
            Xs, ys = D.make_xy(df, idx, feats, target)
            loader_s = T.make_loader(Xs, ys, bs, shuffle=False)
            yt, yp, yq = T.predict(model, loader_s, device)
            test_per_subject[subj] = M.compute_metrics(yt, yp, yq, names)

    return {
        "seed": seed,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "class_weights": cw.tolist() if cw is not None else None,
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "best_val_macro_f1": fit["best_val_score"],
        "val": val_metrics,
        "test": test_metrics,
        "test_per_subject": test_per_subject,
        "model": {
            "n_parameters": Mod.count_parameters(model),
            "state_bytes": Mod.state_size_bytes(model),
        },
    }


def build_loso_split(fold: dict) -> dict:
    # No test_by_subject: under LOSO the test set IS one subject, so pooled and
    # per-subject aggregation coincide.
    return {"indices": fold["indices"], "label": f"fold_{fold['fold_id']:02d}"}


def build_pooled_within_subject(cfg) -> dict:
    """Pool every subject's chronological partitions into one centralised split.

    test_by_subject preserves which test windows belong to whom, so the pooled
    model can additionally be scored per person - the aggregation the
    local-only baseline uses, and therefore the only one comparable with it.
    """
    parts = {"train": [], "val": [], "test": []}
    by_subject = {}
    for subj in D.list_subjects(cfg):
        w = D.load_within_subject(cfg, subj)
        for p in parts:
            parts[p].extend(w["indices"][p])
        by_subject[subj] = w["indices"]["test"]
    return {
        "indices": parts,
        "label": "pooled_within_subject",
        "test_by_subject": by_subject,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 - centralised baseline")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--protocol", choices=["loso", "within_subject"], default="loso")
    ap.add_argument("--task", choices=["multiclass", "binary"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = U.load_configs(args.splits_config, args.model_config)
    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])

    logger = U.setup_logging(f"03_centralised_{args.protocol}_{task}", cfg["output"]["logs_dir"])
    logger.info(f"Stage A3 | centralised | {args.protocol} | task={task} | device={device}")

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)

    # Normalisation is applied ONCE, per the policy declared in splits.yaml.
    # per_subject uses no cross-subject statistics, so it is independent of
    # the split and may be applied before partitioning without leakage.
    strategy = cfg["normalisation"]["strategy"]
    if strategy != "per_subject":
        sys.exit(
            f"normalisation.strategy is '{strategy}'. Only 'per_subject' is "
            "split-independent. 'global_train' must be fitted inside each fold "
            "on that fold's training indices - not implemented for this stage "
            "because it violates the federated threat model and exists only as "
            "a centralised ablation."
        )
    df = D.apply_normalisation(df, feats, cfg)
    logger.info(
        f"{len(df)} windows | {len(feats)} features | {len(names)} classes | "
        f"normalisation={strategy}/{cfg['normalisation'].get('per_subject_calibration','all')}"
    )

    if args.protocol == "loso":
        fold_ids = args.folds if args.folds is not None else D.list_folds(cfg)
        splits = [build_loso_split(D.load_fold(cfg, f)) for f in fold_ids]
    else:
        splits = [build_pooled_within_subject(cfg)]

    out_dir = Path(cfg["output"]["results_dir"]) / args.protocol / "centralised" / task
    runs = []

    for split in splits:
        for seed in seeds:
            logger.info(f"  {split['label']} | seed {seed}")
            r = run_one(df, feats, target, names, split, cfg, seed, device, logger)
            r["split"] = split["label"]
            r["protocol"] = args.protocol
            r["task"] = task
            r["provenance"] = U.provenance("A3_centralised")
            U.write_json(out_dir / f"{split['label']}_seed{seed}.json", r)
            runs.append(r)
            logger.info(
                f"    -> test macro_f1 {r['test']['macro_f1']:.4f} "
                f"| acc {r['test']['accuracy']:.4f} "
                f"| val macro_f1 {r['best_val_macro_f1']:.4f} "
                f"(epoch {r['best_epoch']})"
            )

    flat_test = [r["test"] for r in runs]
    summary = {
        "provenance": U.provenance("A3_centralised_summary"),
        "protocol": args.protocol,
        "task": task,
        "n_runs": len(runs),
        "seeds": seeds,
        "normalisation": cfg["normalisation"],
        "aggregation": (
            "pooled: every test window weighted equally (one score per run)"
        ),
        "aggregate": {
            k: M.aggregate(flat_test, k)
            for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
        },
        "per_class_f1": {
            c: M.aggregate([{"f1": t["per_class"][c]["f1"]} for t in flat_test], "f1")
            for c in names
        },
    }

    # Subject-weighted view: one score per (subject, seed), averaged the way
    # stage 04 averages. This - not the pooled figure above - is what may be
    # placed beside the local-only baseline.
    subject_scores = [
        m
        for r in runs
        if r.get("test_per_subject")
        for m in r["test_per_subject"].values()
    ]
    if subject_scores:
        summary["subject_weighted"] = {
            "aggregation": (
                "one score per (subject, seed), each PERSON weighted equally - "
                "matches stage 04 (local-only) exactly, and is the only view "
                "comparable with it"
            ),
            "n_scores": len(subject_scores),
            "aggregate": {
                k: M.aggregate(subject_scores, k)
                for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
            },
            "per_class_f1": {
                c: M.aggregate(
                    [{"f1": t["per_class"][c]["f1"]} for t in subject_scores], "f1"
                )
                for c in names
            },
            "per_subject_macro_f1": {
                s_: M.aggregate(
                    [
                        r["test_per_subject"][s_]
                        for r in runs
                        if r.get("test_per_subject")
                    ],
                    "macro_f1",
                )
                for s_ in (runs[0]["test_per_subject"] or {})
            },
        }

    U.write_json(out_dir / "summary.json", summary)

    logger.info("")
    logger.info(f"CENTRALISED | {args.protocol} | {task} | {len(runs)} runs")
    for k, v in summary["aggregate"].items():
        if v["mean"] is not None:
            logger.info(f"  {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")
    logger.info("  per-class F1:")
    for c, v in summary["per_class_f1"].items():
        if v["mean"] is not None:
            logger.info(f"    {c:14s} {v['mean']:.4f} +/- {v['std']:.4f}")

    if "subject_weighted" in summary:
        sw = summary["subject_weighted"]
        logger.info("")
        logger.info(
            f"  SUBJECT-WEIGHTED (comparable with local-only) | "
            f"{sw['n_scores']} scores = {len(subject_scores)//len(seeds)} subjects x {len(seeds)} seeds"
        )
        for k, v in sw["aggregate"].items():
            if v["mean"] is not None:
                logger.info(f"    {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")
        logger.info("    per-class F1:")
        for c, v in sw["per_class_f1"].items():
            if v["mean"] is not None:
                logger.info(f"      {c:14s} {v['mean']:.4f} +/- {v['std']:.4f}")

    logger.info(f"  written to {out_dir}")


if __name__ == "__main__":
    main()
