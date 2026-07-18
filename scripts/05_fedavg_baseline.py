#!/usr/bin/env python3
"""
=====================================================================
Stage B - Federated averaging baseline (non-private)
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS ESTABLISHES
---------------------
The cost of federation alone, before any privacy mechanism is applied.
Stage C then measures the cost of privacy on top of it. Without this
separation, a DP-FedAvg number confounds two penalties and the
privacy-utility trade-off is unmeasurable.

THE BAR IT MUST CLEAR
---------------------
Stage A3 established the bracket:

    LOSO           centralised 0.7812   <- ceiling; local-only undefined
    within-subject centralised 0.8540
    within-subject local-only  0.9088   <- floor: private, free, no comms

Under LOSO federation is the only option for a new user, so the question is
how much of the 0.7812 ceiling survives.

Under within-subject the question is harder and the honest one: local-only
already achieves 0.9088 with perfect privacy and zero communication. If
FedAvg cannot beat that, federation has no case in this setting, and that is
a result to report rather than a problem to conceal.

WHAT THIS SCRIPT WILL NOT DO
----------------------------
The predecessor evaluated the global model on the test set every round and
used that number for early stopping, best-state selection and the reported
headline simultaneously. Here run_federated() is never passed the test data;
it is evaluated once, after training, by this script.

USAGE
-----
    python scripts/05_fedavg_baseline.py --self-test
    python scripts/05_fedavg_baseline.py
    python scripts/05_fedavg_baseline.py --protocol within_subject
    python scripts/05_fedavg_baseline.py --task binary --seeds 1 --folds 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import federated as F
from src import metrics as M
from src import models as Mod
from src import training as T
from src import utils as U


# ---------------------------------------------------------------------
# Self-test: aggregation correctness, verified on this machine
# ---------------------------------------------------------------------


def self_test() -> None:
    """Check fedavg_aggregate against a hand-computed expectation.

    Aggregation is the one piece of arithmetic in the study that no metric
    would reveal if it were wrong: a mis-weighted average still trains, still
    converges, and still reports a plausible number. It is verified explicitly.
    """
    import torch

    print("[self-test] fedavg_aggregate")

    # Three clients, known values, deliberately uneven sample counts.
    states = [
        {"w": torch.tensor([1.0, 10.0]), "b": torch.tensor([0.0])},
        {"w": torch.tensor([2.0, 20.0]), "b": torch.tensor([1.0])},
        {"w": torch.tensor([3.0, 30.0]), "b": torch.tensor([2.0])},
    ]
    counts = [100, 200, 700]           # total 1000
    out = F.fedavg_aggregate(states, counts)

    # Hand-computed: 1*0.1 + 2*0.2 + 3*0.7 = 2.6 ; 10*0.1+20*0.2+30*0.7 = 26.0
    expect_w = torch.tensor([2.6, 26.0])
    expect_b = torch.tensor([1.6])     # 0*0.1 + 1*0.2 + 2*0.7

    ok_w = torch.allclose(out["w"], expect_w, atol=1e-6)
    ok_b = torch.allclose(out["b"], expect_b, atol=1e-6)
    print(f"  weighted mean w: {out['w'].tolist()} expected {expect_w.tolist()} -> {'OK' if ok_w else 'FAIL'}")
    print(f"  weighted mean b: {out['b'].tolist()} expected {expect_b.tolist()} -> {'OK' if ok_b else 'FAIL'}")

    # Equal counts must reduce to the plain mean.
    out2 = F.fedavg_aggregate(states, [50, 50, 50])
    ok_eq = torch.allclose(out2["w"], torch.tensor([2.0, 20.0]), atol=1e-6)
    print(f"  equal counts -> plain mean: {out2['w'].tolist()} -> {'OK' if ok_eq else 'FAIL'}")

    # A single client must be returned unchanged.
    out3 = F.fedavg_aggregate([states[1]], [42])
    ok_one = torch.allclose(out3["w"], states[1]["w"], atol=1e-6)
    print(f"  single client -> identity: {'OK' if ok_one else 'FAIL'}")

    # Dominant client must dominate.
    out4 = F.fedavg_aggregate(states, [1, 1, 9998])
    ok_dom = bool(abs(out4["w"][0].item() - 3.0) < 0.01)
    print(f"  dominant client: w[0]={out4['w'][0].item():.4f} ~ 3.0 -> {'OK' if ok_dom else 'FAIL'}")

    if all([ok_w, ok_b, ok_eq, ok_one, ok_dom]):
        print("[self-test] PASSED")
    else:
        sys.exit("[self-test] FAILED - do not trust any federated result until fixed.")


# ---------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------


def build_clients(df, feats, target, n_classes, subject_indices, cfg, subj_col) -> list[F.Client]:
    """One client per subject, holding only that subject's own windows."""
    clients = []
    for subj, idx in subject_indices.items():
        X, y = D.make_xy(df, idx, feats, target)
        cw = (
            D.class_weights(y, n_classes)
            if cfg["federated"]["per_client_class_weights"]
            and cfg["training"]["class_weighted_loss"]
            else None
        )
        clients.append(
            F.Client(client_id=subj, X=X, y=y, n_classes=n_classes, class_weight=cw)
        )
    return clients


def loso_client_indices(df, fold, subj_col) -> dict:
    """Split a fold's training pool back into its constituent subjects."""
    out = {}
    train_idx = np.asarray(fold["indices"]["train"])
    sub = df.loc[train_idx, subj_col]
    for s in fold["train_subjects"]:
        out[s] = train_idx[(sub == s).to_numpy()].tolist()
    return out


# ---------------------------------------------------------------------


def run_one(df, feats, target, names, split, cfg, seed, device, subj_col, logger) -> dict:
    U.set_seed(seed)
    n_classes = len(names)

    clients = build_clients(
        df, feats, target, n_classes, split["client_indices"], cfg, subj_col
    )

    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)
    bs = int(cfg["training"]["batch_size"])
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False)
    test_loader = T.make_loader(X_te, y_te, bs, shuffle=False)

    model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=n_classes).to(device)

    fit = F.run_federated(
        clients,
        model,
        val_loader,
        cfg=cfg,
        device=device,
        score_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        seed=seed,
        logger=logger,
    )

    # Test touched exactly here, once, after training has ended.
    y_true, y_pred, y_proba = T.predict(model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    test_per_subject = None
    if split.get("test_by_subject"):
        test_per_subject = {}
        for s, idx in split["test_by_subject"].items():
            Xs, ys = D.make_xy(df, idx, feats, target)
            yt, yp, yq = T.predict(model, T.make_loader(Xs, ys, bs, shuffle=False), device)
            test_per_subject[s] = M.compute_metrics(yt, yp, yq, names)

    return {
        "seed": seed,
        "n_clients": len(clients),
        "client_sizes": {c.client_id: c.n_samples for c in clients},
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "best_round": fit["best_round"],
        "rounds_run": fit["rounds_run"],
        "best_val_macro_f1": fit["best_val_score"],
        "communication": {
            "state_bytes": fit["state_bytes"],
            "bytes_transferred": fit["bytes_transferred"],
            "bytes_per_round": fit["bytes_transferred"] // max(fit["rounds_run"], 1),
        },
        "test": test_metrics,
        "test_per_subject": test_per_subject,
        "history": fit["history"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B - FedAvg baseline")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--fedavg-config", default="configs/fedavg.yaml")
    ap.add_argument("--protocol", choices=["loso", "within_subject"], default="loso")
    ap.add_argument("--task", choices=["multiclass", "binary"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true", help="Verify aggregation and exit.")
    # Sensitivity overrides. A single untuned configuration cannot support a
    # claim about federation itself: if E=5 overfits a 170-window client, the
    # result characterises that hyper-parameter, not FedAvg. Sweeping E is the
    # minimum needed before any "federation is dominated" statement.
    ap.add_argument("--local-epochs", type=int, default=None,
                    help="Override federated.local_epochs (E).")
    ap.add_argument("--local-optimizer", choices=["adam", "sgd"], default=None,
                    help="Override federated.local_optimizer.")
    ap.add_argument("--tag", default=None,
                    help="Sub-directory for results; defaults to a tag describing the override.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    cfg = U.load_configs(args.splits_config, args.model_config, args.fedavg_config)

    # Overrides are applied to the config BEFORE anything reads it, and are
    # recorded in every result file, so a swept run can never be mistaken for
    # the default configuration later.
    overrides = {}
    if args.local_epochs is not None:
        cfg["federated"]["local_epochs"] = args.local_epochs
        overrides["local_epochs"] = args.local_epochs
    if args.local_optimizer is not None:
        cfg["federated"]["local_optimizer"] = args.local_optimizer
        overrides["local_optimizer"] = args.local_optimizer

    tag = args.tag
    if tag is None and overrides:
        tag = "_".join(f"{k[0].upper()}{v}" if k == "local_epochs" else str(v)
                       for k, v in overrides.items())

    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])
    subj_col = cfg["input"]["subject_col"]

    logger = U.setup_logging(f"05_fedavg_{args.protocol}_{task}", cfg["output"]["logs_dir"])
    logger.info(f"Stage B | FedAvg | {args.protocol} | task={task} | device={device}")
    logger.info(
        f"E={cfg['federated']['local_epochs']} local epochs | "
        f"C={cfg['federated']['client_fraction']} participation | "
        f"optimiser={cfg['federated']['local_optimizer']}"
        + (f" | OVERRIDES {overrides} -> results/{tag}" if overrides else "")
    )

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)

    if cfg["normalisation"]["strategy"] != "per_subject":
        sys.exit(
            "Federated conditions require per_subject normalisation: a scaler "
            "fitted across clients would require pooling raw statistics at the "
            "server and would contradict the threat model."
        )
    df = D.apply_normalisation(df, feats, cfg)

    splits = []
    if args.protocol == "loso":
        fold_ids = args.folds if args.folds is not None else D.list_folds(cfg)
        for fid in fold_ids:
            fold = D.load_fold(cfg, fid)
            splits.append(
                {
                    "label": f"fold_{fid:02d}",
                    "indices": fold["indices"],
                    "client_indices": loso_client_indices(df, fold, subj_col),
                }
            )
    else:
        parts = {"train": [], "val": [], "test": []}
        client_indices, by_subject = {}, {}
        for subj in D.list_subjects(cfg):
            w = D.load_within_subject(cfg, subj)
            for p in parts:
                parts[p].extend(w["indices"][p])
            client_indices[subj] = w["indices"]["train"]
            by_subject[subj] = w["indices"]["test"]
        splits.append(
            {
                "label": "pooled_within_subject",
                "indices": parts,
                "client_indices": client_indices,
                "test_by_subject": by_subject,
            }
        )

    out_dir = Path(cfg["output"]["results_dir"]) / args.protocol / "fedavg" / task
    if tag:
        out_dir = out_dir / tag
    runs = []

    for split in splits:
        for seed in seeds:
            logger.info(f"  {split['label']} | seed {seed} | {len(split['client_indices'])} clients")
            r = run_one(df, feats, target, names, split, cfg, seed, device, subj_col, logger)
            r["split"] = split["label"]
            r["protocol"] = args.protocol
            r["task"] = task
            r["local_epochs"] = cfg["federated"]["local_epochs"]
            r["local_optimizer"] = cfg["federated"]["local_optimizer"]
            r["overrides"] = overrides or None
            r["provenance"] = U.provenance("B_fedavg")
            U.write_json(out_dir / f"{split['label']}_seed{seed}.json", r)
            runs.append(r)
            logger.info(
                f"    -> test macro_f1 {r['test']['macro_f1']:.4f} "
                f"| acc {r['test']['accuracy']:.4f} "
                f"| {r['rounds_run']} rounds (best @ {r['best_round']}) "
                f"| {r['communication']['bytes_transferred']/1e6:.2f} MB"
            )

    flat = [r["test"] for r in runs]
    summary = {
        "provenance": U.provenance("B_fedavg_summary"),
        "protocol": args.protocol,
        "task": task,
        "n_runs": len(runs),
        "seeds": seeds,
        "federated": cfg["federated"],
        "overrides": overrides or None,
        "aggregation": "pooled: every test window weighted equally",
        "aggregate": {
            k: M.aggregate(flat, k)
            for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
        },
        "per_class_f1": {
            c: M.aggregate([{"f1": t["per_class"][c]["f1"]} for t in flat], "f1")
            for c in names
        },
        "communication": {
            "state_bytes": runs[0]["communication"]["state_bytes"],
            "mean_bytes_transferred": float(
                np.mean([r["communication"]["bytes_transferred"] for r in runs])
            ),
            "mean_rounds": float(np.mean([r["rounds_run"] for r in runs])),
        },
    }

    subject_scores = [
        m for r in runs if r.get("test_per_subject") for m in r["test_per_subject"].values()
    ]
    if subject_scores:
        summary["subject_weighted"] = {
            "aggregation": "one score per (subject, seed); each PERSON weighted equally - comparable with stage 04",
            "n_scores": len(subject_scores),
            "aggregate": {
                k: M.aggregate(subject_scores, k)
                for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
            },
            "per_class_f1": {
                c: M.aggregate([{"f1": t["per_class"][c]["f1"]} for t in subject_scores], "f1")
                for c in names
            },
        }

    U.write_json(out_dir / "summary.json", summary)

    logger.info("")
    logger.info(f"FEDAVG | {args.protocol} | {task} | {len(runs)} runs")
    for k, v in summary["aggregate"].items():
        if v["mean"] is not None:
            logger.info(f"  {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")
    logger.info("  per-class F1:")
    for c, v in summary["per_class_f1"].items():
        if v["mean"] is not None:
            logger.info(f"    {c:14s} {v['mean']:.4f} +/- {v['std']:.4f}")
    logger.info(
        f"  comms: {summary['communication']['state_bytes']/1024:.1f} KB/state "
        f"| {summary['communication']['mean_rounds']:.1f} rounds mean "
        f"| {summary['communication']['mean_bytes_transferred']/1e6:.2f} MB mean total"
    )

    if "subject_weighted" in summary:
        logger.info("")
        logger.info("  SUBJECT-WEIGHTED (comparable with local-only 0.9088):")
        for k, v in summary["subject_weighted"]["aggregate"].items():
            if v["mean"] is not None:
                logger.info(f"    {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")

    logger.info(f"  written to {out_dir}")


if __name__ == "__main__":
    main()
