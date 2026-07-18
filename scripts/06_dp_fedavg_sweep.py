#!/usr/bin/env python3
"""
=====================================================================
Stage C - Differentially private federated averaging
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS MEASURES
------------------
The cost of privacy, isolated. Stage B established the cost of federation
alone (LOSO macro-F1 0.7649 against a centralised ceiling of 0.7812,
statistically equivalent within +/-0.05). Everything here is inherited from
Stage B unchanged - architecture, splits, E=5, full participation, per-subject
normalisation, five seeds - so the only difference is the privacy mechanism.

WHAT IS PROTECTED
-----------------
The participation of a person, not of a window. See src/privacy.py.

WHAT TO EXPECT, STATED IN ADVANCE
---------------------------------
The accountant predicts, at 30 rounds with 12 clients and no subsampling
amplification, that the noise surviving into the averaged update is sigma/n
times the clip norm:

    eps 0.5 -> 2.99x   eps 1 -> 1.60x   eps 2 -> 0.86x
    eps 4   -> 0.47x   eps 8 -> 0.27x   eps 16 -> 0.15x

Ratios above 1.0 mean the noise exceeds the entire clipped update, so
epsilon <= 1 is expected to collapse to near-chance. This prediction is
recorded here BEFORE the runs so that the outcome cannot be reinterpreted
afterwards. Collapse at strong budgets is a finding about user-level DP at
realistic clinical cohort sizes, not a failed experiment.

STANDING HYPOTHESIS FROM STAGES A3 AND B
----------------------------------------
Amusement fidelity degrades monotonically as training data becomes less
personal (0.8722 own -> 0.7060 mixed -> 0.4320 others -> 0.3375 averaged).
Stage A3 identified six marginal subjects sitting on a knife-edge. The
prediction is that DP noise removes those first, leaving the strong and the
already-failed, so the distribution of per-subject fidelity becomes MORE
bimodal as epsilon falls.

USAGE
-----
    python scripts/06_dp_fedavg_sweep.py --self-test
    python scripts/06_dp_fedavg_sweep.py --pilot
    python scripts/06_dp_fedavg_sweep.py
    python scripts/06_dp_fedavg_sweep.py --epsilons 4 8 --folds 0 1 --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import federated as F
from src import metrics as M
from src import models as Mod
from src import privacy as P
from src import training as T
from src import utils as U


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


def self_test() -> None:
    """Verify the accountant and the mechanism before any result is produced.

    Privacy accounting is the one component here whose failure no metric would
    reveal. An under-noised run trains beautifully and reports a confident
    epsilon that is simply wrong - and the paper's central claim is that
    epsilon. It is checked explicitly.
    """
    print("[self-test] privacy accountant")
    ok = []

    # 1. epsilon must fall monotonically as sigma rises
    eps = [P.compute_epsilon(s, 30, 1e-4)[0] for s in [1, 2, 4, 8, 16, 32]]
    m1 = all(a > b for a, b in zip(eps, eps[1:]))
    print(f"  eps falls with sigma: {[round(e,3) for e in eps]} -> {'OK' if m1 else 'FAIL'}")
    ok.append(m1)

    # 2. epsilon must rise with composition
    eps_t = [P.compute_epsilon(8.0, t, 1e-4)[0] for t in [1, 10, 30, 100]]
    m2 = all(a < b for a, b in zip(eps_t, eps_t[1:]))
    print(f"  eps rises with rounds: {[round(e,3) for e in eps_t]} -> {'OK' if m2 else 'FAIL'}")
    ok.append(m2)

    # 3. calibration must be a true inverse of the accountant
    m3 = True
    for tgt in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        s = P.calibrate_noise(tgt, 30, 1e-4)
        got, _ = P.compute_epsilon(s, 30, 1e-4)
        hit = abs(got - tgt) < 0.01
        m3 &= hit
        print(f"  target eps={tgt:5.1f} -> sigma={s:7.3f} -> achieved {got:7.4f} "
              f"| noise/clip on avg (n=12) = {s/12:5.3f}x -> {'OK' if hit else 'FAIL'}")
    ok.append(m3)

    # 4. clipping must actually bound the norm
    v = torch.randn(1000) * 10
    clipped, raw = P.clip_delta(v.clone(), clip_norm=1.0)
    n_after = float(torch.linalg.vector_norm(clipped))
    m4 = n_after <= 1.0 + 1e-5
    print(f"  clip: norm {raw:.3f} -> {n_after:.6f} (bound 1.0) -> {'OK' if m4 else 'FAIL'}")
    ok.append(m4)

    # 5. a small update must pass through unchanged
    v2 = torch.randn(100)
    v2 = v2 / torch.linalg.vector_norm(v2) * 0.5
    c2, _ = P.clip_delta(v2.clone(), clip_norm=1.0)
    m5 = torch.allclose(c2, v2)
    print(f"  clip leaves sub-threshold updates untouched -> {'OK' if m5 else 'FAIL'}")
    ok.append(m5)

    # 6. sigma=0 must reproduce plain averaging exactly
    g = {"w": torch.zeros(4)}
    cs = [{"w": torch.tensor([1.0, 2, 3, 4])}, {"w": torch.tensor([3.0, 4, 5, 6])}]
    out, _ = P.dp_aggregate(g, cs, clip_norm=1e9, noise_multiplier=0.0)
    m6 = torch.allclose(out["w"], torch.tensor([2.0, 3, 4, 5]))
    print(f"  sigma=0, no clipping -> plain mean {out['w'].tolist()} -> {'OK' if m6 else 'FAIL'}")
    ok.append(m6)

    # 7. noise must actually be injected, at the right scale
    gen = torch.Generator().manual_seed(0)
    g2 = {"w": torch.zeros(10_000)}
    c3 = [{"w": torch.zeros(10_000)}]
    out2, st = P.dp_aggregate(g2, c3, clip_norm=1.0, noise_multiplier=2.0, generator=gen)
    emp = float(out2["w"].std())
    m7 = abs(emp - 2.0) < 0.1          # n=1, so noise on the average is sigma*C
    print(f"  injected noise std {emp:.4f} vs expected 2.0 -> {'OK' if m7 else 'FAIL'}")
    ok.append(m7)

    print("[self-test] " + ("PASSED" if all(ok) else "FAILED"))
    if not all(ok):
        sys.exit("Do not trust any privacy result until this passes.")


# ---------------------------------------------------------------------
# Pilot: calibrate the clip norm from observed update magnitudes
# ---------------------------------------------------------------------


def pilot_update_norm(clients, model, cfg, device, seed, rounds: int) -> dict:
    """Measure client update norms with no noise and no clipping.

    The clip norm should reflect the magnitude updates actually take. Assuming
    C = 1.0 without checking risks either destroying the signal (C far below
    the typical update) or inflating the noise pointlessly (C far above it),
    and either would be attributed to differential privacy rather than to a
    poorly chosen constant.
    """
    import copy

    global_state = copy.deepcopy(model.state_dict())
    norms = []
    for rnd in range(1, rounds + 1):
        states, counts = [], []
        for c in clients:
            st, n, _ = c.local_train(
                global_state, model, cfg, device, round_seed=seed * 10_000 + rnd
            )
            vec, _ = P.flatten_delta(st, global_state)
            norms.append(float(torch.linalg.vector_norm(vec)))
            states.append(st)
            counts.append(n)
        global_state = F.fedavg_aggregate(states, counts)
        model.load_state_dict(global_state)

    arr = np.array(norms)
    return {
        "n_observations": int(len(arr)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------


def dp_aggregate_fn(global_state, states, counts, *, clip_norm, sigma, generator):
    """Adapter matching run_federated's aggregation signature.

    counts is deliberately ignored: user-level DP requires equal weighting, or
    the sensitivity of the sum would depend on how much data a client holds and
    one clip norm could not bound a client's influence.
    """
    return P.dp_aggregate(
        global_state,
        states,
        clip_norm=clip_norm,
        noise_multiplier=sigma,
        generator=generator,
    )


def run_one(df, feats, target, names, split, cfg, seed, device, subj_col,
            eps_target, sigma, clip_norm, logger) -> dict:
    U.set_seed(seed)
    n_classes = len(names)
    rounds = int(cfg["privacy"]["rounds"])


    clients = []
    for subj, idx in split["client_indices"].items():
        X, y = D.make_xy(df, idx, feats, target)
        cw = (
            D.class_weights(y, n_classes)
            if cfg["federated"]["per_client_class_weights"]
            and cfg["training"]["class_weighted_loss"]
            else None
        )
        clients.append(F.Client(client_id=subj, X=X, y=y, n_classes=n_classes, class_weight=cw))

    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)
    bs = int(cfg["training"]["batch_size"])
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False)
    test_loader = T.make_loader(X_te, y_te, bs, shuffle=False)

    model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=n_classes).to(device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed * 7919 + int(eps_target * 1000))

    agg = partial(dp_aggregate_fn, clip_norm=clip_norm, sigma=sigma, generator=gen)

    fit = F.run_federated(
        clients,
        model,
        val_loader,
        cfg=cfg,
        device=device,
        score_fn=lambda a, b_: f1_score(a, b_, average="macro", zero_division=0),
        seed=seed,
        aggregate_fn=None if sigma == 0 else agg,
        fixed_rounds=rounds,
        logger=logger,
    )

    y_true, y_pred, y_proba = T.predict(model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    report = P.privacy_report(sigma, rounds, float(cfg["privacy"]["delta"]),
                              len(clients), clip_norm) if sigma > 0 else {
        "epsilon": float("inf"), "delta": None, "noise_multiplier": 0.0,
        "clip_norm": None, "rounds": rounds, "n_clients": len(clients),
        "granularity": "none (non-private anchor)",
    }

    return {
        "seed": seed,
        "epsilon_target": eps_target,
        "privacy": report,
        "n_clients": len(clients),
        "best_round": fit["best_round"],
        "rounds_run": fit["rounds_run"],
        "best_val_macro_f1": fit["best_val_score"],
        "communication": {
            "state_bytes": fit["state_bytes"],
            "bytes_transferred": fit["bytes_transferred"],
        },
        "aggregation_stats": fit["aggregation_stats"],
        "test": test_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C - DP-FedAvg sweep")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--fedavg-config", default="configs/fedavg.yaml")
    ap.add_argument("--dp-config", default="configs/dp.yaml")
    ap.add_argument("--task", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--epsilons", type=float, nargs="+", default=None)
    ap.add_argument("--clip-norm", type=float, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="Measure update norms to calibrate clip_norm, then exit.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    cfg = U.load_configs(args.splits_config, args.model_config,
                         args.fedavg_config, args.dp_config)
    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])
    subj_col = cfg["input"]["subject_col"]
    pcfg = cfg["privacy"]

    logger = U.setup_logging(f"06_dp_fedavg_{task}", cfg["output"]["logs_dir"])

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)
    if cfg["normalisation"]["strategy"] != "per_subject":
        sys.exit("DP conditions require per_subject normalisation.")
    df = D.apply_normalisation(df, feats, cfg)

    fold_ids = args.folds if args.folds is not None else D.list_folds(cfg)
    splits = []
    for fid in fold_ids:
        fold = D.load_fold(cfg, fid)
        train_idx = np.asarray(fold["indices"]["train"])
        sub = df.loc[train_idx, subj_col]
        splits.append({
            "label": f"fold_{fid:02d}",
            "indices": fold["indices"],
            "client_indices": {
                s: train_idx[(sub == s).to_numpy()].tolist() for s in fold["train_subjects"]
            },
        })

    # ---- pilot -------------------------------------------------------
    if args.pilot:
        logger.info("PILOT: measuring client update norms (no noise, no clipping)")
        sp = splits[0]
        clients = []
        for subj, idx in sp["client_indices"].items():
            X, y = D.make_xy(df, idx, feats, target)
            cw = D.class_weights(y, len(names)) if cfg["training"]["class_weighted_loss"] else None
            clients.append(F.Client(client_id=subj, X=X, y=y, n_classes=len(names), class_weight=cw))
        U.set_seed(seeds[0])
        model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=len(names)).to(device)
        st = pilot_update_norm(clients, model, cfg, device, seeds[0], int(pcfg["pilot_rounds"]))
        logger.info(f"  update norms over {st['n_observations']} client-rounds:")
        for k in ["min", "p25", "median", "mean", "p75", "max"]:
            logger.info(f"    {k:8s} {st[k]:.4f}")
        logger.info(f"\n  suggested clip_norm = median = {st['median']:.4f}")
        logger.info("  set privacy.clip_norm in configs/dp.yaml, or pass --clip-norm")
        U.write_json(Path(cfg["output"]["results_dir"]) / "loso" / "dp_fedavg" / task / "pilot.json", st)
        return

    # ---- clip norm ---------------------------------------------------
    clip_norm = args.clip_norm if args.clip_norm is not None else float(pcfg["clip_norm"])

    # ---- calibrate sigma for each target ------------------------------
    rounds = int(pcfg["rounds"])
    delta = float(pcfg["delta"])
    targets = args.epsilons if args.epsilons is not None else list(pcfg["epsilon_targets"])

    logger.info(f"Stage C | DP-FedAvg | LOSO | task={task} | device={device}")
    logger.info(f"granularity=user-level | delta={delta} | rounds={rounds} (fixed) | clip_norm={clip_norm}")
    logger.info("")
    logger.info(f"  {'target eps':>10s} {'sigma':>9s} {'achieved':>10s} {'noise/clip on avg':>19s}")
    grid = []
    for t in targets:
        s = P.calibrate_noise(t, rounds, delta)
        got, _ = P.compute_epsilon(s, rounds, delta)
        n_cl = len(splits[0]["client_indices"])
        logger.info(f"  {t:10.1f} {s:9.3f} {got:10.4f} {s/n_cl:18.3f}x")
        grid.append((t, s))
    logger.info("  ratio > 1.0 means the noise exceeds the entire clipped update")
    logger.info("")

    out_root = Path(cfg["output"]["results_dir"]) / "loso" / "dp_fedavg" / task
    all_runs = []

    for eps_target, sigma in grid:
        tag = f"eps{eps_target:g}"
        runs = []
        for split in splits:
            for seed in seeds:
                r = run_one(df, feats, target, names, split, cfg, seed, device,
                            subj_col, eps_target, sigma, clip_norm, logger)
                r["split"] = split["label"]
                r["protocol"] = "loso"
                r["task"] = task
                r["provenance"] = U.provenance("C_dp_fedavg")
                U.write_json(out_root / tag / f"{split['label']}_seed{seed}.json", r)
                runs.append(r)
                logger.info(
                    f"  {tag} | {split['label']} | seed {seed} "
                    f"-> macro_f1 {r['test']['macro_f1']:.4f} "
                    f"| amusement {r['test']['per_class'].get('amusement',{}).get('f1',float('nan')):.4f}"
                )
        flat = [r["test"] for r in runs]
        summary = {
            "provenance": U.provenance("C_dp_fedavg_summary"),
            "epsilon_target": eps_target,
            "epsilon_achieved": runs[0]["privacy"]["epsilon"],
            "noise_multiplier": sigma,
            "delta": delta,
            "clip_norm": clip_norm,
            "rounds": rounds,
            "granularity": "user-level",
            "n_runs": len(runs),
            "aggregate": {
                k: M.aggregate(flat, k)
                for k in ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
            },
            "per_class_f1": {
                c: M.aggregate([{"f1": t_["per_class"][c]["f1"]} for t_ in flat], "f1")
                for c in names
            },
        }
        U.write_json(out_root / tag / "summary.json", summary)
        all_runs.append((eps_target, sigma, summary))

        a = summary["aggregate"]["macro_f1"]
        logger.info(
            f"  == {tag}: macro_f1 {a['mean']:.4f} +/- {a['std']:.4f} "
            f"(sigma={sigma:.3f}) =="
        )
        logger.info("")

    logger.info("=" * 72)
    logger.info(f"STAGE C | DP-FedAvg | user-level | delta={delta} | {rounds} rounds")
    logger.info(f"{'eps':>7s} {'sigma':>8s} {'macro_f1':>17s} {'amusement':>17s}")
    logger.info("-" * 72)
    logger.info(f"{'inf':>7s} {'0':>8s} {'0.7649 +/- 0.0994':>17s} {'0.3375 +/- 0.3474':>17s}   <- Stage B")
    for eps_target, sigma, s in all_runs:
        mf = s["aggregate"]["macro_f1"]
        am = s["per_class_f1"].get("amusement", {"mean": float("nan"), "std": float("nan")})
        logger.info(
            f"{eps_target:7.1f} {sigma:8.3f} "
            f"{mf['mean']:8.4f} +/- {mf['std']:.4f} "
            f"{am['mean']:8.4f} +/- {am['std']:.4f}"
        )
    logger.info(f"\nwritten to {out_root}")


if __name__ == "__main__":
    main()
