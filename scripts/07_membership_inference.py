#!/usr/bin/env python3
"""
=====================================================================
Stage D - Membership inference against DP-FedAvg
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS MEASURES
------------------
Empirical privacy: can an adversary determine whether a person was in the
training cohort? The result is placed beside the formal epsilon from Stage C.
If the attack is near chance at a budget the formal accounting calls weak, then
formal budgets overstate the real risk in this setting - the question the study
exists to answer.

DESIGN (see 07a_build_attack_splits.py and src/attacks.py for detail)
---------------------------------------------------------------------
- Training is DP-FedAvg, identical to Stage C, but on attack-slice splits:
  each member withholds a stratified 20% of its windows from training.
- Members are scored on their withheld slice; the non-member (test subject) on
  its windows. Both are unseen at training, so the generalisation gap is
  removed and only membership distinguishes them.
- Threshold attack (Yeom et al. 2018), no shadow models, because N = 15 cannot
  support uncontaminated shadows.
- Metrics: AUC (headline) and TPR at low FPR (Carlini et al. 2022). Window-level
  per fold; user-level pooled across folds.

This runs its own training rather than reusing Stage C, because the attack-slice
holdout changes the training data. Stage D therefore has its own utility numbers,
close to but not identical to Stage C, and reported separately.

USAGE
-----
    python scripts/07_membership_inference.py --self-test
    python scripts/07_membership_inference.py
    python scripts/07_membership_inference.py --epsilons inf 8 --seeds 1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import attacks as A
from src import data as D
from src import federated as F
from src import metrics as M
from src import models as Mod
from src import privacy as P
from src import training as T
from src import utils as U
from src.privacy_sample import SampleDPClient, check_opacus_available


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


def self_test() -> None:
    """End-to-end check of the attack path on constructed data.

    Verifies that a model which memorises its training data yields a
    high-AUC attack, and that a model whose members and non-members are
    statistically identical yields an AUC near chance. If the second case did
    not return ~0.5, the pipeline would manufacture false evidence of leakage.
    """
    rng = np.random.default_rng(0)

    print("[self-test] attack pipeline")

    # Case 1: clear membership signal (members confident, non-members diffuse)
    K = 4
    def probs(n, conc):
        y = rng.integers(0, K, n)
        lg = np.zeros((n, K)); lg[np.arange(n), y] = conc
        lg += rng.normal(size=(n, K)) * 0.5
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), y

    pm, ym = probs(300, 4.0)      # members: confident
    pn, yn = probs(100, 0.5)      # non-members: near-uniform
    proba = np.vstack([pm, pn]); y = np.r_[ym, yn]
    member = np.r_[np.ones(300), np.zeros(100)].astype(int)
    ce = A.cross_entropy_per_example(proba, y)
    m1 = A.attack_metrics(-ce, member)
    ok1 = m1["auc"] > 0.85
    print(f"  memorising model -> AUC {m1['auc']:.4f} (expect > 0.85) -> {'OK' if ok1 else 'FAIL'}")

    # Case 2: no signal (members and non-members identical)
    pa, ya = probs(300, 2.0); pb, yb = probs(100, 2.0)
    proba2 = np.vstack([pa, pb]); y2 = np.r_[ya, yb]
    mem2 = np.r_[np.ones(300), np.zeros(100)].astype(int)
    ce2 = A.cross_entropy_per_example(proba2, y2)
    m2 = A.attack_metrics(-ce2, mem2)
    ok2 = abs(m2["auc"] - 0.5) < 0.1
    print(f"  no-leak model    -> AUC {m2['auc']:.4f} (expect ~0.5) -> {'OK' if ok2 else 'FAIL'}")

    # Case 3: user-level aggregation preserves a signal
    ws = np.r_[rng.normal(1, 0.5, 120), rng.normal(-1, 0.5, 30)]
    subj = np.r_[np.repeat([f"M{i}" for i in range(12)], 10),
                 np.repeat(["T0", "T1", "T2"], 10)]
    is_mem = {**{f"M{i}": 1 for i in range(12)}, "T0": 0, "T1": 0, "T2": 0}
    us, ul, _ = A.aggregate_to_user(ws, subj, is_mem)
    mu = A.attack_metrics(us, ul)
    ok3 = mu["auc"] > 0.85 and mu["n_members"] == 12 and mu["n_non_members"] == 3
    print(f"  user-level agg   -> AUC {mu['auc']:.4f}, {mu['n_members']}m/{mu['n_non_members']}nm -> {'OK' if ok3 else 'FAIL'}")

    print("[self-test] " + ("PASSED" if all([ok1, ok2, ok3]) else "FAILED"))
    if not all([ok1, ok2, ok3]):
        sys.exit("Attack pipeline unreliable; do not run.")


# ---------------------------------------------------------------------
# Attack-slice split loading
# ---------------------------------------------------------------------


def load_attack_fold(cfg, fid: int) -> dict:
    path = Path(cfg["output"]["splits_dir"]) / cfg["attack"]["splits_subdir"] / f"fold_{fid:02d}.json"
    if not path.exists():
        sys.exit(f"Attack split not found: {path}. Run 07a_build_attack_splits.py first.")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------
# Train one model on the attack-slice split, then score membership
# ---------------------------------------------------------------------


def dp_aggregate_fn(global_state, states, counts, *, clip_norm, sigma, generator):
    return P.dp_aggregate(global_state, states, clip_norm=clip_norm,
                          noise_multiplier=sigma, generator=generator)


def run_one(df, feats, target, names, afold, cfg, seed, device, subj_col,
            eps_target, sigma, clip_norm, logger, mechanism: str = "user") -> dict:
    U.set_seed(seed)
    n_classes = len(names)
    rounds = int(cfg["privacy"]["rounds"])

    # Build clients from the MEMBER TRAIN slices only (attack slices withheld).
    # The client class differs by mechanism: user-level DP noises the aggregate
    # (plain Client + a DP aggregator), whereas sample-level DP noises each
    # client's local gradients (SampleDPClient, then plain FedAvg aggregation,
    # since averaging is post-processing of already-private updates).
    X_va, y_va = D.make_xy(df, afold["val_indices"], feats, target)
    bs = int(cfg["training"]["batch_size"])
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False, device=device)

    if mechanism == "sample" and eps_target is not None:
        check_opacus_available()
        pcfg = cfg["privacy_sample"]
        local_epochs = int(cfg["federated"]["local_epochs"])
        clients = []
        for s, idx in afold["member_train_indices"].items():
            X, y = D.make_xy(df, idx, feats, target)
            cw = (D.class_weights(y, n_classes)
                  if cfg["federated"]["per_client_class_weights"] and cfg["training"]["class_weighted_loss"]
                  else None)
            c = SampleDPClient(client_id=s, X=X, y=y, n_classes=n_classes, class_weight=cw)
            cm = Mod.build_model(cfg, input_dim=X.shape[1], num_classes=n_classes)
            c.attach_privacy_engine(
                cm,
                target_epsilon=float(eps_target),
                delta=float(pcfg["delta"]),
                max_grad_norm=float(clip_norm),
                total_local_epochs=rounds * local_epochs,
                batch_size=int(pcfg["batch_size"]),
                base_lr=float(pcfg["local_learning_rate"]),
                weight_decay=float(pcfg["weight_decay"]),
                device=device,
            )
            clients.append(c)

        model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=n_classes).to(device)
        global_state = copy.deepcopy(model.state_dict())
        best_val, best_state = -1.0, copy.deepcopy(global_state)
        for rnd in range(1, rounds + 1):
            states, counts = [], []
            for c in clients:
                st, n_, _ = c.local_train(global_state, cfg, round_seed=seed * 10_000 + rnd)
                states.append(st); counts.append(n_)
            global_state = F.fedavg_aggregate(states, counts)
            model.load_state_dict(global_state)
            y_v, p_v, _ = T.predict(model, val_loader, device)
            v = f1_score(y_v, p_v, average="macro", zero_division=0)
            if v > best_val:
                best_val, best_state = v, copy.deepcopy(global_state)
        model.load_state_dict(best_state)
        fit = {"best_round": None, "rounds_run": rounds, "best_val_score": float(best_val),
               "state_bytes": None, "bytes_transferred": None}
    else:
        clients = []
        for s, idx in afold["member_train_indices"].items():
            X, y = D.make_xy(df, idx, feats, target)
            cw = (D.class_weights(y, n_classes)
                  if cfg["federated"]["per_client_class_weights"] and cfg["training"]["class_weighted_loss"]
                  else None)
            clients.append(F.Client(client_id=s, X=X, y=y, n_classes=n_classes, class_weight=cw))

        model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=n_classes).to(device)

        if sigma and sigma > 0:
            gen = torch.Generator(device="cpu"); gen.manual_seed(seed * 7919 + int(eps_target * 1000))
            agg = partial(dp_aggregate_fn, clip_norm=clip_norm, sigma=sigma, generator=gen)
        else:
            agg = None

        fit = F.run_federated(
            clients, model, val_loader, cfg=cfg, device=device,
            score_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
            seed=seed, aggregate_fn=agg, fixed_rounds=rounds, logger=None,
        )

    # ---- score membership -------------------------------------------------
    # Members: their withheld attack slice. Non-member: the test subject.
    member_idx, member_subj = [], []
    for s, idx in afold["member_attack_indices"].items():
        member_idx.extend(idx); member_subj.extend([s] * len(idx))
    test_idx = afold["test_indices"]
    test_subj_name = afold["test_subject"]

    all_idx = list(member_idx) + list(test_idx)
    all_subj = np.array(list(member_subj) + [test_subj_name] * len(test_idx))
    is_member = np.r_[np.ones(len(member_idx)), np.zeros(len(test_idx))].astype(int)

    Xa, ya = D.make_xy(df, all_idx, feats, target)
    _, _, proba = T.predict(model, T.make_loader(Xa, ya, bs, shuffle=False, device=device), device)

    # For the within-subject memorisation test, also score each member on the
    # windows the model TRAINED ON. Same person, different windows: only
    # difference is whether the specific windows were seen. This is the
    # correct user-level attack for LOSO — no cross-person confound.
    train_ce = {}
    for s_, idx in afold["member_train_indices"].items():
        Xt, yt = D.make_xy(df, idx, feats, target)
        _, _, pt = T.predict(model, T.make_loader(Xt, yt, bs, shuffle=False, device=device), device)
        train_ce[s_] = A.cross_entropy_per_example(pt, yt)
    attack_ce = {}
    for s_, idx in afold["member_attack_indices"].items():
        Xk, yk = D.make_xy(df, idx, feats, target)
        _, _, pk = T.predict(model, T.make_loader(Xk, yk, bs, shuffle=False, device=device), device)
        attack_ce[s_] = A.cross_entropy_per_example(pk, yk)
    within_subject_result = A.within_subject_memorisation(train_ce, attack_ce)

    subject_is_member = {**{s: 1 for s in afold["members"]}, test_subj_name: 0}

    signals = {
        "cross_entropy": -A.cross_entropy_per_example(proba, ya),  # higher = member
        "calibrated_ce": -A.calibrated_cross_entropy(proba, ya),
        "confidence": A.confidence_per_example(proba, ya),
        "negative_entropy": A.negative_entropy_per_example(proba),
    }

    result = {
        "seed": seed, "epsilon_target": eps_target, "mechanism": mechanism,
        "fold_id": afold["fold_id"],
        "test_subject": test_subj_name,
        "n_member_windows": len(member_idx), "n_test_windows": len(test_idx),
        "utility": {  # this model's own utility, on the test subject
            "test_subject_macro_f1": None,
        },
        "window_level": {},
        # WITHIN-SUBJECT memorisation test (the honest user-level attack).
        # Compares each member's train-window losses to their own attack-slice
        # losses — same person, different windows. Replaces the pooled
        # cross-person user-level attack, which was confounded by the LOSO
        # generalisation gap (see paper_notes/D_mia_confound.md).
        "within_subject_memorisation": within_subject_result,
    }

    for sig_name in signals:
        sig_arr = signals[sig_name]
        result["window_level"][sig_name] = A.attack_metrics(sig_arr, is_member)

    # utility on the test subject, for a Stage-D utility table
    yt, pt, _ = T.predict(model, T.make_loader(*D.make_xy(df, test_idx, feats, target), bs, shuffle=False), device)
    result["utility"]["test_subject_macro_f1"] = float(f1_score(yt, pt, average="macro", zero_division=0))

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage D - membership inference")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--fedavg-config", default="configs/fedavg.yaml")
    ap.add_argument("--dp-config", default="configs/dp.yaml")
    ap.add_argument("--attack-config", default="configs/attack.yaml")
    ap.add_argument("--task", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--epsilons", nargs="+", default=None,
                    help="Targets to attack, e.g. inf 8 4. 'inf' = non-private.")
    ap.add_argument("--clip-norm", type=float, default=None,
                    help="Defaults to 2.2 for user-level, 3.0 for sample-level.")
    ap.add_argument("--dp-mechanism", choices=["user", "sample"], default="user",
                    help="user = noise at aggregation (Stage C/D). "
                         "sample = Opacus DP-SGD per client (Stage C-prime).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    config_paths = [args.splits_config, args.model_config,
                    args.fedavg_config, args.dp_config, args.attack_config]
    if args.dp_mechanism == "sample":
        config_paths.append("configs/dp_sample.yaml")
    cfg = U.load_configs(*config_paths)

    # Clip norms differ by mechanism because the quantities clipped differ:
    # user-level clips a whole client update (Stage C pilot median 2.2),
    # sample-level clips a per-example gradient (Stage C-prime pilot median 3.0).
    clip_norm = args.clip_norm
    if clip_norm is None:
        clip_norm = 3.0 if args.dp_mechanism == "sample" else 2.2
    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])
    subj_col = cfg["input"]["subject_col"]
    rounds = int(cfg["privacy"]["rounds"])
    delta = float(cfg["privacy"]["delta"])

    logger = U.setup_logging(f"07_mia_{task}", cfg["output"]["logs_dir"])

    df = D.load_windows(cfg); D.validate_schema(df, cfg); df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task); names = D.class_names(cfg, task)
    df = D.apply_normalisation(df, feats, cfg)

    fold_ids = args.folds if args.folds is not None else D.list_folds(cfg)
    afolds = [load_attack_fold(cfg, f) for f in fold_ids]

    # Resolve epsilon targets to noise multipliers.
    raw_targets = args.epsilons if args.epsilons is not None else (["inf"] + [str(e) for e in cfg["privacy"]["epsilon_targets"]])
    grid = []
    for t in raw_targets:
        if str(t).lower() in ("inf", "none", "np"):
            grid.append(("inf", 0.0))
        else:
            tv = float(t)
            if args.dp_mechanism == "sample":
                # Opacus calibrates sigma per client inside attach_privacy_engine,
                # so no server-side sigma is computed here.
                grid.append((tv, None))
            else:
                grid.append((tv, P.calibrate_noise(tv, rounds, delta)))

    logger.info(f"Stage D | MIA | LOSO | task={task} | device={device} "
                f"| dp_mechanism={args.dp_mechanism} | clip_norm={clip_norm}")
    logger.info(f"attack-slice training | {len(afolds)} folds | seeds {seeds}")
    logger.info(f"targets: {[g[0] for g in grid]}")

    mia_dir = "mia" if args.dp_mechanism == "user" else "mia_sample"
    out_root = Path(cfg["output"]["results_dir"]) / "loso" / mia_dir / task
    pooled = {}   # (eps, signal, granularity) -> accumulator

    for eps_target, sigma in grid:
        tag = "epsinf" if eps_target == "inf" else f"eps{eps_target:g}"
        runs = []
        for afold in afolds:
            for seed in seeds:
                r = run_one(df, feats, target, names, afold, cfg, seed, device,
                            subj_col, None if eps_target == "inf" else eps_target,
                            sigma, clip_norm, logger, mechanism=args.dp_mechanism)
                r["epsilon_tag"] = tag
                r["task"] = task
                r["provenance"] = U.provenance("D_mia")
                U.write_json(out_root / tag / f"fold_{afold['fold_id']:02d}_seed{seed}.json", r)
                runs.append(r)

        # window-level: mean AUC across runs, per signal
        summary = {"epsilon_target": eps_target, "noise_multiplier": sigma,
                   "n_runs": len(runs), "task": task, "window_level": {},
                   "within_subject_memorisation": {}}
        signal_names = list(runs[0]["window_level"].keys())
        for sig in signal_names:
            aucs = [r["window_level"][sig]["auc"] for r in runs]
            tpr1 = [r["window_level"][sig]["tpr_at_1pct_fpr"] for r in runs]
            summary["window_level"][sig] = {
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0),
                "auc_advantage_mean": float(np.mean([abs(a - 0.5) * 2 for a in aucs])),
                "tpr_at_1pct_fpr_mean": float(np.mean(tpr1)),
            }
        # within-subject memorisation aggregated across runs
        diffs_per_run = [r["within_subject_memorisation"]["paired_diff_mean"] for r in runs]
        n_mem_per_run = [r["within_subject_memorisation"]["n_memorised"] for r in runs]
        p_per_run = [r["within_subject_memorisation"].get("wilcoxon_p_one_sided", None) for r in runs]
        p_valid = [p for p in p_per_run if p is not None]
        summary["within_subject_memorisation"] = {
            "paired_diff_mean_across_runs": float(np.mean(diffs_per_run)),
            "paired_diff_std_across_runs": float(np.std(diffs_per_run, ddof=1)) if len(diffs_per_run) > 1 else 0.0,
            "mean_n_memorised": float(np.mean(n_mem_per_run)),
            "n_runs_with_p": len(p_valid),
            "median_wilcoxon_p": float(np.median(p_valid)) if p_valid else None,
        }

        util = [r["utility"]["test_subject_macro_f1"] for r in runs]
        summary["utility_test_subject_macro_f1"] = {
            "mean": float(np.mean(util)), "std": float(np.std(util, ddof=1) if len(util) > 1 else 0)
        }
        U.write_json(out_root / tag / "summary.json", summary)

        ce = summary["window_level"]["cross_entropy"]
        cc = summary["window_level"].get("calibrated_ce", ce)
        ws = summary["within_subject_memorisation"]
        logger.info(
            f"  {tag:8s} | util {summary['utility_test_subject_macro_f1']['mean']:.4f} "
            f"| CE AUC {ce['auc_mean']:.4f}+/-{ce['auc_std']:.4f} "
            f"| calib AUC {cc['auc_mean']:.4f} "
            f"| train-atk gap {ws['paired_diff_mean_across_runs']:+.4f}"
        )
        pooled[tag] = summary

    logger.info("")
    logger.info("=" * 90)
    logger.info(f"STAGE D | MIA | {task}")
    logger.info(f"{'eps':>8s} {'utility':>9s} {'window CE AUC':>18s} {'window calib AUC':>18s} {'train-atk gap':>16s}")
    logger.info("-" * 90)
    for tag, s in pooled.items():
        ce = s["window_level"]["cross_entropy"]
        cc = s["window_level"].get("calibrated_ce", ce)
        ws = s["within_subject_memorisation"]
        logger.info(
            f"{tag:>8s} {s['utility_test_subject_macro_f1']['mean']:9.4f} "
            f"{ce['auc_mean']:8.4f} +/- {ce['auc_std']:.4f} "
            f"{cc['auc_mean']:8.4f} +/- {cc['auc_std']:.4f} "
            f"{ws['paired_diff_mean_across_runs']:+8.4f} +/- {ws['paired_diff_std_across_runs']:.4f}"
        )
    logger.info("\nAUC ~ 0.5 = attack fails empirically. train-atk gap < 0 = model memorised (lower loss on training windows). ")
    logger.info(f"written to {out_root}")


if __name__ == "__main__":
    main()
