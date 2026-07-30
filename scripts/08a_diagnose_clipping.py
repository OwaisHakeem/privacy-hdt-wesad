#!/usr/bin/env python3
"""
=====================================================================
Stage C-prime diagnostic - clipping-only, zero noise
Project : privacy_hdt_wesad
=====================================================================

WHY THIS EXISTS
---------------
Sample-level DP-FedAvg collapsed stress and amusement to F1 = 0.0000 even at
epsilon = 16, the WEAKEST budget in the grid (sigma = 1.406 - mild noise).
That rules out "needs a bigger privacy budget": something about the mechanism
itself, not the noise level, is destroying the two rarest classes.

This script removes noise entirely (noise_multiplier = 0.0) and trains with
clipping ONLY, at the same clip_norm = 3.0 used in the failed run. Two
possible outcomes:

  * Classes still collapse with zero noise -> the bug is in how per-example
    gradient CLIPPING interacts with class-weighted loss. Rare-class (stress,
    amusement) examples get up-weighted by the class-weighted loss, which
    inflates their raw gradients; if those inflated gradients are then
    clipped harder than majority-class gradients, the rare-class learning
    signal is disproportionately destroyed BEFORE any privacy noise is added.

  * Classes survive with zero noise -> the bug is specifically in the noise
    or accounting path (e.g. per-round noise re-injection, incorrect sigma
    scaling, or an interaction between Poisson subsampling and the small
    per-client dataset size).

Either answer points directly at the fix, so this diagnostic is run BEFORE
touching the clip_norm, the class weighting, or anything else.

USAGE
-----
    python scripts/08a_diagnose_clipping.py
    python scripts/08a_diagnose_clipping.py --clip-norm 3.0 --folds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
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
from src.privacy_sample import SampleDPClient, check_opacus_available


def run_one_diagnostic(df, feats, target, names, split, cfg, seed, device,
                       clip_norm: float, use_class_weights: bool, logger) -> dict:
    """One federated run with clipping but NO noise, NO privacy accounting.

    use_class_weights toggles whether each client's loss is class-weighted.
    Run both ways: if collapse happens ONLY with class weighting on, that
    confirms hypothesis (b) precisely (clip + class-weight interaction).
    """
    U.set_seed(seed)
    n_classes = len(names)
    rounds = int(cfg["privacy_sample"]["rounds"])
    local_epochs = int(cfg["federated"]["local_epochs"])
    total_local_epochs = rounds * local_epochs
    batch_size = int(cfg["privacy_sample"]["batch_size"])
    lr = float(cfg["privacy_sample"]["local_learning_rate"])
    wd = float(cfg["privacy_sample"]["weight_decay"])

    dp_clients = []
    for subj, idx in split["client_indices"].items():
        X, y = D.make_xy(df, idx, feats, target)
        cw = D.class_weights(y, n_classes) if use_class_weights else None
        client = SampleDPClient(client_id=subj, X=X, y=y, n_classes=n_classes, class_weight=cw)
        client_model = Mod.build_model(cfg, input_dim=X.shape[1], num_classes=n_classes)
        client.attach_privacy_engine(
            client_model,
            target_epsilon=None,           # <-- no privacy accounting
            delta=1e-4,                    # unused when noise_multiplier is given directly
            max_grad_norm=clip_norm,
            total_local_epochs=total_local_epochs,
            batch_size=batch_size,
            base_lr=lr,
            weight_decay=wd,
            device=device,
            noise_multiplier=0.0,          # <-- clipping only, zero noise
        )
        dp_clients.append(client)

    global_model = Mod.build_model(cfg, input_dim=dp_clients[0].X.shape[1], num_classes=n_classes).to(device)
    global_state = copy.deepcopy(global_model.state_dict())

    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)
    bs = int(cfg["training"]["batch_size"])
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False)
    test_loader = T.make_loader(X_te, y_te, bs, shuffle=False)

    best_val, best_state, best_round = -1.0, copy.deepcopy(global_state), 0

    for rnd in range(1, rounds + 1):
        states, counts = [], []
        for c in dp_clients:
            st, n, _ = c.local_train(global_state, cfg, round_seed=seed * 10_000 + rnd)
            states.append(st)
            counts.append(n)
        global_state = F.fedavg_aggregate(states, counts)
        global_model.load_state_dict(global_state)

        y_v, p_v, _ = T.predict(global_model, val_loader, device)
        val_score = f1_score(y_v, p_v, average="macro", zero_division=0)
        if val_score > best_val:
            best_val, best_state, best_round = val_score, copy.deepcopy(global_state), rnd

    global_model.load_state_dict(best_state)
    y_true, y_pred, y_proba = T.predict(global_model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    return {"test": test_metrics, "best_round": best_round, "best_val": best_val}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose sample-level DP class collapse")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--fedavg-config", default="configs/fedavg.yaml")
    ap.add_argument("--dp-sample-config", default="configs/dp_sample.yaml")
    ap.add_argument("--task", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--clip-norm", type=float, default=3.0)
    args = ap.parse_args()

    check_opacus_available()

    cfg = U.load_configs(args.splits_config, args.model_config, args.fedavg_config, args.dp_sample_config)
    task = args.task or cfg["experiment"]["task"]
    device = U.get_device(cfg["experiment"]["device"])
    subj_col = cfg["input"]["subject_col"]

    logger = U.setup_logging(f"08a_diagnose_{task}", cfg["output"]["logs_dir"])

    df = D.load_windows(cfg); D.validate_schema(df, cfg); df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)
    df = D.apply_normalisation(df, feats, cfg)

    splits = []
    for fid in args.folds:
        fold = D.load_fold(cfg, fid)
        train_idx = np.asarray(fold["indices"]["train"])
        sub = df.loc[train_idx, subj_col]
        splits.append({
            "label": f"fold_{fid:02d}",
            "indices": fold["indices"],
            "client_indices": {s: train_idx[(sub == s).to_numpy()].tolist() for s in fold["train_subjects"]},
        })

    logger.info("=" * 78)
    logger.info("DIAGNOSTIC: clipping only, zero noise, clip_norm={:.2f}".format(args.clip_norm))
    logger.info("Purpose: isolate whether CLIPPING (not noise) collapses stress/amusement")
    logger.info("=" * 78)

    for use_cw, label in [(True, "WITH class weighting"), (False, "WITHOUT class weighting")]:
        logger.info(f"\n--- {label} ---")
        all_flat = []
        for split in splits:
            for seed in args.seeds:
                r = run_one_diagnostic(df, feats, target, names, split, cfg, seed, device,
                                       args.clip_norm, use_cw, logger)
                pc = r["test"]["per_class"]
                logger.info(
                    f"  {split['label']} seed{seed} | macro_f1 {r['test']['macro_f1']:.4f} | "
                    + " | ".join(f"{c} {pc[c]['f1']:.3f}" for c in names)
                )
                all_flat.append(r["test"])

        mf = M.aggregate(all_flat, "macro_f1")
        logger.info(f"  == {label}: macro_f1 {mf['mean']:.4f} +/- {mf['std']:.4f} ==")
        for c in names:
            cf = M.aggregate([{"f1": t["per_class"][c]["f1"]} for t in all_flat], "f1")
            logger.info(f"     {c:12s} {cf['mean']:.4f} +/- {cf['std']:.4f}")

    logger.info("")
    logger.info("=" * 78)
    logger.info("READING THE RESULT:")
    logger.info("  stress/amusement F1 ~ 0 in BOTH conditions (with and without class")
    logger.info("    weighting) -> bug is NOT the class-weight/clip interaction;")
    logger.info("    look at the noise/accounting path or something else entirely.")
    logger.info("  stress/amusement F1 ~ 0 ONLY WITH class weighting, recovers WITHOUT")
    logger.info("    -> confirmed: class-weighted loss inflates rare-class gradients,")
    logger.info("    clipping then disproportionately destroys them. Fix: either drop")
    logger.info("    class weighting for the DP client loss, or raise clip_norm to")
    logger.info("    accommodate the weighted gradient scale, or clip pre-weighting.")
    logger.info("  stress/amusement F1 nonzero in BOTH -> clipping is not the cause at")
    logger.info("    all; the eps=16 result was likely a NOISE-path bug (check the")
    logger.info("    accountant's per-round injection, or the total_local_epochs math).")


if __name__ == "__main__":
    main()
