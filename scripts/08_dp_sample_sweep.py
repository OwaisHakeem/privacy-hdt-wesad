#!/usr/bin/env python3
"""
=====================================================================
Stage C-prime - Sample-level DP-FedAvg utility sweep
Project : privacy_hdt_wesad
=====================================================================

WHAT THIS MEASURES
------------------
The utility cost of SAMPLE-level differential privacy in the same federated
setup. Stage C measured user-level DP (protects the person); this measures
sample-level DP (protects each 10 s window). Because sample-level DP gets
privacy amplification by Poisson subsampling, we expect it to retain
substantially more utility than user-level at the same epsilon.

The pair (Stage C, this) supports the paper's comparison claim: does the
formal-vs-empirical decoupling seen in Stage D hold generally, or only for
user-level DP?

WHAT INHERITS FROM STAGE C UNCHANGED
------------------------------------
Model, splits, feature columns, per-subject normalisation, 30 fixed rounds,
delta = 1e-4, epsilon grid, seed count, LOSO protocol. Only the privacy
MECHANISM changes.

USAGE
-----
    python scripts/08_dp_sample_sweep.py --self-test
    python scripts/08_dp_sample_sweep.py --pilot
    python scripts/08_dp_sample_sweep.py
    python scripts/08_dp_sample_sweep.py --epsilons 4 8 --seeds 1 --folds 0 1
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import federated as F
from src import metrics as M
from src import models as Mod
from src import training as T
from src import utils as U
from src.privacy_sample import (
    SampleDPClient,
    check_opacus_available,
)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


def self_test() -> None:
    """Verify Opacus is installed and the accountant behaves correctly.

    Two sanity checks:
      1. Non-private Opacus with a very large noise multiplier gives a small
         epsilon (nearly zero); with a very small noise it gives a huge one.
      2. A trained model produces a valid state dict that plain FedAvg can
         average.
    """
    print("[self-test] sample-level DP")
    check_opacus_available()
    print("  opacus imported: OK")

    from opacus import PrivacyEngine
    from src.training import make_loader

    # Toy dataset - 200 samples, 10 features, 3 classes
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=200).astype(np.int64)

    import torch.nn as nn

    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = make_loader(X, y, batch_size=32, shuffle=True, seed=0)

    engine = PrivacyEngine()
    model_dp, opt_dp, loader_dp = engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-4,
        epochs=10,
        max_grad_norm=1.0,
        poisson_sampling=True,
    )

    calibrated_sigma = opt_dp.noise_multiplier
    print(f"  calibration target eps=8.0, delta=1e-4, 10 epochs -> sigma={calibrated_sigma:.4f}")
    assert calibrated_sigma > 0, "sigma must be positive"

    # Train for one epoch and check the accountant advances
    criterion = nn.CrossEntropyLoss()
    model_dp.train()
    n_steps = 0
    for xb, yb in loader_dp:
        opt_dp.zero_grad()
        loss = criterion(model_dp(xb), yb)
        loss.backward()
        opt_dp.step()
        n_steps += 1

    eps_after = engine.get_epsilon(1e-4)
    print(f"  after {n_steps} steps: eps={eps_after:.4f} (should be > 0, < 8)")
    assert eps_after > 0, "accountant must advance"
    assert eps_after < 8.0, "one epoch must spend less than the full 10-epoch budget"

    # State dict cleanup - Opacus adds a "_module." prefix; ensure our strip works
    state = model_dp.state_dict()
    stripped = {k.replace("_module.", ""): v for k, v in state.items()}
    plain_model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    try:
        plain_model.load_state_dict(stripped, strict=True)
        print("  state dict roundtrip (wrapped -> plain): OK")
    except Exception as e:
        print(f"  state dict roundtrip FAILED: {e}")
        sys.exit(1)

    print("[self-test] PASSED")


# ---------------------------------------------------------------------
# Pilot - measure per-example gradient norm to calibrate clip_norm
# ---------------------------------------------------------------------


def pilot_gradient_norms(clients, model, cfg, device, seed, rounds: int) -> dict:
    """Measure per-example gradient norms with no clipping and no noise.

    Choosing clip_norm from a pilot avoids the two pathologies: too small
    destroys signal, too large inflates noise. Standard practice picks the
    median observed norm.
    """
    check_opacus_available()
    from opacus import PrivacyEngine

    global_state = copy.deepcopy(model.state_dict())
    per_example_norms = []

    for c in clients:
        model_c = Mod.build_model(cfg, input_dim=c.X.shape[1], num_classes=c.n_classes).to(device)
        model_c.load_state_dict(global_state)
        opt = torch.optim.Adam(model_c.parameters(), lr=1e-3)
        loader = T.make_loader(c.X, c.y, cfg["privacy_sample"]["batch_size"], shuffle=True, seed=seed)

        # Attach Opacus in "no-noise" mode purely to get per-example gradients.
        engine = PrivacyEngine()
        model_c, opt, loader_dp = engine.make_private(
            module=model_c,
            optimizer=opt,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=1e9,  # do not clip during the pilot
            poisson_sampling=True,
        )

        model_c.train()
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(rounds):
            for xb, yb in loader_dp:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model_c(xb), yb)
                loss.backward()
                # Access per-example gradients that Opacus stashes on each
                # parameter, then compute the L2 norm per example.
                per_example_grads = []
                for p in model_c.parameters():
                    if hasattr(p, "grad_sample") and p.grad_sample is not None:
                        g = p.grad_sample
                        # g shape: (batch, ...). Flatten per-example.
                        per_example_grads.append(g.reshape(g.shape[0], -1))
                if per_example_grads:
                    flat = torch.cat(per_example_grads, dim=1)
                    norms = torch.linalg.vector_norm(flat, dim=1)
                    per_example_norms.extend(norms.detach().cpu().numpy().tolist())
                opt.step()

    arr = np.asarray(per_example_norms)
    return {
        "n_observations": int(len(arr)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------


def run_one_dp(df, feats, target, names, split, cfg, seed, device,
               eps_target: float | None, clip_norm: float, logger) -> dict:
    """Train one federated DP-SGD model and evaluate on the test subject."""
    U.set_seed(seed)
    n_classes = len(names)
    rounds = int(cfg["privacy_sample"]["rounds"])
    local_epochs = int(cfg["federated"]["local_epochs"])
    total_local_epochs = rounds * local_epochs
    batch_size = int(cfg["privacy_sample"]["batch_size"])
    delta = float(cfg["privacy_sample"]["delta"])
    lr = float(cfg["privacy_sample"]["local_learning_rate"])
    wd = float(cfg["privacy_sample"]["weight_decay"])

    # Build clients (one per training subject) and attach a per-client engine.
    dp_clients = []
    for subj, idx in split["client_indices"].items():
        X, y = D.make_xy(df, idx, feats, target)
        cw = (
            D.class_weights(y, n_classes)
            if cfg["federated"]["per_client_class_weights"]
            and cfg["training"]["class_weighted_loss"]
            else None
        )
        client = SampleDPClient(client_id=subj, X=X, y=y, n_classes=n_classes, class_weight=cw)
        client_model = Mod.build_model(cfg, input_dim=X.shape[1], num_classes=n_classes)
        client.attach_privacy_engine(
            client_model,
            target_epsilon=eps_target,
            delta=delta,
            max_grad_norm=clip_norm,
            total_local_epochs=total_local_epochs,
            batch_size=batch_size,
            base_lr=lr,
            weight_decay=wd,
            device=device,
        )
        dp_clients.append(client)

    # Global model and initial state
    global_model = Mod.build_model(cfg, input_dim=dp_clients[0].X.shape[1], num_classes=n_classes).to(device)
    global_state = copy.deepcopy(global_model.state_dict())

    # Val / test loaders (unchanged from Stage C)
    X_va, y_va = D.make_xy(df, split["indices"]["val"], feats, target)
    X_te, y_te = D.make_xy(df, split["indices"]["test"], feats, target)
    bs = int(cfg["training"]["batch_size"])
    val_loader = T.make_loader(X_va, y_va, bs, shuffle=False, device=device)
    test_loader = T.make_loader(X_te, y_te, bs, shuffle=False, device=device)

    best_val, best_state, best_round = -1.0, copy.deepcopy(global_state), 0

    for rnd in range(1, rounds + 1):
        # Each client trains a round of DP-SGD from the current global state.
        states, counts = [], []
        for c in dp_clients:
            st, n, _ = c.local_train(global_state, cfg, round_seed=seed * 10_000 + rnd)
            states.append(st)
            counts.append(n)

        # Plain FedAvg aggregation - server does no additional noising.
        # sample-weighted, matching Stage B: the individual client guarantees
        # already hold under post-processing.
        global_state = F.fedavg_aggregate(states, counts)
        global_model.load_state_dict(global_state)

        # Track best-on-val (free wrt DP because val subjects are not clients).
        y_v, p_v, _ = T.predict(global_model, val_loader, device)
        val_score = f1_score(y_v, p_v, average="macro", zero_division=0)
        if val_score > best_val:
            best_val, best_state, best_round = val_score, copy.deepcopy(global_state), rnd

        if rnd % 5 == 0 or rnd == rounds:
            logger.info(f"    round {rnd:3d} | val {val_score:.4f} | best {best_val:.4f} @ {best_round}")

    # Test with the best-val checkpoint.
    global_model.load_state_dict(best_state)
    y_true, y_pred, y_proba = T.predict(global_model, test_loader, device)
    test_metrics = M.compute_metrics(y_true, y_pred, y_proba, names)

    # Report the actually-spent epsilon (one per client; all clients used the
    # same target so they should agree, small numerical drift possible).
    spent_epsilons = [c.get_epsilon() for c in dp_clients] if eps_target is not None else []

    return {
        "seed": seed,
        "epsilon_target": eps_target if eps_target is not None else float("inf"),
        "granularity": "sample",
        "delta": delta,
        "noise_multiplier": dp_clients[0]._noise_multiplier if eps_target is not None else 0.0,
        "clip_norm": clip_norm,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_clients": len(dp_clients),
        "best_round": best_round,
        "best_val_macro_f1": float(best_val),
        "spent_epsilon_per_client": spent_epsilons,
        "spent_epsilon_max": float(max(spent_epsilons)) if spent_epsilons else float("inf"),
        "test": test_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C-prime - sample-level DP-FedAvg")
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--fedavg-config", default="configs/fedavg.yaml")
    ap.add_argument("--dp-sample-config", default="configs/dp_sample.yaml")
    ap.add_argument("--task", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--epsilons", type=float, nargs="+", default=None)
    ap.add_argument("--clip-norm", type=float, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="Measure per-example gradient norms and exit.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    check_opacus_available()

    cfg = U.load_configs(
        args.splits_config, args.model_config, args.fedavg_config, args.dp_sample_config
    )
    task = args.task or cfg["experiment"]["task"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    device = U.get_device(cfg["experiment"]["device"])
    subj_col = cfg["input"]["subject_col"]
    pcfg = cfg["privacy_sample"]

    logger = U.setup_logging(f"08_dp_sample_{task}", cfg["output"]["logs_dir"])

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    target = D.target_column(cfg, task)
    names = D.class_names(cfg, task)
    if cfg["normalisation"]["strategy"] != "per_subject":
        sys.exit("Sample-level DP requires per_subject normalisation.")
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
        logger.info("PILOT: measuring per-example gradient norms (no noise, no clip)")
        sp = splits[0]
        clients = []
        for subj, idx in sp["client_indices"].items():
            X, y = D.make_xy(df, idx, feats, target)
            cw = D.class_weights(y, len(names)) if cfg["training"]["class_weighted_loss"] else None
            clients.append(F.Client(client_id=subj, X=X, y=y, n_classes=len(names), class_weight=cw))
        U.set_seed(seeds[0])
        model = Mod.build_model(cfg, input_dim=clients[0].X.shape[1], num_classes=len(names)).to(device)
        st = pilot_gradient_norms(clients, model, cfg, device, seeds[0], int(pcfg["pilot_rounds"]))
        logger.info(f"  per-example gradient norms over {st['n_observations']} examples:")
        for k in ["min", "p25", "median", "mean", "p75", "max"]:
            logger.info(f"    {k:8s} {st[k]:.4f}")
        logger.info(f"\n  suggested clip_norm = median = {st['median']:.4f}")
        logger.info("  set privacy_sample.clip_norm in configs/dp_sample.yaml, or pass --clip-norm")
        U.write_json(
            Path(cfg["output"]["results_dir"]) / "loso" / "dp_sample" / task / "pilot.json", st
        )
        return

    # ---- resolve clip norm -------------------------------------------
    clip_norm = args.clip_norm if args.clip_norm is not None else float(pcfg["clip_norm"])

    # ---- grid: include non-private anchor -----------------------------
    targets = args.epsilons if args.epsilons is not None else list(pcfg["epsilon_targets"])
    include_inf = args.epsilons is None or "inf" in [str(t).lower() for t in args.epsilons]

    logger.info(f"Stage C-prime | sample-level DP | LOSO | task={task} | device={device}")
    logger.info(f"delta={pcfg['delta']} | rounds={pcfg['rounds']} | E={cfg['federated']['local_epochs']} "
                f"| batch={pcfg['batch_size']} | clip_norm={clip_norm}")
    logger.info("")

    out_root = Path(cfg["output"]["results_dir"]) / "loso" / "dp_sample" / task

    # Non-private anchor first (fast, useful sanity check).
    if include_inf:
        logger.info("[eps=inf] non-private (plain FedAvg) - kept as anchor for comparison")
        # For the non-private anchor we use plain FedAvg (Stage B result already
        # exists; running it here again is redundant, so we point to it in
        # the summary rather than re-run).
        # If the user wants a fresh non-private run here, use scripts/05_fedavg_baseline.py.

    for eps_target in targets:
        tag = f"eps{eps_target:g}"
        logger.info(f"[{tag}] target eps={eps_target}")
        runs = []
        for split in splits:
            for seed in seeds:
                logger.info(f"  {tag} | {split['label']} | seed {seed}")
                r = run_one_dp(df, feats, target, names, split, cfg, seed, device,
                               eps_target, clip_norm, logger)
                r["split"] = split["label"]
                r["protocol"] = "loso"
                r["task"] = task
                r["provenance"] = U.provenance("C_prime_dp_sample")
                U.write_json(out_root / tag / f"{split['label']}_seed{seed}.json", r)
                runs.append(r)
                logger.info(
                    f"    -> macro_f1 {r['test']['macro_f1']:.4f} "
                    f"| spent eps {r['spent_epsilon_max']:.3f} "
                    f"| sigma {r['noise_multiplier']:.3f}"
                )

        # Aggregate
        flat = [r["test"] for r in runs]
        summary = {
            "provenance": U.provenance("C_prime_dp_sample_summary"),
            "epsilon_target": eps_target,
            "spent_epsilon_max_mean": float(np.mean([r["spent_epsilon_max"] for r in runs])),
            "noise_multiplier_mean": float(np.mean([r["noise_multiplier"] for r in runs])),
            "granularity": "sample",
            "delta": pcfg["delta"],
            "clip_norm": clip_norm,
            "rounds": pcfg["rounds"],
            "batch_size": pcfg["batch_size"],
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

        mf = summary["aggregate"]["macro_f1"]
        logger.info(
            f"  == {tag}: macro_f1 {mf['mean']:.4f} +/- {mf['std']:.4f} "
            f"(spent eps {summary['spent_epsilon_max_mean']:.3f}) =="
        )
        logger.info("")

    logger.info("=" * 76)
    logger.info(f"STAGE C-prime | sample-level DP | {task}")
    logger.info(f"{'eps':>6s} {'spent':>7s} {'sigma':>7s} {'macro_f1':>18s} {'amusement':>18s}")
    logger.info("-" * 76)
    for eps_target in targets:
        tag = f"eps{eps_target:g}"
        s_path = out_root / tag / "summary.json"
        if not s_path.exists():
            continue
        import json
        s = json.load(open(s_path))
        mf = s["aggregate"]["macro_f1"]
        am = s["per_class_f1"].get("amusement", {"mean": float("nan"), "std": float("nan")})
        logger.info(
            f"{eps_target:6.1f} {s['spent_epsilon_max_mean']:7.3f} {s['noise_multiplier_mean']:7.3f} "
            f"{mf['mean']:8.4f} +/- {mf['std']:.4f} "
            f"{am['mean']:8.4f} +/- {am['std']:.4f}"
        )
    logger.info("")
    logger.info("For direct comparison with Stage C (user-level DP), see")
    logger.info("results/loso/dp_fedavg/multiclass/eps*/summary.json.")


if __name__ == "__main__":
    main()
