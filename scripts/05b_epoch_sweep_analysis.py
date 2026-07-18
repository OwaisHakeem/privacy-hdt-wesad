#!/usr/bin/env python3
"""
=====================================================================
Stage B - Local-epoch sensitivity analysis
Project : privacy_hdt_wesad
=====================================================================

WHY THIS EXISTS
---------------
Stage B's headline claim under the within-subject protocol is that FedAvg is
dominated by local-only training. That claim is only admissible if it holds
across a reasonable range of the local-epoch parameter E.

The concern is concrete rather than hypothetical. At E = 5, each client holds
roughly 170 training windows and the logged local loss falls to 0.04-0.10
while validation accuracy plateaus: the clients are overfitting their own
data before aggregation, and averaging then discards what little personal
signal survived. If E = 1 recovers the deficit, the finding concerns a
hyper-parameter and not federation, and the manuscript would be asserting
something the data do not support.

A reviewer will ask this. It is cheaper to answer it now.

USAGE
-----
    python scripts/05b_epoch_sweep_analysis.py
    python scripts/05b_epoch_sweep_analysis.py --protocol loso
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def per_subject_scores(pattern: str, metric: str = "macro_f1") -> dict:
    """Per-subject means over seeds, from files carrying test_per_subject."""
    d = defaultdict(list)
    for f in glob.glob(pattern):
        r = json.load(open(f))
        if not r.get("test_per_subject"):
            continue
        for s, m in r["test_per_subject"].items():
            v = m[metric] if metric in m else m["per_class"][metric]["f1"]
            d[s].append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}


def local_only_scores(base: Path, task: str, metric: str = "macro_f1") -> dict:
    d = defaultdict(list)
    for f in glob.glob(str(base / "within_subject" / "local_only" / task / "S*_seed*.json")):
        r = json.load(open(f))
        v = (
            r["test"][metric]
            if metric in r["test"]
            else r["test"]["per_class"][metric]["f1"]
        )
        d[r["subject"]].append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}


def fold_scores(pattern: str, metric: str = "macro_f1") -> dict:
    d = defaultdict(list)
    for f in glob.glob(pattern):
        r = json.load(open(f))
        d[r["split"]].append(r["test"][metric])
    return {k: float(np.mean(v)) for k, v in d.items()}


def discover(base: Path, protocol: str, task: str) -> dict:
    """Locate the default run and every tagged sweep run."""
    root = base / protocol / "fedavg" / task
    runs = {}
    default = sorted(root.glob("*_seed*.json"))
    if default:
        e = json.load(open(default[0])).get("local_epochs", 5)
        runs[f"E={e} (default)"] = str(root / "*_seed*.json")
    for sub in sorted(p for p in root.glob("*") if p.is_dir()):
        files = sorted(sub.glob("*_seed*.json"))
        if not files:
            continue
        r = json.load(open(files[0]))
        label = f"E={r.get('local_epochs','?')}"
        if r.get("local_optimizer") not in (None, "adam"):
            label += f" ({r['local_optimizer']})"
        runs[label] = str(sub / "*_seed*.json")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B - local-epoch sensitivity")
    ap.add_argument("--results", default="results")
    ap.add_argument("--protocol", choices=["within_subject", "loso"], default="within_subject")
    ap.add_argument("--task", default="multiclass")
    args = ap.parse_args()

    base = Path(args.results)
    runs = discover(base, args.protocol, args.task)
    if not runs:
        raise SystemExit(f"No FedAvg results found under {base / args.protocol / 'fedavg' / args.task}")

    print(f"\nStage B - local-epoch sensitivity | {args.protocol} | {args.task}")
    print("=" * 78)

    if args.protocol == "within_subject":
        lo = local_only_scores(base, args.task)
        lo_am = local_only_scores(base, args.task, "amusement")
        subs = sorted(lo)
        L = np.array([lo[s] for s in subs])
        L_am = np.array([lo_am[s] for s in subs])

        print(f"\n{'condition':22s} {'macro_f1':>10s} {'amusement':>10s} {'vs local':>10s} {'wins':>7s} {'p':>9s}")
        print("-" * 78)
        print(f"{'local-only':22s} {L.mean():10.4f} {L_am.mean():10.4f} {'-':>10s} {'-':>7s} {'-':>9s}")

        rows = []
        for label, pattern in runs.items():
            fa = per_subject_scores(pattern)
            fa_am = per_subject_scores(pattern, "amusement")
            if not fa:
                continue
            F = np.array([fa[s] for s in subs])
            F_am = np.array([fa_am[s] for s in subs])
            diff = F - L
            if wilcoxon is not None and np.any(diff != 0):
                _, p = wilcoxon(F, L)
            else:
                p = float("nan")
            print(
                f"{label:22s} {F.mean():10.4f} {F_am.mean():10.4f} "
                f"{diff.mean():+10.4f} {int((F > L).sum()):4d}/15 {p:9.4f}"
            )
            rows.append((label, F.mean(), diff.mean(), p, int((F > L).sum())))

        print("\nReading:")
        print("  'vs local' > 0 in any row  -> FedAvg beats local-only at that E;")
        print("                                the dominance claim does not hold.")
        print("  all rows < 0               -> dominance holds across the swept range;")
        print("                                the claim is about federation, not E.")
        if rows:
            best = max(rows, key=lambda r: r[1])
            print(f"\n  Best FedAvg configuration: {best[0]} at macro_f1 {best[1]:.4f} "
                  f"({best[2]:+.4f} vs local-only 0.9088 baseline of {L.mean():.4f})")
            if best[2] > 0:
                print("  -> RETRACT the dominance claim; report this configuration instead.")
            else:
                print("  -> Dominance holds at every E tested. State the swept range explicitly.")

    else:
        print(f"\n{'condition':22s} {'macro_f1':>10s} {'vs centralised':>16s} {'wins':>7s} {'p':>9s}")
        print("-" * 78)
        ce = fold_scores(str(base / "loso" / "centralised" / args.task / "fold_*_seed*.json"))
        folds = sorted(ce)
        C = np.array([ce[k] for k in folds])
        print(f"{'centralised':22s} {C.mean():10.4f} {'-':>16s} {'-':>7s} {'-':>9s}")

        for label, pattern in runs.items():
            fa = fold_scores(pattern)
            if not fa:
                continue
            F = np.array([fa[k] for k in folds if k in fa])
            if len(F) != len(C):
                continue
            diff = F - C
            _, p = wilcoxon(F, C) if wilcoxon is not None else (None, float("nan"))
            print(
                f"{label:22s} {F.mean():10.4f} {diff.mean():+16.4f} "
                f"{int((F > C).sum()):4d}/15 {p:9.4f}"
            )

    print("\nNote: p-values here are uncorrected and this sweep constitutes multiple")
    print("      comparisons. Treat them as descriptive; the decision rests on whether")
    print("      any configuration reverses the sign, not on significance.\n")


if __name__ == "__main__":
    main()
