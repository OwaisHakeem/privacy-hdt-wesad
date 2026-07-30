#!/usr/bin/env python3
"""
=====================================================================
Stage D preparation - attack-slice splits for membership inference
Project : privacy_hdt_wesad
=====================================================================

WHY A SEPARATE SPLIT
--------------------
The membership inference attack must not be confounded by the generalisation
gap. A LOSO model fits the subjects it trained on better than the unseen test
subject regardless of any privacy leak, simply because it has seen their
physiology. Comparing member training loss against non-member loss therefore
measures generalisation, not membership.

The fix is to score every subject - member and non-member alike - on windows
the model never trained on. This requires withholding an ATTACK SLICE from each
member subject's training data. Both members (via their withheld slice) and the
non-member test subject (whose data was never trained on at all) are then
evaluated on unseen windows, and the only remaining difference between them is
whether the PERSON participated in training.

WHY STRATIFIED, NOT CHRONOLOGICAL
---------------------------------
WESAD runs its conditions in a fixed protocol order, so a subject's final 20%
of windows may contain no amusement at all - confirmed for 8 of 15 subjects.
A chronological attack slice would leave those subjects unattackable on the
class of greatest interest. The slice is therefore stratified: 20% of EACH
condition block is withheld, guaranteeing all four classes in both the training
remainder and the attack slice for every subject.

DOCUMENTED BIAS
---------------
Within a condition block, consecutive 10 s windows are autocorrelated, so a
member's withheld attack windows are temporally adjacent to windows that were
trained on. This slightly FAVOURS the attacker: the withheld windows resemble
trained-on ones, inflating any membership signal. The direction is deliberate
and safe - if even an attacker granted this advantage cannot beat chance at a
given epsilon, the conclusion that the formal budget overstates real risk is
strengthened, not weakened.

STRUCTURE PRODUCED
------------------
For each LOSO fold:
  members       = the 12 training subjects
    train slice = 80% of each condition block (used to train)
    attack slice= 20% of each condition block (never trained on; scored)
  non-member    = the 1 test subject (all windows; scored)
  validation    = the 2 validation subjects (unchanged; NOT attacked, since
                  they were used for model selection and are thus contaminated)

USAGE
-----
    python scripts/07a_build_attack_splits.py --config configs/splits.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import utils as U


def stratified_attack_slice(
    sdf: pd.DataFrame, label_col: str, order_col: str, frac: float, buffer: int, rng
) -> tuple[list, list]:
    """Withhold `frac` of each condition block for the attack, chronologically.

    Chronological WITHIN each block (not random) so the withheld slice is a
    contiguous tail of each condition, and a discard buffer separates the two
    parts to limit - though not eliminate - the autocorrelation bias noted in
    the module docstring. Returns (train_indices, attack_indices).
    """
    train_idx, attack_idx = [], []
    for _, block in sdf.groupby(label_col):
        block = block.sort_values(order_col)
        ids = block.index.to_numpy()
        n = len(ids)
        n_attack = max(1, int(round(n * frac)))
        n_train = n - n_attack
        # train is the head, attack is the tail, buffer discarded between
        train_end = max(0, n_train - buffer)
        train_idx.extend(ids[:train_end].tolist())
        attack_idx.extend(ids[n_train:].tolist())
    return train_idx, attack_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage D - build attack-slice splits")
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--attack-frac", type=float, default=0.2)
    ap.add_argument("--buffer", type=int, default=2)
    args = ap.parse_args()

    cfg = U.load_config(args.config)
    subj_col = cfg["input"]["subject_col"]
    label_col = cfg["input"]["label_col"]
    order_col = cfg["input"]["order_col"]

    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)

    out_dir = Path(cfg["output"]["splits_dir"]) / "attack"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["reproducibility"]["split_seed"])
    class_names = cfg["task"]["multiclass"]["class_names"]

    summary_rows = []
    for fid in D.list_folds(cfg):
        fold = D.load_fold(cfg, fid)
        members = fold["train_subjects"]
        test_subj = fold["test_subjects"][0]
        val_subjs = fold["val_subjects"]

        member_train, member_attack, per_member = {}, {}, {}
        for s in members:
            sdf = df[df[subj_col] == s]
            tr, atk = stratified_attack_slice(
                sdf, label_col, order_col, args.attack_frac, args.buffer, rng
            )
            member_train[s] = tr
            member_attack[s] = atk
            per_member[s] = {"n_train": len(tr), "n_attack": len(atk)}

        test_idx = df.index[df[subj_col] == test_subj].to_numpy().tolist()
        val_idx = df.index[df[subj_col].isin(val_subjs)].to_numpy().tolist()

        # Verify: no window is both trained on and used as an attack point.
        all_train = set()
        for tr in member_train.values():
            all_train |= set(tr)
        all_attack = set()
        for atk in member_attack.values():
            all_attack |= set(atk)
        if all_train & all_attack:
            sys.exit(f"fold {fid}: attack slice overlaps training slice")
        if all_train & set(test_idx):
            sys.exit(f"fold {fid}: test subject reachable in training")

        # Confirm every member has all four classes in BOTH partitions.
        for s in members:
            for part, idx in [("train", member_train[s]), ("attack", member_attack[s])]:
                present = set(df.loc[idx, "y_multiclass"].unique().tolist())
                if len(present) < len(class_names):
                    missing = [class_names[c] for c in range(len(class_names)) if c not in present]
                    print(f"  WARNING fold {fid} {s} {part}: missing {missing}")

        record = {
            "fold_id": fid,
            "test_subject": test_subj,
            "val_subjects": val_subjs,
            "members": members,
            "attack_frac": args.attack_frac,
            "buffer": args.buffer,
            "member_train_indices": member_train,
            "member_attack_indices": member_attack,
            "test_indices": test_idx,
            "val_indices": val_idx,
            "n_member_train": sum(len(v) for v in member_train.values()),
            "n_member_attack": sum(len(v) for v in member_attack.values()),
            "n_test": len(test_idx),
            "per_member": per_member,
        }
        with open(out_dir / f"fold_{fid:02d}.json", "w") as fh:
            json.dump(record, fh, indent=2)

        summary_rows.append({
            "fold": fid,
            "test_subject": test_subj,
            "n_members": len(members),
            "member_train": record["n_member_train"],
            "member_attack": record["n_member_attack"],
            "test_windows": record["n_test"],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "attack_split_summary.csv", index=False)
    print(f"\nWrote {len(summary_rows)} attack-slice folds to {out_dir}")
    print(summary.to_string(index=False))
    print("\nEach member contributes a stratified 80% train / 20% attack split;")
    print("the test subject is the clean non-member. All four classes are present")
    print("in every member's attack slice (warnings above if not).")


if __name__ == "__main__":
    main()
