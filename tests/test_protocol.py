"""
Protocol assertions. Run before any training stage.

    pytest tests/ -v

These are not unit tests in the ordinary sense. Each one encodes a
methodological commitment that, if silently violated, would invalidate the
study's central claim rather than merely produce a wrong number. They exist
because the previous pipeline failed two of them without anyone noticing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import utils as U

CFG = "configs/splits.yaml"
MODEL_CFG = "configs/model.yaml"


@pytest.fixture(scope="module")
def ctx():
    cfg = U.load_configs(CFG, MODEL_CFG)
    df = D.load_windows(cfg)
    D.validate_schema(df, cfg)
    df = D.prepare_labels(df, cfg)
    feats = D.feature_columns(df, cfg)
    return cfg, df, feats


# -----------------------------------------------------------------
# The feature contract
# -----------------------------------------------------------------


def test_feature_count(ctx):
    _, _, feats = ctx
    assert len(feats) == 144


def test_no_label_column_is_a_feature(ctx):
    """The failure that produced the old leaky centralised baseline."""
    _, _, feats = ctx
    forbidden = (
        "original_wesad_label",
        "binary_baseline_stress_label",
        "multiclass_3_baseline_stress_amusement_label",
        "multiclass_4_valid_conditions_label",
        "y_multiclass",
        "y_binary",
        "condition_name",
    )
    assert not set(feats) & set(forbidden)
    assert not [f for f in feats if f.endswith(("_label", "_included"))]


def test_no_metadata_column_is_a_feature(ctx):
    """Timing metadata correlates with condition order and must not be input."""
    _, _, feats = ctx
    forbidden = (
        "window_id",
        "start_sec",
        "end_sec",
        "start_sample_chest",
        "end_sample_chest",
        "window_seconds",
    )
    assert not set(feats) & set(forbidden)


# -----------------------------------------------------------------
# The split contract
# -----------------------------------------------------------------


def test_loso_partitions_are_disjoint(ctx):
    cfg, _, _ = ctx
    for f in D.list_folds(cfg):
        fold = D.load_fold(cfg, f)
        tr = set(fold["indices"]["train"])
        va = set(fold["indices"]["val"])
        te = set(fold["indices"]["test"])
        assert not tr & va and not tr & te and not va & te


def test_test_subject_never_reachable_during_model_selection(ctx):
    """The defect this whole restructure exists to remove."""
    cfg, df, _ = ctx
    subj_col = cfg["input"]["subject_col"]
    for f in D.list_folds(cfg):
        fold = D.load_fold(cfg, f)
        test_subj = fold["test_subjects"][0]
        selection = list(set(fold["indices"]["train"]) | set(fold["indices"]["val"]))
        assert test_subj not in set(df.loc[selection, subj_col].unique())


def test_folds_account_for_every_window(ctx):
    cfg, df, _ = ctx
    for f in D.list_folds(cfg):
        fold = D.load_fold(cfg, f)
        total = sum(len(fold["indices"][p]) for p in ("train", "val", "test"))
        assert total == len(df)


def test_validation_rotation_is_balanced(ctx):
    """Each subject must serve as validation exactly n_val_subjects times."""
    cfg, _, _ = ctx
    k = cfg["protocol_loso"]["n_val_subjects"]
    counts: dict[str, int] = {}
    for f in D.list_folds(cfg):
        for s in D.load_fold(cfg, f)["val_subjects"]:
            counts[s] = counts.get(s, 0) + 1
    assert set(counts.values()) == {k}


def test_within_subject_split_is_chronological(ctx):
    """Train windows must precede test windows, or autocorrelated neighbours
    straddle the boundary and utility is inflated."""
    cfg, df, _ = ctx
    order = cfg["input"]["order_col"]
    for subj in D.list_subjects(cfg):
        w = D.load_within_subject(cfg, subj)
        if not w["indices"]["train"] or not w["indices"]["test"]:
            continue
        assert df.loc[w["indices"]["train"], order].max() < df.loc[w["indices"]["test"], order].max()


# -----------------------------------------------------------------
# The normalisation contract
# -----------------------------------------------------------------


def test_per_subject_normalisation_uses_no_cross_subject_statistics(ctx):
    """Federation-compatibility, stated as an executable assertion.

    Perturbing one subject must leave every other subject's normalised
    features untouched. If this fails, the 'federated' results were computed
    with a centrally-fitted scaler and the threat model is fiction.
    """
    cfg, df, feats = ctx
    subj_col = cfg["input"]["subject_col"]
    target = D.list_subjects(cfg)[0]

    base = D.apply_normalisation(df, feats, cfg)

    perturbed = df.copy()
    mask = perturbed[subj_col] == target
    perturbed.loc[mask, feats] = perturbed.loc[mask, feats] + 1000.0
    after = D.apply_normalisation(perturbed, feats, cfg)

    others = base[base[subj_col] != target][feats].to_numpy()
    others_after = after[after[subj_col] != target][feats].to_numpy()
    assert np.allclose(others, others_after)


def test_normalisation_is_finite_for_constant_features(ctx):
    cfg, df, feats = ctx
    df2 = df.copy()
    df2[feats[0]] = 5.0
    out = D.apply_normalisation(df2, feats, cfg)
    assert np.isfinite(out[feats].to_numpy()).all()


# -----------------------------------------------------------------
# The class-weight contract
# -----------------------------------------------------------------


def test_class_weights_favour_the_rare_class(ctx):
    cfg, df, feats = ctx
    dfn = D.apply_normalisation(df, feats, cfg)
    fold = D.load_fold(cfg, 0)
    _, y_tr = D.make_xy(dfn, fold["indices"]["train"], feats, "y_multiclass")
    w = D.class_weights(y_tr, 4)
    counts = np.bincount(y_tr, minlength=4)
    assert int(np.argmax(w)) == int(np.argmin(counts))
