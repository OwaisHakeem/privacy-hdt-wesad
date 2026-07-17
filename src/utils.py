"""
Shared utilities: configuration, seeding, logging, result serialisation.

Every stage imports from here so that seeding and result provenance are
implemented once rather than re-derived per script.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs(*paths: str | Path) -> dict:
    """Shallow-merge several config files, later files winning.

    Used so that a stage can read splits.yaml (which owns the data contract
    and the normalisation policy) alongside model.yaml (which owns the
    training contract) without either duplicating the other.
    """
    merged: dict = {}
    for p in paths:
        cfg = load_config(p)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
    return merged


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness this project touches.

    Note the division of labour: the seed governs model initialisation,
    batch shuffling and (later) DP noise. It does NOT govern the data
    partition, which is fixed by stage A2. Reported variance across seeds
    therefore isolates training stochasticity from partition luck.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------


def git_revision() -> str | None:
    """Record the exact code state that produced a result, where available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance(stage: str, extra: dict | None = None) -> dict:
    p = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
    }
    if extra:
        p.update(extra)
    return p


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_NpEncoder)


def setup_logging(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh_ = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh_.setFormatter(fmt)
        logger.addHandler(fh_)

    return logger
