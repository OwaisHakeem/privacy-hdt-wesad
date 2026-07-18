"""
Federated averaging: clients, aggregation, and the round loop.

ARCHITECTURAL COMMITMENT
------------------------
One subject = one client = one Human Digital Twin. A client's raw windows
never leave it; only model state crosses the wire. This is not a simulation
convenience but the object the study is about, and every quantity measured
later (communication at stage F, twin disclosure at stage E) is defined
against it.

WHAT THIS MODULE REFUSES TO DO
------------------------------
The previous implementation evaluated the global model on the TEST set each
round and used that number to stop training, to select the returned state,
and to report the headline. run_federated() is never given the test data;
the caller evaluates it once, afterwards.

Nor does any client see another client's data, or a scaler fitted across
clients. Normalisation was applied per subject at stage A2's policy, so no
cross-client statistic exists to leak.

NON-IID BY CONSTRUCTION
-----------------------
Clients are people, so the data are non-IID in the way that matters
clinically: each client's feature distribution reflects an individual's
physiology, and class proportions differ between them. No artificial
partitioning is imposed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import training as T


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------


@dataclass
class Client:
    """One subject's local training state.

    class_weight is derived from THIS client's own labels. A client cannot
    see the global class distribution, so deriving weights globally would
    quietly grant every client knowledge it does not possess and would make
    the federated result unreproducible in deployment.
    """

    client_id: str
    X: np.ndarray
    y: np.ndarray
    n_classes: int
    class_weight: np.ndarray | None = None
    _loader: DataLoader | None = field(default=None, repr=False)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def loader(self, batch_size: int, seed: int) -> DataLoader:
        # Rebuilt per round with a round-dependent seed so that batch order
        # varies across rounds rather than repeating identically.
        return T.make_loader(self.X, self.y, batch_size, shuffle=True, seed=seed)

    def local_train(
        self,
        global_state: dict,
        model: nn.Module,
        cfg: dict,
        device,
        round_seed: int,
    ) -> tuple[dict, int, float]:
        """Run E local epochs starting from the global state."""
        model.load_state_dict(global_state)
        model.train()

        t = cfg["training"]
        f = cfg["federated"]

        weight = (
            torch.from_numpy(self.class_weight).to(device)
            if self.class_weight is not None
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        opt_name = f.get("local_optimizer", "adam").lower()
        if opt_name == "sgd":
            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=float(f.get("local_learning_rate", t["learning_rate"])),
                momentum=float(f.get("local_momentum", 0.9)),
                weight_decay=float(t["weight_decay"]),
            )
        else:
            optimiser = torch.optim.Adam(
                model.parameters(),
                lr=float(f.get("local_learning_rate", t["learning_rate"])),
                weight_decay=float(t["weight_decay"]),
            )

        loader = self.loader(int(t["batch_size"]), round_seed)
        total, seen = 0.0, 0
        for _ in range(int(f["local_epochs"])):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(xb)
                seen += len(xb)

        return (
            copy.deepcopy(model.state_dict()),
            self.n_samples,
            total / max(seen, 1),
        )


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def fedavg_aggregate(states: list[dict], n_samples: list[int]) -> dict:
    """Weighted parameter average, weights proportional to client sample count.

    This is FedAvg as specified by McMahan et al. (2017): each client's
    contribution is scaled by how much data it holds. Weighting equally
    instead would be a different algorithm and would misreport the effect of
    the cohort's uneven recording lengths (WESAD subjects contribute 284-302
    windows), so the weighting is explicit rather than incidental.
    """
    if not states:
        raise ValueError("No client states to aggregate.")
    if len(states) != len(n_samples):
        raise ValueError("states and n_samples must be the same length.")

    total = float(sum(n_samples))
    if total <= 0:
        raise ValueError("Total sample count must be positive.")

    out: dict = {}
    for key in states[0]:
        ref = states[0][key]
        if not torch.is_floating_point(ref):
            # Integer buffers (e.g. counters) are not averaged; carry the
            # first client's value rather than producing a meaningless mean.
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for st, n in zip(states, n_samples):
            acc += st[key].to(torch.float64) * (float(n) / total)
        out[key] = acc.to(ref.dtype)
    return out


def select_clients(clients: list[Client], fraction: float, rng: np.random.Generator) -> list[Client]:
    """Sample the participating subset for one round."""
    if fraction >= 1.0:
        return list(clients)
    k = max(1, int(round(fraction * len(clients))))
    idx = rng.choice(len(clients), size=k, replace=False)
    return [clients[i] for i in sorted(idx)]


# ---------------------------------------------------------------------
# Round loop
# ---------------------------------------------------------------------


def run_federated(
    clients: list[Client],
    model: nn.Module,
    val_loader: DataLoader,
    *,
    cfg: dict,
    device,
    score_fn,
    seed: int,
    aggregate_fn=None,
    fixed_rounds: int | None = None,
    logger=None,
) -> dict:
    """Train by federated averaging until the VALIDATION score stops improving.

    The test set is not a parameter of this function.

    aggregate_fn allows the aggregation rule to be swapped without duplicating
    this loop. Stage B passes nothing and gets plain FedAvg; stage C passes the
    user-level DP mechanism. Sharing one round loop is deliberate: a second copy
    would let the private and non-private conditions drift apart, and the whole
    point of stage C is that the ONLY difference from stage B is the privacy
    mechanism.

    fixed_rounds forces exactly N rounds with no early stopping. Differential
    privacy requires this: the noise multiplier is calibrated in advance for a
    specific number of compositions, so stopping early would leave the budget
    unspent while stopping late would exceed it. The best state is still
    selected by validation score across those rounds, which costs no privacy
    budget under LOSO because the validation subjects are not clients and their
    data was never part of the protected set.

    A note on who holds the validation data. Under LOSO, validation is a small
    cohort of subjects who take no part in training; the server evaluates the
    global model on them to decide when to stop. This corresponds to an
    institution retaining a labelled validation cohort, which is realistic,
    but it is a server-side capability and should be stated as such rather
    than presented as pure cross-device federation. The alternative — federated
    evaluation, where clients score the model locally and return metrics —
    leaks less but introduces its own accounting, and is noted as future work.
    """
    f = cfg["federated"]
    rng = np.random.default_rng(seed)
    aggregate = aggregate_fn if aggregate_fn is not None else fedavg_aggregate
    max_rounds = int(fixed_rounds) if fixed_rounds else int(f["max_rounds"])

    global_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_state = copy.deepcopy(global_state)
    best_round = 0
    patience_left = int(f["patience"])
    history = []

    # Bytes on the wire, counted rather than estimated. Feeds stage F.
    state_bytes = sum(
        p.numel() * p.element_size() for p in global_state.values()
    )
    bytes_transferred = 0

    agg_stats_history = []
    for rnd in range(1, max_rounds + 1):
        participating = select_clients(clients, float(f["client_fraction"]), rng)

        states, counts, losses = [], [], []
        for c in participating:
            st, n, loss = c.local_train(
                global_state, model, cfg, device, round_seed=seed * 10_000 + rnd
            )
            states.append(st)
            counts.append(n)
            losses.append(loss)

        # Downlink to each participant, uplink back from each.
        bytes_transferred += 2 * len(participating) * state_bytes

        agg_out = aggregate(global_state, states, counts)
        if isinstance(agg_out, tuple):
            global_state, agg_stats = agg_out
            agg_stats_history.append({"round": rnd, **agg_stats})
        else:
            global_state = agg_out
        model.load_state_dict(global_state)

        y_val, pred_val, _ = T.predict(model, val_loader, device)
        val_score = score_fn(y_val, pred_val)
        mean_loss = float(np.average(losses, weights=counts))

        history.append(
            {
                "round": rnd,
                "n_clients": len(participating),
                "mean_local_loss": mean_loss,
                "val_score": val_score,
                "cumulative_bytes": bytes_transferred,
            }
        )

        improved = val_score > best_score + float(cfg["training"]["tolerance"])
        if improved:
            best_score = val_score
            best_state = copy.deepcopy(global_state)
            best_round = rnd
            patience_left = int(f["patience"])
        else:
            patience_left -= 1

        if logger and rnd % max(1, int(f.get("log_every", 10))) == 0:
            logger.info(
                f"    round {rnd:3d} | clients {len(participating):2d} "
                f"| local_loss {mean_loss:.4f} | val {val_score:.4f} "
                f"| best {best_score:.4f} @ {best_round}"
            )

        # Early stopping is suppressed when the round count is fixed by a
        # privacy budget: the accountant was calibrated for exactly this many
        # compositions.
        if fixed_rounds is None and rnd >= int(f["min_rounds"]) and patience_left <= 0:
            if logger:
                logger.info(f"    early stop at round {rnd} (best @ {best_round})")
            break

    model.load_state_dict(best_state)
    return {
        "best_state": best_state,
        "best_val_score": float(best_score),
        "best_round": int(best_round),
        "rounds_run": len(history),
        "state_bytes": int(state_bytes),
        "bytes_transferred": int(bytes_transferred),
        "history": history,
        "aggregation_stats": agg_stats_history or None,
    }
