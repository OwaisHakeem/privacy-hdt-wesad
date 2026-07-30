"""
Sample-level differential privacy for federated learning.

WHAT IS PROTECTED, AND HOW IT DIFFERS FROM STAGE C
--------------------------------------------------
This module implements SAMPLE-level differential privacy: the guarantee is
that the trained model is (epsilon, delta)-indistinguishable with respect to
the participation of any single 10-second WINDOW in a client's training data.

Contrast with Stage C, which protected user (whole-person) participation.
Sample-level is a weaker guarantee — an adversary can still learn whether a
person participated at all, only what specific seconds of their recording
contributed to the model. This is the guarantee commonly used in the FL
literature, but it does not match what a Human Digital Twin requires.

The paper's contribution here is the COMPARISON of the two mechanisms on the
same pipeline: does the formal-vs-empirical decoupling seen at user level
(Stage D) hold generally across mechanisms, or is it specific to user-level DP?

MECHANISM (DP-SGD; Abadi et al., 2016)
--------------------------------------
Each client's local training runs DP-SGD:
    1. sample a Poisson-subsampled mini-batch from the client's data
    2. compute per-example gradients
    3. clip each example's gradient to L2 norm C
    4. sum clipped gradients and add Gaussian noise N(0, (sigma * C)^2)
    5. divide by expected batch size, take an optimiser step

Server aggregation is standard FedAvg. Under the DP definition it is
post-processing of DP outputs, so no additional server-side noise is needed;
each client's local guarantee carries through.

AMPLIFICATION BY SUBSAMPLING
---------------------------
Sample-level DP with Poisson subsampling gets privacy amplification: the
noise required to hit a target epsilon is much lower than for user-level DP
without subsampling. This is exactly why sample-level should retain more
utility than user-level in Stage C — and it is the empirical claim this
module exists to test.

CROSS-ROUND ACCOUNTING
----------------------
Each client's PrivacyEngine persists across all federated rounds. The RDP
accountant tracks composed privacy loss step-by-step. After T rounds x E
local epochs, each client has spent (target_epsilon, delta) against its
own data. The server's model, obtained by averaging clients, inherits the
same guarantee against each client individually (post-processing).

Because the guarantee is client-local, all clients receive the SAME target
epsilon. In deployment terms: every person's data is equally protected.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    _HAS_OPACUS = True
except ImportError:
    _HAS_OPACUS = False
    PrivacyEngine = None  # type: ignore


def check_opacus_available() -> None:
    """Fail fast, with an install hint, if opacus is missing.

    Called at module boundaries rather than raising on import so that this
    file can be imported by tools that only need utility helpers.
    """
    if not _HAS_OPACUS:
        raise ImportError(
            "opacus is required for sample-level DP. Install with:\n"
            "    pip install opacus\n"
            "Opacus 1.4+ recommended; requires PyTorch 1.13+."
        )


# ---------------------------------------------------------------------
# Per-client sample-level DP training
# ---------------------------------------------------------------------


@dataclass
class SampleDPClient:
    """One subject's local training state, with sample-level DP applied.

    Mirrors src.federated.Client so it plugs into the same server loop, but
    each round's local optimisation is DP-SGD (per-example gradient clipping
    + Gaussian noise). The PrivacyEngine is created once and persists across
    all federated rounds, so the accountant composes correctly.

    class_weight is derived from THIS client's own labels. A client cannot
    see the global class distribution, so deriving weights globally would
    quietly grant every client knowledge it does not possess and would make
    the federated result unreproducible in deployment.

    Rationale for the design choices:
      * Per-client PrivacyEngine, not shared across clients — the DP guarantee
        is client-local. A shared engine would compose privacy across
        different subjects, which is not what a client-local guarantee means.
      * Adam is the base optimiser (matched to non-private FedAvg for
        comparability). Opacus wraps Adam correctly; DP-SGD as a name refers
        to the noise mechanism, not the specific base optimiser.
      * The DPOptimizer + PrivacyEngine persist across rounds. Only model
        weights are reloaded each round; optimiser Adam moments persist,
        which is a small deviation from vanilla FedAvg convention but is
        required to keep the accountant intact.
    """

    client_id: str
    X: np.ndarray
    y: np.ndarray
    n_classes: int
    class_weight: np.ndarray | None = None

    # Set by attach_privacy_engine after model architecture is known.
    _engine: Any = field(default=None, repr=False)
    _model: nn.Module | None = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _dataloader: DataLoader | None = field(default=None, repr=False)
    _noise_multiplier: float | None = field(default=None, repr=False)
    _target_epsilon: float | None = field(default=None, repr=False)
    _delta: float | None = field(default=None, repr=False)
    _clip_norm: float | None = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def attach_privacy_engine(
        self,
        model: nn.Module,
        *,
        target_epsilon: float | None,
        delta: float,
        max_grad_norm: float,
        total_local_epochs: int,
        batch_size: int,
        base_lr: float,
        weight_decay: float,
        device,
        noise_multiplier: float | None = None,
    ) -> None:
        """Wrap this client's model, optimiser, and dataloader for DP-SGD.

        Two calibration modes:
          * If target_epsilon is provided (the usual case), Opacus calibrates
            noise_multiplier such that after total_local_epochs epochs the
            budget is spent to exactly target_epsilon.
          * If noise_multiplier is provided (non-private anchor is epsilon=inf,
            handled outside this class), that value is used directly.

        total_local_epochs = number_of_rounds * local_epochs_per_round. Opacus
        converts this to a step count via the sample rate; the accountant then
        tracks each step and, after this many local steps, the client's
        guarantee is exactly (target_epsilon, delta).
        """
        check_opacus_available()
        model = model.to(device)

        # Base optimiser and plain dataloader.
        from src.training import make_loader

        # Tensors placed on the target device once, rather than copied per
        # mini-batch. This is the hot loop: with a 27k-parameter model each
        # step's compute is dwarfed by the host-to-device transfer, which is
        # why GPU utilisation sat near 35% before this change.
        loader = make_loader(self.X, self.y, batch_size, shuffle=True, seed=0, device=device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

        engine = PrivacyEngine(secure_mode=False)

        if target_epsilon is not None:
            # Opacus solves for noise_multiplier so that after 'epochs' epochs
            # the guarantee is exactly (target_epsilon, delta). We are running
            # this many local epochs across all rounds combined.
            model, optimizer, dataloader = engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                target_epsilon=float(target_epsilon),
                target_delta=float(delta),
                epochs=int(total_local_epochs),
                max_grad_norm=float(max_grad_norm),
                poisson_sampling=True,
            )
            noise_multiplier = float(optimizer.noise_multiplier)
        else:
            assert noise_multiplier is not None, "Either target_epsilon or noise_multiplier is required."
            model, optimizer, dataloader = engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=float(noise_multiplier),
                max_grad_norm=float(max_grad_norm),
                poisson_sampling=True,
            )

        self._engine = engine
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._noise_multiplier = noise_multiplier
        self._target_epsilon = target_epsilon
        self._delta = delta
        self._clip_norm = max_grad_norm
        self._device = device

    def get_epsilon(self) -> float:
        """Currently spent epsilon under the client's own accountant."""
        check_opacus_available()
        assert self._engine is not None and self._delta is not None
        return float(self._engine.get_epsilon(self._delta))

    def _load_global_state(self, global_state: dict) -> None:
        """Copy the server's weights into this client's wrapped model.

        Opacus wraps the model in GradSampleModule, whose parameters are named
        with a "_module." prefix (e.g. "_module.net.0.weight"), while the
        global state uses plain names ("net.0.weight"). Loading plain keys into
        the wrapped module with strict=False would match NOTHING and silently
        leave the client's weights unchanged - so every client would ignore the
        aggregated global model and drift on its own data. We therefore add the
        prefix and load strictly, so a future key mismatch fails loudly instead
        of collapsing training.
        """
        assert self._model is not None
        model_keys = list(self._model.state_dict().keys())
        wrapped = any(k.startswith("_module.") for k in model_keys)
        if wrapped:
            prefixed = {f"_module.{k}": v for k, v in global_state.items()}
            self._model.load_state_dict(prefixed, strict=True)
        else:
            self._model.load_state_dict(global_state, strict=True)

    def local_train(
        self,
        global_state: dict,
        cfg: dict,
        round_seed: int,
    ) -> tuple[dict, int, float]:
        """One round of DP-SGD local training.

        Returns (updated state_dict, n_client_samples, mean_training_loss).
        Signature matches src.federated.Client.local_train, so the outer FL
        loop is unchanged.
        """
        check_opacus_available()
        assert self._model is not None and self._optimizer is not None

        self._load_global_state(global_state)
        self._model.train()

        weight = (
            torch.from_numpy(self.class_weight).to(self._device)
            if self.class_weight is not None
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        total, seen = 0.0, 0
        for _ in range(int(cfg["federated"]["local_epochs"])):
            for xb, yb in self._dataloader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                self._optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                self._optimizer.step()
                total += float(loss.item()) * len(xb)
                seen += len(xb)

        # Return a plain state_dict (unwrapped) so it merges cleanly with
        # non-DP models. GradSampleModule's state_dict() already excludes
        # the wrapper's helper buffers.
        state = {k: v.detach().clone().cpu() for k, v in self._model.state_dict().items()}
        # Strip Opacus's wrapper prefix so the state dict matches an unwrapped model.
        state = {k.replace("_module.", ""): v for k, v in state.items()}
        return state, self.n_samples, total / max(seen, 1)


# ---------------------------------------------------------------------
# Calibration helper for the epsilon=inf anchor
# ---------------------------------------------------------------------


def make_nonprivate_client_config(
    batch_size: int,
    base_lr: float,
    weight_decay: float,
    device,
) -> dict:
    """Config for the non-private (epsilon=inf) anchor.

    The non-private anchor should NOT use Opacus (it would still incur the
    per-example gradient overhead for no benefit). This helper returns the
    parameters the caller passes to a plain federated.Client instead. The
    non-private path is handled outside this module.
    """
    return {
        "batch_size": batch_size,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "device": device,
        "use_dp": False,
    }
