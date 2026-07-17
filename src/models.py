"""
Model definitions.

One architecture is used for every condition in the study — centralised,
local-only, FedAvg and DP-FedAvg. This is deliberate: if the architecture
varied between conditions, the privacy-utility curve would confound the cost
of privacy with the cost of a different model, and the paper's central claim
would not be measurable.

The architecture is intentionally small. Three reasons:
  1. DP-SGD noise scales with the number of parameters, so an unnecessarily
     large model is penalised twice under a fixed privacy budget.
  2. The federated clients represent edge devices; a model that could not
     plausibly train on one would undermine the deployment story.
  3. The state dictionary is what crosses the wire each round, so parameter
     count directly determines the communication overhead measured at F.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Feed-forward classifier over the 144 statistical window features.

    DP-compatibility note: Opacus cannot handle BatchNorm, because batch
    statistics mix information across samples and break per-sample gradient
    accounting. LayerNorm or GroupNorm would be admissible; here we simply
    avoid normalisation layers, so that the identical module can be used in
    both the private and non-private conditions without modification.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict, input_dim: int, num_classes: int) -> SimpleMLP:
    m = cfg["model"]
    return SimpleMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=tuple(m["hidden"]),
        dropout=float(m["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_size_bytes(model: nn.Module) -> int:
    """Bytes of the model state — the quantity transmitted per round.

    Recorded here so that stage F measures the same object the federated
    stages actually send, rather than an estimate derived separately.
    """
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())
