"""
Differential privacy: accounting and the user-level Gaussian mechanism.

WHAT IS PROTECTED, AND WHY IT MATTERS
-------------------------------------
This module implements USER-level differential privacy: the guarantee is that
the trained model is (epsilon, delta)-indistinguishable with respect to the
participation of any single CLIENT — that is, any single person — not any
single 10-second window.

The distinction is not pedantic, and it is the reason the previous pipeline's
approach is not reused here. Applying Opacus per client protects individual
windows: an adversary learns little about which second of your recording was
used, but may still learn that you participated at all. For a Human Digital
Twin, the subject of protection is the human. A sample-level guarantee
answers a question this study is not asking.

THE MECHANISM (McMahan et al., 2018)
------------------------------------
Each round:
    1. every client computes its update  d_i = w_i - w_global
    2. d_i is clipped to L2 norm C, bounding any one person's influence
    3. the clipped updates are summed
    4. Gaussian noise N(0, (sigma*C)^2) is added to the sum
    5. the result is averaged and applied to the global model

Clipping bounds the sensitivity of the sum to one client's participation;
the noise renders that participation statistically deniable.

THE ARITHMETIC IS UNFORGIVING AT SMALL COHORTS
----------------------------------------------
Sensitivity is C regardless of how many clients there are, so the noise added
to the SUM does not shrink with cohort size — but the average divides by n.
With n = 12 the noise surviving into the averaged update is sigma*C/12. At
epsilon = 1 over 30 rounds, sigma is roughly 24, so the noise on the average
is about twice the entire clipped update. This is not a tuning deficiency; it
is what hiding one person among twelve costs. Reporting it is a finding.

Full participation (client_fraction = 1.0) forgoes privacy amplification by
subsampling, which would otherwise reduce sigma substantially. That is a real
cost of the Stage B configuration and is stated rather than hidden.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------
# RDP accounting
# ---------------------------------------------------------------------

# Orders at which Renyi divergence is tracked. The conversion to (eps, delta)
# is minimised over these; a coarse grid would silently overstate epsilon.
DEFAULT_ALPHAS: tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 256))
)


def rdp_gaussian_per_round(sigma: float, alpha: float) -> float:
    """RDP of one Gaussian mechanism release, at order alpha.

    For sensitivity C and noise standard deviation sigma*C, the Renyi
    divergence at order alpha is alpha / (2 sigma^2) — independent of C,
    which is why sigma alone determines the privacy cost once clipping is in
    place.

    No subsampling amplification is applied: every client participates in
    every round (client_fraction = 1.0), so the sampling rate is 1.
    """
    if sigma <= 0:
        return float("inf")
    return alpha / (2.0 * sigma**2)


def rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Convert RDP at one order to (eps, delta)-DP.

    Uses the improved conversion of Balle et al. (2020) rather than the
    original bound of Mironov (2017); the latter is looser and would report a
    larger epsilon for identical noise, understating the method.
    """
    if alpha <= 1:
        return float("inf")
    return (
        rdp
        + math.log((alpha - 1) / alpha)
        - (math.log(delta) + math.log(alpha)) / (alpha - 1)
    )


def compute_epsilon(
    sigma: float,
    rounds: int,
    delta: float,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[float, float]:
    """Total epsilon after `rounds` compositions. Returns (epsilon, best alpha)."""
    if sigma <= 0 or rounds <= 0:
        return float("inf"), float("nan")

    best_eps, best_alpha = float("inf"), float("nan")
    for a in alphas:
        eps = rdp_to_dp(rounds * rdp_gaussian_per_round(sigma, a), a, delta)
        if eps < best_eps:
            best_eps, best_alpha = eps, a
    return max(best_eps, 0.0), best_alpha


def calibrate_noise(
    target_epsilon: float,
    rounds: int,
    delta: float,
    lo: float = 0.1,
    hi: float = 500.0,
    tol: float = 1e-4,
) -> float:
    """Smallest sigma achieving `target_epsilon` after `rounds` rounds.

    Direction of the study, stated as code: the noise is chosen to MEET a
    privacy target, rather than a convenient noise level being chosen and
    whatever epsilon results being reported afterwards. The previous pipeline
    did the latter and reported epsilon values near 47 and 78, which
    correspond to no meaningful guarantee.
    """
    if compute_epsilon(hi, rounds, delta)[0] > target_epsilon:
        raise ValueError(
            f"Even sigma={hi} cannot reach epsilon={target_epsilon} in {rounds} "
            "rounds. Reduce rounds or relax the target."
        )
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(mid, rounds, delta)[0] > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------
# User-level Gaussian mechanism
# ---------------------------------------------------------------------


def flatten_delta(state: dict, global_state: dict) -> tuple[torch.Tensor, list]:
    """Flatten the client update into one vector; return it and the layout."""
    parts, layout = [], []
    for k in global_state:
        if not torch.is_floating_point(global_state[k]):
            continue
        d = (state[k] - global_state[k]).flatten()
        layout.append((k, global_state[k].shape, d.numel()))
        parts.append(d)
    return torch.cat(parts), layout


def unflatten_delta(vec: torch.Tensor, layout: list) -> dict:
    out, i = {}, 0
    for k, shape, n in layout:
        out[k] = vec[i : i + n].view(shape)
        i += n
    return out


def clip_delta(vec: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, float]:
    """Scale the update down to L2 norm `clip_norm` if it exceeds it.

    Clipping is applied to the WHOLE update as a single vector, not per
    layer: sensitivity is defined over the client's entire contribution, and
    per-layer clipping would not bound it.
    """
    norm = float(torch.linalg.vector_norm(vec).item())
    if norm > clip_norm:
        vec = vec * (clip_norm / (norm + 1e-12))
    return vec, norm


def dp_aggregate(
    global_state: dict,
    client_states: list[dict],
    *,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> tuple[dict, dict]:
    """One round of user-level DP-FedAvg aggregation.

    Note the departure from Stage B: updates are averaged with EQUAL weight,
    not weighted by client sample count. Sample-count weighting would make the
    sensitivity of the sum depend on how much data a client holds, so a single
    clip norm could no longer bound one client's influence and the stated
    guarantee would not hold. The cost is that WESAD's uneven recording
    lengths (284-302 windows) are no longer reflected in the average — a real
    consequence of the privacy requirement, and one worth stating.
    """
    n = len(client_states)
    if n == 0:
        raise ValueError("No client states to aggregate.")

    summed, layout, norms = None, None, []
    for st in client_states:
        vec, layout_i = flatten_delta(st, global_state)
        vec, raw_norm = clip_delta(vec, clip_norm)
        norms.append(raw_norm)
        layout = layout_i if layout is None else layout
        summed = vec.clone() if summed is None else summed + vec

    if noise_multiplier > 0:
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * clip_norm,
            size=summed.shape,
            generator=generator,
            device=summed.device,
        )
        summed = summed + noise

    averaged = summed / n
    deltas = unflatten_delta(averaged, layout)

    new_state = {}
    for k, v in global_state.items():
        new_state[k] = (v + deltas[k]).to(v.dtype) if k in deltas else v.clone()

    stats = {
        "median_update_norm": float(np.median(norms)),
        "mean_update_norm": float(np.mean(norms)),
        "max_update_norm": float(np.max(norms)),
        "fraction_clipped": float(np.mean([x > clip_norm for x in norms])),
        "noise_std_on_sum": float(noise_multiplier * clip_norm),
        "noise_std_on_average": float(noise_multiplier * clip_norm / n),
        "signal_to_noise": float(
            (clip_norm / max(noise_multiplier * clip_norm / n, 1e-12))
        ),
    }
    return new_state, stats


def privacy_report(sigma: float, rounds: int, delta: float, n_clients: int, clip_norm: float) -> dict:
    eps, alpha = compute_epsilon(sigma, rounds, delta)
    return {
        "epsilon": eps,
        "delta": delta,
        "best_alpha": alpha,
        "noise_multiplier": sigma,
        "clip_norm": clip_norm,
        "rounds": rounds,
        "n_clients": n_clients,
        "granularity": "user-level (client participation)",
        "sampling_rate": 1.0,
        "amplification_by_subsampling": False,
        "noise_std_on_average_update": sigma * clip_norm / max(n_clients, 1),
        "noise_to_clip_ratio_on_average": sigma / max(n_clients, 1),
    }
