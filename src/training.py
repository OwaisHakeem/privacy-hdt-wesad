"""
Training loop with validation-based early stopping.

This module exists to make one specific error impossible.

In the previous codebase, MONITOR_METRIC was macro-F1 evaluated on the global
TEST set, and that single quantity was used simultaneously to trigger early
stopping, to select the best model state, and to report the headline result.
Model selection therefore consumed the test set, and every reported number was
optimistic by an unknown margin. Because this project's central claim is an
honest privacy-utility trade-off, that optimism would inflate the utility axis
of the very curve the paper exists to measure.

Here the split of duties is enforced structurally, not by convention:

    validation set  -> early stopping, best-state selection
    test set        -> evaluated exactly once, after training has finished

The evaluate_test() call is deliberately separate from train(), and train()
is never passed the test data at all.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int | None = None,
    device=None
) -> DataLoader:
    """Build a DataLoader, optionally with the tensors resident on `device`.

    Passing device places the whole tensor on the GPU once, instead of copying
    every mini-batch across the PCIe bus on each step. For this study the entire
    dataset is roughly 2.5 MB and a single client's share is around 130 KB, so
    residency costs nothing and removes what is otherwise the dominant per-step
    cost: the model has only 27k parameters, so a batch of 32 finishes in
    microseconds and the host-to-device copy dwarfs the compute.

    This changes no numbers. The data, the batching, the shuffling generator
    and the arithmetic are identical; only the location of the bytes differs.

    Leave device as None (the default) when using num_workers > 0, since CUDA
    tensors cannot cross a worker process boundary.
    """
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    if device is not None:
        xt = xt.to(device)
        yt = yt.to(device)
    ds = TensorDataset(xt, yt)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, probas = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        proba = torch.softmax(logits, dim=1)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        probas.append(proba.cpu().numpy())
        # .cpu() before .numpy(): labels may already be GPU-resident when the
        # loader was built with device= (see make_loader). Calling .numpy()
        # directly on a CUDA tensor raises.
        ys.append(yb.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(probas),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: dict,
    class_weight: np.ndarray | None,
    device,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    logger=None,
) -> dict:
    """Train until the VALIDATION score stops improving.

    Returns the best state dict and the full history. The test set is not
    referenced anywhere in this function, by construction.
    """
    t = cfg["training"]
    weight = (
        torch.from_numpy(class_weight).to(device) if class_weight is not None else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_left = int(t["patience"])
    history = []

    for epoch in range(1, int(t["max_epochs"]) + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        y_val, pred_val, _ = predict(model, val_loader, device)
        val_score = score_fn(y_val, pred_val)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_score": val_score}
        )

        improved = val_score > best_score + float(t["tolerance"])
        if improved:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = int(t["patience"])
        else:
            patience_left -= 1

        if logger and epoch % max(1, int(t.get("log_every", 10))) == 0:
            logger.info(
                f"    epoch {epoch:3d} | train_loss {train_loss:.4f} "
                f"| val {val_score:.4f} | best {best_score:.4f} @ {best_epoch}"
            )

        if epoch >= int(t["min_epochs"]) and patience_left <= 0:
            if logger:
                logger.info(f"    early stop at epoch {epoch} (best @ {best_epoch})")
            break

    model.load_state_dict(best_state)
    return {
        "best_state": best_state,
        "best_val_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "history": history,
    }
