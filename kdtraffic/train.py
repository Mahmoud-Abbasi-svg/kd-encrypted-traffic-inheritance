"""Training and inference on in-memory arrays."""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

from kdtraffic.data import Arrays


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    eval_batch_size: int = 8192


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device: str, enabled: bool):
    if enabled and device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict_logits(model: nn.Module, arrays: Arrays, device: str, amp: bool = True, batch_size: int = 8192,
                   with_features: bool = False):
    """Logits (and optionally penultimate features) as float32 numpy arrays, in the order of `arrays`."""
    model.eval()
    logits_parts, feature_parts = [], []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            ppi = torch.from_numpy(arrays.ppi[start:start + batch_size]).to(device)
            stats = torch.from_numpy(arrays.flowstats[start:start + batch_size]).to(device)
            with autocast(device, amp):
                features = model.forward_features(ppi, stats)
                logits = model.forward_head(features)
            logits_parts.append(logits.float().cpu().numpy())
            if with_features:
                feature_parts.append(features.float().cpu().numpy())
    logits = np.concatenate(logits_parts)
    return (logits, np.concatenate(feature_parts)) if with_features else logits


Objective = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def train_classifier(model: nn.Module, train: Arrays, val: Arrays, cfg: TrainConfig,
                     log: Callable[[str], None] = print, name: str = "model", objective: Objective | None = None) -> dict:
    """Train with cross-entropy, or with `objective(logits, labels, train_indices)` if given.

    Keeps the epoch with the lowest cross-entropy on validation known flows.
    """
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model.to(device)
    ppi = torch.from_numpy(train.ppi).to(device)
    stats = torch.from_numpy(train.flowstats).to(device)
    labels = torch.from_numpy(train.y.astype(np.int64)).to(device)
    n = len(train)
    batch_size = min(cfg.batch_size, n)
    steps_per_epoch = n // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
                                                    pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    generator = torch.Generator().manual_seed(cfg.seed)
    val_known = val.subset(np.flatnonzero(val.y >= 0))

    # A feature-distillation objective needs the penultimate activations as well as the logits and
    # declares so with `needs_features`; nothing else changes for the other objectives.
    wants_features = getattr(objective, "needs_features", False)
    best_score, best_state, best_epoch, history = float("inf"), None, 0, []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_started = time.time()
        running = torch.zeros((), device=device)
        order = torch.randperm(n, generator=generator).to(device)
        for step in range(steps_per_epoch):
            idx = order[step * batch_size:(step + 1) * batch_size]
            with autocast(device, cfg.amp):
                if wants_features:
                    # only a feature-distillation objective takes this path; every other condition
                    # calls the model exactly as before, so their results are unaffected
                    hidden = model.forward_features(ppi[idx], stats[idx])
                    logits = model.forward_head(hidden)
                else:
                    logits = model(ppi[idx], stats[idx])
            if objective is None:
                loss = loss_fn(logits.float(), labels[idx])
            elif wants_features:
                loss = objective(logits.float(), labels[idx], idx, hidden.float())
            else:
                loss = objective(logits.float(), labels[idx], idx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.detach()
        record = {"epoch": epoch, "train_loss": float(running) / steps_per_epoch,
                  "seconds": round(time.time() - epoch_started, 1)}
        if len(val_known):
            val_logits = predict_logits(model, val_known, device, cfg.amp, cfg.eval_batch_size)
            record["val_loss"] = float(F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(val_known.y)))
            record["val_macro_f1"] = float(f1_score(val_known.y, val_logits.argmax(1), labels=np.unique(val_known.y),
                                                    average="macro", zero_division=0))
        history.append(record)
        log(f"{name} epoch {epoch}/{cfg.epochs}: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items() if k != "epoch"))
        score = record.get("val_loss", -epoch)  # without validation flows, keep the last epoch
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "train_seconds": round(time.time() - started, 1)}
