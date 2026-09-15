"""Turning model outputs into metric rows for the known / unknown groups of a split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax

from kdtraffic import metrics
from kdtraffic.data import Arrays


@dataclass
class Outputs:
    probs: np.ndarray
    scores: dict[str, np.ndarray]
    probs_ts: np.ndarray | None = None
    temperature: float | None = None


def from_logits(logits: np.ndarray, temperature: float | None = None) -> Outputs:
    logits = logits.astype(np.float32, copy=False)
    probs = softmax(logits, axis=1)
    probs_ts = None if temperature is None else softmax(logits / temperature, axis=1)
    return Outputs(probs, {"msp": probs.max(axis=1), "energy": logsumexp(logits, axis=1)}, probs_ts, temperature)


def ensemble_probs(member_logits: list[np.ndarray]) -> np.ndarray:
    total = None
    for logits in member_logits:
        p = softmax(logits.astype(np.float32, copy=False), axis=1)
        total = p if total is None else total + p
    return total / len(member_logits)


def ensemble_log_probs(member_logits: list[np.ndarray]) -> np.ndarray:
    """Log of the ensemble's mean predictive distribution (used as logits for temperature scaling)."""
    return np.log(ensemble_probs(member_logits) + 1e-12)


def from_ensemble(member_logits: list[np.ndarray], temperature: float | None = None) -> Outputs:
    probs = ensemble_probs(member_logits)
    energy = np.mean([logsumexp(l.astype(np.float32, copy=False), axis=1) for l in member_logits], axis=0)
    probs_ts = None if temperature is None else softmax(np.log(probs + 1e-12) / temperature, axis=1)
    return Outputs(probs, {"msp": probs.max(axis=1), "energy": energy}, probs_ts, temperature)


def report_rows(model: str, split: str, arrays: Arrays, near: list[str], far: list[str], outputs: Outputs) -> list[dict]:
    """Metrics for all flows and flows with >= 5 packets, against all / near / far unknown services."""
    known = arrays.y >= 0
    unknown_groups = {
        "all": ~known,
        "near": np.isin(arrays.app, np.array(near, dtype=str)),
        "far": np.isin(arrays.app, np.array(far, dtype=str)),
    }
    flow_groups = {"all": np.ones(len(arrays), dtype=bool), "ge5": arrays.ppi_len >= 5}
    rows = []
    for flows, flow_mask in flow_groups.items():
        for unknown, unknown_mask in unknown_groups.items():
            mask = flow_mask & (known | unknown_mask)
            report = metrics.open_set_report(
                arrays.y[mask],
                outputs.probs[mask],
                {name: score[mask] for name, score in outputs.scores.items()},
                None if outputs.probs_ts is None else outputs.probs_ts[mask],
            )
            rows.append({"model": model, "split": split, "flows": flows, "unknown": unknown,
                         "temperature": outputs.temperature, **report})
    return rows
