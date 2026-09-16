"""Cluster bootstrap over day x service groups (flows are not independent)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def cluster_ids(day: np.ndarray, app: np.ndarray) -> np.ndarray:
    """Integer id of each flow's day x service cluster."""
    return pd.factorize(pd.Series(day).astype(str) + "|" + pd.Series(app).astype(str))[0]


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Percentile interval of `stat(flow_indices)` when whole clusters are resampled with replacement."""
    clusters = np.asarray(clusters)
    order = np.argsort(clusters, kind="stable")
    sorted_ids = clusters[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    lengths = np.diff(np.r_[starts, len(clusters)])
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(starts), size=len(starts))
        picked_lengths = lengths[pick]
        offsets = np.repeat(starts[pick] - np.r_[0, np.cumsum(picked_lengths)[:-1]], picked_lengths)
        values[b] = stat(order[np.arange(picked_lengths.sum()) + offsets])
    low, high = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(np.arange(len(clusters)))), "low": float(low), "high": float(high),
            "se": float(np.nanstd(values, ddof=1)), "clusters": int(len(starts))}
