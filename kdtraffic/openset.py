"""Feature-space unknown-traffic scores: Mahalanobis distance and k-nearest-neighbour distance.

The pre-registration lists a kNN-embedding score among its planned exploratory analyses
(`docs/preregistration.md` §9); this module implements it, and the Mahalanobis score alongside, so
that the study has a dedicated open-set detector to compare the logit-based scores against.

Both are computed from a model's penultimate features, which `predict_logits(..., with_features=True)`
already exposes for every architecture in this project. Neither is used by the confirmatory analysis.

Memory is the binding constraint: the teachers' features are 600- or 1200-dimensional and the
training windows hold up to 1.9M flows, so materialising them would need 4-17 GB. Nothing here ever
holds more than one batch of features:

    stats   = accumulate_feature_stats(model, train_arrays, device)   # one pass, keeps (C,D) and (D,D)
    scorer  = FeatureScorer.fit(stats, reference)                     # reference = a small feature sample
    scores  = score_arrays(model, eval_arrays, scorer, device)        # per batch, returns only (N,) scores

Scores follow the project convention: **higher means more likely known**, so both are negated
distances. They are returned as float32 - a squared Mahalanobis distance in 600 dimensions overflows
float16, which would silently collapse the far-unknown flows into a single tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kdtraffic.data import Arrays
from kdtraffic.train import autocast, resolve_device


@dataclass
class FeatureStats:
    """Sufficient statistics for a tied-covariance Gaussian per class, accumulated in one pass."""
    counts: np.ndarray  # (C,) flows per class
    sums: np.ndarray  # (C, D) feature sums per class
    scatter: np.ndarray  # (D, D) sum of outer products over all known flows
    n: int
    dim: int


def _features(model, ppi: torch.Tensor, stats: torch.Tensor, device: str, amp: bool) -> torch.Tensor:
    with autocast(device, amp):
        return model.forward_features(ppi, stats).float()


def accumulate_feature_stats(model, arrays: Arrays, device: str = "auto", amp: bool = False,
                             batch_size: int = 8192, max_flows: int | None = None,
                             seed: int = 0, num_classes: int | None = None) -> FeatureStats:
    """Class sums and the total scatter matrix of the penultimate features of `arrays`' known flows.

    `amp` defaults to False on purpose: bfloat16 keeps only 8 mantissa bits, which makes a 600x600
    covariance both noisy and non-deterministic.
    """
    device = resolve_device(device)
    model = model.to(device).eval()
    known = np.flatnonzero(arrays.y >= 0)
    if max_flows is not None and len(known) > max_flows:
        known = np.random.default_rng(seed).choice(known, size=max_flows, replace=False)
        known.sort()
    classes = num_classes if num_classes is not None else int(arrays.y.max()) + 1
    sums = scatter = None
    counts = np.zeros(classes, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(known), batch_size):
            index = known[start:start + batch_size]
            ppi = torch.from_numpy(arrays.ppi[index]).to(device)
            flowstats = torch.from_numpy(arrays.flowstats[index]).to(device)
            feats = _features(model, ppi, flowstats, device, amp).double()
            if sums is None:
                dim = feats.shape[1]
                sums = torch.zeros(classes, dim, dtype=torch.float64, device=device)
                scatter = torch.zeros(dim, dim, dtype=torch.float64, device=device)
            labels = torch.from_numpy(arrays.y[index]).to(device)
            sums.index_add_(0, labels, feats)
            scatter += feats.T @ feats
            counts += np.bincount(arrays.y[index], minlength=classes)
            del ppi, flowstats, feats, labels
    return FeatureStats(counts, sums.cpu().numpy(), scatter.cpu().numpy(), int(len(known)), int(sums.shape[1]))


def extract_features(model, arrays: Arrays, index: np.ndarray, device: str = "auto", amp: bool = False,
                     batch_size: int = 8192) -> np.ndarray:
    """Penultimate features of the selected flows only (used for the small kNN reference set)."""
    device = resolve_device(device)
    model = model.to(device).eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start:start + batch_size]
            ppi = torch.from_numpy(arrays.ppi[rows]).to(device)
            flowstats = torch.from_numpy(arrays.flowstats[rows]).to(device)
            parts.append(_features(model, ppi, flowstats, device, amp).cpu().numpy().astype(np.float32))
            del ppi, flowstats
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


class FeatureScorer:
    """Mahalanobis and feature-kNN scores for one model, fitted on its training features."""

    def __init__(self, means: np.ndarray, whitening: np.ndarray, reference: np.ndarray, k: int):
        self.means = means  # (C, D), classes with no training flows removed
        self.whitening = whitening  # (D, D) symmetric square root of the precision matrix
        self.reference = reference  # (R, D) float32 sample of training features
        self.k = k

    @classmethod
    def fit(cls, stats: FeatureStats, reference: np.ndarray, shrinkage: float = 0.1, k: int = 10) -> "FeatureScorer":
        present = stats.counts > 0
        means = stats.sums[present] / stats.counts[present, None]
        # within-class scatter = total scatter - sum_c n_c mu_c mu_c^T
        within = stats.scatter - (means * stats.counts[present, None]).T @ means
        covariance = within / max(stats.n - int(present.sum()), 1)
        # shrink toward a scaled identity: the features are post-ReLU and often rank-deficient,
        # so the raw within-class covariance is not reliably invertible
        scale = np.trace(covariance) / stats.dim
        covariance = (1 - shrinkage) * covariance + shrinkage * scale * np.eye(stats.dim)
        precision = np.linalg.pinv(covariance, hermitian=True)
        # Symmetric square root W of the precision, so that the Mahalanobis distance becomes a plain
        # Euclidean one: (x - mu)' P (x - mu) = ||W(x - mu)||^2. That lets torch.cdist do the work.
        precision = 0.5 * (precision + precision.T)
        eigenvalues, vectors = np.linalg.eigh(precision)
        whitening = (vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ vectors.T
        return cls(means.astype(np.float64), whitening.astype(np.float64), reference, k)

    def scores(self, features: np.ndarray, device: str = "cpu") -> dict[str, np.ndarray]:
        """Negated distances (higher = more likely known) for one batch of features."""
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        rotation = torch.from_numpy(self.whitening.astype(np.float32)).to(device)
        centres = torch.from_numpy(self.means.astype(np.float32)).to(device) @ rotation
        projected = x @ rotation
        mahalanobis = torch.cdist(projected, centres).min(dim=1).values ** 2
        out = {"maha": (-mahalanobis).cpu().numpy().astype(np.float32)}
        if self.reference is not None and len(self.reference):
            reference = torch.from_numpy(self.reference).to(device)
            k = min(self.k, len(self.reference))
            kth = torch.cdist(x, reference).topk(k, largest=False).values[:, -1]
            out["knnfeat"] = (-kth).cpu().numpy().astype(np.float32)
            del reference
        del x, rotation, centres, projected
        return out


def score_arrays(model, arrays: Arrays, scorer: FeatureScorer, device: str = "auto", amp: bool = False,
                 batch_size: int = 8192) -> dict[str, np.ndarray]:
    """Feature-space scores for every flow of `arrays`, one batch of features at a time."""
    device = resolve_device(device)
    model = model.to(device).eval()
    collected: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            stop = start + batch_size
            ppi = torch.from_numpy(arrays.ppi[start:stop]).to(device)
            flowstats = torch.from_numpy(arrays.flowstats[start:stop]).to(device)
            feats = _features(model, ppi, flowstats, device, amp).cpu().numpy()
            for name, value in scorer.scores(feats, device).items():
                collected.setdefault(name, []).append(value)
            del ppi, flowstats, feats
    return {name: np.concatenate(parts) for name, parts in collected.items()}
