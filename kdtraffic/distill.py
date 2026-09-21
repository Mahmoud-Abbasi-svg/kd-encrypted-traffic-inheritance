"""Distillation objectives and teacher targets, plus the synthetic-shortcut helpers (RQ3).

Objectives follow kdtraffic.train.Objective: called with (student logits, labels, train indices);
teacher targets are precomputed for the whole training set and indexed per batch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax, softmax

from kdtraffic.data import Arrays


def softmax_chunked(logits: np.ndarray, temperature: float = 1.0, dtype=np.float32, chunk: int = 262_144) -> np.ndarray:
    out = np.empty(logits.shape, dtype=dtype)
    for start in range(0, len(logits), chunk):
        part = logits[start:start + chunk].astype(np.float32) / temperature
        out[start:start + chunk] = softmax(part, axis=1)
    return out


class HardLabelCE:
    """Cross-entropy against the teacher's top-1 prediction instead of the true label.

    The anchor for the teacher-swap test: a student trained this way copies the teacher's decisions
    but receives none of the soft-target information distillation is supposed to convey. Whatever
    shift toward its own teacher it shows is the part explained by label agreement alone.
    """

    def __init__(self, teacher_labels: np.ndarray, device: str):
        self.labels = torch.from_numpy(np.ascontiguousarray(teacher_labels, dtype=np.int64)).to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        target = self.labels[idx.to(self.labels.device)].to(logits.device)
        return F.cross_entropy(logits, target)


class HintonKD:
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, labels).

    `teacher_probs` are the teacher's softened probabilities at temperature T, for every training
    flow (for an ensemble, the mean of the members' softened probabilities).
    """

    def __init__(self, teacher_probs: np.ndarray, temperature: float, alpha: float, device: str):
        self.targets = torch.from_numpy(np.ascontiguousarray(teacher_probs, dtype=np.float32)).to(device)
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        targets = self.targets[idx.to(self.targets.device)].to(logits.device)
        log_q = F.log_softmax(logits / self.temperature, dim=1)
        kd = (targets * (torch.log(targets.clamp_min(1e-8)) - log_q)).sum(dim=1).mean() * self.temperature ** 2
        return self.alpha * kd + (1 - self.alpha) * F.cross_entropy(logits, labels)


class EnsembleDistributionDistillation:
    """Ensemble distribution distillation (Malinin et al., ICLR 2020).

    The student's exp(logits) are the concentrations of a Dirichlet over class probabilities, fitted by
    maximum likelihood to the members' predictive distributions. The student's mean prediction is
    softmax(logits), so the usual evaluation applies unchanged; log of the total concentration equals
    the energy score.
    """

    def __init__(self, member_probs: np.ndarray, device: str, smoothing: float = 1e-4, max_logit: float = 15.0):
        self.probs = torch.from_numpy(member_probs).to(device)  # (M, N, C), float16
        self.smoothing = smoothing
        self.max_logit = max_logit

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        pi = self.probs[:, idx.to(self.probs.device)].to(logits.device, torch.float32)
        num_classes = pi.shape[-1]
        pi = (1 - self.smoothing) * pi + self.smoothing / num_classes
        alpha = torch.exp(logits.clamp(-self.max_logit, self.max_logit))
        log_norm = torch.lgamma(alpha.sum(dim=1)) - torch.lgamma(alpha).sum(dim=1)
        log_likelihood = log_norm + ((alpha - 1).unsqueeze(0) * torch.log(pi)).sum(dim=-1)  # (M, B)
        return -log_likelihood.mean()


class DirichletProxyAccumulator:
    """Builds proxy Dirichlet targets from ensemble members (after Ryabinin et al., NeurIPS 2021).

    For each flow, the target mean is the members' mean probability p and the precision is the
    approximate maximum-likelihood estimate (C - 1) / (2 * sum_c p_c * (log p_c - mean_m log p_mc)).
    Target concentrations are p * precision + 1.
    """

    def __init__(self, num_flows: int, num_classes: int):
        self.mean_p = np.zeros((num_flows, num_classes), dtype=np.float32)
        self.mean_log_p = np.zeros((num_flows, num_classes), dtype=np.float32)
        self.members = 0

    def add(self, logits: np.ndarray, chunk: int = 262_144) -> None:
        for start in range(0, len(logits), chunk):
            part = logits[start:start + chunk].astype(np.float32)
            self.mean_p[start:start + chunk] += softmax(part, axis=1)
            self.mean_log_p[start:start + chunk] += log_softmax(part, axis=1)
        self.members += 1

    def concentrations(self, max_precision: float = 1e4, chunk: int = 262_144) -> np.ndarray:
        num_classes = self.mean_p.shape[1]
        out = np.empty_like(self.mean_p)
        for start in range(0, len(out), chunk):
            p = self.mean_p[start:start + chunk] / self.members
            log_p = self.mean_log_p[start:start + chunk] / self.members
            gap = 2 * np.sum(p * (np.log(np.clip(p, 1e-8, None)) - log_p), axis=1)
            precision = np.clip((num_classes - 1) / np.maximum(gap, 1e-8), 0.0, max_precision)
            out[start:start + chunk] = p * precision[:, None] + 1.0
        return out


class ProxyDirichletKL:
    """Reverse KL, KL(Dir(exp(logits)) || Dir(target)), to proxy Dirichlet targets (proxy EnDD).

    Maximum-likelihood EnDD is unstable with many classes; proxy targets avoid that. As in EnDD, the
    student's mean prediction is softmax(logits) and its energy score is the log total concentration.
    """

    def __init__(self, targets: np.ndarray, device: str, max_logit: float = 15.0):
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)).to(device)
        self.max_logit = max_logit

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        beta = self.targets[idx.to(self.targets.device)].to(logits.device)
        alpha = torch.exp(logits.clamp(-self.max_logit, self.max_logit))
        alpha0, beta0 = alpha.sum(dim=1), beta.sum(dim=1)
        kl = (torch.lgamma(alpha0) - torch.lgamma(alpha).sum(dim=1)
              - torch.lgamma(beta0) + torch.lgamma(beta).sum(dim=1)
              + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha0).unsqueeze(1))).sum(dim=1))
        return kl.mean()


# --- synthetic shortcut (RQ3) -------------------------------------------------------------------

def shortcut_codes(y: np.ndarray, groups: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Shortcut group per flow: class index mod `groups` with probability rho, otherwise random.

    Unknown flows (y < 0) always get a random group.
    """
    random_codes = rng.integers(0, groups, size=len(y))
    aligned = (rng.random(len(y)) < rho) & (y >= 0)
    return np.where(aligned, np.mod(y, groups), random_codes)


def aligned_and_flipped_codes(y: np.ndarray, groups: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Codes that agree with the class (rho = 1) and the same codes shifted to a wrong group."""
    aligned = shortcut_codes(y, groups, 1.0, rng)
    flipped = np.where(y >= 0, np.mod(aligned + 1, groups), aligned)
    return aligned, flipped


def one_hot(codes: np.ndarray, groups: int) -> np.ndarray:
    return np.eye(groups, dtype=np.float32)[codes]


def add_feature(arrays: Arrays, feature: np.ndarray) -> Arrays:
    return arrays.with_flowstats(np.concatenate([arrays.flowstats, feature.astype(np.float32)], axis=1))
