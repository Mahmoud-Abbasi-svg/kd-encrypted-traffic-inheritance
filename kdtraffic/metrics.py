"""Evaluation metrics: closed-set accuracy, unknown-traffic detection, calibration and selective risk.

Conventions: known flows have labels >= 0 and unknown flows -1; for every unknown-score, a higher
value means "more likely known".
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import f1_score, roc_auc_score

NAN = float("nan")


def closed_set_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"accuracy": NAN, "macro_f1": NAN}
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
    }


def detection_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """AUROC and FPR at 95% TPR, with known flows as the positive class."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return {"auroc": NAN, "fpr95": NAN}
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    auroc = roc_auc_score(labels, np.concatenate([id_scores, ood_scores]))
    threshold = np.quantile(id_scores, 0.05)  # keeps 95% of known flows
    return {"auroc": float(auroc), "fpr95": float(np.mean(ood_scores >= threshold))}


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    if len(y) == 0:
        return NAN
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return NAN
    p_true = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return NAN
    p_true = probs[np.arange(len(y)), y]
    return float(np.mean(np.sum(probs.astype(np.float64) ** 2, axis=1) - 2 * p_true + 1))


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_samples: int = 100_000, seed: int = 0) -> float:
    """Temperature minimising NLL on (validation) known flows."""
    if len(y) == 0:
        return 1.0
    if len(y) > max_samples:
        index = np.random.default_rng(seed).choice(len(y), size=max_samples, replace=False)
        logits, y = logits[index], y[index]
    z = logits.astype(np.float64)
    rows = np.arange(len(y))

    def nll(temperature: float) -> float:
        return float(-log_softmax(z / temperature, axis=1)[rows, y].mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-3}).x)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(confidence) == 0:
        return NAN
    order = np.argsort(-confidence, kind="stable")
    errors = (~correct[order]).astype(np.float64)
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


def oscr(id_scores: np.ndarray, id_correct: np.ndarray, ood_scores: np.ndarray) -> float:
    """Open-set classification rate: area under correct-classification rate vs false-positive rate."""
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return NAN
    order = np.argsort(-np.concatenate([id_scores, ood_scores]), kind="stable")
    correct = np.concatenate([id_correct.astype(np.float64), np.zeros(len(ood_scores))])[order]
    is_ood = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])[order]
    ccr = np.concatenate([[0.0], np.cumsum(correct) / len(id_scores)])
    fpr = np.concatenate([[0.0], np.cumsum(is_ood) / len(ood_scores)])
    return float(np.trapezoid(ccr, fpr))


def open_set_report(y: np.ndarray, probs: np.ndarray, scores: dict[str, np.ndarray],
                    probs_ts: np.ndarray | None = None) -> dict[str, float]:
    known = y >= 0
    pred = probs.argmax(axis=1)
    y_known, probs_known = y[known], probs[known]
    correct = pred[known] == y_known
    report: dict[str, float] = {"n_known": int(known.sum()), "n_unknown": int((~known).sum())}
    report.update(closed_set_metrics(y_known, pred[known]))
    for name, score in scores.items():
        detection = detection_metrics(score[known], score[~known])
        report[f"auroc_{name}"] = detection["auroc"]
        report[f"fpr95_{name}"] = detection["fpr95"]
        report[f"oscr_{name}"] = oscr(score[known], correct, score[~known])
    report["ece"] = expected_calibration_error(probs_known, y_known)
    report["nll"] = negative_log_likelihood(probs_known, y_known)
    report["brier"] = brier_score(probs_known, y_known)
    report["aurc"] = aurc(probs_known.max(axis=1), correct) if len(y_known) else NAN
    if probs_ts is not None:
        report["ece_ts"] = expected_calibration_error(probs_ts[known], y_known)
        report["nll_ts"] = negative_log_likelihood(probs_ts[known], y_known)
    return report
