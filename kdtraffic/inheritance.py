"""Per-flow agreement between a student and a teacher (what the student inherits)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from kdtraffic.evaluation import Outputs

NAN = float("nan")


def rank_correlation(a: np.ndarray, b: np.ndarray, max_samples: int = 200_000, seed: int = 0) -> float:
    if len(a) < 3:
        return NAN
    if len(a) > max_samples:
        index = np.random.default_rng(seed).choice(len(a), size=max_samples, replace=False)
        a, b = a[index], b[index]
    return float(spearmanr(a, b).statistic)


def error_overlap(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> float:
    """Jaccard index of the two models' misclassified known flows."""
    wrong_a, wrong_b = pred_a != y, pred_b != y
    union = np.sum(wrong_a | wrong_b)
    return float(np.sum(wrong_a & wrong_b) / union) if union else NAN


def agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    return float(np.mean(pred_a == pred_b)) if len(pred_a) else NAN


def predictions(outputs: Outputs) -> np.ndarray:
    """Top-1 predictions, whether `probs` holds the (N, C) matrix or already just the (N,) argmax.

    Callers that keep many models in memory at once may store predictions instead of full
    probability matrices; the values are identical either way.
    """
    probs = outputs.probs
    return probs.argmax(axis=1) if probs.ndim == 2 else probs


def inheritance_row(student: str, teacher: str, split: str, y: np.ndarray, student_out: Outputs,
                    teacher_out: Outputs) -> dict:
    known = y >= 0
    row = {"student": student, "teacher": teacher, "split": split,
           "n_known": int(known.sum()), "n_unknown": int((~known).sum())}
    for score in sorted(set(student_out.scores) & set(teacher_out.scores)):
        s, t = student_out.scores[score], teacher_out.scores[score]
        row[f"spearman_{score}"] = rank_correlation(s, t)
        row[f"spearman_{score}_unknown"] = rank_correlation(s[~known], t[~known])
    pred_s, pred_t = predictions(student_out), predictions(teacher_out)
    row["top1_agreement_known"] = agreement(pred_s[known], pred_t[known])
    row["error_jaccard_known"] = error_overlap(pred_s[known], pred_t[known], y[known])
    return row
