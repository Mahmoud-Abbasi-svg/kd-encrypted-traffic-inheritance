import numpy as np
import pytest
from scipy.special import softmax

from kdtraffic import metrics


def test_detection_perfect_and_reversed():
    perfect = metrics.detection_metrics(np.array([2.0, 3.0]), np.array([0.0, 1.0]))
    assert perfect == {"auroc": 1.0, "fpr95": 0.0}
    reversed_ = metrics.detection_metrics(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert reversed_ == {"auroc": 0.0, "fpr95": 1.0}


def test_detection_without_unknowns_is_nan():
    result = metrics.detection_metrics(np.array([1.0]), np.array([]))
    assert np.isnan(result["auroc"]) and np.isnan(result["fpr95"])


def test_ece_calibrated_and_overconfident():
    probs = np.tile([0.8, 0.2], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    assert metrics.expected_calibration_error(probs, y) == pytest.approx(0.0, abs=1e-9)
    overconfident = np.tile([1.0, 0.0], (10, 1))
    assert metrics.expected_calibration_error(overconfident, np.array([0, 1] * 5)) == pytest.approx(0.5)


def test_nll_and_brier_perfect_predictions():
    probs = np.eye(3)
    y = np.arange(3)
    assert metrics.negative_log_likelihood(probs, y) == pytest.approx(0.0, abs=1e-9)
    assert metrics.brier_score(probs, y) == pytest.approx(0.0, abs=1e-9)


def test_temperature_recovers_scaling():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(20_000, 5)) * 2
    probs = softmax(true_logits, axis=1)
    y = (probs.cumsum(axis=1) < rng.random((len(probs), 1))).sum(axis=1)
    assert metrics.fit_temperature(3 * true_logits, y) == pytest.approx(3.0, abs=0.3)


def test_aurc_perfect_ranking():
    confidence = np.array([4.0, 3.0, 2.0, 1.0])
    correct = np.array([True, True, False, False])
    assert metrics.aurc(confidence, correct) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)


def test_oscr_extremes():
    id_scores, ood_scores = np.array([3.0, 4.0]), np.array([1.0, 2.0])
    assert metrics.oscr(id_scores, np.array([True, True]), ood_scores) == pytest.approx(1.0)
    assert metrics.oscr(ood_scores, np.array([True, True]), id_scores) == pytest.approx(0.0)


def test_closed_set_macro_f1_uses_present_classes():
    result = metrics.closed_set_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]))
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_open_set_report_keys():
    y = np.array([0, 1, -1, -1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]])
    report = metrics.open_set_report(y, probs, {"msp": probs.max(axis=1)}, probs_ts=probs)
    assert report["n_known"] == 2 and report["n_unknown"] == 2
    assert report["auroc_msp"] == pytest.approx(1.0)
    for key in ("macro_f1", "ece", "nll", "brier", "aurc", "oscr_msp", "ece_ts", "nll_ts"):
        assert key in report
