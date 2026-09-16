import numpy as np
import pytest

from kdtraffic.data import LOW_COVERAGE_WEEKS, evaluation_windows, period_name
from kdtraffic.evaluation import from_logits
from kdtraffic.inheritance import agreement, error_overlap, inheritance_row, rank_correlation
from kdtraffic.stats import cluster_bootstrap, cluster_ids


def test_rank_correlation_extremes():
    a = np.arange(100, dtype=float)
    assert rank_correlation(a, a) == pytest.approx(1.0)
    assert rank_correlation(a, -a) == pytest.approx(-1.0)
    assert np.isnan(rank_correlation(a[:2], a[:2]))


def test_error_overlap_and_agreement():
    y = np.array([0, 1, 2, 3])
    assert error_overlap(np.array([1, 1, 2, 0]), np.array([1, 1, 2, 0]), y) == pytest.approx(1.0)
    assert error_overlap(np.array([1, 1, 2, 3]), np.array([0, 1, 2, 0]), y) == pytest.approx(0.0)
    assert np.isnan(error_overlap(y, y, y))
    assert agreement(np.array([0, 1]), np.array([0, 2])) == pytest.approx(0.5)


def test_inheritance_row_identical_models():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 4)).astype(np.float32)
    y = rng.integers(-1, 4, size=200)
    out = from_logits(logits)
    row = inheritance_row("s", "t", "val", y, out, out)
    assert row["spearman_energy"] == pytest.approx(1.0)
    assert row["top1_agreement_known"] == pytest.approx(1.0)


def test_cluster_bootstrap_mean():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(200), 5)
    values = rng.normal(loc=2.0, size=200)[clusters] + rng.normal(scale=0.1, size=1000)
    result = cluster_bootstrap(lambda idx: values[idx].mean(), clusters, n_boot=300, seed=1)
    assert result["estimate"] == pytest.approx(values.mean())
    assert result["low"] < 2.0 < result["high"]
    assert result["clusters"] == 200
    assert cluster_bootstrap(lambda idx: values[idx].mean(), clusters, n_boot=50, seed=1) == \
        cluster_bootstrap(lambda idx: values[idx].mean(), clusters, n_boot=50, seed=1)


def test_cluster_bootstrap_keeps_whole_clusters():
    clusters = np.array([0, 1, 0, 1, 2, 2])
    sizes = []
    cluster_bootstrap(lambda idx: sizes.append(np.bincount(clusters[idx], minlength=3)) or 0.0, clusters, n_boot=20)
    for counts in sizes:
        assert np.all(counts % 2 == 0)  # every cluster has two flows


def test_cluster_ids():
    ids = cluster_ids(np.array([20220101, 20220101, 20220102]), np.array(["a", "b", "a"]))
    assert len(set(ids)) == 3


def test_evaluation_windows_skip_low_coverage_weeks():
    windows = evaluation_windows((15, 15))
    assert windows[0] == [16, 17, 18, 19]
    assert all(w not in LOW_COVERAGE_WEEKS for block in windows for w in block)
    assert windows[-1] == [48, 49, 51]
    assert evaluation_windows((41, 41))[0] == [42, 43, 44, 45]


def test_period_name_distinguishes_skipped_weeks():
    assert period_name([16, 17, 18, 19]) == "W-2022-16-19"
    assert period_name([48, 49, 51]) == "W-2022-48-51-skip50"
