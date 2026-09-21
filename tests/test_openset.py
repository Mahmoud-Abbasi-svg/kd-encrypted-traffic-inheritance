import numpy as np
import pytest
import torch
from torch import nn

from kdtraffic.data import Arrays
from kdtraffic.openset import FeatureScorer, accumulate_feature_stats, extract_features, score_arrays


class FlowstatsProjection(nn.Module):
    """Minimal model with the forward_features / forward_head contract the project relies on."""

    def __init__(self, dim: int, flowstats_dim: int = 44, num_classes: int = 6):
        super().__init__()
        self.proj = nn.Linear(flowstats_dim, dim)
        self.head = nn.Linear(dim, num_classes)

    def forward_features(self, ppi, flowstats):
        return self.proj(flowstats)

    def forward_head(self, features):
        return self.head(features)

    def forward(self, ppi, flowstats):
        return self.forward_head(self.forward_features(ppi, flowstats))


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    n, classes = 400, 6
    y = rng.integers(0, classes, size=n)
    y[::10] = -1  # some unknown flows, which must be skipped when fitting
    return Arrays(rng.normal(size=(n, 3, 30)).astype(np.float32),
                  rng.normal(size=(n, 44)).astype(np.float32),
                  y.astype(np.int64), np.array([f"s{c}" for c in y]))


def test_streaming_statistics_match_a_direct_computation(arrays):
    torch.manual_seed(0)
    model = FlowstatsProjection(dim=12)
    stats = accumulate_feature_stats(model, arrays, device="cpu", amp=False, batch_size=64, num_classes=6)

    known = arrays.y >= 0
    with torch.no_grad():
        features = model.forward_features(None, torch.from_numpy(arrays.flowstats[known])).double().numpy()
    labels = arrays.y[known]
    assert stats.n == int(known.sum())
    assert np.array_equal(stats.counts, np.bincount(labels, minlength=6))
    expected_sums = np.stack([features[labels == c].sum(axis=0) for c in range(6)])
    assert np.allclose(stats.sums, expected_sums, atol=1e-8)
    assert np.allclose(stats.scatter, features.T @ features, atol=1e-6)


def test_knn_score_equals_brute_force(arrays):
    torch.manual_seed(1)
    model = FlowstatsProjection(dim=8)
    stats = accumulate_feature_stats(model, arrays, device="cpu", batch_size=128, num_classes=6)
    index = np.arange(0, 200)
    reference = extract_features(model, arrays, index, device="cpu")
    scorer = FeatureScorer.fit(stats, reference, k=5)
    scores = score_arrays(model, arrays, scorer, device="cpu", batch_size=64)

    features = extract_features(model, arrays, np.arange(len(arrays)), device="cpu")
    distances = np.linalg.norm(features[:, None, :] - reference[None, :, :], axis=2)
    expected = -np.sort(distances, axis=1)[:, 4]
    assert scores["knnfeat"] == pytest.approx(expected, abs=1e-3)


def test_mahalanobis_ranks_known_flows_above_unknown_ones():
    """Flows from the fitted classes must score higher than flows drawn far from all of them."""
    rng = np.random.default_rng(2)
    dim, classes, per_class = 8, 4, 200
    centres = rng.normal(scale=6.0, size=(classes, 44))
    known_stats = np.concatenate([centres[c] + rng.normal(scale=0.5, size=(per_class, 44)) for c in range(classes)])
    y = np.repeat(np.arange(classes), per_class)
    far = rng.normal(loc=40.0, scale=0.5, size=(100, 44))

    train = Arrays(np.zeros((len(y), 3, 30), np.float32), known_stats.astype(np.float32), y, np.array(["k"] * len(y)))
    evaluate = Arrays(np.zeros((len(y) + 100, 3, 30), np.float32),
                      np.concatenate([known_stats, far]).astype(np.float32),
                      np.concatenate([y, -np.ones(100, dtype=np.int64)]),
                      np.array(["k"] * len(y) + ["u"] * 100))

    torch.manual_seed(3)
    model = FlowstatsProjection(dim=dim, num_classes=classes)
    stats = accumulate_feature_stats(model, train, device="cpu", batch_size=256, num_classes=classes)
    scorer = FeatureScorer.fit(stats, extract_features(model, train, np.arange(len(train)), device="cpu"), k=5)
    scores = score_arrays(model, evaluate, scorer, device="cpu", batch_size=256)

    known_mask = evaluate.y >= 0
    for name in ("maha", "knnfeat"):
        assert scores[name].dtype == np.float32
        assert np.isfinite(scores[name]).all()
        assert scores[name][known_mask].mean() > scores[name][~known_mask].mean()


def test_scores_are_float32_and_survive_large_distances():
    """Squared Mahalanobis distances overflow float16; the scorer must not return such a dtype."""
    rng = np.random.default_rng(4)
    stats_dim = 44
    y = rng.integers(0, 3, size=300)
    train = Arrays(np.zeros((300, 3, 30), np.float32), rng.normal(size=(300, stats_dim)).astype(np.float32),
                   y, np.array(["k"] * 300))
    far = Arrays(np.zeros((50, 3, 30), np.float32),
                 (rng.normal(size=(50, stats_dim)) + 5000).astype(np.float32),
                 -np.ones(50, dtype=np.int64), np.array(["u"] * 50))
    torch.manual_seed(5)
    model = FlowstatsProjection(dim=16, num_classes=3)
    stats = accumulate_feature_stats(model, train, device="cpu", num_classes=3)
    scorer = FeatureScorer.fit(stats, extract_features(model, train, np.arange(300), device="cpu"), k=3)
    scores = score_arrays(model, far, scorer, device="cpu")
    assert scores["maha"].dtype == np.float32
    assert np.isfinite(scores["maha"]).all()
    # why float32 is required: the same values in float16 either overflow or collapse into ties,
    # and tied scores are counted as half a win by the AUROC used throughout this project
    as_float16 = scores["maha"].astype(np.float16)
    assert not np.isfinite(as_float16).all() or len(np.unique(as_float16)) < len(np.unique(scores["maha"]))
