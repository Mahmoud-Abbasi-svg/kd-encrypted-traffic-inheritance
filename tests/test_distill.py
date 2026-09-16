import numpy as np
import pytest
import torch
from scipy.special import softmax

from kdtraffic.distill import (DirichletProxyAccumulator, EnsembleDistributionDistillation, HintonKD,
                               ProxyDirichletKL, aligned_and_flipped_codes, shortcut_codes, softmax_chunked)


def test_proxy_targets_track_member_agreement():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 5)) * 3
    agreeing = np.repeat(base, 4, axis=0) + rng.normal(scale=0.01, size=(4, 5))
    disagreeing = rng.normal(size=(4, 5)) * 3
    accumulator = DirichletProxyAccumulator(num_flows=2, num_classes=5)
    for m in range(4):
        accumulator.add(np.stack([agreeing[m], disagreeing[m]]).astype(np.float32))
    beta = accumulator.concentrations()
    assert np.all(beta >= 1.0)
    assert beta[0].sum() > beta[1].sum()  # members that agree give a more concentrated target
    mean_p = softmax(np.stack([agreeing, disagreeing], axis=1), axis=2).mean(axis=0)
    np.testing.assert_allclose((beta - 1) / (beta - 1).sum(axis=1, keepdims=True), mean_p, rtol=1e-3, atol=1e-4)


def test_proxy_kl_is_zero_at_target_and_positive_elsewhere():
    beta = np.array([[5.0, 2.0, 1.5], [1.2, 1.1, 30.0]], dtype=np.float32)
    loss = ProxyDirichletKL(beta, device="cpu")
    idx, labels = torch.arange(2), torch.zeros(2, dtype=torch.long)
    at_target = loss(torch.log(torch.tensor(beta)), labels, idx)
    assert at_target.item() == pytest.approx(0.0, abs=1e-4)
    assert loss(torch.zeros(2, 3), labels, idx).item() > 0.1


def test_proxy_kl_is_finite_for_extreme_logits():
    beta = np.full((6, 10), 1.5, dtype=np.float32)
    logits = torch.tensor(np.random.default_rng(3).normal(scale=50, size=(6, 10)), dtype=torch.float32,
                          requires_grad=True)
    value = ProxyDirichletKL(beta, device="cpu")(logits, torch.zeros(6, dtype=torch.long), torch.arange(6))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_softmax_chunked_matches_scipy():
    logits = np.random.default_rng(0).normal(size=(1000, 7)).astype(np.float32)
    out = softmax_chunked(logits, temperature=2.0, chunk=128)
    np.testing.assert_allclose(out, softmax(logits / 2.0, axis=1), rtol=1e-5, atol=1e-6)
    assert out.dtype == np.float32


def test_hinton_kd_is_minimal_when_student_matches_teacher():
    logits = torch.randn(32, 5)
    targets = softmax(logits.numpy() / 3.0, axis=1)
    loss = HintonKD(targets, temperature=3.0, alpha=1.0, device="cpu")
    idx = torch.arange(32)
    matching = loss(logits, torch.zeros(32, dtype=torch.long), idx)
    assert matching.item() == pytest.approx(0.0, abs=1e-5)
    assert loss(torch.randn(32, 5), torch.zeros(32, dtype=torch.long), idx).item() > matching.item()


def test_hinton_kd_mixes_cross_entropy():
    logits = torch.randn(8, 4, requires_grad=True)
    targets = np.full((8, 4), 0.25, dtype=np.float32)
    loss = HintonKD(targets, temperature=2.0, alpha=0.5, device="cpu")(logits, torch.arange(8) % 4, torch.arange(8))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_endd_prefers_concentration_on_the_ensemble_mean():
    p = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    members = np.tile(p, (5, 4, 1)).astype(np.float16)  # (M=5, N=4, C=3)
    loss = EnsembleDistributionDistillation(members, device="cpu")
    idx, labels = torch.arange(4), torch.zeros(4, dtype=torch.long)
    matched = loss(torch.log(torch.tensor(100 * p)).repeat(4, 1), labels, idx)
    flat = loss(torch.zeros(4, 3), labels, idx)
    assert torch.isfinite(matched) and matched.item() < flat.item()


def test_endd_is_finite_for_extreme_logits():
    members = softmax(np.random.default_rng(1).normal(size=(3, 6, 10)), axis=-1).astype(np.float16)
    logits = torch.tensor(np.random.default_rng(2).normal(scale=50, size=(6, 10)), dtype=torch.float32,
                          requires_grad=True)
    value = EnsembleDistributionDistillation(members, device="cpu")(logits, torch.zeros(6, dtype=torch.long),
                                                                    torch.arange(6))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_shortcut_codes_follow_rho():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 50, size=20_000)
    y[:1000] = -1  # unknown flows
    always = shortcut_codes(y, 8, 1.0, rng)
    assert np.all(always[y >= 0] == y[y >= 0] % 8)
    half = shortcut_codes(y, 8, 0.5, rng)
    match = np.mean(half[y >= 0] == y[y >= 0] % 8)
    assert match == pytest.approx(0.5 + 0.5 / 8, abs=0.02)
    never = shortcut_codes(y, 8, 0.0, rng)
    assert np.mean(never[y >= 0] == y[y >= 0] % 8) == pytest.approx(1 / 8, abs=0.02)


def test_flipped_codes_always_wrong_for_known_flows():
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 7, 8, 15, -1])
    aligned, flipped = aligned_and_flipped_codes(y, 8, rng)
    known = y >= 0
    assert np.all(aligned[known] == y[known] % 8)
    assert np.all(flipped[known] != y[known] % 8)
    assert flipped[~known] == aligned[~known]
