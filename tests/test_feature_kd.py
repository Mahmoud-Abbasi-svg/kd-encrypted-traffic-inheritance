import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kdtraffic.distill import SimilarityPreservingKD


def test_loss_is_cross_entropy_when_the_student_matches_the_teacher():
    """Identical feature geometry must leave only the cross-entropy term."""
    rng = np.random.default_rng(0)
    teacher = rng.normal(size=(16, 24)).astype(np.float32)
    objective = SimilarityPreservingKD(teacher, beta=100.0, device="cpu")
    idx = torch.arange(8)
    features = torch.from_numpy(teacher[:8].astype(np.float32))
    logits = torch.from_numpy(rng.normal(size=(8, 5)).astype(np.float32))
    labels = torch.from_numpy(rng.integers(0, 5, size=8))
    got = objective(logits, labels, idx, features)
    # the teacher is stored as float16, so the two Gram matrices agree only to that precision
    assert got == pytest.approx(float(F.cross_entropy(logits, labels)), abs=1e-3)


def test_a_mismatched_student_pays_a_penalty_that_grows_with_beta():
    rng = np.random.default_rng(1)
    teacher = rng.normal(size=(16, 24)).astype(np.float32)
    idx = torch.arange(8)
    features = torch.from_numpy(rng.normal(size=(8, 6)).astype(np.float32))  # different width on purpose
    logits = torch.from_numpy(rng.normal(size=(8, 5)).astype(np.float32))
    labels = torch.from_numpy(rng.integers(0, 5, size=8))
    base = float(F.cross_entropy(logits, labels))
    small = SimilarityPreservingKD(teacher, beta=1.0, device="cpu")(logits, labels, idx, features)
    large = SimilarityPreservingKD(teacher, beta=10.0, device="cpu")(logits, labels, idx, features)
    assert float(small) > base
    assert float(large) > float(small)


def test_the_objective_declares_that_it_needs_features():
    """`train_classifier` routes on this flag; without it the objective would be called wrongly."""
    objective = SimilarityPreservingKD(np.zeros((4, 3), dtype=np.float32), beta=1.0, device="cpu")
    assert objective.needs_features is True


def test_gradients_reach_the_student_features():
    rng = np.random.default_rng(2)
    teacher = rng.normal(size=(8, 12)).astype(np.float32)
    features = torch.from_numpy(rng.normal(size=(8, 4)).astype(np.float32)).requires_grad_(True)
    logits = torch.zeros(8, 3, requires_grad=True)
    labels = torch.zeros(8, dtype=torch.long)
    SimilarityPreservingKD(teacher, beta=100.0, device="cpu")(logits, labels, torch.arange(8), features).backward()
    assert features.grad is not None and torch.any(features.grad != 0)


def test_other_objectives_do_not_request_features():
    """The training loop must keep the old path for every pre-registered condition."""
    from kdtraffic.distill import HardLabelCE, HintonKD
    probs = np.full((4, 3), 1 / 3, dtype=np.float32)
    assert getattr(HintonKD(probs, 4.0, 0.9, "cpu"), "needs_features", False) is False
    assert getattr(HardLabelCE(np.zeros(4, dtype=np.int64), "cpu"), "needs_features", False) is False
    assert getattr(None, "needs_features", False) is False
