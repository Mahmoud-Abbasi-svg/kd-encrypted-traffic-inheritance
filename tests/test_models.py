import torch

from kdtraffic.models import StudentCNN, build_model, count_parameters


def test_student_size_and_shapes():
    model = StudentCNN(num_classes=130, flowstats_dim=44).eval()
    assert 70_000 <= count_parameters(model) <= 140_000
    ppi, stats = torch.randn(4, 3, 30), torch.randn(4, 44)
    with torch.no_grad():
        features = model.forward_features(ppi, stats)
        assert model.forward_head(features).shape == (4, 130)
        assert model(ppi, stats).shape == (4, 130)


def test_teacher_shapes_and_capacity_gap():
    teacher = build_model("mm_cesnet_v2", num_classes=130, flowstats_dim=44).eval()
    with torch.no_grad():
        assert teacher(torch.randn(4, 3, 30), torch.randn(4, 44)).shape == (4, 130)
    assert count_parameters(teacher) > 10 * count_parameters(StudentCNN(130, 44))


def test_wide_teacher_builds():
    wide = build_model("wide_teacher", num_classes=130, flowstats_dim=44).eval()
    with torch.no_grad():
        assert wide(torch.randn(2, 3, 30), torch.randn(2, 44)).shape == (2, 130)
