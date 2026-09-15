"""Teacher and student architectures. All expose forward_features / forward_head like cesnet-models."""

from __future__ import annotations

import torch
from cesnet_models.architectures.multimodal_cesnet import Multimodal_CESNET
from cesnet_models.models import mm_cesnet_v2
from torch import nn

PPI_CHANNELS = 3


class StudentCNN(nn.Module):
    """Small multimodal 1D-CNN (~100k parameters with 130 classes and 44 flow statistics)."""

    def __init__(self, num_classes: int, flowstats_dim: int, width: int = 48, stats_width: int = 64,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        wide = 2 * width
        self.ppi = nn.Sequential(
            nn.Conv1d(PPI_CHANNELS, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, wide, kernel_size=5, padding=2), nn.BatchNorm1d(wide), nn.ReLU(),
            nn.Conv1d(wide, wide, kernel_size=3, padding=1), nn.BatchNorm1d(wide), nn.ReLU(),
        )
        self.stats = nn.Sequential(nn.Linear(flowstats_dim, stats_width), nn.BatchNorm1d(stats_width), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(2 * wide + stats_width, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward_features(self, ppi: torch.Tensor, flowstats: torch.Tensor) -> torch.Tensor:
        h = self.ppi(ppi)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.shared(torch.cat([h, self.stats(flowstats)], dim=1))

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, ppi: torch.Tensor, flowstats: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(ppi, flowstats))


def wide_teacher(num_classes: int, flowstats_dim: int) -> Multimodal_CESNET:
    """Teacher B: a single wider multimodal CESNET network (the teacher-swap control)."""
    return Multimodal_CESNET(
        num_classes=num_classes, flowstats_input_size=flowstats_dim, ppi_input_channels=PPI_CHANNELS,
        cnn_ppi_num_blocks=3, cnn_ppi_channels1=400, cnn_ppi_channels2=600, cnn_ppi_channels3=600,
        cnn_ppi_use_pooling=True, cnn_ppi_dropout_rate=0.1,
        mlp_flowstats_num_hidden=2, mlp_flowstats_size1=450, mlp_flowstats_size2=450, mlp_flowstats_dropout_rate=0.1,
        mlp_shared_num_hidden=1, mlp_shared_size=1200, mlp_shared_dropout_rate=0.2,
    )


def build_model(name: str, num_classes: int, flowstats_dim: int, student_width: int = 48) -> nn.Module:
    if name == "mm_cesnet_v2":
        return mm_cesnet_v2(num_classes=num_classes, flowstats_input_size=flowstats_dim, ppi_input_channels=PPI_CHANNELS)
    if name == "wide_teacher":
        return wide_teacher(num_classes, flowstats_dim)
    if name == "student":
        return StudentCNN(num_classes, flowstats_dim, width=student_width)
    raise ValueError(f"Unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
