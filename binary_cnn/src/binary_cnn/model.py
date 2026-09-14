"""A deliberately small LeNet-scale classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class BytePairCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 3 * 4 * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ByteWindowNet(nn.Module):
    """A reduced Magika-style network over consecutive four-byte chunks."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=256)
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        chunks = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1] * 4)
        features = self.trunk(chunks).amax(dim=1)
        return self.classifier(features)


class HistogramResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted to grayscale byte-pair maps."""

    def __init__(self, num_classes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Histogram inputs are log1p-scaled between zero and log(10,001).
        images = inputs.expand(-1, 3, -1, -1) / 9.210440366976517
        images = (images - self.imagenet_mean) / self.imagenet_std
        return self.backbone(images)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True
