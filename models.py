from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """The exact sequential CNN specified in Homework 1."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(seed: int = 7) -> SmallCNN:
    torch.manual_seed(seed)
    return SmallCNN()


if __name__ == "__main__":
    model = build_model().eval()
    with torch.inference_mode():
        for size in (32, 224, 512):
            output = model(torch.randn(2, 3, size, size))
            assert output.shape == (2, 100)
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
