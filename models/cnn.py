"""
models/cnn.py
Simple CNN for CIFAR-10 — intentionally lightweight for Phase 3 baseline.
Do NOT optimise the architecture yet; the goal is stable federation, not SOTA accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    3-block CNN for CIFAR-10 (32x32 RGB → 10 classes).
    Final FC layer is the 'sensitive' layer that Phase 3 Ring 2 will encrypt with HE.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ----- Feature extractor (early layers — low sensitivity) -----
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32x32 → 16x16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16x16 → 8x8
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8x8 → 4x4
        )

        self.dropout = nn.Dropout(0.4)

        # ----- Classifier (high sensitivity — HE target in Ring 2) -----
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)   # ← this layer gets encrypted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_sensitive_weights(self) -> dict:
        """Return only the final classifier weights — these are the HE targets."""
        return {
            "fc2.weight": self.fc2.weight.detach().cpu().numpy(),
            "fc2.bias":   self.fc2.bias.detach().cpu().numpy(),
        }
