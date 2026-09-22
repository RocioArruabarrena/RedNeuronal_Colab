# app/ml/model.py
"""
Red neuronal para el predictor de compatibilidad de amistad.

Arquitectura: MLP simple (dataset chico de 94 ejemplos)
    Linear(5->8) + ReLU + Dropout(0.3)
    Linear(8->4) + ReLU + Dropout(0.2)
    Linear(4->1) + Sigmoid
"""

import torch
import torch.nn as nn


class CompatibilityNet(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)