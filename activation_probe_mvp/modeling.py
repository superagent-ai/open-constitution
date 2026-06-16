from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [batch, hidden_size]
        return self.head(hidden).squeeze(-1)


class RiskSmoother:
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.values: list[float] = []

    def update(self, value: float) -> float:
        self.values.append(float(value))
        self.values = self.values[-self.window_size :]
        return sum(self.values) / len(self.values)
