from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
