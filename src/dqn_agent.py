from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models import QNetwork
from replay_buffer import ReplayBuffer


@dataclass
class DQNHyperParams:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    gradient_steps: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    hidden_dim1: int = 256
    hidden_dim2: int = 256
    max_grad_norm: float = 10.0


class DQNAgent:
    def __init__(self, obs_dim: int, num_actions: int, device: str = "cpu", hparams: DQNHyperParams | None = None) -> None:
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        self.hparams = hparams or DQNHyperParams()

        hidden_dims = (self.hparams.hidden_dim1, self.hparams.hidden_dim2)
        self.online_net = QNetwork(self.obs_dim, self.num_actions, hidden_dims=hidden_dims).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.num_actions, hidden_dims=hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.hparams.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.hparams.buffer_size)
        self.total_steps = 0

    def epsilon(self, step: int | None = None) -> float:
        step = self.total_steps if step is None else step
        frac = min(1.0, step / max(1, self.hparams.epsilon_decay_steps))
        return self.hparams.epsilon_start + frac * (self.hparams.epsilon_end - self.hparams.epsilon_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.epsilon():
            return int(np.random.randint(self.num_actions))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def maybe_train_step(self) -> list[float]:
        losses: list[float] = []
        if self.total_steps < self.hparams.learning_starts:
            return losses
        if len(self.replay_buffer) < self.hparams.batch_size:
            return losses
        if self.total_steps % self.hparams.train_frequency != 0:
            return losses

        for _ in range(self.hparams.gradient_steps):
            losses.append(self.train_step())

        if self.total_steps % self.hparams.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return losses

    def train_step(self) -> float:
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.hparams.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(obs_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_t).max(dim=1, keepdim=True)[0]
            targets = rewards_t + self.hparams.gamma * (1.0 - dones_t) * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.hparams.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "hparams": asdict(self.hparams),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device)
        hparams = DQNHyperParams(**ckpt["hparams"])
        agent = cls(obs_dim=ckpt["obs_dim"], num_actions=ckpt["num_actions"], device=device, hparams=hparams)
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.total_steps = int(ckpt["total_steps"])
        return agent

    def export_hparams(self) -> dict[str, Any]:
        return asdict(self.hparams)
