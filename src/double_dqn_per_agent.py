from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dqn_agent import DQNAgent, DQNHyperParams
from models import QNetwork
from per_buffer import PrioritizedReplayBuffer


@dataclass
class DoubleDQNPERHyperParams(DQNHyperParams):
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_anneal_steps: int = 80_000
    per_epsilon: float = 1e-6


class DoubleDQNPERAgent(DQNAgent):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        device: str = "gpu",
        hparams: DoubleDQNPERHyperParams | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        self.hparams = hparams or DoubleDQNPERHyperParams()

        hidden_dims = (self.hparams.hidden_dim1, self.hparams.hidden_dim2)
        self.online_net = QNetwork(self.obs_dim, self.num_actions, hidden_dims=hidden_dims).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.num_actions, hidden_dims=hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.hparams.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.replay_buffer = PrioritizedReplayBuffer(
            self.hparams.buffer_size,
            alpha=self.hparams.alpha,
            epsilon=self.hparams.per_epsilon,
        )
        self.total_steps = 0

    def _beta(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.hparams.beta_anneal_steps))
        return self.hparams.beta_start + frac * (self.hparams.beta_end - self.hparams.beta_start)

    def train_step(self) -> float:
        obs, actions, rewards, next_obs, dones, is_weights, indices = self.replay_buffer.sample(
            self.hparams.batch_size, beta=self._beta()
        )

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(obs_t).gather(1, actions_t)

        with torch.no_grad():
            best_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_obs_t).gather(1, best_actions)
            targets = rewards_t + self.hparams.gamma * (1.0 - dones_t) * next_q_values

        td_errors = (q_values - targets).detach().cpu().numpy().squeeze(1)
        self.replay_buffer.update_priorities(indices, td_errors)

        elementwise_loss = self.loss_fn(q_values, targets)
        loss = (weights_t * elementwise_loss).mean()

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
    def load(cls, path: str | Path, device: str = "cpu") -> "DoubleDQNPERAgent":
        ckpt = torch.load(path, map_location=device)
        hparams = DoubleDQNPERHyperParams(**ckpt["hparams"])
        agent = cls(obs_dim=ckpt["obs_dim"], num_actions=ckpt["num_actions"], device=device, hparams=hparams)
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.total_steps = int(ckpt["total_steps"])
        return agent
