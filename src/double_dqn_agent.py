from __future__ import annotations

import torch
from torch import nn

from dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def train_step(self) -> float:
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.hparams.batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(obs_t).gather(1, actions_t)
        with torch.no_grad():
            best_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_obs_t).gather(1, best_actions)
            targets = rewards_t + self.hparams.gamma * (1.0 - dones_t) * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.hparams.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())
