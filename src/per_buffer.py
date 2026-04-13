from __future__ import annotations

from typing import Any

import numpy as np

from replay_buffer import Transition


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Any] = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, Any]:
        idx = self._retrieve(s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.n_entries

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(
            obs=np.asarray(obs, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            done=float(done),
        )
        self.tree.add(self.max_priority, transition)

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)
        transitions: list[Transition] = []

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, transition = self.tree.get(s)
            indices[i] = idx
            priorities[i] = max(priority, 1e-8)
            transitions.append(transition)

        probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        obs = np.stack([t.obs for t in transitions])
        actions = np.asarray([t.action for t in transitions], dtype=np.int64)
        rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in transitions])
        dones = np.asarray([t.done for t in transitions], dtype=np.float32)

        return obs, actions, rewards, next_obs, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), float(priority))
            self.max_priority = max(self.max_priority, float(priority))
