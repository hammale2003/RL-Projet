from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env
import numpy as np
import torch

from config_core import SHARED_CORE_ENV_ID, env_config


ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(env_config())
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


def get_obs_dim(env: gym.Env) -> int:
    obs, _ = env.reset()
    return int(flatten_obs(obs).shape[0])


def get_num_actions(env: gym.Env) -> int:
    if not hasattr(env.action_space, "n"):
        raise ValueError("Expected a discrete action space for the shared core task.")
    return int(env.action_space.n)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RunningMean:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, x: float) -> float:
        self.values.append(float(x))
        return self.mean

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))
