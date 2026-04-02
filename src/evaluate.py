from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3 import PPO as SB3PPO
from tqdm import tqdm

from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from utils import ROOT, flatten_obs, make_env, save_csv_rows, save_json, set_global_seeds


def evaluate_custom(model_path: Path, seed: int, num_episodes: int) -> list[dict[str, float | int | bool]]:
    env = make_env(seed=seed)
    agent = DQNAgent.load(model_path)
    rows: list[dict[str, float | int | bool]] = []
    for ep in tqdm(range(num_episodes), desc=f"Eval custom seed {seed}"):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        done = False
        total_reward = 0.0
        steps = 0
        crashed = False
        while not done:
            action = agent.act_greedy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = flatten_obs(next_obs)
            total_reward += float(reward)
            steps += 1
            crashed = crashed or bool(
                info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False)
            )
        rows.append({"episode": ep, "reward": total_reward, "length": steps, "crashed": int(crashed)})
    env.close()
    return rows


def evaluate_double_dqn(model_path: Path, seed: int, num_episodes: int) -> list[dict[str, float | int | bool]]:
    env = make_env(seed=seed)
    agent = DoubleDQNAgent.load(model_path)
    rows: list[dict[str, float | int | bool]] = []
    for ep in tqdm(range(num_episodes), desc=f"Eval DoubleDQN seed {seed}"):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        done = False
        total_reward = 0.0
        steps = 0
        crashed = False
        while not done:
            action = agent.act_greedy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = flatten_obs(next_obs)
            total_reward += float(reward)
            steps += 1
            crashed = crashed or bool(
                info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False)
            )
        rows.append({"episode": ep, "reward": total_reward, "length": steps, "crashed": int(crashed)})
    env.close()
    return rows


def evaluate_sb3(model_path: Path, seed: int, num_episodes: int) -> list[dict[str, float | int | bool]]:
    env = make_env(seed=seed)
    model = SB3DQN.load(model_path, env=env, device="cpu")
    rows: list[dict[str, float | int | bool]] = []
    for ep in tqdm(range(num_episodes), desc=f"Eval SB3 DQN seed {seed}"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        crashed = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += float(reward)
            steps += 1
            crashed = crashed or bool(
                info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False)
            )
        rows.append({"episode": ep, "reward": total_reward, "length": steps, "crashed": int(crashed)})
    env.close()
    return rows


def evaluate_ppo(model_path: Path, seed: int, num_episodes: int) -> list[dict[str, float | int | bool]]:
    env = make_env(seed=seed)
    model = SB3PPO.load(model_path, env=env, device="cpu")
    rows: list[dict[str, float | int | bool]] = []
    for ep in tqdm(range(num_episodes), desc=f"Eval PPO seed {seed}"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        crashed = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += float(reward)
            steps += 1
            crashed = crashed or bool(
                info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False)
            )
        rows.append({"episode": ep, "reward": total_reward, "length": steps, "crashed": int(crashed)})
    env.close()
    return rows


EVALUATORS: dict[str, Callable[[Path, int, int], list[dict[str, float | int | bool]]]] = {
    "custom_dqn": evaluate_custom,
    "double_dqn": evaluate_double_dqn,
    "sb3_dqn": evaluate_sb3,
    "ppo_shaped": evaluate_ppo,
}


def main(args: argparse.Namespace) -> None:
    if args.model_type not in EVALUATORS:
        raise ValueError(f"Unknown model_type={args.model_type}")

    results_root = ensure_eval_dir(args.model_type, args.output_subdir)
    summary_rows: list[dict[str, float | int]] = []

    for seed in args.seeds:
        set_global_seeds(seed)
        model_path = Path(args.model_template.format(seed=seed))
        rows = EVALUATORS[args.model_type](
            model_path,
            seed=args.eval_seed_base + 1000 * seed,
            num_episodes=args.num_episodes,
        )

        seed_dir = results_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        save_csv_rows(
            seed_dir / f"evaluation_{args.num_episodes}_episodes.csv",
            ["episode", "reward", "length", "crashed"],
            rows,
        )

        rewards = np.asarray([r["reward"] for r in rows], dtype=np.float32)
        lengths = np.asarray([r["length"] for r in rows], dtype=np.float32)
        crashes = np.asarray([r["crashed"] for r in rows], dtype=np.float32)

        seed_summary = {
            "seed": seed,
            "num_episodes": args.num_episodes,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=0)),
            "length_mean": float(lengths.mean()),
            "length_std": float(lengths.std(ddof=0)),
            "crash_rate": float(crashes.mean()),
        }
        save_json(seed_dir / f"evaluation_summary_{args.num_episodes}.json", seed_summary)
        summary_rows.append(seed_summary)

    overall = {
        "model_type": args.model_type,
        "output_subdir": args.output_subdir,
        "num_episodes_per_seed": args.num_episodes,
        "num_seeds": len(args.seeds),
        "reward_mean_across_seeds": float(np.mean([x["reward_mean"] for x in summary_rows])),
        "reward_std_across_seeds": float(np.std([x["reward_mean"] for x in summary_rows], ddof=0)),
        "crash_rate_mean_across_seeds": float(np.mean([x["crash_rate"] for x in summary_rows])),
    }

    save_csv_rows(
        results_root / f"evaluation_table_{args.num_episodes}_episodes.csv",
        ["seed", "num_episodes", "reward_mean", "reward_std", "length_mean", "length_std", "crash_rate"],
        summary_rows,
    )
    save_json(results_root / f"evaluation_overall_{args.num_episodes}.json", overall)


def ensure_eval_dir(model_type: str, output_subdir: str) -> Path:
    path = ROOT / "results" / model_type / output_subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on 50 episodes and summarize metrics.")
    parser.add_argument("--model-type", type=str, choices=["custom_dqn", "double_dqn", "sb3_dqn", "ppo_shaped"], required=True)
    parser.add_argument("--model-template", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--output-subdir", type=str, default="evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())