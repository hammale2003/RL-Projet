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


def _episode_summary(rows: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float32)
    crashes = np.asarray([r["crashed"] for r in rows], dtype=np.float32)
    return {
        "num_episodes": len(rows),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "crash_rate": float(crashes.mean()),
    }


def main(args: argparse.Namespace) -> None:
    if args.model_type not in EVALUATORS:
        raise ValueError(f"Unknown model_type={args.model_type}")

    results_root = ensure_eval_dir(args.model_type, args.output_subdir)
    summary_rows: list[dict[str, float | int]] = []

    for train_seed in args.seeds:
        model_path = Path(args.model_template.format(seed=train_seed))
        if args.eval_env_seeds is not None:
            env_seeds: list[int] = list(args.eval_env_seeds)
        else:
            env_seeds = [args.eval_seed_base + 1000 * train_seed]

        per_env_summaries: list[dict[str, float | int]] = []
        seed_dir = results_root / f"seed_{train_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for env_seed in env_seeds:
            set_global_seeds(env_seed)
            rows = EVALUATORS[args.model_type](
                model_path,
                seed=env_seed,
                num_episodes=args.num_episodes,
            )
            summ = _episode_summary(rows)
            summ["eval_env_seed"] = env_seed
            per_env_summaries.append(summ)

            if len(env_seeds) > 1:
                env_sub = seed_dir / f"eval_env_{env_seed}"
                env_sub.mkdir(parents=True, exist_ok=True)
                save_csv_rows(
                    env_sub / f"evaluation_{args.num_episodes}_episodes.csv",
                    ["episode", "reward", "length", "crashed"],
                    rows,
                )
            else:
                save_csv_rows(
                    seed_dir / f"evaluation_{args.num_episodes}_episodes.csv",
                    ["episode", "reward", "length", "crashed"],
                    rows,
                )

        reward_means = [float(s["reward_mean"]) for s in per_env_summaries]
        crash_rates = [float(s["crash_rate"]) for s in per_env_summaries]

        seed_summary: dict[str, float | int | list[float]] = {
            "seed": train_seed,
            "train_seed": train_seed,
            "num_episodes_per_eval": args.num_episodes,
            "num_eval_env_seeds": len(env_seeds),
            "eval_env_seeds": env_seeds,
            "reward_mean": float(np.mean(reward_means)),
            "reward_std_across_eval_env_seeds": float(np.std(reward_means, ddof=0)),
            "per_eval_env_reward_means": reward_means,
            "crash_rate_mean": float(np.mean(crash_rates)),
            "crash_rate_std_across_eval_env_seeds": float(np.std(crash_rates, ddof=0)),
        }
        # Back-compat fields for single-eval table
        seed_summary["num_episodes"] = args.num_episodes * len(env_seeds)
        seed_summary["reward_std"] = float(per_env_summaries[0]["reward_std"]) if len(per_env_summaries) == 1 else seed_summary["reward_std_across_eval_env_seeds"]
        seed_summary["length_mean"] = float(np.mean([float(s["length_mean"]) for s in per_env_summaries]))
        seed_summary["length_std"] = float(np.mean([float(s["length_std"]) for s in per_env_summaries]))
        seed_summary["crash_rate"] = seed_summary["crash_rate_mean"]

        out_name = (
            f"evaluation_summary_{args.num_episodes}_{len(env_seeds)}evalseeds.json"
            if len(env_seeds) > 1
            else f"evaluation_summary_{args.num_episodes}.json"
        )
        save_json(seed_dir / out_name, seed_summary)
        summary_rows.append(
            {
                "seed": train_seed,
                "num_episodes": args.num_episodes,
                "num_eval_env_seeds": len(env_seeds),
                "reward_mean": seed_summary["reward_mean"],
                "reward_std": seed_summary["reward_std_across_eval_env_seeds"],
                "length_mean": seed_summary["length_mean"],
                "length_std": seed_summary["length_std"],
                "crash_rate": seed_summary["crash_rate"],
            }
        )

    overall = {
        "model_type": args.model_type,
        "output_subdir": args.output_subdir,
        "num_episodes_per_eval_env": args.num_episodes,
        "num_eval_env_seeds": len(args.eval_env_seeds) if args.eval_env_seeds else 1,
        "num_train_seeds": len(args.seeds),
        "reward_mean_across_seeds": float(np.mean([x["reward_mean"] for x in summary_rows])),
        "reward_std_across_seeds": float(np.std([x["reward_mean"] for x in summary_rows], ddof=0)),
        "crash_rate_mean_across_seeds": float(np.mean([x["crash_rate"] for x in summary_rows])),
    }

    save_csv_rows(
        results_root / f"evaluation_table_{args.num_episodes}_episodes.csv",
        [
            "seed",
            "num_episodes",
            "num_eval_env_seeds",
            "reward_mean",
            "reward_std",
            "length_mean",
            "length_std",
            "crash_rate",
        ],
        summary_rows,
    )
    save_json(results_root / f"evaluation_overall_{args.num_episodes}.json", overall)


def ensure_eval_dir(model_type: str, output_subdir: str) -> Path:
    path = ROOT / "results" / model_type / output_subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on 50 episodes and summarize metrics.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["custom_dqn", "double_dqn", "sb3_dqn", "ppo_shaped"],
        required=True,
    )
    parser.add_argument("--model-template", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument(
        "--eval-env-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Distinct environment seeds for evaluation (e.g. 3 seeds). "
            "If set, runs --num-episodes per env seed and aggregates means. "
            "If omitted, uses one seed per checkpoint: --eval-seed-base + 1000 * train seed."
        ),
    )
    parser.add_argument("--output-subdir", type=str, default="evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())