from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from double_dqn_per_agent import DoubleDQNPERAgent, DoubleDQNPERHyperParams
from utils import (
    ROOT,
    ensure_dir,
    flatten_obs,
    get_num_actions,
    get_obs_dim,
    make_env,
    save_csv_rows,
    save_json,
    set_global_seeds,
)


def evaluate_agent(
    agent: DoubleDQNPERAgent, seed: int, num_episodes: int
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    env = make_env(seed=seed)
    rows: list[dict[str, float | int]] = []

    for ep in range(num_episodes):
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
            crashed = crashed or bool(info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False))

        rows.append({"episode": ep, "reward": total_reward, "length": steps, "crashed": int(crashed)})

    env.close()

    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float32)
    crashes = np.asarray([r["crashed"] for r in rows], dtype=np.float32)

    summary = {
        "num_episodes": num_episodes,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "crash_rate": float(crashes.mean()),
    }
    return rows, summary


def train_one_seed(args: argparse.Namespace, seed: int) -> None:
    set_global_seeds(seed)
    env = make_env(seed=seed)
    obs_dim = get_obs_dim(env)
    num_actions = get_num_actions(env)

    hparams = DoubleDQNPERHyperParams(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        target_update_frequency=args.target_update_frequency,
        gradient_steps=args.gradient_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        max_grad_norm=args.max_grad_norm,
        alpha=args.alpha,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_anneal_steps=args.beta_anneal_steps,
        per_epsilon=args.per_epsilon,
    )
    agent = DoubleDQNPERAgent(obs_dim=obs_dim, num_actions=num_actions, device=args.device, hparams=hparams)

    results_dir = ensure_dir(ROOT / "results" / "double_dqn_per" / f"seed_{seed}")
    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "double_dqn_per" / f"seed_{seed}")
    eval_dir = ensure_dir(results_dir / "eval_callback")

    save_json(results_dir / "train_config.json", {"seed": seed, **agent.export_hparams()})

    metrics_rows: list[dict[str, float | int]] = []
    eval_rows: list[dict[str, float | int | str]] = []

    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0
    losses_window: list[float] = []

    best_eval_reward = -float("inf")
    best_eval_step = -1
    best_eval_episode = -1

    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)
    start_time = time.time()

    pbar = tqdm(range(1, args.total_timesteps + 1), desc=f"Training Double DQN + PER seed {seed}")

    for step in pbar:
        action = agent.act(obs, greedy=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = flatten_obs(next_obs)

        agent.observe(obs, action, reward, next_obs, done)
        losses_window.extend(agent.maybe_train_step())

        episode_reward += float(reward)
        episode_length += 1
        obs = next_obs

        if done:
            crash = bool(info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False))
            current_mean_loss = float(np.mean(losses_window)) if losses_window else np.nan

            metrics_rows.append(
                {
                    "episode": episode_idx,
                    "global_step": step,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "epsilon": agent.epsilon(),
                    "mean_loss": current_mean_loss,
                    "crashed": int(crash),
                }
            )

            pbar.set_postfix(
                episode=episode_idx,
                ep_reward=f"{episode_reward:.3f}",
                ep_len=episode_length,
                eps=f"{agent.epsilon():.3f}",
                best_eval=f"{best_eval_reward:.3f}" if best_eval_step >= 0 else "NA",
                crash=int(crash),
                loss=f"{current_mean_loss:.4f}" if not np.isnan(current_mean_loss) else "nan",
            )

            losses_window = []
            episode_idx += 1
            episode_reward = 0.0
            episode_length = 0
            obs, _ = env.reset()
            obs = flatten_obs(obs)

        if step % args.checkpoint_every == 0:
            agent.save(ckpt_dir / f"double_dqn_per_step_{step}.pt")

        if step % args.eval_every == 0 and step >= args.learning_starts:
            eval_seed = args.eval_seed_base + 1000 * seed + step
            _, eval_summary = evaluate_agent(agent, seed=eval_seed, num_episodes=args.eval_episodes)
            eval_reward_mean = float(eval_summary["reward_mean"])
            is_best_eval = int(eval_reward_mean > best_eval_reward)

            if is_best_eval:
                best_eval_reward = eval_reward_mean
                best_eval_step = step
                best_eval_episode = episode_idx
                agent.save(ckpt_dir / "double_dqn_per_best.pt")

            eval_rows.append(
                {
                    "global_step": step,
                    "train_episode": episode_idx,
                    "eval_seed": eval_seed,
                    "eval_num_episodes": args.eval_episodes,
                    "eval_reward_mean": eval_summary["reward_mean"],
                    "eval_reward_std": eval_summary["reward_std"],
                    "eval_length_mean": eval_summary["length_mean"],
                    "eval_length_std": eval_summary["length_std"],
                    "eval_crash_rate": eval_summary["crash_rate"],
                    "is_best_eval": is_best_eval,
                    "best_eval_reward_so_far": best_eval_reward,
                }
            )

    agent.save(ckpt_dir / "double_dqn_per_final.pt")

    save_csv_rows(
        results_dir / "train_metrics.csv",
        ["episode", "global_step", "episode_reward", "episode_length", "epsilon", "mean_loss", "crashed"],
        metrics_rows,
    )

    save_csv_rows(
        eval_dir / "evaluations.csv",
        [
            "global_step",
            "train_episode",
            "eval_seed",
            "eval_num_episodes",
            "eval_reward_mean",
            "eval_reward_std",
            "eval_length_mean",
            "eval_length_std",
            "eval_crash_rate",
            "is_best_eval",
            "best_eval_reward_so_far",
        ],
        eval_rows,
    )

    save_json(
        results_dir / "train_summary.json",
        {
            "seed": seed,
            "total_timesteps": args.total_timesteps,
            "episodes_finished": episode_idx,
            "wall_clock_seconds": time.time() - start_time,
            "last_100_episode_reward_mean": float(np.mean([r["episode_reward"] for r in metrics_rows[-100:]])) if metrics_rows else None,
            "last_100_crash_rate": float(np.mean([r["crashed"] for r in metrics_rows[-100:]])) if metrics_rows else None,
            "best_eval_reward": best_eval_reward if best_eval_step >= 0 else None,
            "best_eval_step": best_eval_step if best_eval_step >= 0 else None,
            "best_eval_episode": best_eval_episode if best_eval_episode >= 0 else None,
            "best_model_path": str(ckpt_dir / "double_dqn_per_best.pt") if best_eval_step >= 0 else None,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
        },
    )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Double DQN + PER on the shared highway benchmark.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=80_000)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-steps", type=int, default=80_000)
    parser.add_argument("--per-epsilon", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        train_one_seed(args, seed)
