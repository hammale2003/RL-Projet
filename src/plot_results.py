from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ROOT, ensure_dir


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_offpolicy_training(results_dir_name: str, display_name: str, seed: int, smooth_window: int) -> None:
    csv_path = ROOT / "results" / results_dir_name / f"seed_{seed}" / "train_metrics.csv"
    df = pd.read_csv(csv_path)
    out_dir = ensure_dir(ROOT / "results" / "plots")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], rolling_mean(df["episode_reward"], smooth_window))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{display_name} training reward (seed={seed})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_training_reward_seed_{seed}.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], rolling_mean(df["crashed"], smooth_window))
    plt.xlabel("Episode")
    plt.ylabel("Crash rate")
    plt.title(f"{display_name} training crash trend (seed={seed})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_training_crash_seed_{seed}.png", dpi=200)
    plt.close(fig)


def plot_offpolicy_eval(results_dir_name: str, display_name: str, seed: int) -> None:
    csv_path = ROOT / "results" / results_dir_name / f"seed_{seed}" / "eval_callback" / "evaluations.csv"
    df = pd.read_csv(csv_path)
    out_dir = ensure_dir(ROOT / "results" / "plots")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["eval_reward_mean"])
    plt.xlabel("Global step")
    plt.ylabel("Eval reward mean")
    plt.title(f"{display_name} evaluation reward (seed={seed})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_eval_reward_seed_{seed}.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["eval_crash_rate"])
    plt.xlabel("Global step")
    plt.ylabel("Eval crash rate")
    plt.title(f"{display_name} evaluation crash rate (seed={seed})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_eval_crash_seed_{seed}.png", dpi=200)
    plt.close(fig)


def plot_sb3_like_eval(results_dir_name: str, display_name: str, seed: int) -> None:
    npz_path = ROOT / "results" / results_dir_name / f"seed_{seed}" / "eval_callback" / "evaluations.npz"
    data = np.load(npz_path, allow_pickle=True)

    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)

    out_dir = ensure_dir(ROOT / "results" / "plots")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards)
    plt.xlabel("Global step")
    plt.ylabel("Eval reward mean")
    plt.title(f"{display_name} evaluation reward (seed={seed})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_eval_reward_seed_{seed}.png", dpi=200)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if args.model in ("custom_dqn", "both"):
        plot_offpolicy_training("dqn", "Custom DQN", seed=args.seed, smooth_window=args.smooth_window)
        plot_offpolicy_eval("dqn", "Custom DQN", seed=args.seed)

    if args.model in ("double_dqn", "both"):
        plot_offpolicy_training("double_dqn", "Double DQN", seed=args.seed, smooth_window=args.smooth_window)
        plot_offpolicy_eval("double_dqn", "Double DQN", seed=args.seed)

    if args.model in ("sb3_dqn", "both"):
        plot_sb3_like_eval("sb3_dqn", "SB3 DQN", seed=args.seed)

    if args.model in ("ppo_shaped", "both"):
        plot_sb3_like_eval("ppo_shaped", "PPO shaped", seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot saved training/evaluation curves for DQN, Double DQN, SB3 DQN and PPO shaped."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom_dqn", "double_dqn", "sb3_dqn", "ppo_shaped", "both"],
        default="both",
    )
    args = parser.parse_args()
    main(args)