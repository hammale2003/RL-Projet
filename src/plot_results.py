from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ROOT, ensure_dir

PLOT_DDQN_EVAL_DIR = ROOT / "results" / "plots_double_dqn" / "plot_ddqn_eval"


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


def plot_offpolicy_eval(
    results_dir_name: str,
    display_name: str,
    seed: int,
    out_dir: Path | None = None,
) -> None:
    csv_path = ROOT / "results" / results_dir_name / f"seed_{seed}" / "eval_callback" / "evaluations.csv"
    df = pd.read_csv(csv_path)
    if out_dir is None:
        out_dir = ROOT / "results" / "plots"
    out_dir = ensure_dir(out_dir)

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


def plot_offline_evaluation_artifacts(
    results_dir_name: str,
    display_name: str,
    train_seed: int,
    out_dir: Path,
) -> None:
    """Bar chart + histograms from evaluate.py output: results/<variant>/evaluation/seed_<k>/."""
    eval_root = ROOT / "results" / results_dir_name / "evaluation" / f"seed_{train_seed}"
    if not eval_root.is_dir():
        print(f"[offline eval plots] Skip: no directory {eval_root}")
        return

    rows_by_label: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("eval_env_"):
            continue
        csv_files = list(sub.glob("evaluation_*_episodes.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[0])
        label = sub.name.removeprefix("eval_env_")
        rewards = df["reward"].to_numpy(dtype=np.float64)
        crashes = df["crashed"].to_numpy(dtype=np.float64)
        rows_by_label.append((label, rewards, crashes))

    if not rows_by_label:
        flat = list(eval_root.glob("evaluation_*_episodes.csv"))
        if flat:
            df = pd.read_csv(flat[0])
            rewards = df["reward"].to_numpy(dtype=np.float64)
            crashes = df["crashed"].to_numpy(dtype=np.float64)
            rows_by_label.append(("single", rewards, crashes))

    if not rows_by_label:
        print(f"[offline eval plots] Skip: no evaluation_*_episodes.csv under {eval_root}")
        return

    out_dir = ensure_dir(out_dir)
    n_ep = len(rows_by_label[0][1])

    labels = [t[0] for t in rows_by_label]
    means = [float(t[1].mean()) for t in rows_by_label]
    stds = [float(t[1].std(ddof=0)) for t in rows_by_label]
    crash_means = [float(t[2].mean()) for t in rows_by_label]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Eval env seed")
    ax.set_title(f"{display_name} offline eval: reward (mean ± std over {n_ep} episodes per seed)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_offline_eval_bar_reward_seed_{train_seed}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, crash_means, color="indianred", edgecolor="black", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Crash rate (mean over episodes)")
    ax.set_xlabel("Eval env seed")
    ax.set_title(f"{display_name} offline eval: crash rate per env seed")
    fig.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_offline_eval_bar_crash_seed_{train_seed}.png", dpi=200)
    plt.close(fig)

    n_groups = len(rows_by_label)
    fig, axes = plt.subplots(1, n_groups, figsize=(max(4 * n_groups, 6), 4), squeeze=False)
    for ax, (label, rew, _) in zip(axes[0], rows_by_label):
        ax.hist(rew, bins=min(20, max(8, n_ep // 5)), color="coral", edgecolor="black", alpha=0.88)
        ax.set_title(f"env {label}")
        ax.set_xlabel("Episode reward")
        ax.set_ylabel("Count")
    fig.suptitle(f"{display_name} offline eval: reward histograms (train seed={train_seed})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_offline_eval_hist_reward_seed_{train_seed}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rew, _ in rows_by_label:
        ax.hist(rew, bins=min(20, max(8, n_ep // 5)), alpha=0.45, label=f"env {label}")
    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(f"{display_name} offline eval: overlaid reward histograms")
    fig.tight_layout()
    fig.savefig(out_dir / f"{results_dir_name}_offline_eval_hist_overlay_seed_{train_seed}.png", dpi=200)
    plt.close(fig)

    print(f"[offline eval plots] Wrote figures to {out_dir}")


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


ABLATION_VARIANTS: dict[str, str] = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
}


def plot_ablation_comparison(seeds: list[int], smooth_window: int = 20) -> None:
    out_dir = ensure_dir(ROOT / "results" / "plots")

    reward_means: dict[str, float] = {}
    reward_stds: dict[str, float] = {}
    crash_means: dict[str, float] = {}

    fig_curves, ax_curves = plt.subplots(figsize=(10, 6))

    for variant, display_name in ABLATION_VARIANTS.items():
        per_seed_curves: list[np.ndarray] = []
        per_seed_eval_rewards: list[float] = []
        per_seed_crash_rates: list[float] = []

        for seed in seeds:
            train_csv = ROOT / "results" / variant / f"seed_{seed}" / "train_metrics.csv"
            eval_csv = ROOT / "results" / variant / f"seed_{seed}" / "eval_callback" / "evaluations.csv"

            if train_csv.exists():
                df_train = pd.read_csv(train_csv)
                smoothed = df_train["episode_reward"].rolling(window=smooth_window, min_periods=1).mean().to_numpy()
                per_seed_curves.append(smoothed)

            if eval_csv.exists():
                df_eval = pd.read_csv(eval_csv)
                if not df_eval.empty:
                    per_seed_eval_rewards.append(float(df_eval["eval_reward_mean"].iloc[-1]))
                    per_seed_crash_rates.append(float(df_eval["eval_crash_rate"].iloc[-1]))

        if per_seed_curves:
            min_len = min(len(c) for c in per_seed_curves)
            matrix = np.stack([c[:min_len] for c in per_seed_curves], axis=0)
            mean_curve = matrix.mean(axis=0)
            std_curve = matrix.std(axis=0, ddof=0)
            episodes = np.arange(min_len)
            ax_curves.plot(episodes, mean_curve, label=display_name)
            ax_curves.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

        if per_seed_eval_rewards:
            reward_means[display_name] = float(np.mean(per_seed_eval_rewards))
            reward_stds[display_name] = float(np.std(per_seed_eval_rewards, ddof=0))

        if per_seed_crash_rates:
            crash_means[display_name] = float(np.mean(per_seed_crash_rates))

    ax_curves.set_xlabel("Episode")
    ax_curves.set_ylabel("Smoothed reward")
    ax_curves.set_title("Ablation: training reward comparison")
    ax_curves.legend()
    fig_curves.tight_layout()
    fig_curves.savefig(out_dir / "ablation_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_curves)

    if reward_means:
        names = list(reward_means.keys())
        means = [reward_means[n] for n in names]
        stds = [reward_stds[n] for n in names]
        x = np.arange(len(names))

        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        ax_bar.bar(x, means, yerr=stds, capsize=5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, rotation=15, ha="right")
        ax_bar.set_ylabel("Mean eval reward")
        ax_bar.set_title("Ablation: final eval reward per variant")
        fig_bar.tight_layout()
        fig_bar.savefig(out_dir / "ablation_eval_rewards.png", dpi=150, bbox_inches="tight")
        plt.close(fig_bar)

    if crash_means:
        names = list(crash_means.keys())
        rates = [crash_means[n] for n in names]
        x = np.arange(len(names))

        fig_crash, ax_crash = plt.subplots(figsize=(8, 5))
        ax_crash.bar(x, rates)
        ax_crash.set_xticks(x)
        ax_crash.set_xticklabels(names, rotation=15, ha="right")
        ax_crash.set_ylabel("Crash rate")
        ax_crash.set_title("Ablation: crash rate per variant")
        fig_crash.tight_layout()
        fig_crash.savefig(out_dir / "ablation_crash_rates.png", dpi=150, bbox_inches="tight")
        plt.close(fig_crash)


def main(args: argparse.Namespace) -> None:
    if args.mode == "ablation":
        plot_ablation_comparison(seeds=args.seeds, smooth_window=args.smooth_window)
        return

    if args.mode == "offline-eval":
        if args.model not in ("double_dqn", "custom_dqn"):
            raise ValueError("--mode offline-eval requires --model double_dqn or custom_dqn")
        variant_map = {
            "double_dqn": ("double_dqn", "Double DQN"),
            "custom_dqn": ("dqn", "Custom DQN"),
        }
        rdir, dname = variant_map[args.model]
        out = PLOT_DDQN_EVAL_DIR if args.model == "double_dqn" else ROOT / "results" / "plots" / f"offline_eval_{rdir}"
        plot_offline_evaluation_artifacts(rdir, dname, args.seed, ensure_dir(out))
        return

    if args.model in ("custom_dqn", "both"):
        plot_offpolicy_training("dqn", "Custom DQN", seed=args.seed, smooth_window=args.smooth_window)
        plot_offpolicy_eval("dqn", "Custom DQN", seed=args.seed)

    if args.model in ("double_dqn", "both"):
        plot_offpolicy_training("double_dqn", "Double DQN", seed=args.seed, smooth_window=args.smooth_window)
        plot_offpolicy_eval("double_dqn", "Double DQN", seed=args.seed, out_dir=PLOT_DDQN_EVAL_DIR)
        plot_offline_evaluation_artifacts("double_dqn", "Double DQN", args.seed, PLOT_DDQN_EVAL_DIR)

    if args.model in ("sb3_dqn", "both"):
        plot_sb3_like_eval("sb3_dqn", "SB3 DQN", seed=args.seed)

    if args.model in ("ppo_shaped", "both"):
        plot_sb3_like_eval("ppo_shaped", "PPO shaped", seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot saved training/evaluation curves for DQN, Double DQN, SB3 DQN and PPO shaped."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom_dqn", "double_dqn", "sb3_dqn", "ppo_shaped", "both"],
        default="both",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "ablation", "offline-eval"],
        default="single",
        help="'offline-eval': only bar/histogram plots from results/<model>/evaluation/ (saved under plot_ddqn_eval for double_dqn).",
    )
    args = parser.parse_args()
    main(args)