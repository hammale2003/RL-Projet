from __future__ import annotations

import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from utils import ROOT, ensure_dir, make_env, save_json, set_global_seeds


def train_one_seed(args: argparse.Namespace, seed: int) -> None:
    set_global_seeds(seed)

    env = Monitor(make_env(seed=seed))
    eval_env = Monitor(make_env(seed=10_000 + seed))

    results_dir = ensure_dir(ROOT / "results" / "sb3_dqn" / f"seed_{seed}")
    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "sb3_dqn" / f"seed_{seed}")
    best_dir = ensure_dir(ckpt_dir / "best_model")
    eval_log_dir = ensure_dir(results_dir / "eval_callback")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_frequency,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [args.hidden_dim1, args.hidden_dim2]},
        verbose=1,
        seed=seed,
        device=args.device,
        tensorboard_log=str(ROOT / "results" / "tensorboard" / "sb3_dqn"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(ckpt_dir),
        name_prefix="sb3_dqn",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_every,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    save_json(
        results_dir / "train_config.json",
        {
            "seed": seed,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "train_frequency": args.train_frequency,
            "gradient_steps": args.gradient_steps,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_initial_eps": args.exploration_initial_eps,
            "exploration_final_eps": args.exploration_final_eps,
            "hidden_dims": [args.hidden_dim1, args.hidden_dim2],
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "best_model_path": str(best_dir / "best_model.zip"),
        },
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    model.save(ckpt_dir / "sb3_dqn_final")

    env.close()
    eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 DQN on the shared highway benchmark.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seed in tqdm(args.seeds, desc="Training SB3 seeds"):
        train_one_seed(args, seed)