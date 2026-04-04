from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3 import PPO as SB3PPO

from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from utils import ROOT, flatten_obs, make_env


def make_video_env(seed: int, video_duration: int | None, fps: int) -> gym.Env:
    env = make_env(seed=seed, render_mode="rgb_array")

    if video_duration is not None:
        env.unwrapped.configure({"duration": int(video_duration)})

    env.metadata["render_fps"] = int(fps)
    return env


def record_custom(
    model_path: Path, output_dir: Path, seed: int, video_duration: int | None, fps: int, device: str = "cpu"
) -> None:
    base_env = make_video_env(seed=seed, video_duration=video_duration, fps=fps)
    model_name = model_path.parent.name

    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{model_name}_eval_{seed}",
    )

    agent = DQNAgent.load(model_path, device=device)

    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)
    done = False

    while not done:
        action = agent.act_greedy(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = flatten_obs(next_obs)

    env.close()


def record_double_dqn(
    model_path: Path, output_dir: Path, seed: int, video_duration: int | None, fps: int, device: str = "cpu"
) -> None:
    base_env = make_video_env(seed=seed, video_duration=video_duration, fps=fps)
    model_name = model_path.parent.name

    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{model_name}_eval_{seed}",
    )

    agent = DoubleDQNAgent.load(model_path, device=device)

    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)
    done = False

    while not done:
        action = agent.act_greedy(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = flatten_obs(next_obs)

    env.close()


def record_sb3(model_path: Path, output_dir: Path, seed: int, video_duration: int | None, fps: int) -> None:
    base_env = make_video_env(seed=seed, video_duration=video_duration, fps=fps)
    model_name = model_path.parents[1].name

    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{model_name}_eval_{seed}",
    )

    model = SB3DQN.load(model_path, env=env, device="cpu")

    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


def record_ppo(model_path: Path, output_dir: Path, seed: int, video_duration: int | None, fps: int) -> None:
    base_env = make_video_env(seed=seed, video_duration=video_duration, fps=fps)
    model_name = model_path.parents[1].name

    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{model_name}_eval_{seed}",
    )

    model = SB3PPO.load(model_path, env=env, device="cpu")

    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record rollout videos for qualitative analysis.")
    parser.add_argument("--model-type", type=str, choices=["custom_dqn", "double_dqn", "sb3_dqn", "ppo_shaped"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--video-duration", type=int, default=80)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder for mp4 files (default: <project>/videos/<model-type>).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for loading DQN checkpoints (cpu or cuda).",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "videos" / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        if args.model_type == "custom_dqn":
            record_custom(
                Path(args.model_path),
                out_dir,
                seed=seed,
                video_duration=args.video_duration,
                fps=args.fps,
                device=args.device,
            )
        elif args.model_type == "double_dqn":
            record_double_dqn(
                Path(args.model_path),
                out_dir,
                seed=seed,
                video_duration=args.video_duration,
                fps=args.fps,
                device=args.device,
            )
        elif args.model_type == "sb3_dqn":
            record_sb3(Path(args.model_path), out_dir, seed=seed, video_duration=args.video_duration, fps=args.fps)
        else:
            record_ppo(Path(args.model_path), out_dir, seed=seed, video_duration=args.video_duration, fps=args.fps)