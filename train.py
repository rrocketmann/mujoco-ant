"""
Training script for Gymnasium MuJoCo Ant-v4 using SAC.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


DEFAULT_ENV_ID = "Ant-v4"
DEFAULT_TOTAL_STEPS = 10_000


class TrainProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, enabled: bool = True) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.enabled = enabled
        self._last_print_ts = 0.0
        self._bar_width = 28

    def _on_training_start(self) -> None:
        self._last_print_ts = 0.0

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        now = time.time()
        if now - self._last_print_ts < 0.2 and self.model.num_timesteps < self.total_timesteps:
            return True

        self._last_print_ts = now
        progress = min(1.0, self.model.num_timesteps / max(1, self.total_timesteps))
        filled = int(self._bar_width * progress)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        print(
            f"\rTraining [{bar}] {progress * 100:6.2f}% "
            f"({self.model.num_timesteps:,}/{self.total_timesteps:,} steps)",
            end="",
            flush=True,
        )
        return True

    def _on_training_end(self) -> None:
        if self.enabled:
            self._on_step()
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_ID)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--model-path", default="sac_ant_basic")
    parser.add_argument("--vecnorm-path", default="ant_vecnormalize.pkl")
    parser.add_argument("--tensorboard-log", default="./tb_ant/")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mujoco-gl", choices=["egl", "glfw", "osmesa"], default="egl")
    parser.add_argument("--egl-device", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show clean one-line progress bar during training.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Number of post-training evaluation episodes (0 to skip).",
    )
    return parser.parse_args()


def configure_mujoco_backend(mujoco_gl: str, egl_device: int) -> None:
    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda), but no CUDA device is available to PyTorch."
        )
    return device_arg


def make_env(env_id: str) -> gym.Env:
    return gym.make(env_id)


def make_env_fn(env_id: str) -> Callable[[], gym.Env]:
    def _factory() -> gym.Env:
        return make_env(env_id)

    return _factory


def build_train_env(env_id: str, n_envs: int, vec_env_type: str) -> VecNormalize:
    env_fns = [make_env_fn(env_id) for _ in range(n_envs)]
    if vec_env_type == "subproc" and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )


def main() -> None:
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1.")

    configure_mujoco_backend(args.mujoco_gl, args.egl_device)
    device = resolve_device(args.device)

    print(f"MuJoCo GL backend: {args.mujoco_gl}")
    print(f"SAC device: {device}")
    print(f"Vectorized envs: {args.n_envs} ({args.vec_env})")

    train_env = build_train_env(args.env, args.n_envs, args.vec_env)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        tensorboard_log=args.tensorboard_log,
        device=device,
    )

    progress_callback = TrainProgressCallback(
        total_timesteps=args.timesteps,
        enabled=args.progress,
    )
    model.learn(total_timesteps=args.timesteps, callback=progress_callback, log_interval=10)

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vecnorm_path).parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_path)
    train_env.save(args.vecnorm_path)

    print(f"Saved model: {args.model_path}.zip")
    print(f"Saved normalization stats: {args.vecnorm_path}")

    if args.eval_episodes > 0:
        eval_env = DummyVecEnv([make_env_fn(args.env)])
        eval_env = VecNormalize.load(args.vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        print(
            f"Eval reward over {args.eval_episodes} episode(s): "
            f"{mean_reward:.2f} +/- {std_reward:.2f}"
        )


if __name__ == "__main__":
    main()
