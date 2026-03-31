"""
Render a trained Ant SAC model.
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


DEFAULT_ENV_ID = "Ant-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_ID)
    parser.add_argument("--model-path", default="sac_ant_basic")
    parser.add_argument("--vecnorm-path", default="ant_vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mujoco-gl", choices=["egl", "glfw", "osmesa"], default="glfw")
    parser.add_argument("--egl-device", type=int, default=0)
    parser.add_argument(
        "--render-mode",
        choices=["auto", "human", "rgb_array"],
        default="auto",
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


def assert_artifacts_exist(model_path: str, vecnorm_path: str) -> None:
    model_zip = Path(f"{model_path}.zip")
    vecnorm_file = Path(vecnorm_path)
    if not model_zip.exists() or not vecnorm_file.exists():
        raise FileNotFoundError(
            "Missing trained artifacts. Expected: "
            f"{model_zip} and {vecnorm_file}. "
            "Run train.py first, or pass --model-path/--vecnorm-path to existing artifacts."
        )


def resolve_render_mode(render_mode_arg: str) -> str:
    if render_mode_arg in {"human", "rgb_array"}:
        return render_mode_arg
    return "human" if os.environ.get("DISPLAY") else "rgb_array"


def main() -> None:
    args = parse_args()
    configure_mujoco_backend(args.mujoco_gl, args.egl_device)
    device = resolve_device(args.device)
    render_mode = resolve_render_mode(args.render_mode)
    assert_artifacts_exist(args.model_path, args.vecnorm_path)

    print(f"MuJoCo GL backend: {args.mujoco_gl}")
    print(f"Inference device: {device}")
    print(f"Render mode: {render_mode}")

    render_env = DummyVecEnv([lambda: gym.make(args.env, render_mode=render_mode)])
    render_env = VecNormalize.load(args.vecnorm_path, render_env)
    render_env.training = False
    render_env.norm_reward = False

    model = SAC.load(args.model_path, env=render_env, device=device)

    for ep in range(1, args.episodes + 1):
        obs = render_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = render_env.step(action)
            episode_reward += float(rewards[0])
            done = bool(dones[0])
            step_count += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep}: steps={step_count}, reward={episode_reward:.2f}")

    render_env.close()


if __name__ == "__main__":
    main()
