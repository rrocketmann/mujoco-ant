"""
Basic, heavily commented script to load a trained Ant SAC model and render it.

Run this after training and saving:
  - sac_ant_basic.zip
  - ant_vecnormalize.pkl
"""

import time

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "Ant-v4"
MODEL_PATH = "sac_ant_basic"
VECNORM_PATH = "ant_vecnormalize.pkl"
EPISODES = 3


def make_render_env():
    """
    Build env with human rendering so you can see the Ant move.
    """
    return gym.make(ENV_ID, render_mode="human")


def main():
    # ---------------------------------------------------------------------
    # 1) Rebuild env wrappers exactly like training side
    # ---------------------------------------------------------------------
    render_env = DummyVecEnv([make_render_env])

    # Load normalization statistics collected during training.
    # This keeps observation scaling consistent with what the policy expects.
    render_env = VecNormalize.load(VECNORM_PATH, render_env)

    # During inference, normalization stats should stay frozen.
    render_env.training = False

    # Turn off reward normalization so printed rewards are "real" env rewards.
    render_env.norm_reward = False

    # ---------------------------------------------------------------------
    # 2) Load trained model
    # ---------------------------------------------------------------------
    model = SAC.load(MODEL_PATH, env=render_env)

    # ---------------------------------------------------------------------
    # 3) Rollout episodes with deterministic actions
    # ---------------------------------------------------------------------
    # deterministic=True means "use the mean action" for stable evaluation.
    for ep in range(1, EPISODES + 1):
        obs = render_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = render_env.step(action)

            # Vec env returns arrays; index 0 because we use one environment.
            episode_reward += float(rewards[0])
            done = bool(dones[0])
            step_count += 1

            # Small sleep keeps rendering smooth/human-viewable on fast machines.
            time.sleep(0.01)

        print(f"Episode {ep}: steps={step_count}, reward={episode_reward:.2f}")

    render_env.close()


if __name__ == "__main__":
    main()
