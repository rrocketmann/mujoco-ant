"""
Basic, heavily commented training script for Gymnasium MuJoCo Ant-v4 using SAC.

This is written for learning, not for squeezing out maximum benchmark score.
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Ant is a continuous-control environment, and SAC is usually a strong baseline.
ENV_ID = "Ant-v4"

# Keep this moderate for quick iteration while learning.
# For better performance, you will often train longer (1M-3M+ steps).
TOTAL_STEPS = 500_000


def make_env():
    """
    Creates one raw Gymnasium environment instance.

    We keep this in a function so it can be passed to vectorized wrappers.
    """
    return gym.make(ENV_ID)


def main():
    # ---------------------------------------------------------------------
    # 1) Build training environment
    # ---------------------------------------------------------------------
    # SB3 expects vectorized envs, even if we only use one env.
    train_env = DummyVecEnv([make_env])

    # Normalize observations and rewards:
    # - norm_obs=True  -> scales observations to make optimization easier
    # - norm_reward=True -> scales rewards to reduce training instability
    # - clip_obs=10.0 -> avoids extreme normalized obs values
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # ---------------------------------------------------------------------
    # 2) Create SAC model
    # ---------------------------------------------------------------------
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,      # common default for SAC
        buffer_size=1_000_000,   # replay buffer size
        batch_size=256,          # minibatch sampled from replay buffer
        tau=0.005,               # target network smoothing coeff
        gamma=0.99,              # discount factor
        train_freq=1,            # train every env step
        gradient_steps=1,        # one gradient update per train step
        verbose=1,
        tensorboard_log="./tb_ant/",  # launch with: tensorboard --logdir ./tb_ant
    )

    # ---------------------------------------------------------------------
    # 3) Train
    # ---------------------------------------------------------------------
    model.learn(total_timesteps=TOTAL_STEPS, log_interval=10)

    # ---------------------------------------------------------------------
    # 4) Save artifacts
    # ---------------------------------------------------------------------
    # Save the trained policy/critic weights.
    model.save("sac_ant_basic")

    # Save normalization stats too.
    # IMPORTANT: if you load the model later without the same VecNormalize
    # stats, performance can look broken.
    train_env.save("ant_vecnormalize.pkl")

    print("Saved model: sac_ant_basic.zip")
    print("Saved normalization stats: ant_vecnormalize.pkl")

    # ---------------------------------------------------------------------
    # 5) Quick evaluation
    # ---------------------------------------------------------------------
    # Create a fresh eval env and load the exact normalization stats used in training.
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load("ant_vecnormalize.pkl", eval_env)

    # Disable training mode for normalization during eval:
    # - do not update running mean/variance
    # - report unnormalized rewards so scores are interpretable
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        deterministic=True,  # no stochastic exploration during evaluation
    )
    print(f"Eval reward over 5 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
