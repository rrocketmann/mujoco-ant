# mujoco-ant

 Minimal RL project for training and visualizing a MuJoCo Ant agent with SAC (Stable-Baselines3).

 ## Setup

 ```bash
 # Python 3.11 recommended
 source myenv/bin/activate
 python -m pip install --upgrade pip
 python -m pip install "gymnasium[mujoco]" stable-baselines3 tensorboard

Train

 python train.py

Outputs:

 - sac_ant_basic.zip
 - ant_vecnormalize.pkl

Render

 python render.py

This loads the saved model + normalization stats and renders a few episodes of Ant-v4. ```
