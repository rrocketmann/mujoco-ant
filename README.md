 # mujoco-ant
 
 A minimal neural-net project for training a walking agent on Gymnasium MuJoCo `Ant-v4`.
 
 ## Install (venv)
 
 ```bash
 python3 -m venv .venv
 source .venv/bin/activate
 pip install --upgrade pip
 pip install gymnasium[mujoco]

Train

 python train.py --env Ant-v4 --timesteps 1000000 --save models/ant.pt

Render

 python render.py --env Ant-v4 --model models/ant.pt
