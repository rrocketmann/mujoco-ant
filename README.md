# mujoco-ant

Minimal RL project for training and visualizing a MuJoCo Ant agent with SAC (Stable-Baselines3).

A minimal Gymnasium MuJoCo `Ant-v4` + Stable-Baselines3 SAC setup.

## Install (Python 3.12 required)

This project requires **Python 3.12**.

If `python3.12` is not installed yet (Ubuntu):

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

Create and activate the virtual environment with Python 3.12:

```bash
python3.12 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
pip install "gymnasium[mujoco]" stable-baselines3 torch tensorboard
```

## Train (saved model + stats)

```bash
python3 train.py \
  --env Ant-v4 \
  --timesteps 10000 \
  --n-envs 8 \
  --vec-env subproc \
  --model-path models/ant \
  --vecnorm-path models/ant_vecnormalize.pkl
```

Notes:
- Defaults: `--device auto` (uses CUDA if available), `--mujoco-gl egl`.
- Parallel rollout collection: `--n-envs` + `--vec-env subproc`.
- Outputs: `<model-path>.zip` and `<vecnorm-path>`.
- Training UI: clean one-line progress bar (use `--no-progress` to disable).
- Fast default: `--timesteps 10000` (~100x less than 1,000,000).
- Training ends when `--timesteps` is reached, then model/stats are saved.
- Post-train eval defaults to `--eval-episodes 1` (`--eval-episodes 0` skips eval).

## Render (load saved artifacts)

```bash
python3 render.py --env Ant-v4 --model-path models/ant --vecnorm-path models/ant_vecnormalize.pkl --episodes 3
```

Notes:
- `render.py` defaults to `--render-mode auto`:
  - Uses `human` when `DISPLAY` is available.
  - Uses `rgb_array` in headless environments.
- If artifacts are missing, run `train.py` first (or pass existing paths).
