# M2TD3 and SoftM2TD3

Official implementation of Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification.

# Requirement
```
pip install -r requirements.txt
```
You will also need to install mujoco, if necessary.
We used mjpro150. These can be installed for free.

# Training
```train
python main.py algorithm=m2td3
```
The algorithm can be selected from `m2td3`, `soft_m2td3`.

With the `environment` option, you can train in various scenarios of mujoco tasks.
Check `configs/environment` to see what scenarios are available.

This repository contains the modified version of [Gym](https://github.com/openai/gym).
