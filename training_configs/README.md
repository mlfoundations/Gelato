# Reproducing Training Runs
We use [EasyR1](https://github.com/hiyouga/EasyR1) to train our models. Below are the configs for the different models we trained.

## Gelato-30B-A3B
We train the Gelato-30B-A3B model on 32 40GB A100 GPUs. We use the config at [gelato_30b_a3b.yaml](./gelato_30b_a3b.yaml) and the reward function at [norm_gui_reward.py](./rewards/norm_gui_reward.py).

## UI-TARS-1.5-7B + Gelato Baseline
We train the UI-TARS-1.5-7B Gelato baseline model on 16 40GB A100 GPUs. We use the config at [gelato_uitars_1_5_7b.yaml](./gelato_uitars_1_5_7b.yaml) and since the UI-TARS model uses native pixel coordinates for the grounding predictions we use a different reward function which you can find in [gui_reward.py](./rewards/gui_reward.py).