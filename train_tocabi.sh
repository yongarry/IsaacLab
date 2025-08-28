#!/usr/bin/env bash


python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi --headless --logger=wandb --run_name p_sn
python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi --headless --logger=wandb --run_name p_lcp
