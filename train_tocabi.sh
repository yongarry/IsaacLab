#!/usr/bin/env bash


python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi-SNfix-pos --headless --logger=wandb --run_name p_snfix
python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi-SNfix-torq --headless --logger=wandb --run_name t_snfix
python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi-SNlearn-pos --headless --logger=wandb --run_name p_snlearn
python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi-SNlearn-torq --headless --logger=wandb --run_name t_snlearn
python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-Tocabi-LCP-torq --headless --logger=wandb --run_name t_lcp
