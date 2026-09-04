#!/usr/bin/env python3
import argparse
from uav_search.runner.train import train_experiment

p = argparse.ArgumentParser(description="Train MASAC, MATD3 and MADDPG on both primary paper scenarios.")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--steps", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="auto")
p.add_argument("--output-root", default="runs")
p.add_argument("--wandb", action="store_true")
a = p.parse_args()
for algorithm in ("masac", "matd3", "maddpg"):
    for scenario in ("f1_m5", "f1_m9"):
        path = train_experiment(algorithm, scenario, a.episodes, a.steps, a.seed, a.device, a.output_root, a.wandb)
        print(f"{algorithm}/{scenario}: {path}")
