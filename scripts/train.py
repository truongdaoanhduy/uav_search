#!/usr/bin/env python3
import argparse
from uav_search.runner.train import train_experiment

p = argparse.ArgumentParser(description="Train one paper baseline.")
p.add_argument("--algorithm", choices=["masac", "matd3", "maddpg"], required=True)
p.add_argument("--scenario", choices=["f1_m5", "f1_m9"], required=True)
p.add_argument("--episodes", type=int, default=None, help="Default 300; paper reference is 50,000")
p.add_argument("--steps", type=int, default=None, help="Override paper's 50 steps/episode (smoke tests only)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
p.add_argument("--output-root", default="runs")
p.add_argument("--wandb", action="store_true")
p.add_argument("--wandb-project", default="uav-search-paper-baselines")
a = p.parse_args()
run = train_experiment(a.algorithm, a.scenario, a.episodes, a.steps, a.seed, a.device, a.output_root, a.wandb, a.wandb_project)
print(run)
