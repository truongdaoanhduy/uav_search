#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from uav_search.runner.visualize import plot_training_curves, plot_trajectory

p = argparse.ArgumentParser(description="Regenerate plots for an existing run.")
p.add_argument("run_dir")
a = p.parse_args()
r = Path(a.run_dir)
plot_training_curves(r / "metrics" / "episodes.csv", r / "plots")
z = np.load(r / "rollouts" / "final_trajectory.npz")
plot_trajectory(z["trajectory"], z["targets"], z["obstacles"], z["agent_names"], z["agent_types"], r / "plots" / "trajectory.png", 5000)
print(r / "plots")
