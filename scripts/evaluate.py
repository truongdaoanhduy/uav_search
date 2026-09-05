#!/usr/bin/env python3
import argparse
import json
from uav_search.runner.evaluate import evaluate_checkpoint

p = argparse.ArgumentParser(description="Evaluate a trained paper baseline.")
p.add_argument("--checkpoint", required=True)
p.add_argument("--episodes", type=int, default=5000)
p.add_argument("--device", default="auto")
p.add_argument("--output-dir", default="evaluation")
a = p.parse_args()
print(json.dumps(evaluate_checkpoint(a.checkpoint, a.episodes, a.device, a.output_dir), indent=2))
