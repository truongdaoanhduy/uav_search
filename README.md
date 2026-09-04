# Heterogeneous UAV Search — MASAC / MATD3 / MADDPG

Minimal, modular reproduction of the **software-simulation** part of:

> T. Ao *et al.*, “Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network,” IEEE Transactions on Vehicular Technology, DOI: 10.1109/TVT.2025.3594534.

This repository intentionally implements **only the three requested baselines**:

- **MASAC** — Multi-Agent Soft Actor-Critic
- **MATD3** — Multi-Agent Twin Delayed DDPG
- **MADDPG** — Multi-Agent DDPG

It does **not** implement GATAC, SA-GATAC, or the paper's physical motion-capture/micro-UAV experiment.

## What is reproduced

- heterogeneous leader/follower swarm: fixed-wing relay(s) + multi-rotor search UAVs;
- 5 km × 5 km post-disaster search area;
- uniformly randomized targets and circular obstacles/buildings;
- probabilistic LoS/NLoS A2A communication + Shannon rate;
- simplified fixed-wing and multi-rotor dynamics;
- paper multi-rotor energy model;
- communication / energy / safety / target-search rewards;
- CTDE training with local actor observations and centralized critics;
- paper's two primary small scenarios: **1 fixed + 5 multi-rotor** and **1 fixed + 9 multi-rotor**;
- logging, low-episode diagnostics, crash logs, checkpoints, optional W&B, evaluation, and plots.

See [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) for equation/parameter mapping and [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md) for what every project folder does.

## Install

```bash
cd /home/aduy/Documents/NCKH/uav_search
python3 -m pip install -e .
```

Optional W&B:

```bash
python3 -m pip install -e '.[wandb]'
```

The core project does not require PettingZoo or the archived original OpenAI MPE package; the environment exposes a lightweight parallel multi-agent API directly and uses `gymnasium.spaces.Box` for spaces.

## Fast start

Train one baseline:

```bash
python3 scripts/train.py --algorithm masac --scenario f1_m5 --episodes 10 --device auto
```

Other combinations:

```bash
python3 scripts/train.py --algorithm matd3 --scenario f1_m9 --episodes 10 --device cpu
python3 scripts/train.py --algorithm maddpg --scenario f1_m5 --episodes 10 --device auto
```

Run all six requested combinations:

```bash
python3 scripts/run_all.py --episodes 10 --device auto
```

`--device auto` uses CUDA when available, otherwise CPU. To request a particular GPU use `--device cuda:0`.

For a quick code-only smoke run you may override episode length:

```bash
python3 scripts/run_all.py --episodes 2 --steps 5 --device cpu
```

`--steps` is **only a development override**. The paper value is 50 steps/episode.

## Paper-scale command

Table I reports 50,000 training rounds and 50 steps per episode. A paper-budget run is therefore:

```bash
python3 scripts/run_all.py --episodes 50000 --device cuda:0
```

Do not expect the included short smoke runs to reproduce Fig. 7 numerically; their purpose is correctness/runnability. Exact numerical reproduction is also limited by parameters the article does not publish; those are isolated under the `assumed:` section in `configs/paper.yaml`.

## Evaluate a checkpoint

```bash
python3 scripts/evaluate.py \
  --checkpoint runs/masac/f1_m5/<run>/checkpoints/final.pt \
  --episodes 100 \
  --device auto \
  --output-dir evaluation/masac_f1_m5
```

The paper evaluates final algorithms on 5,000 random test cases; use `--episodes 5000` for the same evaluation count.

## Output of every training run

```text
runs/<algorithm>/<scenario>/<timestamp>/
├── config.yaml
├── summary.json
├── evaluation.json
├── checkpoints/
│   ├── best.pt
│   └── final.pt          # always written after a successful run
├── logs/
│   ├── errors.log        # traceback + episode/step context
│   └── low_episodes.jsonl
├── metrics/
│   ├── episodes.csv
│   └── updates.csv
├── plots/
│   ├── reward.png
│   ├── search_rate.png
│   ├── energy.png
│   ├── broken_link.png
│   └── trajectory.png
└── rollouts/
    └── final_trajectory.npz
```

If training raises an exception after the model exists, the runner records the traceback/context and also attempts to save `checkpoints/crash.pt`.

## What is inside `low_episodes.jsonl`?

Each low/bad episode stores the complete episode metrics plus diagnostic signals. The main signals are:

- `search_low` — target search rate ≤ 20%;
- `link_unstable` — mean broken-link time is high or mean rate falls below 1 Mbps;
- `collision_high` — UAV safety-distance or obstacle problems;
- `energy_high` — minimum battery approaches the paper's safety limit;
- `action_saturated` — policy repeatedly outputs near ±1;
- `critic_unstable` — critic loss is non-finite or extremely large;
- `return_low` — return is in the low tail of the recent rolling window.

This makes the log useful for deciding whether to tune reward coefficients, communication geometry, exploration, critic learning rate, or action scaling rather than recording only a scalar reward.

## Important metrics

`metrics/episodes.csv` includes, at minimum:

- `return_mean`, `return_sum`
- `targets_found`, `search_rate`
- `energy_j`, `min_battery_pct`
- `mean_comm_rate_mbps`
- `mean_broken_link_s`, `max_broken_link_s`
- `collisions`, `obstacle_hits`, `boundary_hits`
- `action_saturation`
- `critic_loss`, `actor_loss`, and MASAC entropy/alpha when applicable

These directly support the paper's three main outcome families: **target search**, **energy consumption**, and **broken-link duration**.

## Determinism

The runner seeds Python, NumPy, PyTorch, replay sampling, environment generation, and action exploration. It also enables deterministic PyTorch algorithms where supported. Same code/config/seed/device is intended to be repeatable; CPU vs GPU and different PyTorch/CUDA versions can still introduce small numeric differences.

## Tests

```bash
pytest -q
```

The suite covers config provenance, environment physics/metrics, deterministic seeded reset, replay/network components, all three algorithms, checkpoint round-trip, logging/diagnostics, plots, and end-to-end training smoke tests.
