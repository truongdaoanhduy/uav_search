# File / Folder Guide

```text
uav_search/
├── configs/
│   ├── paper.yaml
│   ├── algorithms/
│   └── scenarios/
├── src/uav_search/
│   ├── config.py
│   ├── envs/
│   ├── algorithms/
│   └── runner/
├── scripts/
├── tests/
├── docs/
└── runs/              # generated, gitignored
```

## `configs/`

**`paper.yaml`** — single source of truth for environment/training constants. `paper:` contains values explicitly printed by the article; `assumed:` contains values required by code but not published; `runtime:` contains engineering defaults such as buffer/batch/hidden sizes.

**`algorithms/*.yaml`** — only algorithm-specific hyperparameters. Adding a new baseline later should generally start here plus one algorithm module.

**`scenarios/*.yaml`** — swarm/target/obstacle counts. Scenario scaling therefore does not require editing Python code.

## `src/uav_search/envs/`

**`models.py`** — pure physical/model functions: A2A communication rate, multi-rotor power, circle collision. Kept separate so radio/energy models can later be replaced independently.

**`paper_env.py`** — stateful simulator: randomized scenario reset, fixed-wing/multi-rotor motion, target confirmation, link tracking, rewards, observations, episode metrics, and trajectory history.

## `src/uav_search/algorithms/`

**`networks.py`** — generic MLP deterministic actor, Gaussian actor, centralized critic.

**`common.py`** — replay buffer and target-network update helpers.

**`base.py`** — common off-policy algorithm metadata/replay/checkpoint interface.

**`maddpg.py`** — MADDPG learning rule.

**`matd3.py`** — inherits deterministic actor infrastructure and adds TD3 twin-Q/smoothing/delay behavior.

**`masac.py`** — stochastic entropy-regularized MASAC learning rule.

**`factory.py`** — one place to resolve `masac|matd3|maddpg` into the corresponding class. Runner code therefore has no algorithm `if/else` tree.

## `src/uav_search/runner/`

**`train.py`** — experiment orchestration: seeds, env → policy → replay → update loop, best/final/crash checkpoint, episode/update metrics, post-training deterministic rollout, summary and plot generation.

**`evaluate.py`** — reload a checkpoint and evaluate it over arbitrary random test cases.

**`logging.py`** — CSV logging, traceback context, and low-episode diagnostics.

**`wandb_logger.py`** — optional lazy W&B adapter; W&B is not imported unless enabled.

**`visualize.py`** — headless matplotlib training curves and trajectory renderer.

## `scripts/`

**`train.py`** — train one algorithm/scenario.

**`run_all.py`** — run exactly the requested 3 algorithms × 2 scenarios.

**`evaluate.py`** — multi-case checkpoint evaluation.

**`visualize.py`** — regenerate figures from an existing run without retraining.

## `tests/`

- `test_config.py` — paper constants/provenance/scenario merge.
- `test_env.py` — simulator determinism, output contract, target confirmation, episode length, physics monotonicity.
- `test_common.py` — replay/network/soft target update.
- `test_algorithms.py` — all three action/update/checkpoint flows.
- `test_logging.py` — diagnostics/error logs and plots.
- `test_smoke.py` — end-to-end train → checkpoint → five plots → trajectory artifacts.

## Where to customize later

- Change physical constants only: `configs/paper.yaml`.
- Change UAV count/scenario: `configs/scenarios/`.
- Change neural/training defaults: `runtime:` or `configs/algorithms/`.
- Replace radio/energy formulas: `envs/models.py`.
- Add a new MARL algorithm: add one module under `algorithms/`, one YAML, then register it in `factory.py`.
- Add another plot/metric: `runner/visualize.py` / `runner/logging.py` without touching algorithms.
