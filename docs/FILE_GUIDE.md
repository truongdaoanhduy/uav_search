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

**`scenarios/*.yaml`** — swarm/target/obstacle counts. `f*_m*` files are paper-reproduction scenarios; `u6.yaml` is the homogeneous peer research adaptation and also contains the adapted GCS/report/buffer parameters.

## `src/uav_search/envs/`

**`models.py`** — pure physical/model functions: A2A communication rate, multi-rotor power, circle collision. Kept separate so radio/energy models can later be replaced independently.

**`paper_env.py`** — stateful root-paper mission simulator: randomized scenario reset, fixed-wing/multi-rotor motion, target confirmation, rewards, observations, episode metrics, and trajectory history. In `u6` mode it switches to six identical peers, launch-zone deployment, local/stale Dec-POMDP peer observations, per-agent target knowledge, finite report buffers/TTL, one-hop-per-step forwarding, and direct/multi-hop/disconnected GCS diagnostics, while delegating transport outcomes to `network_backends.py`.

**`network_backends.py`** — hybrid networking adapter. `AnalyticalNetworkBackend` preserves the previous paper-equation peer channel for regression/ablation while honoring `u6` contact caps. `UavNetSimBackend` lazily imports the pinned UavNetSim stack and uses its A2A path-gain model, `CsmaCa`, packet events and `Channel`; the backend, not `PaperUAVEnv`, accepts/rejects selected links. MARL still owns transmit gating, requested byte amount, and immediate next-hop selection.

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

**`install_uavnetsim.sh`** — installs SimPy plus pinned UavNetSim commit `04daafb815eb377409b40b285574eeb62b9a8d58` with `--no-deps`; the fast `a2a` integration does not require Sionna RT.

**`run_all.py`** — run exactly the requested 3 algorithms across the current paper-reproduction scenario set.

**`run_u6.py`** — run MASAC, MATD3 and MADDPG sequentially on the homogeneous `u6` joint search/networking scenario; default deterministic seed is 44.

**`evaluate.py`** — multi-case checkpoint evaluation.

**`visualize.py`** — regenerate figures from an existing run without retraining.

## `tests/`

- `test_config.py` — paper constants/provenance/scenario merge.
- `test_env.py` — legacy paper simulator determinism, output contract, target confirmation, episode length, physics monotonicity.
- `test_homogeneous_paper_env.py` — homogeneous `u6` topology/action/report-routing behavior and seed determinism.
- `test_homogeneous_algorithms.py` — verifies MASAC/MATD3/MADDPG accept the `u6` five-dimensional action.
- `test_network_backends.py` — analytical-regression tests plus optional real-UavNetSim A2A/CSMA transport tests.
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
