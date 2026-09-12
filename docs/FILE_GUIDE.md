# File / Folder Guide

```text
uav_search/
├── configs/
│   ├── paper.yaml
│   ├── algorithms/
│   └── scenarios/
├── src/uav_search/
│   ├── actions.py
│   ├── config.py
│   ├── dependencies.py
│   ├── envs/
│   ├── algorithms/
│   └── runner/
├── scripts/
├── tests/
├── docs/
│   ├── README.md       # authoritative docs index
│   ├── legacy/         # legacy paper reproduction mapping
│   └── archive/        # superseded plans/specs
└── runs/              # generated, gitignored
```

## `configs/`

**`paper.yaml`** — single source of truth for environment/training constants. `paper:` contains values explicitly printed by the article; `assumed:` contains values required by code but not published; `runtime:` contains engineering defaults such as buffer/batch/hidden sizes.

**`algorithms/*.yaml`** — only algorithm-specific hyperparameters. Adding a new baseline later should generally start here plus one algorithm module.

**`scenarios/*.yaml`** — swarm/target/obstacle counts. `f*_m*` files are paper-reproduction scenarios; `u6.yaml` and `u9.yaml` are the active homogeneous peer research scales and contain per-field provenance for altitude/sensing, GCS, DTN, RF, battery and calibration-required parameters.

## `src/uav_search/` core helpers

**`actions.py`** — authoritative U6/U9 hybrid-action codec. The peer action is `[a_x,a_y,a_z,tx_gate,tx_power,recipient]`; it canonicalizes hard gate/recipient semantics for replay/critics, provides straight-through torch canonicalization, and computes meaningful continuous-action saturation.

**`dependencies.py`** — single source for the pinned UavNetSim repository/version/commit and SimPy version, plus strict runtime preflight used by Kaggle/local setup.

## `src/uav_search/envs/`

**`models.py`** — pure physical/model functions: A2A communication rate, multi-rotor power, circle collision. Kept separate so radio/energy models can later be replaced independently.

**`paper_env.py`** — stateful root-paper mission simulator. Paper-reproduction scenarios retain their existing planar control contract. In `u6`/`u9` mode it switches to homogeneous peers, full 3D velocity/altitude motion, common-base launch pads, rejection-sampled valid obstacles, communication-gated local/stale Dec-POMDP peer observations, Bayesian local sensing with receiver-only minimum-uncertainty belief fusion after successful peer synchronization and cell-first true/false confirmation, discrete-time CBF-style inter-UAV safety shielding with intervention diagnostics, finite report buffers/TTL, one-hop-per-step forwarding, battery depletion, and direct/multi-hop/disconnected GCS diagnostics while delegating transport outcomes to `network_backends.py`.

**`sensing.py`** — pure altitude-aware sensing primitives: continuous altitude/Pd/Pf interpolation, exact circular-footprint/absolute-cell intersection at the UAV's true XY pose, deterministic circle–cell coverage fraction, coverage-weighted log-odds Bayesian evidence, legacy discrete profile helpers, Bayesian occupancy update and binary entropy. Keeping these functions pure makes sensing tests deterministic and separates paper-derived sensor parameters from environment control logic.

**`network_backends.py`** — hybrid networking adapter. `AnalyticalNetworkBackend` preserves the paper-equation peer channel for regression/ablation while honoring each scenario's power-scaled reference envelope. `UavNetSimBackend` lazily imports the pinned UavNetSim stack and keeps one SimPy/channel/node set alive per episode. UAV-to-UAV links retain native UavNetSim A2A propagation; any data/ACK pair involving the synthetic GCS uses the configured Al-Hourani-style urban A2G gain. MARL owns transmit gating, transmit power and immediate next-hop selection; communication consumes only data queued before the current sensing phase, and existing reports are forwarded before TTL expiry is checked. UavNetSim supplies CSMA/CA, PHY/channel delivery, native ACK/ARQ retries, delay/PDR/throughput and per-UAV radio TX energy; a commanded link rejected by the project candidate envelope still incurs sender probe-airtime energy rather than becoming a free no-op. Packet-delivery metrics distinguish MAC-admitted/generated payload from application-offered load: `network_byte_pdr = delivered/admitted`, while `network_offered_delivery_ratio = delivered/offered`. Application state advances only through the longest ACKed contiguous payload prefix. RF power changes the operational candidate envelope around the 0.1 W reference radius and also affects PHY/energy. The environment computes both minimum- and maximum-power snapshots so the actor can distinguish links that already work at low power from links that require more power; the legacy topology diagnostic remains based on the maximum-power graph.

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

**`network_calibration.py`** — non-learning `u6`/`u9` calibration runner. It drives deterministic-by-seed **3D** random-waypoint motion (including altitude changes across the configured flight levels) through the real UavNetSim backend and aggregates direct/multi-hop/disconnected node-step fractions, GCS hops, neighbor degree, and optional traffic metrics across seeds/ranges.

**`evaluate.py`** — reload a checkpoint and evaluate it over arbitrary random test cases.

**`logging.py`** — CSV logging, traceback context, and low-episode diagnostics.

**`wandb_logger.py`** — optional lazy W&B adapter; W&B is not imported unless enabled.

**`visualize.py`** — headless matplotlib training curves and trajectory renderer.

## `scripts/`

**`train.py`** — train one algorithm/scenario.

**`install_uavnetsim.sh`** — installs SimPy plus pinned UavNetSim commit `04daafb815eb377409b40b285574eeb62b9a8d58` with `--no-deps`; the active lightweight A2A/MAC/PHY integration does not execute the Sionna-RT path.

**`setup_kaggle.sh`** — clean Kaggle bootstrap: project dependencies → pinned UavNetSim → strict `check_system.py --require-uavnetsim` preflight.

**`run_all.py`** — train/evaluate the three baselines over the active `u6`/`u9` research scenarios and build comparison figures.

**`run_u6.py`** — run MASAC, MATD3 and MADDPG sequentially on the homogeneous `u6` joint search/networking scenario; default deterministic seed is 44.

**`calibrate_u6_network.py`** — backward-compatible calibration CLI for both active research scales via `--scenario u6|u9`; defaults to 50 seeds × 600 steps over 1.0/1.5/2.0/2.5 km reference-power envelopes (at 0.1 W) and writes CSV/JSON under `runs/calibration/<scenario>-network/`. Use `--traffic` only when packet-level PDR/delay/energy calibration is desired.

**`evaluate.py`** — multi-case checkpoint evaluation.

**`visualize.py`** — regenerate figures from an existing run without retraining.

## `tests/`

- `test_config.py` — paper constants/provenance/scenario merge.
- `test_env.py` — legacy paper simulator determinism, output contract, target confirmation, episode length, physics monotonicity.
- `test_homogeneous_paper_env.py` — homogeneous `u6` topology/action/report-routing behavior and seed determinism.
- `test_homogeneous_algorithms.py` — verifies MASAC/MATD3/MADDPG accept the six-dimensional peer action and MATD3 smooths only continuous coordinates.
- `test_action_semantics_regressions.py` — Cartesian-zero neutrality, hard gate/recipient canonicalization, straight-through gradients, replay action manifold, MASAC continuous-only entropy, and hybrid saturation diagnostics.
- `test_terminal_lifecycle_regressions.py` — final-transition reward, finite-horizon termination, inactive-agent topology/safety exclusion, replay validity masks, and masked actor/critic updates.
- `test_dependencies.py` — exact UavNetSim/SimPy reproducibility contract and preflight metadata.
- `test_u6_3d_motion.py` — full-3D action, altitude initialization/bounds and legacy 2-D compatibility.
- `test_u6_3d_sensing.py` — altitude profiles, Bayesian local sensing, receiver-only minimum-uncertainty fusion, cell-first true/false confirmation, no hidden-target leakage and sensing diagnostics.
- `test_research_scenario_comm_gating.py` — active U6/U9 scope, successful/failed peer synchronization, no topology-only cache refresh, no global belief teleportation, and one-hop-per-macro-step knowledge propagation.
- `test_sensing.py` — pure FOV/Bayes/entropy primitives.
- `test_train_metrics.py` — runner extraction of altitude/belief/sensing episode metrics.
- `test_network_backends.py` — analytical-regression tests plus real-UavNetSim A2A peer, A2G GCS, power-aware topology and CSMA/ACK transport tests.
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
