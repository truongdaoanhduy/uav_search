# Paper Baselines Design

## Goal
Reproduce the software-simulation portion of Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network* (DOI 10.1109/TVT.2025.3594534), restricted to the three requested baselines: MASAC, MATD3, and MADDPG. Hardware experiments, GATAC, and SA-GATAC are excluded.

## Scope
- Primary scenarios: one fixed-wing relay + five multi-rotor search UAVs (`f1_m5`) and one fixed-wing relay + nine multi-rotor search UAVs (`f1_m9`).
- Paper simulation field: 5 km x 5 km, circular obstacles, random targets/buildings.
- Episode length: 50 steps; paper reference training budget: 50,000 episodes/rounds. CLI can run much smaller smoke budgets.
- CTDE: local observation at execution, centralized critic during training.
- Metrics: reward, targets found/search rate, energy, broken-link duration, collision/safety signals, communication rate, actor/critic losses, action saturation.

## Architecture
A compact custom PyTorch CTDE core is used rather than importing multiple incompatible MARL frameworks. All algorithms share replay, networks, checkpoint schema, config loading, trainer, metrics, diagnostics, and visualization. Algorithm modules contain only the learning-rule differences.

The environment exposes a dictionary-parallel API with normalized continuous 2-D actions for every UAV. Multi-rotor action maps to thrust magnitude + planar direction. Fixed-wing action maps to yaw-rate control + thrust/speed control. Flight is modeled in 2.5D: x/y dynamics are learned while type-specific altitude remains fixed, keeping the paper's heterogeneous dynamics while avoiding an unnecessary 6-DoF simulator.

## Paper-faithful models
- Communication: probabilistic LoS/NLoS path loss, SINR, and Shannon rate; fixed-wing is cluster center/relay.
- Multi-rotor energy: `Pdyn ~= P0 + mu*v^2 + d*a*v^3 + Pcom`; fixed-wing energy is not optimized.
- Rewards: multi-rotor communication + remaining-energy + safety + search reward; fixed-wing safety + target-detection shaping.
- Constraints/metrics: safe battery, minimum 1 Mbps A2A rate, map/obstacle safety, velocity limits.

## Provenance policy
Every value published in Table I is tagged `PAPER_EXPLICIT`. Values necessary for simulation but omitted by the paper are grouped under `assumed` in YAML and documented as `ASSUMED`; they can be changed without modifying code.

## Algorithms
- MADDPG: deterministic type-shared actors, centralized critic, target networks, Gaussian exploration.
- MATD3: MADDPG plus twin critics, clipped target-policy smoothing, delayed actor update.
- MASAC: stochastic tanh-Gaussian type-shared actors, twin centralized critics, entropy regularization.

Actors are shared among UAVs of the same physical type to keep code and parameter count small while preserving heterogeneous policies. Centralized critics output one Q value per agent so the environment can retain per-agent rewards.

## Run artifacts
Each run writes under `runs/<algorithm>/<scenario>/<timestamp>/`:
- resolved config,
- `metrics/episodes.csv` and `metrics/updates.csv`,
- `logs/errors.log`, `logs/low_episodes.jsonl`,
- `checkpoints/best.pt`, `checkpoints/final.pt`, and `checkpoints/crash.pt` on exception,
- plots for reward, search rate, energy, broken links, and final trajectory,
- optional W&B logging when explicitly enabled and installed.

Low-episode diagnostics save the complete episode metrics and machine-readable warning signals (`search_low`, `link_unstable`, `collision_high`, `energy_high`, `action_saturated`, `critic_unstable`) so tuning decisions can be made from the logs.

## Testing / done criteria
- Unit tests for deterministic seeded environment dynamics and metrics.
- Unit tests for all three algorithms producing bounded finite actions, performing at least one finite update, and checkpoint round-trip.
- Logging/plot tests.
- End-to-end smoke training for all 3 algorithms on both primary scenarios without errors.
- Local HEAD and `origin/main` must resolve to the same final commit after verification.
