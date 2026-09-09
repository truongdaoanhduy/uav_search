# u6 Full-3D Belief-Sensing Design

## Goal

Upgrade only the `u6` homogeneous peer scenario from fixed-altitude 2.5D search to a full-3D search/network research scenario while preserving all paper-faithful heterogeneous scenarios. Replace the peer-mode fixed `Dfound` target confirmation rule with altitude-aware sensing, Bayesian belief updates, and confidence-based confirmation. Keep persistent UavNetSim for networking and keep MASAC, MATD3, and MADDPG as the baseline algorithms.

## Non-goals

- Do not modify the paper-faithful `f*_m*` scenarios.
- Do not add camera images, YOLO, segmentation, or CV inference.
- Do not replace the existing MARL baselines with MAPPO/AM-MAPPO.
- Do not add full rigid-body quadrotor physics or PyBullet/AirSim.
- Do not add moving targets in this change; `u6` remains post-disaster static-victim search.
- Do not silently relabel research adaptations as paper-explicit parameters.

## Source hierarchy

1. Root scenario paper: Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network*.
2. 3D sensing model: Liu et al., *Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios*, Drones 2024, 8, 378.
3. Adaptive altitude and communication-gating concept: Wang et al., *Multi-UAV collaborative maritime search via deep reinforcement learning*, Ad Hoc Networks 190 (2026) 104277.
4. 3D movement + next-hop + transmit-power precedent and altitude bounds: Yuan and Gao, *Learn to Access and Backhaul the Sky: Multi-Scale Radio Map Guided Multi-UAV Cooperation*, arXiv:2606.06954.
5. Network execution: Zhou et al., *UavNetSim-v1: A Python-based Simulation Platform for UAV Communication Networks* and the official `Zihao-Felix-Zhou/UavNetSim` repository.
6. Fixed communication-radius sensitivity: Dou et al., *Cooperative Multi-UAV Search for Prioritized Targets Under Constrained Communications*, Drones 2025, 9, 855.

## Provenance taxonomy

Every new configuration field must use one of:

- `ROOT_PAPER_EXPLICIT`
- `CORE_PAPER_EXPLICIT:<short-paper-name>`
- `UAVNETSIM_NATIVE`
- `LITERATURE_BACKED_BASELINE:<short-paper-name>`
- `RESEARCH_ADAPTATION`
- `RESEARCH_ASSUMPTION`
- `RESEARCH_DESIGN_CALIBRATED`

A number composed from two papers is still a `RESEARCH_ADAPTATION`, not paper-explicit.

## 3D motion

`u6` changes from 5-D to 6-D continuous Box actions:

1. horizontal thrust magnitude command,
2. horizontal heading command,
3. vertical acceleration command,
4. transmit gate,
5. RF transmit-power command,
6. recipient/next-hop command.

The existing two horizontal controls retain their current mapping. The vertical command maps `[-1, 1]` to `[-a_max, +a_max]` and uses the root paper's published multirotor acceleration limit as the magnitude bound. The total 3D velocity is clipped to the root paper's published multirotor maximum speed.

`u6` uses the altitude interval 50--150 m from MRMG. For sensing, this interval is represented by three reference levels `[50, 100, 150]` m. The endpoints are MRMG-explicit; the 100 m midpoint and mapping of Liu's abstract low/mid/high levels onto physical metres are a `RESEARCH_ADAPTATION`.

Initial altitude is sampled deterministically from these three levels using the episode RNG. This adapts Liu et al.'s random initialization over altitude levels to the MRMG physical altitude interval.

Altitude is bounded to `[50, 150]` m. Vertical boundary saturation counts as a boundary event. Existing circular obstacles remain vertical no-fly columns for `u6`; no building-height model is invented in this change.

## Altitude-aware sensing and belief

The fixed peer-mode rules `target_detect_m` and `target_found_m` no longer determine detection/confirmation.

The search area is discretized into 100 m cells. This preserves Liu et al.'s published spatial discretization density: 2000 m / 20 cells = 100 m/cell, while scaling the root 5 km map to 50 x 50 cells. The resulting 50 x 50 map size is therefore a `RESEARCH_ADAPTATION`.

Each UAV maintains a local belief map initialized to 0.5, following Liu et al. The ground-truth occupancy map is environment-only and is never inserted into actor observations.

The nearest sensing altitude level selects Liu et al.'s published sensing profile:

| Level | Physical level in u6 | FOV cells | Pd | Pf | Provenance |
|---|---:|---:|---:|---:|---|
| Low | 50 m | 1 | 0.90 | 0.10 | Liu explicit for FOV/Pd/Pf; metre mapping adapted |
| Mid | 100 m | 5 | 0.80 | 0.20 | same |
| High | 150 m | 9 | 0.70 | 0.30 | same |

The 1-cell FOV is the center cell; the 5-cell FOV is center plus the four cardinal neighbors; the 9-cell FOV is the full 3 x 3 neighborhood. Liu publishes the 1/5/9 sizes and a fixed 3 x 3 state representation, but not the exact 5-cell geometry in text, so the cross shape is explicitly `RESEARCH_ADAPTATION`.

For each scanned cell, the simulator samples a Bernoulli observation from `Pd` when a true target occupies the cell and from `Pf` when the cell is empty. The cell belief is updated by Bayes' rule exactly in the Liu/Dou model family. The update is deterministic under a fixed RNG seed.

A target is confirmed when any active UAV's local belief at the target cell reaches `0.99`, using Liu et al.'s published threshold `tau_p = 0.99`. If more than one UAV confirms the same target in one step, the deterministic source choice is: highest posterior belief, then shortest horizontal distance, then lowest agent index. This tie-break rule is a `RESEARCH_ADAPTATION`.

Confirmation creates the existing TargetReport and enters the existing buffer/TTL/DTN/UavNetSim delivery pipeline. There is no `Dfound` shortcut in `u6` after this change.

## Observation

Paper-faithful scenarios keep their existing observation dimensions.

For `u6`, the existing common observation remains, with two changes:

- own z-velocity is now the real vertical velocity instead of zero;
- append a fixed 3 x 3 local belief patch (9 values), aligned with Liu et al.'s fixed FOV encoding idea.

Cells outside the current altitude-specific FOV are zero-masked. Target-distance slots expose a target only when the local sensor produced a positive observation in the current step or the target is already locally known/confirmed. This prevents ground-truth position leakage outside sensing.

## Task reward

For `u6`, remove distance-to-hidden-target shaping from `_task_reward`, because it leaks ground-truth target distance into learning even when the target is outside sensing.

Use two sensing terms:

- target-confirmation term with relative weight `w1 = 1.0` from Liu et al.;
- information-gain term with relative weight `w2 = 0.1` from Liu et al.

To preserve the existing `u6` reward order of magnitude relative to communication/delivery, multiply the sum by the existing `search_reward_coeff = 20`. That overall scale is a `RESEARCH_ADAPTATION`; only the 1.0:0.1 relative weighting is paper-backed.

Existing communication, delivery, energy, and safety rewards remain unchanged in this feature.

## Networking

The configured `Rcomm = 2000 m` remains fixed as a one-hop candidate radius. It was selected by prior project calibration and is literature-supported by constrained-communication search work; it is not root-paper explicit.

Transmit power must not enlarge the hard contact radius. Power is passed to UavNetSim and affects PHY/SINR/energy inside the candidate radius. This aligns the project with the fixed-radius + link-quality separation used by constrained-communication literature and removes the project's earlier `R proportional sqrt(P)` engineering assumption.

UavNetSim already consumes 3D positions, so peer/GCS distances continue to be Euclidean 3D distances.

## Energy

The existing root-derived multirotor power approximation is retained. For `u6`, speed and acceleration magnitudes become 3D magnitudes so climb/descent cannot be energy-free. Applying the existing approximation to vertical motion is a `RESEARCH_ADAPTATION`, not a paper-explicit vertical-flight energy model.

Battery capacity is not changed in this feature; the existing 1 MJ value remains a separate calibration issue.

## Metrics and diagnostics

Add to `info` and episode/W&B metrics for `u6`:

- mean/min/max altitude,
- mean belief entropy,
- mean target-cell posterior,
- scanned-cell count,
- positive sensor observations,
- information gain,
- targets confirmed this step/episode.

Keep existing search rate, report delivery, queue, PDR, delay, topology, energy, and collision metrics.

## Testing

TDD requirements:

1. `u6` action dimension becomes 6 while paper-faithful scenarios stay 2.
2. vertical action changes z and z-velocity and respects 50--150 m bounds.
3. same seed produces identical sensing samples, belief maps, and rollouts.
4. low/mid/high profiles expose 1/5/9 cells and correct Pd/Pf values.
5. Bayes update matches the analytical formula for positive and negative observations.
6. no `Dfound` confirmation occurs in peer mode.
7. posterior >= 0.99 confirms exactly once and creates exactly one report.
8. actor target slots do not leak unsensed target distance.
9. fixed 2 km contact radius is not enlarged by 0.4 W transmit power.
10. MATD3 smoothing treats vertical and power dimensions as continuous, and gate/recipient as non-smoothed.
11. network calibration runner emits valid 6-D actions.
12. all previous tests continue to pass or are intentionally updated for the new `u6` contract.
13. MASAC/MATD3/MADDPG smoke runs pass with real UavNetSim.
14. same-seed deterministic smoke repeats exactly on mission metrics.

## Rollback and integration

Before feature implementation, preserve the previous code at branch `backup/u6-before-3d-sensing-20260909`, pointing to commit `c501284bddecceb4af3dfd30c5e67b4318c85620` and push it to GitHub.

Develop in isolated branch/worktree `feature/u6-3d-sensing`. After fresh full verification, fast-forward the user-facing branch `feature/homo-u6-paper-sim`, push it, and verify local HEAD, tracking branch, and GitHub remote SHA match.
