# W&B monitoring — compact aggregate telemetry

Current active homogeneous-swarm scope: `u6` and `u9`, with `MASAC`, `MATD3`, and `MADDPG`.
Legacy `f*_m*` reproduction scenarios remain available, but cross-swarm reward comparisons below use the per-agent mean rather than the team sum.
The tracking policy intentionally avoids per-UAV series so a 50,000-episode run stays readable.
Detailed raw optimizer updates remain available locally in `metrics/updates.csv`.

## What is sent to W&B every episode

### Paper-facing results
- `paper/reward_mean` — primary reward metric for comparisons across different swarm sizes (`u6` vs `u9`)
- `paper/reward_total` — retained for backwards-compatible dashboards; do not use it as the primary U6/U9 comparison because shared rewards make it scale with agent count
- `paper/targets_found`
- `paper/search_rate`
- `paper/energy_consumption_pct`
- `paper/broken_link_duration_s`

### Swarm state
- `swarm/avg_battery_pct`
- `swarm/depleted_uavs`
- `swarm/safety_distance_violation_uavs`
- `swarm/obstacle_hit_uavs`
- `swarm/boundary_hit_uavs`
- `swarm/broken_link_uavs`
- `swarm/avg_comm_rate_mbps` — backward-compatible alias for the max-power potential-link diagnostic, not realized goodput
- `swarm/avg_broken_link_s`

Counts such as `safety_distance_violation_uavs` are the number of distinct UAVs that entered that state at least once during the episode, not the number of repeated safety-distance violation events.
Battery monitoring is intentionally only the swarm average plus the number of depleted UAVs.

### Network semantics
- `network/mean_potential_comm_rate_mbps` — max-controllable-power topology/link potential; useful for reachability diagnostics, not evidence that the policy transmitted.
- `network/mean_selected_tx_rate_mbps` — rate of links actually selected by gate-on actions; zero when there is no selected TX.
- `network/byte_pdr` — delivered payload bytes divided by MAC-admitted/generated payload bytes.
- `network/offered_delivery_ratio` — delivered payload bytes divided by all application bytes offered by the policy; this can be lower than PDR when a 1 s slot cannot admit all offered bytes.
- `network/throughput_bps` — episode delivered payload over elapsed mission time.


### Fixed-wing vs multi-rotor groups
- `group/fixed_return_mean`
- `group/rotor_return_mean`

### Reward decomposition
- `reward/task`
- `reward/communication`
- `reward/energy`
- `reward/safety`

### RL health
- `rl/actor_loss`
- `rl/critic_loss`
- `rl/q_mean`
- `rl/td_error`
- MASAC only: `rl/entropy`, `rl/alpha`

### Runtime throughput
- `performance/env_steps_per_sec`
- `performance/updates_per_sec`
- `performance/episode_sec`

No `diagnostics/agent/<uav>/...`, per-UAV battery, per-UAV communication-rate, or `worst_agent` metric is sent to W&B.
Anomaly records remain coarse (`swarm`, `fixed_wing`, or `rotor_group`) and report the likely cause rather than a specific UAV.

## Logging cadence

- Aggregate episode metrics: every episode.
- Optimizer `update/*` metrics: every 100 optimizer updates on W&B; every update still goes to local `metrics/updates.csv`.
- Terminal progress: episode 1, every 100 episodes, and the final episode by default.
- Live images: only at the 1,000-episode checkpoint cadence.
- Final paper comparison figures: after evaluation.

## Images

Per-run paper/diagnostic images remain bounded:
- `fig06_simulation_scenario.png`
- `reward.png`
- `search_rate.png`
- `energy.png`
- `broken_link.png`
- `group_returns.png`
- `reward_components.png`
- `trajectory.png`

After `run_all.py`, the comparison run uploads:
- `fig07_training_reward.png`
- `fig10_average_targets_found.png`
- `fig11_average_energy_consumption.png`
- `fig12_average_broken_link_duration.png`

## Reproducibility mode

Paper runs are deterministic by default. The runtime seeds Python, NumPy, Torch, environment RNG, exploration RNG, and replay sampling; it enables deterministic Torch kernels and configures cuBLAS determinism. Use `--no-deterministic` only when maximum throughput is more important than exact same-seed repetition on the same software/hardware stack.
