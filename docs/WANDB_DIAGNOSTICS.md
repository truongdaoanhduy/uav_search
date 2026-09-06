# W&B monitoring and failure diagnosis

Current reproduction scope is the two small-scale Fig. 7 scenarios: `f1_m5` and `f1_m9`, with `MASAC`, `MATD3`, and `MADDPG`.

## Paper-facing metrics

These keys are the primary paper-comparison signals and are logged live:

- `paper/reward_total`: team return used for the Fig. 7-style reward comparison.
- `paper/targets_found` and `paper/search_rate`: target-search performance.
- `paper/energy_consumption_pct`: average multi-rotor battery consumption percentage. Fixed-wing energy is intentionally excluded because the root paper explicitly does not optimize it.
- `paper/broken_link_duration_s`: mean rotor-to-leader broken-link duration.

A full 50,000-episode `run_all.py` run defaults to 5,000 evaluation cases, matching the root paper. Short development runs default to 10 cases unless `--eval-episodes` is supplied.

## Diagnostic metrics

Paper plots answer whether an algorithm performs well. Diagnostic metrics answer why it performs poorly.

### Swarm / UAV-type level

- `diagnostics/fixed_return_mean`
- `diagnostics/rotor_return_mean`
- `diagnostics/rotor_return_min`
- `diagnostics/reward_task_sum`
- `diagnostics/reward_communication_sum`
- `diagnostics/reward_energy_sum`
- `diagnostics/reward_safety_sum`
- `diagnostics/worst_agent_return`

Interpretation examples:

- Fixed return falls first and most rotors then show low communication rates: inspect the fixed-wing leader trajectory/coverage.
- Rotor mean stays reasonable but rotor minimum collapses: inspect the reported `worst_agent` rather than the whole swarm.
- Task reward stays near zero while communication and safety are healthy: search behavior / target approach is the likely bottleneck.
- Communication reward drops with broken-link duration rising: formation/relay geometry is the likely bottleneck.

### Per-agent level

For every UAV:

- `diagnostics/agent/<name>/return`
- `diagnostics/agent/<name>/task_reward`
- `diagnostics/agent/<name>/communication_reward`
- `diagnostics/agent/<name>/energy_reward`
- `diagnostics/agent/<name>/safety_reward`
- `diagnostics/agent/<name>/battery_pct`
- `diagnostics/agent/<name>/action_saturation`

For each multi-rotor UAV additionally:

- `diagnostics/agent/<name>/comm_rate_mbps`
- `diagnostics/agent/<name>/broken_link_s`
- `diagnostics/agent/<name>/energy_consumption_pct`

This is sufficient to distinguish a single failed rotor from a rotor-group problem or a fixed-wing/whole-swarm problem.

### RL training health

The update stream and episode CSV expose:

- `actor_loss`
- `critic_loss`, plus twin-critic losses where applicable
- `q_mean`
- `target_q_mean`
- `td_error_abs_mean`
- `q_gap_abs_mean` for MATD3/MASAC
- MASAC: `entropy`, `alpha`, and `alpha_loss`

If returns collapse while environment diagnostics remain healthy but critic loss / TD error / Q values become unstable, investigate optimization/training rather than changing UAV physics or rewards.

## Low-metric episode records

An anomalous episode records:

- episode number and training phase (`warmup`, `learning`, `post_convergence_reference`)
- `failure_scope`: `single_agent`, `rotor_group`, `fixed_wing`, or `swarm`
- `worst_agent`
- `primary_cause` and `secondary_cause`
- all paper, swarm, per-agent, and training-health values in that episode

Random warm-up search/link failures are not promoted to anomalies unless safety, energy, or critic health indicates a real failure. This avoids filling W&B with expected early-training noise.

## Images uploaded to W&B

High-value live previews are refreshed at checkpoints:

- `reward.png`
- `group_returns.png`
- `reward_components.png`

Final per-run images include:

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

The image policy is intentional: scalar metrics are more useful for live diagnosis and are cheap to log every episode, while images are refreshed only at checkpoints/final comparison to avoid unnecessary W&B media volume.
