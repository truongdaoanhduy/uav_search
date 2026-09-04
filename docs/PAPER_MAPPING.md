# Paper → Code Mapping

This document separates **what the article explicitly specifies** from values that have to be assumed to make an executable simulator.

## 1. Simulation setup

| Paper item | Paper value | Code |
|---|---:|---|
| Simulation field | 5 km × 5 km | `paper.area_size_m: 5000` |
| Fixed-wing mass | 10 kg | `paper.fixed_wing_mass_kg` |
| Multi-rotor mass | 1 kg | `paper.multirotor_mass_kg` |
| Communication power | 5 W | `paper.communication_power_w` |
| Minimum A2A rate | 1 Mbps | `paper.min_comm_rate_bps` |
| Fixed-wing max speed | 40 m/s | `paper.fixed_speed_max_mps` |
| Fixed-wing min speed | 10 m/s | `paper.fixed_speed_min_mps` |
| Max acceleration | 8 m/s² | `paper.max_accel_mps2` |
| Multi-rotor max speed | 10 m/s | `paper.multirotor_speed_max_mps` |
| Safe battery level | 10% | `paper.safe_battery_pct` |
| Energy reward scale τ | 0.2 | `paper.energy_reward_scale` |
| Training rounds | 50,000 | `paper.training_rounds` |
| Episode length | 50 | `paper.episode_steps` |
| Learning rate | 0.001 | `paper.learning_rate` / algorithm LR |
| Discount γ | 0.99 | `algorithm.gamma` |
| Paper soft coefficient ε | 0.99 | `paper.soft_update_epsilon` |

The paper writes its target update in the form `target <- ε*target + (1-ε)*online`. The code uses the conventional form `target <- (1-tau)*target + tau*online`, therefore **`tau = 1 - ε = 0.01`**. This is a notation conversion, not a changed paper value.

## 2. Communication model — paper Eq. (3)–(7)

Implementation: `src/uav_search/envs/models.py::communication_rate_bps`.

Flow:

1. 3-D UAV distance and elevation angle;
2. sigmoid LoS probability;
3. free-space loss + LoS/NLoS additional loss;
4. received SNR using 5 W communication power;
5. `R = B log2(1 + SNR)`;
6. compare against the paper's 1 Mbps minimum rate.

The article does not provide every radio constant needed to numerically evaluate the equations (carrier, bandwidth, environmental sigmoid constants, additional losses, noise). Those values live only in `configs/paper.yaml -> assumed` so they can be replaced without changing code.

## 3. UAV dynamics — paper Eq. (8)–(12)

Implementation: `src/uav_search/envs/paper_env.py::step`.

- Fixed-wing: retains planar yaw, heading, minimum/maximum airspeed, and thrust/speed control.
- Multi-rotor: planar force/acceleration vector, velocity integration, max-speed constraint.
- A 2.5D simulator is used: x/y are dynamic and altitude is fixed by type. This is an explicit implementation assumption made to keep the reproduction minimal while preserving the paper's heterogeneous flight roles.

Normalized policy actions are always 2-D:

- fixed-wing: `[yaw_control, thrust/speed_control]` → paper `{γ̇, F}`;
- multi-rotor: `[thrust_control, direction_control]` → paper `{F, θ}`.

## 4. Energy — paper Eq. (13)–(14)

Implementation: `src/uav_search/envs/models.py::multirotor_power_w`.

The code follows:

`Pdyn(v,a) ≈ P0 + μ v² + d a v³`

and adds the constant communication power. Only multi-rotor energy is optimized/tracked as a battery budget, matching the paper's decision not to optimize fixed-wing energy for the SAR mission.

`P0`, blade-drag coefficient, frame-drag coefficient, and battery capacity are unpublished simulator requirements and are marked `ASSUMED`.

## 5. State / observation — paper Eq. (17)–(20)

Implementation: `PaperUAVEnv._observations`.

Every actor receives local structured information containing:

- normalized own position, velocity/speed, battery, heading/type/link state;
- relative information for every other UAV;
- relative target positions/distances/found state;
- circular-obstacle geometry/clearance.

The centralized critic concatenates all agents' observations and actions, implementing CTDE. Observation dimension deliberately grows with swarm size for these baseline algorithms; no GAT compression is used.

## 6. Rewards — paper Eq. (21)–(27)

Implementation: `PaperUAVEnv._task_reward`, `_safety_reward`, and `step`.

Multi-rotor reward:

`r = r_comm + r_power + r_safe + r_task`

Fixed-wing reward:

`r = r_safe + r_task_fixed`

The code preserves the paper's piecewise structure:

- rate below `Rmin` → communication penalty;
- valid link → distance-sensitive communication reward;
- remaining battery above `esafe` → scaled battery reward;
- too-small UAV separation / obstacle collision → safety penalty;
- target within detect/found radius → distance-sensitive / confirmation reward.

The article does not publish all reward magnitudes and detect/safety radii. They are all under `assumed`.

## 7. Algorithms

### MADDPG

`src/uav_search/algorithms/maddpg.py`

- deterministic actor;
- centralized critic;
- replay buffer;
- online/target actor and critic;
- Gaussian exploration during collection;
- soft target update.

### MATD3

`src/uav_search/algorithms/matd3.py`

Adds to MADDPG:

- twin centralized critics;
- min-Q target;
- clipped target-policy smoothing noise;
- delayed actor/target updates.

### MASAC

`src/uav_search/algorithms/masac.py`

- tanh-Gaussian stochastic actor;
- twin centralized critics;
- entropy regularization;
- online/target actor and target critics, following the target-network description/equations in the paper;
- fixed entropy coefficient (`alpha`) because the paper does not publish an automatic temperature-tuning procedure.

Actors are parameter-shared by UAV physical type (`fixed`, `rotor`). The centralized critic outputs one Q value per agent so per-agent rewards are retained while avoiding duplicated critic boilerplate.

## 8. Scenarios implemented

- `configs/scenarios/f1_m5.yaml`: 1 fixed-wing + 5 multi-rotor.
- `configs/scenarios/f1_m9.yaml`: 1 fixed-wing + 9 multi-rotor.

The article's Fig. 7 explicitly names these swarm sizes. It does not clearly state the ground-target count for the two Fig. 7 small scenarios in the available text, so the repository uses **5 targets as an ASSUMED value** for both. Change the YAML if author source code or supplementary material gives the exact value.

## 9. Evaluation / figures

The article reports results over 5,000 random test cases. `scripts/evaluate.py --episodes 5000` reproduces that evaluation count.

Automatically generated code-side figures:

- `reward.png` — training return;
- `search_rate.png` — target-search rate;
- `energy.png` — multi-rotor energy;
- `broken_link.png` — mean broken-link duration;
- `trajectory.png` — post-training deterministic multi-UAV rollout over targets/obstacles.

These are designed to expose the same metric families as the paper without fabricating the paper's 50,000-round curves from a short smoke run.
