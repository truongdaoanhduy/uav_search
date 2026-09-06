# Paper → Code Mapping

This document separates **what the article explicitly specifies** from values that have to be assumed to make an executable simulator.

## 1. Simulation setup

| Paper item | Paper value | Code |
|---|---:|---|
| Simulation field | 5 km × 5 km | `paper.area_size_m: 5000` |
| Fixed-wing mass | 10 kg | `paper.fixed_wing_mass_kg` |
| Multi-rotor mass | 1 kg | `paper.multirotor_mass_kg` |
| Communication-energy power `P_com` | 5 W | `paper.communication_power_w` |
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
4. linear received signal power using the distinct Eq. (6) transmit power `P_tx,u`;
5. SINR including interference and Gaussian noise;
6. `R = B log2(1 + SINR)`;
7. compare against the paper's 1 Mbps minimum rate.

Table I publishes `P_com = 5 W` for communication-energy consumption, not a numeric value for Eq. (6) `P_tx,u`. Root-paper Ref. [39] Table III publishes UAV transmit power as 40 dBm = 10 W, so `P_tx,u = 10 W` is kept under `reference_backed`. Other recovered channel constants from refs. [39]-[41] are also `REFERENCE_BACKED`; none are labeled as original-paper explicit values.

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

The baseline now embeds the two paper type-specific observations into one common padded tensor width required by the baseline training interface:

- common self slots are the union `{p(3), v(3), e, n, psi}`;
- multi-rotor self-state uses `{p, v, e, n}` from Eq. (17) and masks `psi`;
- fixed-wing self-state uses `{p, v, psi}` from Eq. (19) and masks energy/network self slots;
- each other-UAV block is the union `{d_i,j, p_j(3), e_j, n_j}`; fields absent from Eq. (20) are zero-masked for fixed-wing observers;
- target input is only `d_i,k`, visible only within the type's sensing range and zero-masked once found/out of range;
- obstacle geometry, agent-type flags, and always-visible target coordinates are no longer leaked into the Actor observation because they are not present in Eqs. (17)-(20).

The centralized critic concatenates all agents' padded observations and actions, implementing CTDE. Observation dimension still grows with swarm size for these baseline algorithms; GAT importance ranking/aggregation remains outside baseline scope.

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
- too-small inter-UAV separation → safety penalty; obstacle-domain entry is handled as the separate hard constraint C3, not as an unpublished reward term;
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
- independent automatic entropy-temperature tuning per UAV, following root-paper Ref. [44]; `alpha_init = 0.01` is reference-backed while the separate alpha learning rate falls back to the paper actor learning rate.

Each UAV has its own Actor and centralized Critic (twin centralized Critics for MATD3/MASAC). This matches the root paper's Algorithm 1 wording that Actor/Critic parameters are initialized and updated for each UAV; actors are not parameter-shared by physical type.

## 8. Scenario scope

Current active/default reproduction scope:

- `configs/scenarios/f1_m5.yaml`: 1 fixed-wing + 5 multi-rotor.
- `configs/scenarios/f1_m9.yaml`: 1 fixed-wing + 9 multi-rotor.

The article's Fig. 7 explicitly names these swarm sizes. It does not separately print the ground-target count for Fig. 7, so both active scenario files currently use **10 targets as `PAPER_INFERRED`**, based on the paper's neighboring small-scale experiment description. This must not be presented as `PAPER_EXPLICIT`.

The four larger Fig. 8-9 configurations are present as later-phase config files but are not included in `ACTIVE_SCENARIOS` or the default `run_all.py` scenario list.

## 9. Evaluation / figures

The article reports results over 5,000 random test cases. `scripts/evaluate.py --episodes 5000` reproduces that evaluation count.

Automatically generated per-run figures include the Fig. 6-style scenario map plus reward/search/energy/broken-link/group-return/reward-component/trajectory diagnostics. After all six active runs finish, `run_all.py` creates Fig. 7/10/11/12-style paper comparison figures.

These plots expose the same result families as the paper without fabricating 50,000-round results from short smoke runs. A paper-scale comparison requires `--episodes 50000 --eval-episodes 5000`.
