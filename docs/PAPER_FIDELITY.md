# Paper Fidelity Audit

This repository targets the simulation scenario in:

> T. Ao, H. Li, K. Zhang, H. Shi, L. Shi, F. Liu, and Y. Zhou, **"Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network,"** IEEE Transactions on Vehicular Technology, vol. 75, no. 1, pp. 1412-1426, 2026. DOI: 10.1109/TVT.2025.3594534.

The baseline algorithms in this repository are MASAC, MATD3, and MADDPG. This document audits the **scenario/environment model**, not algorithm hyperparameter fidelity.

## 1. Direct matches to the original paper

The following are implemented from statements/equations/experimental parameters published directly by the original paper.

| Item | Implementation |
|---|---|
| Search domain | 5 km x 5 km |
| Search-target positions | Uniform random over the full task domain |
| Building positions | Uniform random over the full task domain |
| Building geometry | Circles |
| UAV types | Fixed-wing + multi-rotor |
| Fixed-wing role | Leader/mobile relay and signal detection |
| Multi-rotor role | Follower/low-altitude target search and confirmation |
| Formation communication | Rotor-to-leader star links |
| Leader communication | Fixed-wing mesh links |
| A2A propagation | Probabilistic LoS/NLoS average path loss, paper Eqs. (3)-(5) |
| Fixed-wing-to-fixed-wing LoS | Forced LoS probability 1 as stated by the paper |
| Link quality | SINR including interference + Gaussian noise, Eq. (6) |
| Link rate | `B log2(1 + SINR)`, Eq. (7) |
| Connectivity threshold | `R_A2A > R_min` |
| Fixed-wing action semantics | yaw/heading-rate command + thrust/speed command proxy |
| Multi-rotor action semantics | force magnitude + planar force direction |
| Multi-rotor reward | `r_com + r_power + r_safe + r_task-m`, Eqs. (21)-(26) |
| Fixed-wing reward | `r_safe + r_task-f`, Eq. (27) |
| Obstacle domain | Hard flight constraint rather than an unpublished reward penalty |
| Task-area domain | UAV positions constrained to the task area |
| Fixed-wing mass | 10 kg |
| Multi-rotor mass | 1 kg |
| Communication power | 5 W |
| Minimum communication rate | 1 Mbps |
| Fixed-wing maximum speed | 40 m/s |
| Fixed-wing minimum speed | 10 m/s |
| Maximum acceleration | 8 m/s^2 |
| Multi-rotor maximum speed | 10 m/s |
| Safe battery level | 10% |
| Energy reward scaling factor | 0.2 |
| Training rounds reference | 50,000 |
| Episode length | 50 steps |
| Paper learning-rate reference | 0.001 |
| Reward discount factor | 0.99 |
| Soft-update coefficient | 0.99 |
| `f1_m5` swarm | 1 fixed-wing + 5 multi-rotor |
| `f1_m9` swarm | 1 fixed-wing + 9 multi-rotor |

## 2. Paper-inferred value

### Number of targets in `f1_m5` and `f1_m9`: 10

The paper explicitly states that the two-swarm experiments in Fig. 8 use **10 ground targets**, and discusses Fig. 7 and Fig. 8 as scale comparisons of the same post-disaster target-search setup. It does not separately print a sentence that says Fig. 7 has exactly 10 targets.

Therefore the small-scenario files use:

```yaml
targets: 10
targets_provenance: PAPER_INFERRED
```

This is intentionally **not** labeled `PAPER_EXPLICIT`.

## 3. Reference-backed channel constants

The original paper gives the equations but does not publish every numeric channel constant. For those missing numbers, this implementation uses papers that the original paper itself cites.

| Parameter | Value used | Provenance |
|---|---:|---|
| Carrier frequency | 700 MHz | Original-paper ref. [39] |
| Bandwidth | 1 MHz | Original-paper ref. [39] |
| Gaussian noise power | `1e-13 W` = -100 dBm | Original-paper ref. [39] |
| LoS sigmoid `a` | 11.95 | Original-paper ref. [41] urban setup |
| LoS sigmoid `b` | 0.14 | Original-paper ref. [41] urban setup |
| Mean LoS additional loss | 1 dB | Original-paper ref. [40], Urban |
| Mean NLoS additional loss | 20 dB | Original-paper ref. [40], Urban |

They live under `reference_backed:` in `configs/paper.yaml`; they are not presented as original-paper experimental constants.

## 4. Still not exactly reproducible from the paper

The original paper does not publish enough numerical information to reproduce these values exactly. They remain explicit implementation assumptions:

| Missing original-paper value | Current fallback |
|---|---:|
| Time-slot duration `delta_t` | 1 s |
| Fixed-wing simulation altitude | 200 m |
| Multi-rotor simulation altitude | 60 m |
| UAV initial XY distribution/coordinates | Uniform with 300 m edge margin |
| Number of circular buildings in `f1_m5` | 6 |
| Number of circular buildings in `f1_m9` | 8 |
| Building radius distribution | 80-220 m |
| Multi-rotor detection distance `D_detect` | 700 m |
| Confirmation distance `D_found` | 100 m |
| Fixed-wing detection distance `D_detect-f` | 1500 m |
| Safety distance `D_safe` | 100 m |
| Fixed-wing maximum yaw-rate mapping | 0.35 rad/s |
| Battery capacity in joules | 1,000,000 J |
| `P0` in Eq. (13) | 120 W |
| `mu` in Eq. (13) | 0.08 |
| `d` in Eq. (13) | 0.00002 |
| `R_max` for Eq. (21) | 10 Mbps |
| Communication reward coefficient `r_c` | 5 |
| Safety reward coefficient `eta` | 8 |
| Search reward coefficient `zeta` | 20 |
| Division stabilizer `Delta_d` | 100 m |

Because these values are absent from the original paper, matching its reward curves numerically cannot be guaranteed even when the equations and disclosed scenario structure match.

## 5. Deliberate implementation approximations

### 2.5D flight

The paper writes 3D state/dynamics equations, but its action definition does not publish an altitude control action or simulation altitude values. The baseline therefore keeps type-specific altitude fixed and optimizes planar trajectories. This is a reproduction gap, not a paper-explicit choice.

### Fixed-wing thrust dynamics

The paper provides simplified Newton-Euler equations but does not publish the aerodynamic drag coefficient, yaw moment of inertia, or rudder torque mapping needed for an exact numerical integrator. The normalized second fixed-wing action is therefore mapped to a bounded forward speed/thrust proxy while enforcing the paper's 10-40 m/s limits.

### Multi-rotor force dynamics

The multi-rotor normalized force action is converted to an acceleration bounded by the paper's 8 m/s^2 maximum-acceleration parameter. Exact force-to-acceleration dynamics beyond the published mass/constraint cannot be reconstructed because the needed aerodynamic constants are not reported.

### Observation vector

The world state now reflects the corrected paper scenario and communication topology, but the baseline retains a fixed-size observation vector convenient for MASAC/MATD3/MADDPG. It does **not** reproduce GATAC's GAT-based importance pruning/aggregation; implementing GATAC is outside the baseline-scenario task.

### OpenAI MPE

The paper states that simulation is based on the OpenAI Multi-Agent Particle Environment. This repository keeps a custom MPE-style parallel world API rather than importing a stock OpenAI MPE scenario, because the paper requires heterogeneous fixed-wing/multi-rotor dynamics and actions that stock MPE scenarios do not provide. The original MPE design itself expects scenario-specific world/reset/reward/observation functions, so a custom scenario/world implementation is consistent with how MPE is extended, but it is not byte-for-byte the authors' unpublished environment code.

## 6. What can and cannot be claimed

After the paper-fidelity rewrite, it is reasonable to say:

- the **published scenario structure** is reproduced;
- the **published Table-I environment constraints** are used;
- the **published A2A equations/topology** are represented;
- the **published action and reward definitions** are represented;
- the two requested small swarm sizes are represented.

It is **not** reasonable to claim exact numerical reproduction of the authors' simulator or reported curves until the unpublished constants and original source code become available.
