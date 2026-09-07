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
| Communication-energy power `P_com` | 5 W |
| Minimum communication rate | 1 Mbps |
| Fixed-wing maximum speed | 40 m/s |
| Fixed-wing minimum speed | 10 m/s |
| Maximum acceleration | 8 m/s^2 |
| Multi-rotor maximum speed | 10 m/s |
| Safe battery level | 10% |
| Energy reward scaling factor | 0.2 |
| Training rounds reference | 50,000 |
| Evaluation random test cases | 5,000 |
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

The original paper gives the equations but does not publish every numeric channel constant. In particular, Table I gives `P_com = 5 W` for communication-energy consumption, while Eq. (6) uses the distinct A2A transmit-power symbol `P_tx,u` without a numerical value. For those missing numbers, this implementation uses papers that the original paper itself cites.

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
| A2A transmit power `P_tx,u` in Eq. (6) | 10 W from root-paper Ref. [39] Table III (`REFERENCE_BACKED`, not root-paper explicit) |
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

The paper writes 3D state/dynamics equations, but the software-simulation figures and trajectory plots are planar, the physical validation first sends UAVs to designated cruising altitudes before trajectory optimization, and the original OpenAI MPE world is fundamentally a 2-D particle world. The baseline therefore keeps type-specific altitude fixed and optimizes planar trajectories. This is a **plausible implementation inference**, not a paper-explicit numerical choice: the fixed/multi-rotor altitude values themselves remain unpublished and assumed.

### Fixed-wing thrust dynamics

The paper provides simplified Newton-Euler equations but does not publish the aerodynamic drag coefficient, yaw moment of inertia, or rudder torque mapping needed for an exact numerical integrator. The normalized second fixed-wing action is therefore mapped to a bounded forward speed/thrust proxy while enforcing the paper's 10-40 m/s limits.

### Multi-rotor force dynamics

The multi-rotor normalized force action is converted to an acceleration bounded by the paper's 8 m/s^2 maximum-acceleration parameter. Exact force-to-acceleration dynamics beyond the published mass/constraint cannot be reconstructed because the needed aerodynamic constants are not reported.

### Observation vector

The baseline now follows the contents of Eqs. (17)-(20) using a common padded vector width required by MASAC/MATD3/MADDPG: multi-rotor and fixed-wing fields that do not belong to that UAV type are zero-masked, target input is distance-only, targets outside sensing range are hidden, and found targets are masked. Direct obstacle geometry and extra type flags are no longer injected into Actor observations. It still does **not** reproduce GATAC's learned importance ranking/GAT aggregation, which is outside the baseline-scenario task.

### OpenAI MPE

The paper states that simulation is based on the OpenAI Multi-Agent Particle Environment. The original MPE provides a generic particle-world/scenario framework rather than this paper's heterogeneous UAV scenario, and its maintained successor MPE2 does not contain Ao et al.'s unpublished custom environment. This repository therefore keeps a custom MPE-style parallel world API. That choice is structurally consistent with scenario-specific MPE extension, but it is not byte-for-byte the authors' unpublished MPE implementation.

## 6. What can and cannot be claimed

After the paper-fidelity rewrite, it is reasonable to say:

- the **published scenario structure** is reproduced;
- the **published Table-I environment constraints** are used;
- the **published A2A equations/topology** are represented;
- the **published action and reward definitions** are represented;
- the two requested small swarm sizes are represented.

The single-run configuration (`paper.yaml` / `train.py` without `--episodes`) follows the paper's 50,000 training rounds. `run_all.py` intentionally retains a 10-episode development default, so a paper-scale comparison must explicitly pass `--episodes 50000`; it then defaults to 5,000 evaluation cases. This CLI development default is an implementation convenience, not a paper parameter.

It is **not** reasonable to claim exact numerical reproduction of the authors' simulator or reported curves until the unpublished constants and original source code become available.

## 7. Original-paper ambiguities and explicit remaining structural gaps

The final audit against the uploaded IEEE paper identifies several points that must not be presented as exact paper values/behavior:

- **Small-scenario target count:** Fig. 7 explicitly gives only the UAV counts `(1,5)` and `(1,9)`. It does not print the number of targets for those two scenes. The current value `10` is retained as `PAPER_INFERRED`; it is not an explicit paper parameter.
- **Sensor description is internally inconsistent:** Section III states that multi-rotor UAVs carry RSSI target-signal detectors, while the simulation-design paragraph says fixed-wing UAVs carry signal-detection devices and multi-rotors carry cameras. The baseline therefore must not claim one sensor modality as uniquely specified by the paper.
- **Large-scale Fig. 9 text contains an internal inconsistency:** the figure caption/plots use `(4,12)` and `(8,24)`, while the prose says "four and six fixed-wing" alongside 12 and 24 multi-rotors. Those expansion scenarios are configured for a later phase, but they are not in the current `ACTIVE_SCENARIOS` or default experiment scope.
- **Observation container is padded, not byte-identical to the authors' implementation:** the contents now follow Eqs. (17)-(20), with type-inapplicable slots zero-masked so MASAC/MATD3/MADDPG can share one tensor width. GATAC importance ranking/aggregation is still outside scope.
- **Sensing values remain numerically assumed:** target visibility now obeys `D_detect` / `D_detect-f` and found-target masking, but those sensing radii are not numerically published by the original paper and therefore remain unchanged assumptions.
- **OpenAI MPE backend is not literal:** the paper explicitly says the simulation is based on OpenAI MPE; this repository is a custom Gymnasium/MPE-style environment rather than the authors' MPE source.
- **3D dynamics are approximated as 2.5D:** the paper defines 3D position/velocity dynamics, but does not publish the numerical constants/control details needed for exact integration. Fixed type-specific altitudes are retained as assumptions.

These points are intentionally left visible instead of being silently filled with guessed values. Parameters for which the paper gives only a symbol/formula but no numerical value are preserved for later source/reference completion.

## 8. Homogeneous `u6` research adaptation (not a paper reproduction)

The repository also contains `configs/scenarios/u6.yaml`. This scenario intentionally uses the **same lightweight root-paper simulation implementation** while changing the research question:

- six identical multi-rotor UAVs (`fixed_wing: 0`, `multirotor: 6`);
- no fixed-wing leader and no leader/follower formation;
- every UAV pair is a candidate A2A link and the existing paper Eq. (3)-(7) rate calculation determines usable connectivity;
- the first two action components remain the root-paper multi-rotor `{F, theta}` motion action;
- three added continuous components control `transmit/not`, transmission amount, and recipient UAV/GCS;
- confirmed targets generate finite-buffer mission reports and forwarding is limited to one hop per RL step;
- a ground command station, report size, buffer size, and report-delivery reward are **adapted assumptions**, because the root paper does not publish packet-buffer/GCS delivery semantics.

Therefore `u6` should be described as **root-paper-simulator-based / root-paper-inspired**, not as a reproduction of the paper's heterogeneous leader-follower scenario. The legacy `f1_m5` and `f1_m9` paths remain unchanged for reproduction comparisons.
