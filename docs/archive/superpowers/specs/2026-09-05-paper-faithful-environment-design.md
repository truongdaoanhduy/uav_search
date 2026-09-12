# Paper-Faithful Environment Redesign

## Goal

Align `/home/aduy/Documents/NCKH/uav_search` with the scenario and system model in **Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network**, while leaving MASAC/MATD3/MADDPG algorithm hyperparameters outside this change.

The implementation must distinguish values explicitly stated by the paper from values inferred from experiments and values the paper does not publish.

## Source of Truth

The original paper is authoritative for:

- heterogeneous leader-follower roles;
- 5 km x 5 km search domain;
- uniformly random target/building positions over the domain;
- circular building approximation;
- fixed-wing and multi-rotor dynamics equations;
- A2A probabilistic LoS/NLoS communication equations, SINR, and rate;
- star-like multi-rotor-to-fixed-wing formation links and fixed-wing mesh links;
- action definitions;
- reward equations (21)-(27);
- Table I parameters;
- experiment swarm sizes and episode length.

Reference papers may only provide fallback values for quantities not numerically disclosed by the original paper. Such values must never be marked `PAPER_EXPLICIT`.

## Scope

### In scope

1. Environment geometry and reset semantics.
2. Heterogeneous UAV movement semantics.
3. Communication/SINR/link topology.
4. Paper reward semantics.
5. Scenario configurations for `f1_m5` and `f1_m9`.
6. Provenance labels for parameters.
7. Tests that lock the above behavior.

### Out of scope

- Changing MASAC/MATD3/MADDPG learning rates, batch sizes, replay sizes, neural-network widths, entropy coefficients, etc.
- Implementing GATAC itself.
- Reproducing the physical motion-capture experiment.
- Claiming exact reproduction for numerical constants absent from the paper.

## Architecture

Keep the current project-facing parallel multi-agent interface so existing trainers continue to operate. The environment will remain a custom MPE-style world rather than importing a stock MPE scenario, because the paper's heterogeneous fixed-wing/multi-rotor action and dynamics semantics do not match any stock MPE environment.

The environment is split conceptually into:

1. `models.py`: stateless physical/channel calculations.
2. `paper_env.py`: world state, reset, actions, dynamics, topology, reward, observations, episode lifecycle.
3. scenario YAMLs: swarm sizes and scenario quantities.
4. `paper.yaml`: paper-explicit and fallback parameters, clearly labeled by provenance.
5. tests: paper-equation and scenario-behavior checks.

## Environment Reset and Geometry

- Domain: `[0, 5000] x [0, 5000]` m.
- Target centers: uniform random distribution over the full domain.
- Building centers: uniform random distribution over the full domain.
- Buildings: circles.
- Remove the current hard-coded 300 m margin for paper-described random objects.
- UAV initial XY positions remain a documented fallback because the paper does not numerically specify the initialization distribution/coordinates.
- Seeded resets must remain deterministic.

## Scenario Sizes

`f1_m5`:
- 1 fixed-wing UAV.
- 5 multi-rotor UAVs.

`f1_m9`:
- 1 fixed-wing UAV.
- 9 multi-rotor UAVs.

The target count will be set to 10 and labeled `PAPER_INFERRED`, not `PAPER_EXPLICIT`. The paper explicitly states 10 ground targets for the two-swarm experiments and presents the small/large experiments as scale comparisons, but does not separately state the target count in the Fig. 7 sentence.

Obstacle count/radius remain fallback values because the paper states multiple buildings and circular geometry but does not publish their count/radii.

## Actions and Dynamics

Every agent keeps a two-dimensional normalized action vector for trainer compatibility, but decoding becomes type-specific.

### Multi-rotor

Paper action: `{F_i(m), theta_i(m)}`.

- action[0] maps to non-negative thrust/force magnitude bounded by the paper's max acceleration constraint.
- action[1] maps to planar force direction in `[-pi, pi]`.
- velocity is integrated and clipped by the paper's 10 m/s maximum speed.

### Fixed-wing

Paper action: `{gamma_dot_i(m), F_i(m)}`.

- action[0] maps to signed heading/yaw change rate.
- action[1] maps to thrust/forward acceleration/speed command.
- speed remains in the paper's `[10, 40]` m/s range.
- planar yaw-only simplification follows the paper's simplified fixed-wing model.

Altitude remains fixed in the baseline because the original paper gives 3D state equations but does not publish simulation altitude values or an altitude action. These altitude values remain non-paper fallback parameters and must be reported as a reproduction gap.

## Communication Model

Replace the current single-link SNR approximation with paper-equation semantics:

1. Probabilistic LoS/NLoS average path loss.
2. Path gain derived from average loss.
3. Received signal power from the transmitting UAV.
4. Interference from other simultaneously transmitting UAVs sharing the modeled channel.
5. Gaussian noise power.
6. `SINR = signal / (interference + noise)`.
7. `R_A2A = B log2(1 + SINR)`.

Link validity uses `R_A2A > R_min` as in the paper's adjacency rule.

### Formation topology

- Multi-rotor UAVs connect to the fixed-wing leader assigned to their formation.
- Fixed-wing UAVs connect to fixed-wing UAVs in the leader mesh.
- For current one-leader scenarios, every rotor belongs to the sole leader's formation.
- The implementation must support multiple leaders later without changing the link model interface.

## Rewards

Implement the paper's equations without adding unrelated shaping into the paper reward.

### Multi-rotor total reward

`r_m = r_com + r_power + r_safe + r_task-m`

- `r_com`: Eq. (21), based on communication rate and distance.
- `r_power`: Eq. (22), remaining-energy term only above safe battery threshold.
- `r_safe`: Eq. (23), inverse-distance penalty inside `D_safe`.
- `r_task-m`: Eq. (24), detection/confirmation reward.

### Fixed-wing total reward

`r_f = r_safe + r_task-f`

as Eq. (27).

Boundary clipping and obstacle-hit diagnostics may remain in `info`, but penalties not specified in Eqs. (21)-(27) must not silently alter the paper reward. Physical collision/building constraints may prevent or flag invalid movement separately.

## Observation Semantics

The current observation implementation is retained structurally for baseline compatibility, but values must reflect the corrected world state and corrected communication links. A full reproduction of GAT-based observation pruning is outside scope because the current task is scenario/env fidelity, not GATAC.

## Parameter Provenance

Configuration labels must distinguish:

- `PAPER_EXPLICIT`: directly published in the original paper/Table I.
- `PAPER_INFERRED`: strong experiment-level inference, such as 10 targets for the small scenarios.
- `REFERENCE_BACKED`: numeric fallback obtained from a paper cited by the original paper.
- `ASSUMED`: implementation fallback with no original-paper numeric disclosure.

No fallback constant may be promoted to `PAPER_EXPLICIT` merely because it is common in UAV literature.

## Tests

Add/update tests for:

1. Table-I values remain exact.
2. `f1_m5` and `f1_m9` agent counts.
3. target count = 10 with inferred provenance.
4. targets/building centers can occupy the full `[0,5000]` domain and no 300 m margin is imposed.
5. seeded reset reproducibility.
6. SINR decreases with injected interference.
7. communication rate follows SINR and minimum-rate link threshold.
8. one-leader star links.
9. multi-leader fixed-wing mesh topology.
10. correct type-specific action decoding.
11. fixed-wing speed remains in `[10,40]` m/s.
12. rotor speed remains <= 10 m/s.
13. Eq. (21)-(27) reward composition.
14. target confirmation behavior.
15. 50-step episode truncation.

Run the complete existing test suite after the focused tests.

## Known Reproduction Gaps After This Change

Even after the redesign, exact reproduction cannot be claimed for values absent from the original paper, including at least:

- simulation altitudes;
- time-slot duration;
- target detection/confirmation distances;
- fixed-wing detection distance;
- safety distance;
- obstacle count and circle radii;
- initial UAV XY coordinates/distribution;
- battery capacity in joules;
- numerical `P0`, `mu`, and `d` propulsion coefficients;
- exact carrier frequency/bandwidth/noise if not supplied by the original paper;
- exact LoS/NLoS environment coefficients if not supplied by the original paper.

These gaps must remain visible in configuration and final reporting.

## Acceptance Criteria

The change is accepted when:

- paper-explicit scenario behavior is represented directly;
- current known mismatches (300 m random margin, SNR instead of SINR, missing fixed-wing mesh, reward extras) are corrected;
- existing trainers still receive the same high-level parallel env API and 2-D per-agent action shape;
- focused environment tests and the full suite pass;
- final report clearly separates `matches paper`, `paper-inferred/reference-backed`, and `still not exact` items.
