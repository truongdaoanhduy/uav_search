# U6/U9 Scenario Audit Fixes — Implementation Plan

**Goal:** Correct scenario/environment semantics without changing MASAC, MADDPG, or MATD3.

**Approved network decision:** Keep one UavNetSim episode/clock/RNG across the mission, but make each 1 s transmission action a closed service window. A packet chunk is admitted only when it can reach terminal ACK/drop inside the current slot; no untracked packet work may survive into the next action.

## Task 1: Lock the diagnosed failures with behavior tests

**Files**
- Create: `tests/test_scenario_audit_regressions.py`
- Modify only where an obsolete assertion encodes the diagnosed bug:
  - `tests/test_network_backends.py`
  - `tests/test_u6_scenario_semantics.py`

Add tests proving:

1. Initial target confirmation uses cumulative Bayesian posterior plus persistent direct fine evidence; it does not require both facts to occur in the same simulator step.
2. Report regeneration after TTL expiry still requires a fresh fine positive.
3. Idle UAV geometry cannot interfere with a lone analytical-backend sender, while active concurrent senders can.
4. A large good-link UavNetSim request commits meaningful in-order application progress in one slot.
5. Calling an idle next slot emits no late ACK/energy from the previous action, and captured event history stays bounded to the current slot.
6. GCS progress shaping is proportional to newly committed report bytes; peer forwarding never creates positive hop reward.
7. A nominally unsafe action corrected by the shield receives a negative safety component, and intervention count represents actually corrected UAVs rather than projection-loop iterations.

Run each new test against the current implementation and observe the intended failure before production changes.

## Task 2: Restore cumulative sensing semantics

**Files**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `tests/test_u6_3d_sensing.py`
- Modify: `tests/test_u6_scenario_semantics.py`

For first confirmation, use persistent `fine_positive_cells_by_agent` and `fine_target_evidence_by_agent` together with the accumulated posterior. Remove the transient fine-cell state added by the current dirty diff. Preserve the existing fresh `last_sensor_positive` requirement only for creating a new report generation after expiry.

## Task 3: Make network service match action timing

**Files**
- Modify: `src/uav_search/envs/network_backends.py`
- Modify: `tests/test_network_backends.py`

For UavNetSim:

- schedule at most one outstanding data chunk per intent;
- wait for native ACK or terminal ARQ drop before admitting its next chunk;
- stop that intent after a terminal chunk failure so successful application bytes are a contiguous prefix;
- use a conservative terminal-service guard near the slot boundary;
- assert/guarantee that no admitted packet remains pending at return;
- retain the same episode, SimPy clock, per-node MAC RNG, channel and node state;
- rotate captured events per macro-step so history cannot grow quadratically.

For the analytical ablation:

- calculate interference from other senders present in the same `intents` call, not every geometrically present UAV;
- keep `link_snapshot` as an explicitly conservative potential-connectivity estimate.

## Task 4: Align reward and safety signals with mission outcomes

**Files**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `configs/scenarios/u6.yaml`
- Modify: `configs/scenarios/u9.yaml`

- Keep the literature-backed peer-attempt cost and zero positive peer-hop shaping already present in the dirty worktree.
- Replace full `+comm_reward_max` per successful GCS fragment with
  `comm_reward_max * committed_report_bytes / report_bytes`.
- Record each UAV's final shield displacement from its nominal candidate position.
- Define intervention count as the number of UAVs with nonzero shield correction.
- In peer scenarios, add a normalized correction penalty
  `-safety_reward_coeff * correction_m / safety_distance_m`.
- Retain the existing hard-distance penalty as a fail-safe.

## Task 5: Update scenario documentation

**Files**
- Modify: `docs/U6_PROVENANCE.md`
- Modify: `docs/PAPER_FIDELITY.md`
- Modify: `docs/superpowers/specs/2026-09-09-u6-power-dtn-persistent-network.md`
- Modify: `src/uav_search/envs/paper_env.py` docstring

Document cumulative confirmation, post-expiry freshness, byte-proportional delivery shaping, correction-aware safety reward, active-intent analytical interference, and closed-slot packet admission. Clarify that the environment returns a custom multi-agent tuple and is not a PettingZoo `ParallelEnv` implementation.

## Task 6: Verify

Run:

- focused regression tests;
- full `pytest -q`;
- formatting/lint/type checks exposed by the repository;
- `git diff --check`;
- deterministic direct-to-GCS rollout confirming substantial progress and no event/process backlog;
- `git status --short` to report all pre-existing and new changes without committing.
