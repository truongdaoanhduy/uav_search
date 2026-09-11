# U6/U9 Residual Scenario Semantics Hardening

**Date:** 2026-09-12  
**Status:** Approved by the user on 2026-09-12.
**Scope:** Homogeneous research scenarios `u6` and `u9` only.

## Goal

Remove the remaining belief-sync, finite-buffer, invalid-action, false-confirmation,
navigation-feedback, agent-depletion, and previous-recipient ambiguities found by the
independent scenario audit, without changing MASAC, MADDPG, MATD3, or the six-dimensional
action contract.

## Fixed Invariants

- The action remains
  `[horizontal_thrust, horizontal_acceleration_azimuth, vertical_acceleration,
  tx_gate, tx_power, recipient]` in `Box[-1, 1]^6`.
- `u6.obs_dim == 158` and `u9.obs_dim == 197` remain unchanged.
- Legacy `f*_m*` paper-reproduction scenarios retain their current behavior.
- A peer transmission still carries a fixed one-byte-per-cell belief payload inside the
  existing 4 KiB synchronization bundle.
- UavNetSim remains authoritative for PHY/MAC/ACK/ARQ outcomes; `PaperUAVEnv` remains
  authoritative for application buffer admission and report ownership.
- Static victims, one application hop per macro-step, report size, buffer size, TTL,
  horizon, and current reward coefficients remain unchanged.
- No file under `src/uav_search/algorithms/` is modified.

## Root Causes and Accepted Semantics

### 1. Belief synchronization creates artificial certainty

The current linear codec rounds `p * 255` and decodes with `/ 255`. It therefore maps
high but finite beliefs to exactly `1.0`, maps low beliefs to exactly `0.0`, and maps the
neutral prior `0.5` to approximately `0.50196`. Minimum-entropy fusion then prefers these
artificially certain values. Repeated cooperative fusion can make a false alarm much
harder to reverse than the underlying Bayesian evidence warrants.

The replacement codec uses a symmetric quantized log-odds representation. For configured
level count `L`, define `radius = floor((L - 1) / 2)`,
`effective_codes = 2 * radius + 1`, and
`p_floor = 0.5 / effective_codes`. Require `L >= 3`.

1. Clip probabilities to `[p_floor, 1 - p_floor]` only for the logarithm.
2. Convert probability to log odds and clip it to
   `[-log((1-p_floor)/p_floor), +log((1-p_floor)/p_floor)]`.
3. Scale and round log odds to a signed integer in `[-radius, +radius]`; signed zero
   represents `p = 0.5` exactly.
4. Store `signed + radius` in one `uint8`; code `2*radius+1` is unused when `L` is
   even.
5. Reject an out-of-range received code and otherwise decode through the logistic
   function, producing only values strictly inside `(0, 1)`.

This preserves the payload size, preserves the neutral prior exactly, avoids absorbing
endpoint beliefs, and matches the logarithmic Bayesian-map formulation used in
cooperative-search literature.

### 2. Concurrent fan-in overcommits receiver capacity

All senders currently compute room from the same pre-slot `queue_bytes[receiver]`.
Multiple senders can therefore ask the network to deliver the same remaining capacity.
The application later accepts the first outcome and silently rejects otherwise ACKed
report bytes from later outcomes.

During intent construction, maintain `reserved_report_bytes_by_receiver`. A peer report
budget is limited by:

```text
buffer_bytes - queue_bytes_at_slot_start - bytes_already_reserved_this_slot
```

Reserve only report bytes, not the fixed sync bundle. Reservation is conservative: room
unused because of link loss is not reassigned during the same closed slot. Sender-index
ordering remains deterministic and is documented as the admission tie-breaker.

### 3. A transmission to a depleted recipient is a free no-op

For an active sender whose gate is on, decode and record recipient, power, distance, and
attempt state before recipient validation. If the selected peer is depleted:

- create no network intent and consume no artificial RF energy;
- keep `last_tx_active = True`, `last_tx_success = False`, and the selected recipient;
- record the report bytes the sender attempted to route, if any;
- apply the existing peer attempt cost, plus the existing failed-report penalty when
  report bytes were available.

Inactive senders remain ignored.

### 4. False target confirmation has no consequence

The approved behavior is an explicit symmetric mission penalty, not a dynamic false-report
subsystem. Each newly false-confirmed cell contributes

```text
-search_reward_coeff * sensing_target_reward_weight
```

to the same shared task term in which a true confirmation contributes the positive value.
No new reward coefficient is introduced. A false-confirmed static cell remains recorded as
verified empty so the same event cannot be farmed every step. The detector's belief for the
cell is set to the representable low-confidence floor and its fine-positive evidence for
that cell is consumed. Other agents learn the corrected belief only through the existing
communication-gated belief sync.

### 5. Boundary/obstacle rejection is reward-equivalent to idling

Capture the nominal pose immediately after motion integration. After world-boundary clipping
and obstacle rollback—but before pairwise shielding—record each UAV's
`last_world_constraint_correction_m_by_agent`. The safety reward subtracts that distance
normalized by the existing safety distance and scaled by the existing
`safety_reward_coeff`. Pairwise shield correction remains a separate term, so diagnostics
distinguish world constraints from inter-UAV safety projection. Expose current-step sum/max
correction metrics while retaining the existing episode-level unique hit sets; do not add a
reward hyperparameter.

### 6. Depleted UAVs continue receiving and bootstrapping team reward

For peer scenarios:

- a UAV inactive at reward time receives exactly zero reward;
- `terminated[a] = global_terminal OR NOT uav_active[a]`;
- `truncated[a] = horizon_reached AND NOT terminated[a]`;
- mission success and all-depleted conditions still set `global_terminal` and terminate
  every agent;
- horizon truncates only still-active agents and does not suppress bootstrap;
- a shared runner helper stops collection when
  `all(terminated[a] OR truncated[a] for a in agents)`, covering mixed dead/horizon states;
- network-calibration and evaluation loops use the same combined stopping rule.

The fixed-size joint replay layout is retained. No actor/critic loss or replay-buffer schema
is changed.

### 7. Previous TX feedback omits recipient identity; heading slot is constant

In peer mode, repurpose the ninth own-state scalar, which is currently a constant zero
legacy heading slot, as the previous selected-recipient identity. Encode the decoded
recipient at the center of its existing action bin. `last_tx_active` remains the validity
flag, so the recipient value is ignored when no attempt occurred. Legacy scenarios continue
to expose normalized fixed-wing heading in the same slot.

This fixes feedback attribution and removes the redundant constant without changing
observation dimensions or network shapes. Old checkpoints remain shape-compatible, though
their learned interpretation of this formerly zero feature is not semantically equivalent;
new scientific runs must retrain from scratch.

## Files and Responsibilities

- `src/uav_search/envs/sensing.py`: endpoint-safe one-byte log-odds belief codec.
- `src/uav_search/envs/paper_env.py`: codec integration, buffer reservation, invalid-recipient
  feedback, false-confirmation consequence, world-constraint correction, depleted-agent
  reward/termination, and previous-recipient feature.
- `src/uav_search/runner/termination.py`: shared mixed terminated/truncated episode-stop
  predicate.
- `src/uav_search/runner/train.py`: consume the shared stopping predicate while preserving
  terminated-only replay masks.
- `src/uav_search/runner/network_calibration.py`: consume the same stopping predicate.
- `tests/test_u6_scenario_semantics.py`: environment-level regression behavior.
- `tests/test_scenario_audit_regressions.py`: codec and finite-buffer regressions.
- `tests/test_runner_termination.py`: mixed termination/truncation behavior.
- `docs/U6_PROVENANCE.md`, `docs/PAPER_FIDELITY.md`, and the dated audit report: updated
  semantics and research provenance.

## Error Handling and Compatibility

- Belief codec validates that the configured level count yields at least one negative,
  neutral, and positive signed code and rejects incompatible shapes/dtypes through normal
  NumPy conversion/shape checks at the caller boundary.
- Buffer reservations never make room negative and never modify queue state until ACKed
  application bytes settle.
- Invalid peer identifiers outside the action decoder's candidate set remain impossible;
  only a valid but depleted peer follows the failed-attempt path.
- Legacy scenario reward and termination behavior is protected by regression tests.

## Test Strategy

Every production change follows a separate red-green cycle. Tests must demonstrate:

1. `0`, `0.5`, and `1` round-trip to interior-low, exact-neutral, and interior-high values;
   decoded beliefs are symmetric and one byte per cell.
2. Two full-ACK senders targeting a receiver with 1,000 free bytes attempt and commit at
   most 1,000 report bytes in total while both sync bundles may still be delivered.
3. Selecting a depleted peer records the recipient and returns attempt/failure penalty;
   gate-off remains a no-op.
4. One false confirmation produces one symmetric negative team event, consumes local fine
   evidence, and cannot repeat without new evidence.
5. An obstacle-rejected or boundary-clipped movement receives a lower reward than an idle
   action despite an otherwise identical realized state.
6. A depleted UAV receives zero reward and is terminated individually; a mixed
   terminated/truncated dictionary ends rollout collection while truncation alone still
   bootstraps.
7. Distinct previous recipients produce distinct ninth own-state features while `u6/u9`
   observation shapes and legacy heading behavior remain unchanged.
8. Targeted tests, algorithm smoke tests, the complete `pytest -q` suite, compilation, and
   `git diff --check` pass before commit or push.

## Non-Goals

- Converting MASAC/MADDPG/MATD3 to a categorical/continuous hybrid actor.
- Adding recurrent policies, privileged critic state, a global belief observation, or a
  new false-report message subsystem.
- Changing target motion, RF calibration, reward scales, TTL, report/buffer sizes, obstacle
  density, or mission horizon.
- Reinterpreting max-power channel-sounding features or historical experiment results.
