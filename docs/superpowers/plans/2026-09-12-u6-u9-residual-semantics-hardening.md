# U6/U9 Residual Scenario Semantics Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the seven approved U6/U9 environment-semantics defects without changing the six-dimensional action contract, observation sizes, or any MASAC/MADDPG/MATD3 implementation.

**Architecture:** Keep `PaperUAVEnv` as the application-state owner and UavNetSim as the network-outcome owner. Add pure helpers at narrow seams: an endpoint-safe belief codec in `sensing.py` and a shared episode-stop predicate in `runner/termination.py`. Integrate every other correction inside existing peer-scenario paths so legacy fixed-wing scenarios preserve their semantics.

**Tech Stack:** Python 3.10+, NumPy, Gymnasium/PettingZoo-style multi-agent dictionaries, pytest, Ruff-compatible formatting.

**Spec:** `docs/superpowers/specs/2026-09-12-u6-u9-residual-semantics-hardening.md`

## Global Constraints

- Do not modify anything under `src/uav_search/algorithms/`.
- Keep action shape six and `u6.obs_dim == 158`, `u9.obs_dim == 197`.
- Apply each production change only after its new regression test fails for the intended reason.
- Preserve terminated-only replay masks so time-limit truncation continues to bootstrap.
- Run `git diff --check` before every task commit; do not push until the full verification task passes.

---

## Task 1: Add the endpoint-safe one-byte log-odds belief codec ✅

**Files:**

- Modify: `tests/test_scenario_audit_regressions.py`
- Modify: `src/uav_search/envs/sensing.py`
- Modify: `src/uav_search/envs/paper_env.py`

- [ ] **Step 1: Write failing codec tests**

Add tests importing `belief_probability_floor`, `encode_belief_probabilities`, and
`decode_belief_probabilities`. Assert for `levels=256` that:

```python
encoded = encode_belief_probabilities(np.array([0.0, 0.5, 1.0]), levels=256)
decoded = decode_belief_probabilities(encoded, levels=256)
assert encoded.dtype == np.uint8
assert decoded[0] > 0.0
assert decoded[1] == 0.5
assert decoded[2] < 1.0
assert decoded[0] == pytest.approx(1.0 - decoded[2])
```

Also assert one byte per cell, monotonic round trips, invalid `levels < 3` rejection,
and rejection of the unused code for even level counts.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
pytest -q tests/test_scenario_audit_regressions.py -k belief_codec
```

Expected: import failure or missing-helper assertion, not an unrelated fixture error.

- [ ] **Step 3: Implement the pure codec**

In `sensing.py`, implement:

```python
def belief_probability_floor(levels: int) -> float: ...
def encode_belief_probabilities(probabilities: np.ndarray, levels: int) -> np.ndarray: ...
def decode_belief_probabilities(codes: np.ndarray, levels: int) -> np.ndarray: ...
```

Use `radius = (levels - 1) // 2`, `p_floor = 0.5 / (2 * radius + 1)`,
symmetric clipped log odds, nearest signed-code rounding, and a logistic decoder.
Require `3 <= levels <= 256`; reject decoded codes greater than `2 * radius`.

- [ ] **Step 4: Integrate the codec at the peer synchronization boundary**

Replace `rint(probability * quantizer)` and `code / quantizer` in
`PaperUAVEnv._peer_transmit` with the helpers. Keep the array `uint8` and keep
the existing sync payload byte count unchanged. Tighten configuration validation
to reject fewer than three levels.

- [ ] **Step 5: Run focused and adjacent tests and confirm GREEN**

Run:

```bash
pytest -q tests/test_scenario_audit_regressions.py -k "belief_codec or sync"
pytest -q tests/test_u6_scenario_semantics.py -k "belief or observation"
git diff --check
```

- [ ] **Step 6: Commit locally**

```bash
git add src/uav_search/envs/sensing.py src/uav_search/envs/paper_env.py tests/test_scenario_audit_regressions.py
git commit -m "fix: use endpoint-safe belief synchronization"
```

## Task 2: Bound concurrent peer fan-in and make dead-recipient attempts observable ✅

**Files:**

- Modify: `tests/test_scenario_audit_regressions.py`
- Modify: `tests/test_u6_scenario_semantics.py`
- Modify: `src/uav_search/envs/paper_env.py`

- [ ] **Step 1: Write failing fan-in and invalid-recipient tests**

Create deterministic tests with a stubbed full-ACK backend. For two senders targeting
one receiver with exactly 1,000 bytes free, assert the total report bytes placed in
network intents and accepted by the receiver never exceeds 1,000, while both sync
bundles may be present.

For a live sender selecting a depleted peer, assert:

```python
assert env.last_tx_active[sender]
assert not env.last_tx_success[sender]
assert env.last_selected_recipient[sender] == dead_peer_index
assert env.last_tx_attempt_report_bytes[sender] == queued_report_bytes
assert reward[sender] < gate_off_reward[sender]
```

and assert the invalid recipient produces no UavNetSim intent or RF-energy charge.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
pytest -q tests/test_scenario_audit_regressions.py -k "fan_in or reservation"
pytest -q tests/test_u6_scenario_semantics.py -k "depleted_recipient or dead_recipient"
```

- [ ] **Step 3: Add deterministic report-byte reservations**

In `_peer_transmit`, snapshot receiver queue bytes at slot start and maintain:

```python
reserved_report_bytes_by_receiver = np.zeros(self.num_uavs, dtype=np.int64)
room = max(
    0,
    buffer_bytes
    - queue_bytes_at_slot_start[receiver]
    - reserved_report_bytes_by_receiver[receiver],
)
report_budget = min(sender_report_bytes, room)
reserved_report_bytes_by_receiver[receiver] += report_budget
```

Reserve only report bytes and iterate in existing sender-index order. Do not reclaim
unused reservation after packet loss within that closed slot.

- [ ] **Step 4: Record attempts before validating a peer recipient**

For every active gate-on sender, decode and store recipient, power, distance, and
available report bytes before checking peer liveness. If the peer is depleted, skip
intent construction but leave attempt telemetry populated so the existing attempt
cost and failed-report penalty apply. Gate-off and inactive senders remain no-ops.

- [ ] **Step 5: Run focused and network regression tests**

```bash
pytest -q tests/test_scenario_audit_regressions.py -k "fan_in or reservation or uavnetsim"
pytest -q tests/test_u6_scenario_semantics.py -k "recipient or transmit or buffer"
git diff --check
```

- [ ] **Step 6: Commit locally**

```bash
git add src/uav_search/envs/paper_env.py tests/test_scenario_audit_regressions.py tests/test_u6_scenario_semantics.py
git commit -m "fix: enforce peer admission semantics"
```

## Task 3: Penalize false confirmation once and expose world-constraint correction ✅

**Files:**

- Modify: `tests/test_u6_scenario_semantics.py`
- Modify: `src/uav_search/envs/paper_env.py`

- [ ] **Step 1: Write failing false-confirmation tests**

Build one confirmed empty cell with fresh fine evidence. Assert its first confirmation
adds exactly
`-search_reward_coeff * sensing_target_reward_weight` to the shared task event,
marks it false-confirmed, consumes the detector's positive fine evidence, and resets
that detector's belief to `belief_probability_floor(levels)`. Assert the next step
does not repeat the event without new evidence.

- [ ] **Step 2: Write failing navigation-correction tests**

Create boundary and obstacle cases where attempted motion is rejected or clipped to
the same realized pose as idle. Assert nonzero
`last_world_constraint_correction_m_by_agent`, a lower reward than idle, and
current-step sum/max values in `info`. Assert pairwise shield correction remains
separate.

- [ ] **Step 3: Run the new tests and confirm RED**

```bash
pytest -q tests/test_u6_scenario_semantics.py -k "false_confirmation or world_constraint"
```

- [ ] **Step 4: Implement the symmetric false-confirmation event**

In `_confirm_peer_targets`, on a newly false-confirmed cell:

- add the symmetric negative task term once;
- leave the static cell marked verified empty;
- set the detector-local belief to the codec floor;
- clear detector-local fresh fine-positive evidence;
- retain communication-gated dissemination to other agents.

Do not create a report or a new reward coefficient.

- [ ] **Step 5: Track and penalize boundary/obstacle correction**

Capture nominal positions immediately after dynamics integration. After boundary
clipping and obstacle rollback, but before pairwise shielding, calculate per-agent
Euclidean correction. Reset it each step, subtract
`safety_reward_coeff * correction / safety_distance_m` in `_safety_reward`,
and expose sum/max in `_info` alongside existing pair-shield metrics.

- [ ] **Step 6: Run focused safety/sensing regressions**

```bash
pytest -q tests/test_u6_scenario_semantics.py -k "false_confirmation or world_constraint or safety or obstacle or boundary"
git diff --check
```

- [ ] **Step 7: Commit locally**

```bash
git add src/uav_search/envs/paper_env.py tests/test_u6_scenario_semantics.py
git commit -m "fix: align false alarms and navigation feedback"
```

## Task 4: Terminate depleted agents individually without suppressing truncation bootstrap

**Files:**

- Create: `src/uav_search/runner/termination.py`
- Create: `tests/test_runner_termination.py`
- Modify: `tests/test_u6_scenario_semantics.py`
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `src/uav_search/runner/train.py`
- Modify: `src/uav_search/runner/network_calibration.py`

- [ ] **Step 1: Write failing environment and runner tests**

Assert that an agent depleted before reward calculation receives `0.0`, is
individually terminated, and is not truncated. At a horizon step with one already
dead agent, assert live agents are truncated and the episode-stop predicate returns
true for the mixed dictionaries.

Test the pure helper:

```python
assert episode_finished(
    {"uav_0": True, "uav_1": False},
    {"uav_0": False, "uav_1": True},
)
```

Also assert `_replay_terminal_mask` remains false for truncation-only transitions.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
pytest -q tests/test_runner_termination.py tests/test_u6_scenario_semantics.py -k "depleted or termination or truncated or bootstrap"
```

- [ ] **Step 3: Add and consume the shared stop predicate**

Create:

```python
def episode_finished(
    terminated: Mapping[str, bool],
    truncated: Mapping[str, bool],
) -> bool:
    agents = terminated.keys() | truncated.keys()
    return bool(agents) and all(
        bool(terminated.get(agent, False) or truncated.get(agent, False))
        for agent in agents
    )
```

Import it in training and network-calibration loops and replace separate
`all(terminated)` / `all(truncated)` stop expressions. Do not change
`_replay_terminal_mask`.

- [ ] **Step 4: Implement per-agent reward and done dictionaries**

In peer mode, zero the final reward of agents inactive at reward time. Set
`terminated[a] = global_terminal or not uav_active[a]`; set
`truncated[a] = horizon_reached and not terminated[a]`. Preserve global success
and all-depleted termination for every agent. Preserve legacy scenario behavior.

- [ ] **Step 5: Run runner and algorithm smoke tests**

```bash
pytest -q tests/test_runner_termination.py tests/test_u6_scenario_semantics.py -k "depleted or termination or truncated or bootstrap"
pytest -q tests -k "masac or maddpg or matd3 or runner"
git diff --check
```

- [ ] **Step 6: Commit locally**

```bash
git add src/uav_search/runner/termination.py src/uav_search/runner/train.py src/uav_search/runner/network_calibration.py src/uav_search/envs/paper_env.py tests/test_runner_termination.py tests/test_u6_scenario_semantics.py
git commit -m "fix: handle mixed multi-agent episode endings"
```

## Task 5: Surface previous peer recipient without changing observation shape

**Files:**

- Modify: `tests/test_u6_scenario_semantics.py`
- Modify: `src/uav_search/envs/paper_env.py`

- [ ] **Step 1: Write failing observation-contract tests**

For peer mode, set two distinct `last_selected_recipient` values and assert the
ninth own-state scalar differs and equals the center of the corresponding existing
recipient action bin. Assert `last_tx_active` still gates validity.

Also assert:

```python
assert u6_observation.shape == (158,)
assert u9_observation.shape == (197,)
```

and a legacy fixed-wing scenario still returns normalized heading in that slot.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
pytest -q tests/test_u6_scenario_semantics.py -k "previous_recipient or observation_shape or legacy_heading"
```

- [ ] **Step 3: Reuse the recipient decoder's bin geometry**

Add or reuse one inverse helper that maps a recipient index to the center of the same
action interval consumed by `_decode_peer_recipient`. Populate own-state index 8
with this value only in peer mode; keep legacy normalized heading unchanged. Do not
change any observation-space dimension.

- [ ] **Step 4: Run observation/config regressions**

```bash
pytest -q tests/test_u6_scenario_semantics.py -k "recipient or observation"
pytest -q tests/test_config.py tests/test_paper_env.py
git diff --check
```

- [ ] **Step 5: Commit locally**

```bash
git add src/uav_search/envs/paper_env.py tests/test_u6_scenario_semantics.py
git commit -m "fix: expose previous peer recipient feedback"
```

## Task 6: Update research documentation and verify the complete repository

**Files:**

- Modify: `docs/U6_PROVENANCE.md`
- Modify: `docs/PAPER_FIDELITY.md`
- Modify: `docs/U6_U9_SCENARIO_AUDIT_2026-09-12.md`

- [ ] **Step 1: Document the corrected semantics and provenance**

Record the log-odds codec, deterministic conservative fan-in, invalid-recipient
feedback, symmetric false-alarm consequence, separate world-constraint correction,
per-agent depletion semantics, runner stopping predicate, and observation-slot
reinterpretation. Note that checkpoints remain shape-compatible but scientific
runs need retraining.

Add primary-paper links for the external sources consulted:

- Khan, Yanmaz, and Rinner, *Information Merging in Multi-UAV Cooperative Search*
  (ICRA 2014).
- Xiong et al., *Parametrized Deep Q-Networks Learning* (arXiv:1810.06394).
- Sun et al., *Multi-Agent Reinforcement Learning Based on Hybrid Action
  Representation for UAV Swarms' Integrated Communication and Control*
  (IEEE LWC 2026, DOI 10.1109/LWC.2026.3663841).

Explain that P-DQN and MAHyAR motivate a future explicit hybrid-action actor but
are non-goals in this patch.

- [ ] **Step 2: Run targeted tests**

```bash
pytest -q tests/test_scenario_audit_regressions.py tests/test_u6_scenario_semantics.py tests/test_runner_termination.py
```

Expected: all pass.

- [ ] **Step 3: Run algorithm smoke tests**

```bash
pytest -q tests -k "masac or maddpg or matd3"
```

Expected: all selected tests pass; no algorithm file changed.

- [ ] **Step 4: Run full verification from a clean command invocation**

```bash
python -m compileall -q src tests
pytest -q
git diff --check
git status --short
git diff --name-only HEAD~6..HEAD -- src/uav_search/algorithms
```

Expected: compilation succeeds; full suite passes; whitespace check is empty;
only intended files remain; algorithm diff is empty.

- [ ] **Step 5: Commit documentation locally**

```bash
git add docs/U6_PROVENANCE.md docs/PAPER_FIDELITY.md docs/U6_U9_SCENARIO_AUDIT_2026-09-12.md
git commit -m "docs: record hardened u6 u9 semantics"
```

- [ ] **Step 6: Inspect the final branch and synchronize once**

```bash
git status
git log --oneline --decorate -10
git push origin feature/homo-u6-paper-sim
git status
```

Expected: push is fast-forward; local branch and
`origin/feature/homo-u6-paper-sim` point to the same final commit and the
working tree is clean.
