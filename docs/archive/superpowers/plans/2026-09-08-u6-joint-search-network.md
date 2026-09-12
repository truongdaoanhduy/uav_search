# U6 Joint Search + Networking Scenario Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `u6` into a scientifically meaningful six-peer-UAV post-disaster target-search scenario in which search data must be delivered to a GCS through direct, multi-hop, or store-carry-forward communication, without decentralized actor global-information leakage.

**Architecture:** Keep the root-paper 5 km × 5 km search world and six homogeneous multirotors, but make `u6` a research adaptation with an edge GCS, launch-zone initialization, a longer mission horizon, explicit contact range, local/stale neighbor observations, and per-target report state. The MARL policy remains authoritative for movement, transmit gate, amount, and immediate next-hop; UavNetSim is authoritative for contact feasibility and MAC/PHY delivery outcomes. Root-paper scenarios remain unchanged.

**Tech Stack:** Python 3, NumPy, Gymnasium-style environment, PyTorch learners already in repo, UavNetSim 2.0.0 pinned at `04daafb815eb377409b40b285574eeb62b9a8d58`, SimPy, pytest.

**Spec:** User project context in `Pasted text(20260908-100705).txt`; root paper `Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network`; supporting CCLR, JTFR, JUROR and limited-communication search papers.

## Global Constraints

- Only `u6` is redesigned; root paper scenarios `f1_m5`, `f1_m9`, `f2_m10`, `f2_m18`, `f4_m12`, `f8_m24` must preserve existing behavior.
- Six peer UAVs, no fixed leader/follower roles.
- Preserve the fixed-dimensional 5-element action `[move_force, move_direction, tx_gate, tx_amount, recipient]` for MASAC, MATD3 and MADDPG compatibility.
- No packet may traverse more than one application-level hop per RL macro-step.
- Search success (detected) and mission delivery success (report reaches GCS) remain separate metrics.
- Decentralized actor observations may use own state, local radio/contact state, and stale cached neighbor information; live remote global state is forbidden.
- Training-only centralized critics may continue receiving concatenated actor observations; no new global actor features.
- UavNetSim RF transmit power is independent of the root paper's `P_com=5 W` communication-energy term.
- TDD: every behavior change gets a failing test before production changes.

---

### Task 1: U6 deployment and mission horizon

**Files:**
- Modify: `configs/scenarios/u6.yaml`
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Consumes existing `scenario` config.
- Produces deterministic launch-zone initialization via `scenario.initialization`, `launch_center_m`, `launch_radius_m`, `initial_min_separation_m`, and a `scenario.episode_steps` override.

- [ ] Write failing tests asserting `u6` uses 600 steps, edge GCS, and deterministic launch-zone initialization while root-paper scenarios remain unchanged.
- [ ] Run focused tests and verify failure on current 50-step/center-GCS/random-all-map behavior.
- [ ] Add the research config and minimal initialization logic.
- [ ] Re-run focused tests and full homogeneous-env tests.

### Task 2: Contact model and network authority

**Files:**
- Modify: `configs/scenarios/u6.yaml`
- Modify: `src/uav_search/envs/network_backends.py`
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_network_backends.py`
- Test: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Produces `scenario.max_comm_range_m` and backend-side contact rejection.
- `PaperUAVEnv` emits an offered byte budget independent of current link rate; backend reports success/failure.

- [ ] Write failing tests showing an out-of-contact UavNetSim transmission is counted as attempted but delivered as zero and that a close contact succeeds.
- [ ] Write a failing test showing `PaperUAVEnv` forwards a bad-link transmission intent to the backend instead of silently pre-filtering it.
- [ ] Implement range gating inside both networking backends, not in the policy/environment decision layer.
- [ ] Separate `uavnetsim_tx_power_w` from root-paper `P_com` and use a radio-specific value.
- [ ] Remove env-side `rate <= R_min` pre-filter and compute offered bytes from configured link bit-rate budget.
- [ ] Re-run backend tests including pinned UavNetSim tests.

### Task 3: Dec-POMDP local and stale observations

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Produces fixed-width per-peer slots containing live one-hop or stale cached state plus freshness.
- Produces `target_known_by_agent[n_agents, n_targets]` distinct from environment truth `target_found[n_targets]`.

- [ ] Write failing test: moving a disconnected remote UAV must not instantly change another actor observation.
- [ ] Write failing test: moving a connected one-hop UAV must update the observer's peer slot.
- [ ] Write failing test: a target found by UAV A must not become globally known to disconnected UAV B.
- [ ] Add peer cache (`position`, `battery`, `age`, `valid`) and fixed-width local/stale encoding.
- [ ] Add per-agent target knowledge and use it when masking target observations.
- [ ] Propagate target knowledge only on self-detection or successful report bytes received from another UAV.
- [ ] Re-run tests.

### Task 4: Report lifecycle, finite buffer and causal step semantics

**Files:**
- Modify: `configs/scenarios/u6.yaml`
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Produces report creation step, TTL, dropped/expired byte counters and per-target loss state.
- Preserves one-hop-per-step forwarding with a frozen pre-network snapshot.

- [ ] Write failing tests for report TTL expiry, full-buffer drop accounting, and no same-step two-hop forwarding.
- [ ] Move peer sensing/report creation before communication in the research scenario, as specified by the agreed macro-step semantics.
- [ ] Add report TTL aging and explicit drop/expiration metrics.
- [ ] Keep frozen network input so newly received bytes cannot be forwarded a second hop in the same step.
- [ ] Re-run tests.

### Task 5: Topology and mission diagnostics

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Produces info metrics `network_direct_uavs`, `network_multihop_uavs`, `network_disconnected_uavs`, `network_mean_hops_to_gcs`, report drop/expiry counters.

- [ ] Write failing scripted-topology tests containing one direct UAV, one multi-hop UAV, and one disconnected UAV.
- [ ] Implement shortest-hop-to-GCS diagnostics from the current directed adjacency snapshot.
- [ ] Add report lifecycle diagnostics to `_info`.
- [ ] Re-run full test suite.

### Task 6: Verification and research handoff

**Files:**
- Modify if necessary: `README.md`
- Create: `docs/U6_SCENARIO_DESIGN.md`

- [ ] Run `python -m pytest -q` and record pass/skip counts.
- [ ] Run a deterministic 10-episode smoke test for MASAC, MADDPG and MATD3 on `u6` without large training.
- [ ] Run topology calibration scripts with UavNetSim and report direct/multi-hop/disconnected proportions.
- [ ] Run `git diff`, `git status`, and remote-divergence checks.
- [ ] Do not claim sync/push unless remote commit is explicitly verified.
