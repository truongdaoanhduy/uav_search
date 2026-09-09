# U6 Scenario Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the u6 scenario-modeling defects that can bias scientific results without redesigning the MARL algorithms or legacy paper scenarios.

**Architecture:** Keep `PaperUAVEnv` authoritative for mission state and movement and keep `UavNetSimBackend` authoritative for peer MAC/PHY. Harden reset sampling, make sensing fusion/confirmation cell-driven rather than ground-truth-driven, and dispatch ground-station radio links through an explicit Al-Hourani-style A2G gain while preserving native UavNetSim A2A behavior. Network snapshots will use the configured maximum controllable RF power so the actor is not told a link is impossible when its power action can in fact make it feasible.

**Tech Stack:** Python 3, NumPy, pytest, PyYAML, pinned UavNetSim/SimPy.

**Spec:** `configs/scenarios/u6.yaml`

## Global Constraints

- Do not modify MASAC/MAPPO/MATD3/MADDPG implementations or training hyperparameters.
- Do not change legacy `f*_m*` scenario behavior.
- Preserve deterministic reset behavior for equal seeds.
- Preserve the configured u6 RF action range 0.1--0.4 W and fixed 2 km candidate radius.
- Preserve literature-backed u6 sensing profiles (FOV 1/5/9, Pd 0.9/0.8/0.7, Pf 0.1/0.2/0.3, threshold 0.99).
- Preserve UavNetSim native A2A channel behavior for UAV-to-UAV links.
- Use an explicit urban A2G model for any link involving the synthetic GCS; defaults: a=9.61, b=0.16, eta_LoS=1 dB, eta_NLoS=20 dB.
- Keep the 100 MB buffer as a literature-backed nonbinding capacity baseline; do not invent a smaller capacity just to force congestion.

---

### Task 1: Valid reset geometry

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_u6_power_dtn_refactor.py`

**Interfaces:**
- Consumes: current launch pad positions, GCS position, target positions, obstacle radius bounds.
- Produces: `_sample_peer_obstacles() -> np.ndarray` returning in-map, non-overlapping circles that do not contain launch pads, GCS, or targets.

- [ ] **Step 1: Write failing geometry regression tests**
  - Sweep deterministic seeds and assert every circle lies inside the map.
  - Assert no obstacle overlaps another obstacle.
  - Assert no launch pad, GCS, or target lies inside any obstacle.
- [ ] **Step 2: Run only those tests and verify RED.**
- [ ] **Step 3: Implement rejection sampling with a bounded attempt count and clear failure error.**
- [ ] **Step 4: Run geometry tests and existing u6 reset/motion tests; verify GREEN.**

### Task 2: Cooperative belief fusion and cell-driven confirmation

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_u6_3d_sensing.py`

**Interfaces:**
- Consumes: per-agent Bayesian cell posteriors.
- Produces: `_fuse_peer_beliefs()` selecting the lowest-binary-entropy posterior per cell, `confirmed_cells`, `false_confirmed_cells`, and `belief_source_map`.

- [ ] **Step 1: Write failing tests**
  - Minimum-entropy posterior is shared across agents after fusion.
  - An empty cell above threshold is recorded as a false confirmation instead of being silently ignored.
  - A true target is marked found only because its cell is confirmed, not because `_confirm_peer_targets()` iterates the target list first.
- [ ] **Step 2: Run sensing tests and verify RED.**
- [ ] **Step 3: Implement minimum-uncertainty fusion and cell-first confirmation; retain target ground truth only for evaluation/mapping confirmed cells to true targets.**
- [ ] **Step 4: Add false-confirmation diagnostics to `info` and run sensing tests; verify GREEN.**

### Task 3: Ground-station A2G radio model

**Files:**
- Modify: `src/uav_search/envs/network_backends.py`
- Modify: `configs/scenarios/u6.yaml`
- Test: `tests/test_network_backends.py`

**Interfaces:**
- Produces: `_a2g_gain(start, end) -> float`, `_gain(..., gcs: bool = False) -> float`, and a UavNetSim channel estimate override for pairs involving the synthetic GCS.
- Preserves: native UavNetSim `A2AChannelModel` for UAV-to-UAV links.

- [ ] **Step 1: Write failing tests**
  - A2G gain changes with elevation angle at fixed 3D distance according to the configured urban model.
  - A2A gain remains delegated to UavNetSim `a2a.path_gain`.
  - UavNetSim snapshot uses A2G for GCS links and A2A for peers.
- [ ] **Step 2: Run network tests and verify RED.**
- [ ] **Step 3: Implement Al-Hourani-style average path loss and inject GCS-aware gain estimates into the persistent UavNetSim channel.**
- [ ] **Step 4: Run network tests including actual packet/ACK tests; verify GREEN.**

### Task 4: Power-aware topology observation

**Files:**
- Modify: `src/uav_search/envs/network_backends.py`
- Test: `tests/test_network_backends.py`
- Test: `tests/test_u6_power_dtn_refactor.py`

**Interfaces:**
- Changes: `link_snapshot(..., tx_power_w: float | None = None)` on both backends; default remains existing reference power for backward compatibility.
- `PaperUAVEnv._refresh_links()` requests a snapshot at `tx_power_max_w` in peer mode, representing controllable link feasibility rather than only minimum/reference-power connectivity.

- [ ] **Step 1: Write failing tests showing a link can be unavailable at 0.1 W but feasible at 0.4 W inside the candidate radius.**
- [ ] **Step 2: Run those tests and verify RED.**
- [ ] **Step 3: Add optional snapshot power and make u6 actor topology use max controllable power.**
- [ ] **Step 4: Run targeted networking/u6 tests and verify GREEN.**

### Task 5: Hard safety constraint without changing legacy scenarios

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_u6_3d_motion.py`

**Interfaces:**
- Produces: peer-mode candidate-motion rejection when a candidate step would violate `safety_distance_m`; legacy scenarios retain reward-only behavior.

- [ ] **Step 1: Write a failing u6 test where two safe UAVs attempt to move into the unsafe radius.**
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Reject unsafe peer-mode candidate motion deterministically and zero rejected velocities.**
- [ ] **Step 4: Run motion/safety tests and verify GREEN.**

### Task 6: Verification and documentation consistency

**Files:**
- Modify: `configs/scenarios/u6.yaml` comments/provenance only where behavior changed.
- Verify: entire test suite.

- [ ] **Step 1: Run the targeted u6/network regression suite.**
- [ ] **Step 2: Run full `pytest -q`.**
- [ ] **Step 3: Run a multi-seed geometry audit and deterministic same-seed replay check.**
- [ ] **Step 4: Inspect `git diff --check`, `git diff`, and `git status`; report every changed file and any remaining nonbinding modeling limitation (100 MB buffer, discrete sensing bands, vertically extruded obstacles).**
