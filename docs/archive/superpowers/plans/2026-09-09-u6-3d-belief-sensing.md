# u6 Full-3D Belief-Sensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert only `u6` to full-3D motion with altitude-aware Bayesian sensing and confidence-based target confirmation while preserving persistent UavNetSim and all existing MARL baselines.

**Architecture:** Add a focused pure sensing module for altitude profiles, FOV cells, Bayesian updates, and entropy. Integrate it into `PaperUAVEnv` only in `homogeneous_peer` mode, expanding peer action space from 5 to 6 dimensions and appending a fixed 3x3 belief patch to peer observations. Keep UavNetSim as the network executor and make the configured 2 km contact radius power-independent.

**Tech Stack:** Python 3, NumPy, Gymnasium, PyTorch, pytest, UavNetSim/SimPy.

**Spec:** `docs/archive/superpowers/specs/2026-09-09-u6-3d-belief-sensing-design.md`

## Global Constraints

- Paper-faithful scenarios must retain their existing 2-D action contract and fixed-altitude behavior.
- `u6` must be deterministic under fixed seeds.
- No production behavior change without a failing regression/feature test first.
- No CV/image pipeline or new simulator dependency.
- All new research parameters must carry provenance in `configs/scenarios/u6.yaml`.
- The previous state remains recoverable at `backup/u6-before-3d-sensing-20260909`.

---

### Task 1: Pure altitude-aware sensing primitives

**Files:**
- Create: `src/uav_search/envs/sensing.py`
- Create: `tests/test_sensing.py`

**Interfaces:**
- Produces: `SensingProfile`, `profile_for_altitude(z_m, levels_m, fov_sizes, pd_values, pf_values)`, `fov_offsets(size)`, `bayes_update(prior, measurement, pd, pf)`, `binary_entropy(p)`.

- [ ] Write tests asserting nearest-level profile mapping, 1/5/9 FOV cell counts, valid 5-cell cross geometry, analytical Bayes positive/negative results, and stable entropy endpoints.
- [ ] Run `pytest -q tests/test_sensing.py` and verify RED because the module does not exist.
- [ ] Implement the pure functions with validation and no environment side effects.
- [ ] Run `pytest -q tests/test_sensing.py` and verify GREEN.
- [ ] Commit `feat: add altitude-aware belief sensing primitives`.

### Task 2: u6 3D action and motion contract

**Files:**
- Modify: `configs/scenarios/u6.yaml`
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `tests/test_homogeneous_paper_env.py`
- Modify: `tests/test_homogeneous_algorithms.py`
- Modify: `tests/test_u6_dtn_hardening.py`
- Modify: `tests/test_u6_power_dtn_refactor.py`

**Interfaces:**
- `u6.action_dim == 6` with `[horizontal_thrust, heading, vertical_accel, tx_gate, tx_power, recipient]`.
- `PaperUAVEnv.velocities` becomes `(n_agents, 3)` while paper-faithful observation semantics remain unchanged.

- [ ] Add tests for action dimension 6, initial altitude membership in `[50,100,150]`, vertical movement, altitude clipping, and unchanged paper-faithful action dimension 2.
- [ ] Run targeted tests and verify RED on the old 5-D/fixed-altitude behavior.
- [ ] Add provenance-backed altitude config, expand peer action parsing, implement 3-D velocities and vertical acceleration, and include true z-velocity in peer observations.
- [ ] Update existing u6 test action helpers from 5 to 6 dimensions without changing the meaning of their tx gate/power/recipient commands.
- [ ] Run targeted environment/algorithm tests and verify GREEN.
- [ ] Commit `feat: add full-3d motion to u6`.

### Task 3: Bayesian belief-map sensing and confirmation

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `configs/scenarios/u6.yaml`
- Create/Modify: `tests/test_u6_3d_sensing.py`

**Interfaces:**
- Environment owns `belief_maps[n_agents, grid_n, grid_n]`, `last_sensor_positive[n_agents, n_targets]`, `last_information_gain_by_agent[n_agents]`, and per-step sensing diagnostics.
- `_perform_peer_target_detection()` becomes the altitude/FOV/Bayes scan and confirms targets at configured posterior threshold.

- [ ] Add tests for deterministic seeded scans, no fixed-distance confirmation, belief update on occupied/empty cells, 0.99 confirmation, one report per target, and no unsensed target-distance leakage.
- [ ] Run targeted tests and verify RED.
- [ ] Implement 100 m belief grid, ground-truth occupancy map, altitude-selected FOV scanning, Bayesian updates, deterministic confirmation source tie-break, and report creation hook.
- [ ] Append a zero-masked 3x3 local belief patch to peer observations and expose target distance only for current positive sensing or local confirmed knowledge.
- [ ] Replace peer-mode hidden-distance task shaping with `20 * (1.0 * confirmations + 0.1 * information_gain)` and keep paper-faithful reward behavior untouched.
- [ ] Run targeted tests and verify GREEN.
- [ ] Commit `feat: add belief-based target sensing to u6`.

### Task 4: Fixed contact radius and 6-D network/calibration actions

**Files:**
- Modify: `src/uav_search/envs/network_backends.py`
- Modify: `src/uav_search/runner/network_calibration.py`
- Modify: `src/uav_search/algorithms/matd3.py`
- Modify: `tests/test_network_backends.py`
- Modify: `tests/test_network_calibration.py`
- Modify: `tests/test_homogeneous_algorithms.py`

**Interfaces:**
- `peer_contact_range_m` / `gcs_contact_range_m` are fixed candidate radii independent of tx power.
- MATD3 smoothing mask for 6-D u6 is `[1,1,1,0,1,0]`.

- [ ] Add/adjust tests proving 2500 m fails at both 0.1 W and 0.4 W when Rcomm=2000 m, while power remains passed to PHY inside range.
- [ ] Add tests for 6-D calibration actions and MATD3 smoothing mask.
- [ ] Run targeted tests and verify RED.
- [ ] Remove `sqrt(P)` range scaling from both analytical and UavNetSim backends; update calibration actions and MATD3 mask.
- [ ] Run targeted tests and verify GREEN.
- [ ] Commit `fix: separate contact radius from transmit power`.

### Task 5: Metrics, logging, docs, and provenance

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `src/uav_search/runner/train.py`
- Modify: `README.md`
- Modify: `docs/FILE_GUIDE.md`
- Modify: `configs/scenarios/u6.yaml`
- Modify/Create: `tests/test_logging.py` or `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- `info` and episode logs expose altitude and belief/sensing diagnostics without changing existing metric names.

- [ ] Add tests for altitude/belief metric presence and scalar validity.
- [ ] Run targeted tests and verify RED.
- [ ] Add metrics to `info` and episode logging; document action order, sensing model, provenance, and rollback branch.
- [ ] Run targeted tests and verify GREEN.
- [ ] Commit `docs: document u6 3d sensing provenance`.

### Task 6: Full regression and deterministic smoke

**Files:**
- No production changes unless a regression is reproduced by a failing test.

- [ ] Run `pytest -q`.
- [ ] Run `python -m compileall -q src scripts tests`.
- [ ] Run `git diff --check` and conflict-marker scan.
- [ ] Run short real-UavNetSim MASAC, MATD3, and MADDPG u6 smoke runs.
- [ ] Repeat a same-seed deterministic MASAC smoke and compare mission metrics excluding timing.
- [ ] Run a short 3D network calibration smoke and confirm altitude/network metrics are finite.
- [ ] If any failure occurs, reproduce with a focused failing test before fixing.
- [ ] After all checks pass, fast-forward `feature/homo-u6-paper-sim`, push, and verify local/tracking/remote SHA equality.
