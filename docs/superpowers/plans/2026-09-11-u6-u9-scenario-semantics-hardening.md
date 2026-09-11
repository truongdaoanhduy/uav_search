# U6/U9 Scenario Semantics Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Correct every verified U6/U9 environment-flow defect without changing the MASAC, MADDPG, or MATD3 algorithm implementations.

**Architecture:** Harden the existing `PaperUAVEnv` data flow at its sources: physical sensing, fine-confirmation provenance, report lifecycle, local synchronization state, and reward attribution. Keep UavNetSim as the MAC/PHY authority and retain the six-dimensional hybrid action.

**Tech Stack:** Python 3, NumPy, Gymnasium, Pytest, YAML, UavNetSim backend.

**Spec:** `docs/superpowers/specs/2026-09-11-u6-u9-scenario-semantics-hardening.md`

## Global Constraints

- Homogeneous `u6` and `u9` scenarios only.
- Do not modify files under `src/uav_search/algorithms/`.
- Keep `action_dim == 6` and the existing action meaning.
- Preserve deterministic seeding and one-hop-per-macro-step semantics.
- Write and observe every regression test failing before production changes.

---

### Task 1: Physical sensing and anonymous observations

**Files:**
- Modify: `src/uav_search/envs/sensing.py`
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_u6_scenario_semantics.py`

**Interfaces:**
- Produces: `continuous_fov_cells(center_xy, fov_radius_m, grid_cell_m, grid_n)`
- Produces: fine-cell evidence used by `_confirm_peer_targets()`.

- [x] Add tests showing cross-boundary near targets are sensed, same-cell distant targets are not treated as occupied, peer observations contain no pre-confirmation target-distance block, high-altitude evidence cannot confirm, and a low-altitude positive can confirm.
- [x] Run the new tests and verify the expected failures.
- [x] Implement exact footprint/cell intersection, exact target range checks, fine-evidence confirmation, and remove peer target-distance slots.
- [x] Replace clipped entropy decrease with signed entropy-potential change.
- [x] Run the sensing tests and verify they pass.

### Task 2: DTN lifecycle, EDF, reward, and terminal failure

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `configs/scenarios/u6.yaml`
- Modify: `configs/scenarios/u9.yaml`
- Test: `tests/test_u6_scenario_semantics.py`

**Interfaces:**
- Produces: `_report_priority(sender, slot_start)` sorted by remaining TTL then target index.
- Produces: attempted-report byte tracking and shared mission reward terms.

- [x] Add tests for confirmation-time TTL, pending-report expiry, EDF order, failed-TX penalty, shared discovery/delivery/expiry rewards, and irreversible-failure termination.
- [x] Run them and verify the expected failures.
- [x] Implement lifecycle timestamps, EDF traversal, attempted-byte accounting, shared terms, and terminal failure.
- [x] Set each U6/U9 buffer to three complete reports and update provenance.
- [x] Run the DTN tests and verify they pass.

### Task 3: Local routing, power, safety, and quantized sync observation

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Test: `tests/test_u6_scenario_semantics.py`
- Test: existing homogeneous environment/algorithm tests.

**Interfaces:**
- Produces: cached peer routing fields and per-report local lifecycle fields.
- Produces: 8-bit belief serialization bounded by `peer_sync_bytes`.
- Produces: local proximity features and min/max-power link summaries.

- [x] Add tests that distinguish TTL states in observations, expose nearby obstacles, distinguish min/max power feasibility when the backend does, and show sync fuses quantized rather than exact float64 beliefs.
- [x] Run them and verify the expected failures.
- [x] Implement caches, report-local features, proximity encoding, vehicle-specific normalization, min/max snapshots, and quantized sync.
- [x] Update old exact-shape/exact-fusion expectations to the new documented contract.
- [x] Run all relevant environment and algorithm compatibility tests.

### Task 4: Final verification and synchronization

**Files:**
- Modify if needed: provenance/mapping documentation affected by the new contract.

- [x] Run `pytest -q -rs`.
- [x] Run one deterministic short smoke training/update for MASAC, MADDPG, and MATD3 on U6 and U9.
- [x] Inspect `git diff --check`, `git diff`, and `git status`; confirm no algorithm implementation changed.
- [x] Commit the complete change on `feature/homo-u6-paper-sim`.
- [x] Push without force and verify local HEAD equals `origin/feature/homo-u6-paper-sim`.
