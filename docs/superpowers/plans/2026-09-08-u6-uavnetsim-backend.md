# U6 Hybrid UavNetSim Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable UavNetSim-backed communication path for the leaderless `u6` search environment without changing legacy paper scenarios or the five-dimensional MARL action.

**Architecture:** `PaperUAVEnv` remains the mission/mobility authority. Peer communication delegates link evaluation and one-step transmission outcomes to `NetworkBackend`; the analytical backend preserves current behavior while `UavNetSimBackend` uses pinned UavNetSim CSMA/CA and A2A PHY with RL-controlled next hop.

**Tech Stack:** Python, NumPy, Gymnasium, pytest, SimPy, UavNetSim v2.0.0 pinned at commit `04daafb815eb377409b40b285574eeb62b9a8d58`.

**Spec:** `docs/superpowers/specs/u6-hybrid-uavnetsim.md`

## Global Constraints

- Keep legacy six paper scenarios behavior-compatible.
- Keep `u6` action dimension exactly 5: `[move_x, move_y, tx_gate, tx_amount, recipient]`.
- Do not let UavNetSim routing choose or replace the policy recipient.
- Preserve no-same-step multi-hop causality.
- UavNetSim dependency must be optional for the base install and pinned for reproducibility.
- Deterministic seed behavior must remain testable.

---

### Task 1: Backend interface and analytical regression backend

**Files:**
- Create: `src/uav_search/envs/network_backends.py`
- Test: `tests/test_network_backends.py`

**Interfaces:**
- Produces: `NetworkLinkSnapshot`, `TransmissionIntent`, `NetworkStepResult`, `AnalyticalNetworkBackend`, `create_network_backend`.

- [ ] Write a failing test proving the factory returns an analytical backend and rejects unknown backend names.
- [ ] Run the focused test and verify RED because the module does not exist.
- [ ] Implement the minimal dataclasses/factory/analytical backend.
- [ ] Run the focused test and verify GREEN.

### Task 2: UavNetSim transport adapter

**Files:**
- Modify: `src/uav_search/envs/network_backends.py`
- Test: `tests/test_network_backends.py`
- Create: `scripts/install_uavnetsim.sh`

**Interfaces:**
- Produces: `UavNetSimBackend.link_snapshot(...)` and `UavNetSimBackend.transmit(...)`.

- [ ] Write failing tests for actionable missing-dependency errors and deterministic UavNetSim outcomes when the optional dependency is present.
- [ ] Verify RED.
- [ ] Implement lazy imports, UavNetSim A2A link evaluation, CSMA/CA aggregate-frame delivery and metrics.
- [ ] Add the pinned lightweight installer (`simpy` + UavNetSim `--no-deps`).
- [ ] Verify focused tests GREEN.

### Task 3: Integrate backend into `PaperUAVEnv`

**Files:**
- Modify: `src/uav_search/envs/paper_env.py`
- Modify: `configs/scenarios/u6.yaml`
- Modify: `tests/test_homogeneous_paper_env.py`

**Interfaces:**
- Consumes: backend snapshot/transmission result types from Task 1/2.
- Produces: unchanged environment API plus backend/network metrics in `info`.

- [ ] Write failing tests proving `u6` selects the configured backend, policy recipient is preserved, transmission metrics are exposed, and no-same-step forwarding remains enforced.
- [ ] Verify RED.
- [ ] Route peer link refresh and peer transmissions through the backend while keeping legacy scenarios untouched.
- [ ] Verify focused homogeneous tests GREEN.

### Task 4: Documentation, optional dependency smoke test and conflict verification

**Files:**
- Modify: `README.md`
- Modify: `docs/FILE_GUIDE.md`
- Modify: `pyproject.toml`

- [ ] Document analytical vs UavNetSim backend selection and pinned install command.
- [ ] Run full `pytest`.
- [ ] Create a Python 3.12 temporary venv, install the pinned lightweight UavNetSim dependency, run backend integration tests and a short `u6` smoke episode.
- [ ] Run `git diff --check`, `git status`, and inspect the final diff.
- [ ] `git fetch origin`, compare local branch against `origin/feature/homo-u6-paper-sim`, and verify there are no merge conflicts before push.
- [ ] Commit and push only after all verification passes; then verify remote branch HEAD equals local HEAD.
