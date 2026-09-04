# Paper Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, modular reproduction of the paper simulation for MASAC, MATD3, and MADDPG on the 1x5 and 1x9 scenarios.

**Architecture:** One paper-faithful heterogeneous UAV environment and one shared PyTorch CTDE infrastructure. Algorithm-specific files implement only their learning-rule differences, while training, diagnostics, checkpointing, evaluation, and visualization are shared.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Gymnasium spaces, PyYAML, pandas, matplotlib, pytest; optional Weights & Biases.

**Spec:** `docs/superpowers/specs/2026-09-04-paper-baselines-design.md`

## Global Constraints
- Exactly three algorithms: `masac`, `matd3`, `maddpg`.
- Primary scenarios: `f1_m5`, `f1_m9`.
- Published paper parameters remain unchanged and are marked `PAPER_EXPLICIT`.
- Unpublished parameters are isolated and marked `ASSUMED`.
- No hardware experiment, GATAC, or SA-GATAC implementation.
- Successful training always writes `final.pt`; failures write traceback and `crash.pt` when model state exists.
- W&B is optional and must not be a hard dependency.

---

### Task 1: Project/config foundation
**Files:** `pyproject.toml`, `.gitignore`, `configs/**/*.yaml`, `src/uav_search/config.py`, `tests/test_config.py`

- [ ] Write failing tests for config merge, paper constants, and scenario counts.
- [ ] Run tests and confirm RED.
- [ ] Implement minimal config loader and YAML hierarchy.
- [ ] Run tests and confirm GREEN.

### Task 2: Paper environment
**Files:** `src/uav_search/envs/paper_env.py`, `src/uav_search/envs/models.py`, `tests/test_env.py`

- [ ] Write failing seeded-reset, step-shape, bounds, target, communication, energy, and deterministic-repeat tests.
- [ ] Run tests and confirm RED.
- [ ] Implement communication, dynamics, energy, reward, metrics, and renderer state.
- [ ] Run tests and confirm GREEN.

### Task 3: Shared CTDE building blocks
**Files:** `src/uav_search/algorithms/common.py`, `src/uav_search/algorithms/networks.py`, `tests/test_common.py`

- [ ] Write failing replay/network/soft-update tests.
- [ ] Run tests and confirm RED.
- [ ] Implement minimal replay buffer and neural components.
- [ ] Run tests and confirm GREEN.

### Task 4: MADDPG, MATD3, MASAC
**Files:** `src/uav_search/algorithms/{maddpg,matd3,masac,factory}.py`, `tests/test_algorithms.py`

- [ ] Write parameterized failing tests for bounded actions, one finite update, save/load.
- [ ] Run tests and confirm RED.
- [ ] Implement MADDPG.
- [ ] Extend shared pieces minimally for MATD3.
- [ ] Extend shared pieces minimally for MASAC.
- [ ] Run algorithm tests and confirm GREEN.

### Task 5: Logging, diagnostics, W&B adapter, plots
**Files:** `src/uav_search/runner/{logging,wandb_logger,visualize}.py`, `tests/test_logging.py`

- [ ] Write failing artifact/diagnostic/plot tests.
- [ ] Run tests and confirm RED.
- [ ] Implement CSV/JSONL/error logging, optional W&B, and plots.
- [ ] Run tests and confirm GREEN.

### Task 6: Training/evaluation CLI
**Files:** `src/uav_search/runner/train.py`, `src/uav_search/runner/evaluate.py`, `scripts/*.py`, `tests/test_smoke.py`

- [ ] Write failing 2-episode smoke test.
- [ ] Run test and confirm RED.
- [ ] Implement trainer, checkpoint handling, CLI wrappers, evaluation, and run-all.
- [ ] Run smoke and full tests and confirm GREEN.

### Task 7: Documentation and six-run verification
**Files:** `README.md`, `docs/PAPER_MAPPING.md`, generated `runs/` artifacts (gitignored)

- [ ] Document exact commands, structure, assumptions, metrics, troubleshooting, and provenance.
- [ ] Run `pytest -q`.
- [ ] Run short training for 3 algorithms x 2 scenarios.
- [ ] Verify every run has metrics, final checkpoint, and five plots.
- [ ] Verify no NaN/Inf or traceback in successful runs.
- [ ] Commit implementation, fast-forward `main`, push, and verify local `HEAD == origin/main`.
