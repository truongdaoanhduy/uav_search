# GPU-Generic Fast Training + W&B Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three paper baselines faster and portable across CPU/CUDA GPUs while providing automatic W&B-online/local-fallback tracking with full diagnostics and beginner-friendly setup documentation.

**Architecture:** Introduce a small runtime module for device/AMP/optimizer policy, keep algorithm math unchanged, batch same-type actors, and make replay storage torch-native. Extend the existing local `RunLogger` as the durable source and use a resilient W&B adapter to mirror metrics/media/artifacts online when credentials are present.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Gymnasium, PyYAML, pandas, matplotlib, pytest, Weights & Biases.

**Spec:** `docs/superpowers/specs/2026-09-04-gpu-wandb-training.md`

## Global Constraints
- Do not change paper-facing scenario/reward/dynamics equations or training update ratio.
- Do not hard-code a GPU model or W&B API key.
- Local logging must remain complete even when W&B is online or fails.
- Fast mode is default; strict deterministic mode is opt-in.
- TDD each self-contained task and commit after green tests.

---

### Task 1: Runtime performance policy
**Files:** Create `src/uav_search/runtime.py`, `scripts/check_system.py`, `tests/test_runtime.py`; modify `src/uav_search/runner/train.py`.

**Interfaces:** `resolve_device(str)->torch.device`; `configure_runtime(device, deterministic, amp_mode)->RuntimeProfile`; `make_adam(params, lr, device)->Optimizer`; `autocast_context(profile)`.

- [ ] Write failing tests for auto CPU/device resolution, deterministic flag behavior, AMP auto/off rules, and optimizer fallback.
- [ ] Run `pytest tests/test_runtime.py -q` and confirm RED.
- [ ] Implement runtime module and system-check script.
- [ ] Integrate deterministic/AMP runtime profile into trainer without changing algorithm math.
- [ ] Run runtime + existing config tests and confirm GREEN.
- [ ] Commit runtime task.

### Task 2: Batched actors and faster replay
**Files:** Modify `src/uav_search/algorithms/common.py`, `base.py`, `maddpg.py`, `matd3.py`, `masac.py`; modify `tests/test_common.py`, `tests/test_algorithms.py`.

**Interfaces:** `ReplayBuffer(..., pin_memory=False)`, `sample(batch_size, device, non_blocking=True)->Batch`; `BaseOffPolicy._typed_actor_outputs(...)` helpers or equivalent typed batching.

- [ ] Add failing tests proving sampled values/shapes remain identical and same-type actor batching preserves agent order.
- [ ] Run targeted tests and confirm RED.
- [ ] Convert replay storage to preallocated torch CPU tensors and batch same-type actor calls.
- [ ] Use runtime optimizer factory in all three algorithms.
- [ ] Run targeted tests and checkpoint round-trip tests; confirm GREEN.
- [ ] Commit performance-core task.

### Task 3: AMP-safe algorithm updates
**Files:** Modify `base.py`, `maddpg.py`, `matd3.py`, `masac.py`, `tests/test_algorithms.py`.

**Interfaces:** Base stores runtime profile/scaler and exposes autocast/backward-step helpers; CPU behavior remains FP32.

- [ ] Add failing CPU tests for `amp_mode=off/auto` and finite update metrics for all algorithms.
- [ ] Implement autocast/GradScaler paths on CUDA while preserving FP32 CPU path and checkpoint compatibility.
- [ ] Run algorithm tests; confirm GREEN.
- [ ] Commit AMP task.

### Task 4: Resilient W&B mirror and complete diagnostics
**Files:** Modify `runner/wandb_logger.py`, `runner/logging.py`, `runner/train.py`; create/modify `tests/test_tracking.py`, `tests/test_logging.py`.

**Interfaces:** `WandbLogger(mode='auto', ...)` auto-detects API key; `log_episode`, `log_update`, `log_low_episode`, `log_exception`, `log_file`, `log_artifact`, `finish`; local logger returns low-episode diagnostic payload when one is generated.

- [ ] Add failing tests for no-key local fallback, fake-online W&B calls, low-metric CSV/JSONL schema, and tracking failure isolation.
- [ ] Implement local-first writes plus W&B mirror, including W&B Table rows and exception payloads.
- [ ] Upload final/best/crash checkpoints, config, summary/evaluation, plots, and rollout artifact when online.
- [ ] Run tracking/logging tests; confirm GREEN.
- [ ] Commit tracking task.

### Task 5: Performance metrics and CLI
**Files:** Modify `runner/train.py`, `scripts/train.py`, `scripts/run_all.py`, `configs/paper.yaml`; modify smoke tests.

**Interfaces:** CLI adds `--amp auto|on|off`, `--deterministic`, `--wandb-mode auto|online|offline|disabled`, `--wandb-project`; trainer logs `env_steps_per_sec`, `updates_per_sec`, `episode_sec`, `training_sec`, GPU memory when available.

- [ ] Add failing CLI/smoke tests for new defaults and overrides.
- [ ] Implement timing counters/device stats without synchronizing CUDA more than needed.
- [ ] Run smoke tests for MASAC/MATD3/MADDPG on CPU.
- [ ] Commit CLI/performance metrics task.

### Task 6: Install and usage documentation
**Files:** Create `requirements.txt`; rewrite `README.md`; modify `docs/FILE_GUIDE.md`.

- [ ] Add dependency/install consistency test or static check.
- [ ] Create one-command requirements with W&B included.
- [ ] Rewrite README from fresh-machine setup through CPU/GPU/Kaggle/rented-GPU/W&B/local fallback/evaluation.
- [ ] Run README commands in smoke-sized form where possible.
- [ ] Commit docs task.

### Task 7: Benchmark and final verification
**Files:** Create `scripts/benchmark_training.py`, optionally `docs/PERFORMANCE.md`.

- [ ] Implement a benchmark for actor inference/replay/update throughput that reports hardware and does not alter configs.
- [ ] Run `pytest -q`, `python -m compileall -q src scripts`, `git diff --check`.
- [ ] Run 3 algorithm CPU smoke experiments with W&B disabled/local fallback.
- [ ] Run benchmark and record host limitations honestly.
- [ ] Inspect git diff/status and commit final verification updates.
- [ ] Use finishing-a-development-branch to integrate to `main`, push, and verify local/remote SHA match.
