# GPU-Generic Fast Training + W&B Tracking Spec

## Goal
Optimize MASAC/MATD3/MADDPG execution speed without changing the paper-facing environment, reward definitions, episode length, replay/update ratio, or algorithm equations; support CPU and arbitrary CUDA GPUs; make W&B the online tracking destination when a token is passed with `--wandb-api-key`, with complete local fallback otherwise.

## Runtime behavior
- `--device auto` selects CUDA when available, otherwise CPU.
- Training must work on P100, T4, V100, A100, L4 and other CUDA devices without hard-coded GPU names.
- Fast mode seeds Python/NumPy/Torch/env/replay but does not force deterministic kernels.
- `--deterministic` enables deterministic PyTorch algorithms for stricter reproducibility at a possible speed cost.
- `--amp auto|on|off`: `auto` enables CUDA FP16 AMP when CUDA is available; CPU remains FP32. `on` requires CUDA.
- Optimizers prefer fused Adam on CUDA when supported and fall back safely.
- Policy calls for identical agent types are batched instead of one Python forward per agent.
- Replay sampling uses preallocated torch CPU storage and non-blocking CUDA transfer when beneficial.

## Tracking behavior
- No hard-coded W&B API key.
- If `--wandb-api-key` is provided, authenticate with `wandb.login()` and initialize W&B online unless explicitly disabled for tests.
- If no key exists, training continues and logs to the local run directory.
- W&B receives training episode metrics, update metrics, performance metrics, system/device metadata, low-metric episodes, exceptions, plots, trajectory media, configs, summaries, and checkpoints/artifacts.
- A W&B outage must not destroy local records or checkpointing; local logging is always written first.

## Low-metric diagnostics
- Low-metric episodes are separate from exception logs.
- Store CSV + JSONL locally.
- Upload rows to W&B as a Table when online.
- Include return, search rate, targets found, energy, battery, communication rate, broken-link durations, collisions, action saturation, actor/critic loss, entropy/alpha, steps/sec, update/sec, and diagnostic reasons/severity.
- Diagnostic reasons include low search, low return, unstable link, low communication, collision/obstacle problems, low battery/high energy, saturated actions, and unstable critic.

## Documentation/install
- Create root `requirements.txt` for one-command dependency installation.
- README must cover installation from a fresh Ubuntu/Linux environment, venv, requirements installation, system check, CPU run, generic GPU run, deterministic run, Kaggle/W&B secret setup, rented GPU usage, all-six experiments, evaluation, local fallback, and later W&B sync.
- Add `scripts/check_system.py` to display Python/Torch/CUDA/GPU/VRAM and runtime recommendations.

## Verification
- Existing paper/environment/algorithm tests remain green.
- Add TDD coverage for runtime device profile, batched typed actor outputs, torch replay sample behavior, W&B/no-key fallback, low-metric CSV diagnostics, and CLI flags.
- Run CPU smoke training for all three algorithms.
- Run a benchmark comparing baseline-compatible policy/replay operations to the optimized implementation; record results without claiming GPU speedups on a CPU-only host.
