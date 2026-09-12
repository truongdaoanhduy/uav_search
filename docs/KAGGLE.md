# Kaggle run guide

The active `u6`/`u9` scenarios use the pinned lightweight UavNetSim MAC/PHY path. A clean Kaggle session therefore needs the project dependencies plus the exact UavNetSim Git commit used by the repository.

## 1. Enable Internet for installation

`scripts/setup_kaggle.sh` fetches UavNetSim from GitHub at the pinned commit. Kaggle notebooks run inside versioned Docker images, so do not assume a particular notebook image already contains the simulator or the same dependency versions.

From the repository root:

```bash
bash scripts/setup_kaggle.sh
```

The script performs three steps:

1. installs `requirements.txt`;
2. installs SimPy 4.1.1 and UavNetSim 2.0.0 at commit `04daafb815eb377409b40b285574eeb62b9a8d58` using `--no-deps` so the unused Sionna/web stack is not pulled into the training image;
3. runs the strict system preflight.

If Kaggle Internet is disabled, the GitHub install cannot succeed. In that case provide the pinned UavNetSim source/wheel as a Kaggle Dataset and install that local artifact before running the preflight. Do not silently substitute a newer upstream commit.

## 2. Verify the runtime

```bash
python scripts/check_system.py --device auto --require-uavnetsim
```

The command must report:

- `lightweight_runtime_ready: true`;
- UavNetSim version `2.0.0`;
- installed commit `04daafb815eb377409b40b285574eeb62b9a8d58`;
- SimPy `4.1.1`;
- `u6` and `u9` loading with backend `uavnetsim`.

## 3. Smoke before a long GPU run

```bash
python scripts/train.py --algorithm masac --scenario u6 \
  --episodes 1 --steps 4 --device auto --amp auto --local-only
```

Then optionally smoke all six active combinations:

```bash
python scripts/run_all.py --episodes 1 --steps 4 --eval-episodes 1 \
  --device auto --amp auto --local-only
```

## 4. Long run

For the full active comparison:

```bash
python scripts/run_all.py --episodes 50000 --eval-episodes 5000 \
  --device auto --amp auto --seed 44 --local-only
```

Remove `--local-only` and provide `--wandb-api-key` only when W&B upload is desired.

## Reproducibility notes

- Deterministic mode is on by default.
- `u6`/`u9` checkpoints produced before the Cartesian-action/hybrid-action canonicalization change are intentionally incompatible in meaning and should not be resumed for scientific runs.
- The 600-step U6/U9 deadline is an observable finite-horizon terminal, not an external truncation.
- Historical documents under `docs/archive/` are not runtime specifications.
