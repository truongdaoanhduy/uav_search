# Heterogeneous UAV Search — MASAC / MATD3 / MADDPG

Minimal, modular reproduction of the software-simulation baselines from:

> T. Ao et al., “Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network,” IEEE Transactions on Vehicular Technology, DOI: 10.1109/TVT.2025.3594534.

Implemented baselines only:

- MASAC
- MATD3
- MADDPG

Primary scenarios:

- `f1_m5`: 1 fixed-wing + 5 multi-rotor UAVs
- `f1_m9`: 1 fixed-wing + 9 multi-rotor UAVs

The project does not implement GATAC/SA-GATAC or the paper's physical hardware experiment.

## 1. What the optimized trainer does

The training code keeps the same MARL logic/config while optimizing execution:

- automatic CPU/CUDA device selection;
- works with P100, T4, V100, A100, L4 and other CUDA GPUs supported by the installed PyTorch build;
- batched actor inference by UAV type instead of one forward pass per UAV;
- preallocated Torch replay storage;
- pinned host memory + non-blocking CUDA transfers;
- automatic mixed precision (`--amp auto`) on CUDA;
- fused Adam when supported, then foreach Adam, then standard Adam fallback;
- deterministic mode remains available with `--deterministic`, but is not forced during fast training;
- throughput metrics (`env_steps_per_sec`, `updates_per_sec`, episode/wall time, GPU memory) are logged;
- periodic, best, final and crash checkpoints;
- W&B online tracking when `WANDB_API_KEY` exists;
- complete local fallback when no W&B key exists or W&B/network logging fails.

The optimizer does not intentionally change the paper environment, action/reward definitions, replay/update ratio, learning rates, gamma, tau or algorithm objective.

## 2. Install from a blank Ubuntu/Debian machine

### 2.1 Install system tools

```bash
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv
```

For GPU training, install a working NVIDIA driver first. Kaggle and most rented GPU machines already provide the driver/CUDA runtime.

Check the driver:

```bash
nvidia-smi
```

If `nvidia-smi` is not available on a machine where you expect a GPU, fix the NVIDIA driver before training.

### 2.2 Clone the project

```bash
git clone https://github.com/truongdaoanhduy/uav_search.git
cd uav_search
```

### 2.3 Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.4 Install every Python dependency

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

On Kaggle/rented GPU environments, an existing CUDA-enabled PyTorch installation normally satisfies the `torch>=2.2` requirement and will not be replaced if its version is compatible.

### 2.5 Verify the machine

```bash
python scripts/check_system.py --device auto
```

You should see Python/PyTorch versions, selected device, GPU name, CUDA availability, compute capability, VRAM and whether AMP is enabled.

Run tests:

```bash
pytest -q
```

## 3. W&B tracking behavior

No API key is ever hardcoded into this repository.

### When `WANDB_API_KEY` exists

Training automatically uses W&B online mode. It sends:

- episode metrics;
- actor/critic losses and update metrics;
- search rate and targets found;
- energy and battery metrics;
- communication rate and broken-link duration;
- collision/obstacle/boundary metrics;
- action saturation;
- MASAC entropy/alpha;
- throughput and timing;
- GPU memory metrics;
- low-metric episode diagnostics as a W&B Table;
- exception/error diagnostics as a W&B Table;
- reward/search/energy/broken-link/trajectory plots;
- best/final/periodic/crash checkpoints as model Artifacts;
- the complete run directory as a run-output Artifact.

W&B itself also collects standard system metrics for online runs.

### When `WANDB_API_KEY` does not exist

Training does not fail. It runs normally and stores the same experiment information under `runs/` locally.

### If W&B/network logging fails during training

Training continues. The run remains complete locally and the failure reason is written to:

```text
logs/tracking_fallback.log
```

## 4. Add the W&B token safely

Do not paste the key into source code, YAML or GitHub.

A safer interactive shell method is:

```bash
read -s -p "WANDB_API_KEY: " WANDB_API_KEY
echo
export WANDB_API_KEY
```

Then start training normally.

Optional W&B destination:

```bash
export WANDB_PROJECT="uav-search-paper-baselines"
```

The CLI also accepts `--wandb-project` and `--wandb-entity`.

## 5. CPU training

Quick smoke run:

```bash
python scripts/train.py \
  --algorithm masac \
  --scenario f1_m5 \
  --episodes 10 \
  --device cpu \
  --amp off
```

All six requested combinations:

```bash
python scripts/run_all.py \
  --episodes 10 \
  --device cpu \
  --amp off
```

## 6. Generic GPU training

Recommended command for any supported CUDA GPU:

```bash
python scripts/train.py \
  --algorithm masac \
  --scenario f1_m5 \
  --episodes 50000 \
  --device auto \
  --amp auto
```

`--device auto` selects CUDA when available and CPU otherwise.

To select a specific GPU:

```bash
python scripts/train.py \
  --algorithm matd3 \
  --scenario f1_m9 \
  --episodes 50000 \
  --device cuda:0 \
  --amp auto
```

Run all 3 algorithms × 2 scenarios:

```bash
python scripts/run_all.py \
  --episodes 50000 \
  --device auto \
  --amp auto \
  --wandb-project uav-search-paper-baselines
```

## 7. Kaggle GPU workflow

In Kaggle Notebook settings, enable a GPU accelerator.

Clone/install:

```python
!git clone https://github.com/truongdaoanhduy/uav_search.git
%cd uav_search
!pip install -q -r requirements.txt
!pip install -q -e . --no-deps
!python scripts/check_system.py --device auto
```

### W&B on Kaggle

Create a Kaggle Secret named `WANDB_API_KEY`, then load it without placing the value in notebook source:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Train:

```python
!python scripts/train.py --algorithm masac --scenario f1_m5 --episodes 50000 --device auto --amp auto
```

or all six runs:

```python
!python scripts/run_all.py --episodes 50000 --device auto --amp auto --wandb-project uav-search-paper-baselines
```

The exact GPU model does not need to be hardcoded.

## 8. Rented GPU / cloud VM

After SSH login:

```bash
nvidia-smi
git clone https://github.com/truongdaoanhduy/uav_search.git
cd uav_search
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_system.py --device auto
```

Enter the W&B key interactively:

```bash
read -s -p "WANDB_API_KEY: " WANDB_API_KEY
echo
export WANDB_API_KEY
```

Then run the same GPU command as above.

## 9. Fast mode vs deterministic mode

Fast/default:

```bash
python scripts/train.py --algorithm masac --scenario f1_m5 --device auto --amp auto
```

Strict deterministic mode:

```bash
python scripts/train.py --algorithm masac --scenario f1_m5 --device auto --amp auto --deterministic
```

Deterministic PyTorch kernels can be slower. Same seed is still used in normal fast mode; `--deterministic` additionally requests deterministic PyTorch algorithms where available.

## 10. AMP controls

Automatic and recommended:

```bash
--amp auto
```

Force AMP on CUDA:

```bash
--amp on
```

Disable AMP:

```bash
--amp off
```

`--amp on` fails immediately if CUDA is unavailable. `auto` is portable between CPU and GPU machines.

## 11. Low-metric episode diagnostics

This is separate from software exception logs.

When an episode is poor, the complete episode metrics are written to:

```text
logs/low_episodes.jsonl
logs/low_metric_episodes.csv
```

and, when W&B is online, to:

```text
diagnostics/low_metric_episodes
```

as a W&B Table.

Diagnostic signals include:

- `search_low`: search rate is very low;
- `return_low`: return is in the lower tail of the recent rolling window;
- `link_unstable`: broken-link duration is high or communication rate is low;
- `collision_high`: collisions/obstacle incidents occurred;
- `energy_high`: battery approaches the safety threshold;
- `action_saturated`: actions remain near ±1 too often;
- `critic_unstable`: critic loss is non-finite or extremely large.

Each bad episode still contains all normal metrics, so you can inspect why the score fell instead of seeing only one reward number.

## 12. Software errors/crashes

Exceptions are written separately to:

```text
logs/errors.log
```

The record includes exception type, message, traceback and algorithm/scenario/episode/step context. When W&B is online this is also added to the W&B diagnostics error table.

The trainer attempts to save `checkpoints/crash.pt` before re-raising the exception.

## 13. Checkpoints

Default checkpoint behavior:

```text
checkpoints/best.pt
checkpoints/latest.pt      # periodic, default every 1000 episodes
checkpoints/final.pt
checkpoints/crash.pt       # only on handled crash
```

Online W&B runs upload these as model Artifacts. Periodic checkpoints use the same artifact name with new versions/episode aliases.

## 14. Local run layout

The local directory is always the safety copy/fallback while training:

```text
runs/<algorithm>/<scenario>/<run>/
├── config.yaml
├── summary.json
├── evaluation.json
├── checkpoints/
├── logs/
│   ├── errors.log
│   ├── low_episodes.jsonl
│   ├── low_metric_episodes.csv
│   └── tracking_fallback.log   # only when W&B fails
├── metrics/
│   ├── episodes.csv
│   ├── updates.csv
│   └── performance.csv
├── plots/
│   ├── reward.png
│   ├── search_rate.png
│   ├── energy.png
│   ├── broken_link.png
│   └── trajectory.png
└── rollouts/
    └── final_trajectory.npz
```

If W&B is online, the final run-output Artifact contains this complete run directory as well; checkpoints are also versioned separately as model Artifacts.

## 15. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --checkpoint runs/masac/f1_m5/<run>/checkpoints/final.pt \
  --episodes 5000 \
  --device auto \
  --output-dir evaluation/masac_f1_m5
```

The paper uses 5,000 random evaluation cases.

## 16. Development-only short episodes

The paper uses 50 steps per episode. For a quick code smoke test only:

```bash
python scripts/run_all.py --episodes 2 --steps 5 --device cpu --amp off --local-only
```

Do not use `--steps` when collecting paper-scale results.

## 17. Paper vs assumed parameters

`configs/paper.yaml` separates:

- `PAPER_EXPLICIT`: parameters stated in the article;
- `ASSUMED`: parameters not fully published and therefore isolated for later replacement.

See:

- `docs/PAPER_MAPPING.md`
- `docs/FILE_GUIDE.md`

## 18. Verification

```bash
pytest -q
python -m compileall -q src scripts
```
