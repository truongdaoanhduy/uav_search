# Heterogeneous UAV Search — MASAC / MATD3 / MADDPG

Baselines: `MASAC`, `MATD3`, `MADDPG`.

Scenarios:
- `f1_m5`: 1 fixed-wing + 5 multi-rotor UAVs (paper reproduction)
- `f1_m9`: 1 fixed-wing + 9 multi-rotor UAVs (paper reproduction)
- `u6`: 6 identical multi-rotor UAVs, peer-to-peer, no leader/follower (research adaptation)

Paper-scenario fidelity and the remaining unpublished-parameter gaps are documented in [`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md). The environment matches the paper where values/equations are published, but it does not claim byte-for-byte reproduction of the authors' unpublished simulator.

## Cài đặt

```bash
git clone https://github.com/truongdaoanhduy/uav_search.git
cd uav_search
pip install -r requirements.txt
```


## Homogeneous `u6` research scenario

`u6` uses the approved **hybrid option C**. `PaperUAVEnv` remains authoritative for the 5 km × 5 km post-disaster mission, randomized targets/buildings, circular obstacles, 50-step episodes, multi-rotor 2.5D motion, sensing, finite application buffers, and reward calculation. Networking is delegated to a pluggable backend. The default for `u6` is **UavNetSim**, pinned to commit `04daafb815eb377409b40b285574eeb62b9a8d58` (distribution version 2.0.0).

The continuous action remains fixed at:

```text
[force, direction, transmit_gate, transmit_amount, recipient]
```

The MARL policy decides whether to transmit, how many bytes to attempt, and the immediate recipient/next hop. UavNetSim does **not** choose routing for this integration; its `CsmaCa`, `Phy`, and `Channel` components resolve contention, SINR/interference, transmission success, delay, PDR, throughput, and radio transmit energy. Data received by a relay cannot be forwarded again until the next MARL step.

Install the pinned lightweight UavNetSim backend before running `u6`:

```bash
./scripts/install_uavnetsim.sh
```

The training path intentionally uses UavNetSim `CHANNEL_MODE=a2a`, which exercises UavNetSim's A2A PHY/channel and CSMA/CA without requiring the heavy Sionna RT scene worker on every MARL step. The original analytical paper-style peer channel remains available as `network_backend: analytical` for regression/ablation.

Smoke train:

```bash
python scripts/train.py --algorithm masac --scenario u6 --episodes 10 --device cpu --local-only
```

Target confirmation creates a finite-buffer report; mission delivery is counted only when target data reaches the GCS. `info` includes `network_attempted_bytes`, `network_delivered_bytes`, byte PDR, throughput, mean delay, PHY failures, and network transmit energy. The legacy `f1_m5`/`f1_m9` paths remain on the original paper simulator and are not switched to UavNetSim.

## Train CPU

```bash
python scripts/train.py --algorithm masac --scenario f1_m5 --episodes 50000 --device cpu --amp off
```

## Train GPU

```bash
python scripts/train.py --algorithm masac --scenario f1_m5 --episodes 50000 --device auto --amp auto
```

Chọn GPU cụ thể:

```bash
python scripts/train.py --algorithm matd3 --scenario f1_m9 --episodes 50000 --device cuda:0 --amp auto
```

## Train với W&B

Truyền W&B API key trực tiếp vào lệnh chạy:

```bash
python scripts/train.py \
  --algorithm masac \
  --scenario f1_m5 \
  --episodes 50000 \
  --device auto \
  --amp auto \
  --wandb-api-key YOUR_WANDB_TOKEN
```

Có W&B key: metrics, low-metric episodes, error diagnostics, plots và checkpoints được gửi lên W&B. Không truyền key: training vẫn chạy và lưu local trong `runs/`.

## Chạy cả 3 thuật toán × 2 scenarios

```bash
python scripts/run_all.py --episodes 50000 --eval-episodes 5000 --device auto --amp auto --wandb-api-key YOUR_WANDB_TOKEN
```

W&B được cố định tại `uav_search_paper/uav_search_target`. Nếu không dùng W&B, chỉ cần bỏ `--wandb-api-key` khỏi lệnh.

## Paper plots và W&B diagnostics

`run_all.py` mặc định chỉ chạy hai scenario hiện tại `f1_m5` và `f1_m9`. Sau evaluation, pipeline tạo Fig. 7/10/11/12-style comparison; từng run tạo thêm Fig. 6-style scenario plot. W&B chỉ log telemetry tổng hợp: paper metrics, trạng thái toàn swarm, fixed-vs-rotor returns, reward components, RL health và throughput; không còn tạo series riêng cho từng UAV. Paper runs bật deterministic mặc định và terminal in tiến độ mỗi 100 episode.

Chi tiết key và cách đọc dashboard: [`docs/WANDB_DIAGNOSTICS.md`](docs/WANDB_DIAGNOSTICS.md).
