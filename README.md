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

`u6` deliberately reuses the **root paper's lightweight MPE-style simulation** instead of adding a second network simulator. It keeps the 5 km × 5 km post-disaster world, uniformly randomized targets/buildings, circular obstacles, 50-step episodes, root-paper multi-rotor 2.5D motion/energy model, and the paper's A2A LoS/NLoS/SINR/rate equations. The research adaptation changes the architecture to six identical peer UAVs and extends each continuous action to:

```text
[force, direction, transmit_gate, transmit_amount, recipient]
```

Target confirmation creates a finite-buffer report; the selected UAV can forward that data one hop per RL step to another UAV or to an adapted ground command station. The report/GCS/buffer/delivery-reward values are explicitly marked `ADAPTED_ASSUMPTION` in `configs/scenarios/u6.yaml`, because the root paper does not publish them.

Smoke train:

```bash
python scripts/train.py --algorithm masac --scenario u6 --episodes 10 --device cpu --local-only
```

The legacy `f1_m5`/`f1_m9` behavior is kept intact so paper reproduction and the homogeneous research scenario can be compared without mixing their provenance.

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
