# Heterogeneous UAV Search — MASAC / MATD3 / MADDPG

Baselines: `MASAC`, `MATD3`, `MADDPG`.

Scenarios:
- `f1_m5`: 1 fixed-wing + 5 multi-rotor UAVs
- `f1_m9`: 1 fixed-wing + 9 multi-rotor UAVs

Paper-scenario fidelity and the remaining unpublished-parameter gaps are documented in [`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md). The environment matches the paper where values/equations are published, but it does not claim byte-for-byte reproduction of the authors' unpublished simulator.

## Cài đặt

```bash
git clone https://github.com/truongdaoanhduy/uav_search.git
cd uav_search
pip install -r requirements.txt
```

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

`run_all.py` mặc định chỉ chạy hai scenario hiện tại `f1_m5` và `f1_m9`. Sau evaluation, pipeline tạo Fig. 7/10/11/12-style comparison; từng run tạo thêm Fig. 6-style scenario plot. W&B đồng thời log paper metrics, reward components, fixed-vs-rotor returns, per-UAV communication/energy/safety/search diagnostics, và RL training-health metrics để truy ra episode/UAV/nguyên nhân metric thấp.

Chi tiết key và cách đọc dashboard: [`docs/WANDB_DIAGNOSTICS.md`](docs/WANDB_DIAGNOSTICS.md).
