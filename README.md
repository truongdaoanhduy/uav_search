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

`u6` uses the approved **full-3D search + belief sensing + persistent DTN + power-control design**. `PaperUAVEnv` remains authoritative for the 5 km × 5 km post-disaster mission, six homogeneous multi-rotor UAVs, 3D motion, altitude-aware sensing, local Bayesian belief maps, finite report buffers/TTL, and reward calculation. The paper-reproduction scenarios remain unchanged. `u6` uses a literature-backed 600 s/600-step development horizon, a literature-backed edge-GCS/common-base concept, and a **fixed 2.0 km one-hop candidate radius**. RF transmit power does not enlarge that hard radius; UavNetSim decides PHY/MAC success, retry timing, delay and radio energy for attempts inside the candidate radius. The default backend is **UavNetSim**, pinned to commit `04daafb815eb377409b40b285574eeb62b9a8d58` (distribution version 2.0.0).

The `u6` continuous Box action is:

```text
[horizontal_thrust, heading, vertical_acceleration, transmit_gate, transmit_power, recipient]
```

The physical altitude levels are `[50, 100, 150] m`. At the nearest low/mid/high level the sensor uses FOV sizes `[1, 5, 9]`, detection probabilities `[0.9, 0.8, 0.7]`, false-alarm probabilities `[0.1, 0.2, 0.3]`, prior belief `0.5`, and confirmation threshold `0.99`. The sensing numbers come from Liu et al. (Drones 2024); mapping those abstract low/mid/high sensing levels onto the physical 50/100/150 m flight levels is explicitly a research adaptation. Target confirmation no longer uses the legacy `D_found` shortcut in `u6`.

The MARL policy decides whether to transmit, the RF transmit power, and the immediate recipient/next hop. The application offers capacity-limited queued bytes; byte amount is **not** a policy action. UavNetSim does **not** overwrite the learned next hop in this adapter. One persistent SimPy/UavNetSim MAC/PHY instance is kept for the whole episode, and native CSMA/CA plus stop-and-wait ACK/ARQ determine delivery, retries, delay, PDR, throughput and radio TX energy. Data received by a relay cannot be forwarded again until the next MARL step. Data/retry energy is charged to the sending UAV; native ACK TX energy is charged to the UAV that sends the ACK, while GCS radio energy is excluded from UAV battery accounting. Actor peer slots expose only live one-hop or stale cached neighbor state; centralized critics may train from the joint collection of decentralized observations.

Application-level `TargetReport` queues, finite buffers, TTL and store-carry-forward remain in `PaperUAVEnv`; UavNetSim supplies persistent lower-layer MAC/PHY timing/channel state. MARL remains the sole routing authority, so autonomous UavNetSim routing protocols are intentionally not invoked.

Install the pinned lightweight UavNetSim backend before running `u6`:

```bash
./scripts/install_uavnetsim.sh
```

The training path intentionally uses UavNetSim `CHANNEL_MODE=a2a`, which exercises UavNetSim's A2A PHY/channel and CSMA/CA without requiring the heavy Sionna RT scene worker on every MARL step. The original analytical paper-style peer channel remains available as `network_backend: analytical` for regression/ablation.

Smoke train one baseline:

```bash
python scripts/train.py --algorithm masac --scenario u6 --episodes 10 --device cpu --local-only
```

Run the requested three baselines on `u6` with one command:

```bash
python scripts/run_u6.py --episodes 10 --device auto --amp auto --seed 44 --local-only
```

Calibrate the `u6` network topology before large training runs (no learning; real UavNetSim):

```bash
python scripts/calibrate_u6_network.py --ranges 1000 1500 2000 2500 --seed-start 44 --num-seeds 50 --steps 600
```

The `--ranges` values are **fixed peer/GCS one-hop candidate radii in meters**; they are not sensing ranges and are not scaled by RF transmit power. The default calibration is topology-only and drives deterministic-by-seed **3D random waypoints**, including altitude changes, so range/GCS geometry is measured under the current 3D mission semantics without packet-load confounding. Add `--traffic` for packet-level PDR/delay/energy checks; `--tx-power-w` then changes PHY behavior inside the selected candidate radius. Outputs are written as per-episode CSV plus aggregate JSON under `runs/calibration/`.
The final 50-seed × 600 s full-3D topology sweep produced `(direct, multi-hop, disconnected)` fractions of `(0.262, 0.051, 0.686)` at 1 km, `(0.371, 0.113, 0.516)` at 1.5 km, **`(0.493, 0.213, 0.294)` at 2 km**, and `(0.616, 0.257, 0.127)` at 2.5 km. The 2 km baseline is therefore retained as a scenario-calibrated middle regime, not claimed as a universal WLAN optimum.

Target confirmation and mission delivery are separate: altitude-dependent sensing updates each UAV's local belief map, posterior `>= 0.99` confirms a target, and confirmation creates (or queues creation of) a finite-buffer `TargetReport`. The current application baseline uses 1 MB reports, 100 MB UAV buffers and 300 s TTL from UAV-assisted VDTN literature; UavNetSim packetizes offered bytes at 1024 B while the application may offer multiple packets per 1 s RL macro-step. Relay forwarding remains limited to one application-level hop per macro-step. `info`, CSV and W&B include altitude, belief entropy, target posterior, scanned cells, positive sensor observations, information gain, confirmations, topology, queues, PDR, throughput, delay and radio energy. The `u6` battery baseline is 77 Wh = 277.2 kJ (Mavic 3 Enterprise literature reference). There is **no charging**: a depleted UAV is disabled for motion, sensing, TX, RX and relaying for the rest of the episode. See [`docs/U6_PROVENANCE.md`](docs/U6_PROVENANCE.md) for the exact paper-backed vs adapted split.

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
