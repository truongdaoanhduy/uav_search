# Heterogeneous UAV Search — MASAC / MATD3 / MADDPG

Baselines: `MASAC`, `MATD3`, `MADDPG`.

Active research scenarios:
- `u6`: 6 identical multi-rotor UAVs, peer-to-peer, no leader/follower.
- `u9`: 9 identical multi-rotor UAVs with the same mission/network model, used as the larger swarm scale.

Historical `f*_m*` paper-reproduction configs remain loadable for regression/reproducibility, but `run_all.py` does not include them in the active research sweep.

Paper-scenario fidelity and the remaining unpublished-parameter gaps are documented in [`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md). The environment matches the paper where values/equations are published, but it does not claim byte-for-byte reproduction of the authors' unpublished simulator.

## Cài đặt

```bash
git clone https://github.com/truongdaoanhduy/uav_search.git
cd uav_search
pip install -r requirements.txt
```


## Homogeneous `u6` / `u9` research scenarios

`u6` and `u9` use the approved **full-3D search + belief sensing + persistent DTN + power-control design**. `PaperUAVEnv` remains authoritative for the 5 km × 5 km post-disaster mission, homogeneous multi-rotor swarms, 3D motion, altitude-aware sensing, local Bayesian belief maps, finite report buffers/TTL, and reward calculation. `u6` uses six UAVs and `u9` uses nine; all other mission/network parameters are intentionally aligned except the larger `u9` launch-pad radius needed to maintain safe initial separation. This is therefore a **fixed-workload team-size ablation** (10 targets in both scenarios), not a load-scaled scalability experiment. The paper-reproduction scenarios remain unchanged. Both research scenarios use a literature-backed 600 s/600-step development horizon, a literature-backed edge-GCS/common-base concept, and a **2.0 km one-hop reference-power envelope at 0.1 W**. The operational candidate envelope scales as `d_max = 2 km * (P_tx/0.1 W)^(1/3)` before the native PHY/MAC check, so transmit power now changes both reachability near the envelope and radio energy instead of degenerating into an energy-only action. UavNetSim still decides PHY/MAC success, retry timing, delay and radio energy. The default backend is **UavNetSim**, pinned to commit `04daafb815eb377409b40b285574eeb62b9a8d58` (distribution version 2.0.0).

The `u6`/`u9` continuous Box action is:

```text
[horizontal_thrust, horizontal_acceleration_azimuth, vertical_acceleration, transmit_gate, transmit_power, recipient]
```

The physical calibration anchors are `[50, 100, 150] m`. Detection probabilities `[0.9, 0.8, 0.7]` and false-alarm probabilities `[0.1, 0.2, 0.3]` come from Liu et al. (Drones 2024) and are linearly interpolated between anchors. The physical footprint radius is continuous (`r = h tan(FOV/2)`) and absolute circle–cell intersection is evaluated at the true UAV XY pose, so the number of scanned cells is **not forced** to `[1,5,9]` near grid boundaries. Partial circle–cell overlap scales Bayesian evidence in log-odds space rather than applying a full-cell update, and the Actor receives a fixed 5×5 ego belief crop large enough for the maximum 150 m footprint. Prior belief remains `0.5`; posterior `0.99` is a suspicion threshold, while final target confirmation additionally requires the same UAV's accumulated direct low/fine-altitude positive evidence. The posterior crossing and the fine positive need not occur in the same step. Mapping Liu's abstract sensing layers onto 50/100/150 m remains a research adaptation.

The MARL policy decides whether to transmit, the RF transmit power, and the immediate recipient/next hop. The application offers capacity-limited queued bytes; byte amount is **not** a policy action. UavNetSim does **not** overwrite the learned next hop in this adapter. One persistent SimPy/UavNetSim MAC/PHY instance, clock and MAC RNG state are kept for the whole episode. Within each 1 s action slot, chunks are admitted in rounds with at most one outstanding chunk per intent; the next chunk is created only after native ACK or terminal ARQ drop, and a conservative service guard prevents untracked in-flight work from crossing the slot boundary. Native CSMA/CA plus stop-and-wait ACK/ARQ determine delivery, retries, delay, PDR, throughput and radio TX energy. Within one RL macro-step, the communication phase runs before current-step sensing: it may use the post-motion local state and the queue/belief/target knowledge available before that sensing phase, but it cannot use sensing evidence produced later in the same transition. A newly confirmed target/report therefore becomes eligible for TX at `t+1` rather than being transmitted by an action that could not have observed it. Existing reports receive their forwarding opportunity before TTL expiry. Data received by a relay cannot be forwarded again until the next MARL step. Commanded out-of-envelope transmissions remain counted attempts and consume sender probe airtime/RF energy instead of becoming free no-ops. Data/retry energy is charged to the sending UAV; native ACK TX energy is charged to the UAV that sends the ACK, while GCS radio energy is excluded from UAV battery accounting. Actor peer slots expose state received by a successful peer synchronization plus age-based stale-cache freshness, together with sender-side radio/channel-sounding estimates of recipient-specific min/max-power link feasibility; these flags are a research sensing abstraction and do not expose peer position/battery when the peer cache is stale. Centralized critics train from the concatenated collection of decentralized observations, not from a privileged full internal environment state.

Application-level `TargetReport` queues, finite buffers, TTL and store-carry-forward remain in `PaperUAVEnv`; UavNetSim supplies persistent lower-layer MAC/PHY timing/channel state. MARL remains the sole routing authority, so autonomous UavNetSim routing protocols are intentionally not invoked.

Install the pinned lightweight UavNetSim backend before running `u6` or `u9`:

```bash
./scripts/install_uavnetsim.sh
```

The training path keeps UavNetSim `CHANNEL_MODE=a2a` for native UAV-to-UAV links and CSMA/CA without requiring the heavy Sionna RT scene worker on every MARL step. Links involving the synthetic ground control station are overridden with the configured Al-Hourani-style probabilistic urban A2G gain, including GCS ACK links, while peer links remain native UavNetSim A2A. The original analytical paper-style peer channel remains available as `network_backend: analytical` for regression/ablation; its transmission-time interference includes only UAVs with an active intent in that slot, while topology snapshots remain conservative potential-link estimates.

Smoke train one baseline:

```bash
python scripts/train.py --algorithm masac --scenario u6 --episodes 10 --device cpu --local-only
python scripts/train.py --algorithm masac --scenario u9 --episodes 10 --device cpu --local-only
```

Run the requested three baselines on `u6` with one command:

```bash
python scripts/run_u6.py --episodes 10 --device auto --amp auto --seed 44 --local-only
```

Calibrate each active research scale before large training runs (no learning; real UavNetSim). The historical script name is retained for compatibility, but it now accepts both `u6` and `u9`:

```bash
python scripts/calibrate_u6_network.py --scenario u6 --ranges 1000 1500 2000 2500 --seed-start 44 --num-seeds 50 --steps 600
python scripts/calibrate_u6_network.py --scenario u9 --ranges 1000 1500 2000 2500 --seed-start 44 --num-seeds 50 --steps 600
```

The `--ranges` values are **reference-power peer/GCS one-hop envelopes in meters at 0.1 W**; they are not sensing ranges. Runtime reach scales with the configured transmit power using the scenario path-loss exponent before PHY evaluation. The default calibration is topology-only and drives deterministic-by-seed **3D random waypoints**, including altitude changes, so range/GCS geometry is measured under the current 3D mission semantics without packet-load confounding. Add `--traffic` for packet-level PDR/delay/energy checks. Outputs are written as per-episode CSV plus aggregate JSON under `runs/calibration/`.
**Calibration status:** the earlier 50-seed × 600 s sweep was produced before the September 2026 A2G/power-aware hardening and is now historical evidence only. The current code separates native peer A2A propagation from GCS A2G propagation and now computes both minimum- and maximum-controllable-power link snapshots for actor context, so those old fractions must not be reported as current calibration. The 2 km radius remains a literature-backed development baseline, but `contact_range_calibration_status` is intentionally `RECALIBRATION_REQUIRED:POWER_SCALED_ENVELOPE` until the command above is rerun on the hardened model.

Target confirmation and mission delivery are separate. In `u6`/`u9`, the nadir footprint is evaluated from the UAV's **true continuous XY position** against absolute grid cells; it is no longer rasterized from the containing-cell index. Altitude continuously changes footprint radius and interpolates Liu et al.'s `Pd/Pf` anchors. A posterior `>= 0.99` marks a suspicious cell, but a target is finalized and a report is created only after **independent low/fine-altitude positive evidence from that UAV**. Both are accumulated evidence; only report regeneration after TTL expiry requires a new fine positive. Peer actor observations expose no target coordinates/IDs before confirmation; they use anonymous belief, local report lifecycle, time, obstacle/proximity, queue, cached-neighbor, GCS-progress, and min/max-power link-feasibility features instead.
Inter-UAV separation is enforced by a lightweight **discrete-time CBF-style safety shield**: the policy proposes nominal motion, unsafe candidate pairs are minimally projected onto the safe set, and boundary/obstacle constraints are rechecked before the state reaches sensing/networking. The number of uniquely corrected UAVs and their shield displacement are logged explicitly, and normalized shield displacement contributes a negative safety reward so an unsafe nominal action is not hidden by a safe realized state. A conservative rollback remains only as a numerical/constraint fallback, not the normal safety mechanism.

Belief maps are **not globally fused**. A peer can obtain another UAV's cached state, queue/contact metadata, report-age/progress metadata, target knowledge, and belief snapshot only after the MARL policy selects that peer and the network backend successfully delivers the complete 4 KiB synchronization bundle. The belief snapshot is serialized conceptually as deterministic 8-bit values and only the decoded quantized snapshot is fused at the receiver. Frozen slot-start snapshots still prevent same-step multi-hop information teleportation. Failed transmissions, topology snapshots, and disconnected UAVs do not change peer knowledge.

The application uses literature-backed **1 MB report size** and **300 s TTL** from Du et al. 2021, but deliberately uses a **3 MB per-UAV buffer** as a workload-coupled research design (three complete reports) because the literature's 100 MB UAV buffer cannot bind in a mission with only ten 1 MB reports. TTL starts when information is confirmed, including time spent waiting for buffer space. Report scheduling is earliest-deadline-first, and an existing report gets its final forwarding opportunity before expiry is evaluated. TTL expiry purges the current report generation/copies and applies the team expiry penalty, but it does **not** terminate the whole mission; persistent historical confirmation cannot silently reset TTL. A new generation is created only after a **fresh low/fine positive observation after expiry**, which then receives a new TTL. The 600-step time limit remains `truncated`. Communication reward includes cooperative team delivery/expiry/discovery terms, a small cost for every peer-directed attempt, no positive peer-hop bonus, and GCS progress shaping proportional to newly ACKed report bytes. UavNetSim packetizes offered bytes at 1024 B; closed-slot sequential admission makes all committed application bytes one ACKed contiguous payload prefix. Relay forwarding remains limited to one application-level hop per macro-step. The `u6`/`u9` battery baseline remains 77 Wh = 277.2 kJ and depleted UAVs remain disabled for motion, sensing, TX, RX, and relaying. See [`docs/U6_PROVENANCE.md`](docs/U6_PROVENANCE.md) for the paper-backed versus adapted split.

## Train CPU

```bash
python scripts/train.py --algorithm masac --scenario u6 --episodes 50000 --device cpu --amp off
```

## Train GPU

```bash
python scripts/train.py --algorithm masac --scenario u9 --episodes 50000 --device auto --amp auto
```

Chọn GPU cụ thể:

```bash
python scripts/train.py --algorithm matd3 --scenario u9 --episodes 50000 --device cuda:0 --amp auto
```

## Train với W&B

Truyền W&B API key trực tiếp vào lệnh chạy:

```bash
python scripts/train.py \
  --algorithm masac \
  --scenario u6 \
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

`run_all.py` mặc định chỉ chạy hai research scenario `u6` và `u9` (3 thuật toán × 2 scenario = 6 runs). Các `f*_m*` config lịch sử không được đưa vào sweep này. Sau evaluation, pipeline tạo các comparison plot; từng run tạo thêm Fig. 6-style scenario plot. W&B chỉ log telemetry tổng hợp: paper metrics, trạng thái toàn swarm, fixed-vs-rotor returns, reward components, RL health và throughput; không còn tạo series riêng cho từng UAV. Paper runs bật deterministic mặc định và terminal in tiến độ mỗi 100 episode.

Chi tiết key và cách đọc dashboard: [`docs/WANDB_DIAGNOSTICS.md`](docs/WANDB_DIAGNOSTICS.md).
