# Audit kịch bản u6/u9: state, observation, action và environment flow

**Ngày audit:** 2026-09-12
**Phạm vi:** `configs/scenarios/u6.yaml`, `configs/scenarios/u9.yaml`, `PaperUAVEnv`, hai network backend và các test liên quan.
**Ngoài phạm vi:** không sửa kiến trúc, loss, replay buffer hoặc hyperparameter của MASAC, MADDPG, MATD3.

## 1. Kết luận ngắn

Kịch bản tổng thể hợp lý cho bài toán multi-UAV search + DTN:

- `u6`: 6 multirotor đồng nhất; `u9`: 9 multirotor đồng nhất.
- 10 mục tiêu tĩnh trên bản đồ 5 km × 5 km, GCS ở giữa một cạnh.
- UAV bay 3D trong khoảng 50–150 m, cảm biến nadir cập nhật belief Bayes theo độ cao.
- Mục tiêu được xác nhận từ **posterior tích lũy** và **bằng chứng dương trực tiếp ở tầng thấp/fine** của cùng UAV.
- Báo cáo 1 MB, buffer 3 MB/UAV, TTL 300 s, store-carry-forward đến GCS.
- Mỗi episode tối đa 600 slot × 1 s.
- Mỗi policy action điều khiển đồng thời chuyển động, bật/tắt truyền, công suất và next hop.

Không có chiều action nào thừa trên toàn miền trạng thái. Tuy vậy, action truyền thông đang được mã hóa bằng một continuous `Box` nhưng có quyết định ngưỡng/lượng tử hóa; đây là hạn chế tối ưu hóa đáng báo cáo, không phải lỗi logic cần đổi ngay.

Audit phát hiện và đã sửa sáu nhóm lỗi có thể làm sai kết quả thí nghiệm:

1. Điều kiện xác nhận mục tiêu bị siết sai thành “posterior và fine-positive phải mới trong cùng step”.
2. UavNetSim phát quá nhiều packet song song, chỉ commit được prefix 1 KiB và để ACK/process của action cũ tràn sang slot sau.
3. Event/MAC bookkeeping của UavNetSim có thể tăng theo thời gian.
4. Analytical backend tính cả UAV không truyền như nguồn nhiễu.
5. Reward truyền GCS trả tối đa cho chỉ một byte và peer hop có thể farm reward.
6. Safety shield đưa state về an toàn rồi reward không còn thấy action danh định nguy hiểm.

## 2. Kịch bản hiện tại

| Thuộc tính | u6 | u9 | Nhận xét |
|---|---:|---:|---|
| Multirotor đồng nhất | 6 | 9 | Không leader/follower |
| Action dimension/UAV | 6 | 6 | Cùng interface cho cả ba thuật toán |
| Observation dimension/UAV | 158 | 197 | Tăng theo số peer |
| Target | 10 | 10 | Tĩnh |
| Obstacle | 6 | 6 | Cột no-fly thẳng đứng, số lượng/bán kính là assumption |
| Launch radius | 300 m | 450 m | Tách pad để không collision tại reset |
| Altitude | 50–150 m | 50–150 m | Bay 3D |
| GCS | `[0, 2500, 0]` | như u6 | Giữa cạnh trái bản đồ |
| Horizon | 600 s | 600 s | 600 slot, mỗi slot 1 s |
| Report / buffer / TTL | 1 MB / 3 MB / 300 s | như u6 | 3 report đầy đủ/UAV |
| RF power | 0.1–0.4 W | như u6 | Policy chọn |
| Packet payload | 1024 B | như u6 | UavNetSim native |
| Reference contact envelope | 2 km tại 0.1 W | như u6 | Cần hiệu chuẩn lại |

Khác biệt cấu hình thực chất giữa u6 và u9 chỉ là số UAV và bán kính launch pad. Vì vậy các sửa lỗi semantics áp dụng đồng nhất cho cả hai.

## 3. Action: ý nghĩa và phán quyết

Action là:

```text
[a0 horizontal_thrust,
 a1 horizontal_acceleration_azimuth,
 a2 vertical_acceleration,
 a3 transmit_gate,
 a4 transmit_power,
 a5 recipient]
```

| Chiều | Mapping hiện tại | Có cần thiết? | Lưu ý |
|---|---|---|---|
| `a0` | `[-1,1] -> [0,1]`, nhân `a_max=8 m/s²` | Có | Khi thrust bằng 0 thì `a1` tạm thời không có tác dụng, nhưng `a1` không thừa toàn cục |
| `a1` | góc gia tốc ngang `[-π,π]` | Có | Đây là hướng vector gia tốc, không phải heading bền vững |
| `a2` | gia tốc đứng `[-8,+8] m/s²` | Có | Tạo trade-off altitude–FOV–Pd/Pf–link |
| `a3` | truyền khi `a3 > 0` | Có | Tránh always-on interference/energy/attempt; là quyết định nhị phân qua threshold |
| `a4` | tuyến tính sang 0.1–0.4 W | Có | Điều khiển reach/rate/energy; chỉ có tác dụng khi gate bật |
| `a5` | lượng tử hóa thành một trong `N-1` peer hoặc GCS | Có | Cần cho routing; các khoảng action tạo plateau |

### Hạn chế action còn lại

- `a3` và `a5` là biến rời rạc được nhét vào continuous `Box`. Critic phải học hàm có discontinuity/plateau; gradient actor có thể kém ổn định.
- GCS được chọn khi queue trống sẽ thành no-op; power/recipient không có hiệu lực khi gate tắt. Đây là conditional relevance, không phải action thừa.
- Có thể nghiên cứu hybrid actor (Bernoulli/Categorical + continuous motion/power) sau này, nhưng đổi đó sẽ tác động cả ba thuật toán và checkpoint, nên không thực hiện trong audit này.
- Metric `action_saturation` hiện đếm chung cả biến continuous và hybrid; dùng để chẩn đoán định tính được nhưng không nên diễn giải như saturation vật lý thuần túy.

## 4. State và observation

### State nội bộ của environment

Environment giữ ground truth để mô phỏng:

- pose, velocity, battery và trạng thái active của từng UAV;
- vị trí target/obstacle/GCS;
- belief map 50 × 50 riêng cho từng UAV;
- bằng chứng fine trực tiếp và target knowledge riêng;
- neighbor cache có tuổi;
- report generation, holder bytes, delivery bytes, TTL và pending source;
- link/rate snapshot ở min/max/current power;
- trạng thái episode-persistent của UavNetSim.

Ground truth này không được đưa thẳng vào actor observation.

### Observation cục bộ của mỗi UAV

Công thức:

```text
obs_dim = 9 + 13*(N-1) + 4*10 + 19 + 25
u6: 9 + 13*5 + 40 + 19 + 25 = 158
u9: 9 + 13*8 + 40 + 19 + 25 = 197
```

| Khối | Kích thước | Nội dung |
|---|---:|---|
| Own state | 9 | normalized xyz, vxyz, battery, local contact degree, heading slot |
| Mỗi peer | 13 | 11 cached fields nếu cache còn hạn + min/max-power link feasibility tức thời |
| Report lifecycle | 40 | mỗi target: known, held fraction, remaining TTL, known delivery progress |
| Local/network tail | 19 | queue, GCS delta/rate, previous ACK, time, TX/power/attempt, link feasibility, obstacle/nearest-peer proximity |
| Ego belief patch | 25 | crop 5 × 5; ngoài FOV vật lý hiện tại bị zero-mask |

Các điểm đúng:

- Peer pose/battery/queue/knowledge chỉ cập nhật sau khi đủ bundle sync 4 KiB.
- Belief không tự động global-fuse.
- Snapshot đầu slot ngăn “teleport” thông tin qua nhiều hop trong một step.
- Actor không nhận target coordinate/ID chưa được biết.

Các hạn chế cần công bố, chưa nên đổi vội:

- Hai min/max-power feasibility field của từng recipient là channel-sounding tức thời; đây là sensing assumption hơi lạc quan.
- `last_tx_success` không đi kèm identity của recipient trước đó, gây một phần ambiguity cho policy feed-forward.
- Topology info ở max power mô tả connectivity tiềm năng, không phải link đã thực sự được action chọn.
- Centralized critic ghép local observations; không có privileged global state. Đây vẫn là một CTDE variant hợp lệ, nhưng phải mô tả đúng.

## 5. Flow một step sau khi sửa

1. Kiểm tra đủ action cho mọi agent, kiểm tra shape và clip về `[-1,1]`.
2. Tích phân chuyển động 3D từ ba action mobility.
3. Clip boundary; rollback segment đụng obstacle.
4. Safety shield chiếu candidate pose lên safe set; ghi displacement cho từng UAV.
5. Tính propulsion energy từ **chuyển động đã được shield**, cập nhật battery/deactivation.
6. Refresh link/rate snapshot.
7. Thực thi transmission từ dữ liệu đã tồn tại ở đầu slot:
   - gate/power/recipient từ action;
   - frozen slot-start snapshot;
   - tối đa một application-level hop trong macro-step;
   - UavNetSim chỉ admit packet kế tiếp sau ACK hoặc terminal ARQ drop;
   - không còn packet của action cũ chạy qua ranh giới slot.
8. Tính radio energy, cập nhật battery/deactivation.
9. Cho report hiện có cơ hội forward cuối cùng, sau đó mới purge report quá TTL.
10. Chạy sensing tại pose mới, cập nhật Bayes, xác nhận target và enqueue/retry report.
11. Tính reward và diagnostics.
12. Tăng time; terminate khi giao đủ report hoặc toàn bộ UAV cạn pin; horizon là `truncated`.

Thứ tự **communication trước sensing** là có chủ đích: action `a_t` không được truyền measurement chỉ mới sinh ra trong transition đó; measurement mới chỉ được policy dùng từ `t+1`. Thứ tự **forward trước expiry** cũng hợp lý cho store-carry-forward vì bundle có một cơ hội dịch vụ cuối ở deadline.

## 6. Lỗi đã sửa

### 6.1 Xác nhận mục tiêu tích lũy

**Lỗi:** code tạm thời đòi posterior vượt 0.99 và fine-positive phải xảy ra cùng step. Điều này mâu thuẫn với repeated Bayesian scan: một update sau có thể làm measurement flag mất đi dù UAV đã có bằng chứng fine hợp lệ.

**Sửa:** xác nhận ban đầu dùng posterior tích lũy + persistent direct fine evidence của cùng UAV. Chỉ khi report đã expiry mới bắt buộc một fine-positive mới để tạo generation/TTL mới.

### 6.2 Ranh giới slot UavNetSim

**Lỗi:** một intent lớn tạo hàng trăm process packet song song; ACK có thể đến không theo prefix và process còn sống sau khi step trả về. Trường hợp 250,000 B từng ACK 229,299 B ở lower layer nhưng application chỉ commit được 1,024 B; slot rỗng kế tiếp vẫn nhận work cũ.

**Sửa:**

- giữ một simulator/clock/RNG/channel xuyên episode;
- mỗi intent chỉ có tối đa một packet outstanding;
- chunk sau chỉ admit sau native ACK hoặc terminal drop sau ARQ;
- service-time guard không admit round không thể settle trong slot;
- commit luôn là contiguous ACKed prefix;
- xóa packet completion/MAC wait bookkeeping sau terminal state;
- event history chỉ giữ event của slot hiện tại.

Probe sau sửa truyền khoảng 200 KiB/slot trên link tốt, report 1 MB hoàn tất trong 5 slot; không còn pending completion sau mỗi step.

### 6.3 Nhiễu analytical backend

**Lỗi:** tất cả UAV nằm trong geometry đều bị tính là interferer dù không có transmission intent.

**Sửa:** chỉ sender đang phát đồng thời mới gây nhiễu, với đúng power action của nó. Test bổ sung đồng thời kiểm tra UAV idle không làm giảm rate và sender active thực sự vẫn gây nhiễu.

### 6.4 Reward communication

**Lỗi:** 1 byte đến GCS có thể nhận cùng `+5` như một lượng dữ liệu lớn; peer hop dương có thể khuyến khích vòng lặp/farming.

**Sửa:**

- GCS progress: `5 * newly_ACKed_report_bytes / 1_MB`;
- peer forwarding không nhận positive hop reward;
- mỗi peer-directed action chịu cost 0.1, kể cả control-only sync;
- final delivery vẫn là team reward 20;
- failed peer report giữ penalty thất bại cộng attempt cost.

Các hệ số 0.1/5/20 là calibration của dự án, không được trình bày như hằng số paper.

### 6.5 Safety shield và reward

**Lỗi:** reward chỉ nhìn realized state sau shield nên action danh định nguy hiểm có thể nhận safety reward bằng 0.

**Sửa:** ghi displacement do shield theo UAV; safety reward trừ thêm normalized correction. Intervention count là số UAV thực sự bị sửa, không phải số vòng projection/numerical iteration.

### 6.6 API/docs

Docstring từng gọi class là PettingZoo ParallelEnv dù interface thực tế là custom Gymnasium-like dict API. Mô tả đã được sửa; không thay API runtime.

## 7. Những việc còn cần hiệu chuẩn/ablation

Các mục sau không phải bug có đáp án duy nhất nên audit giữ nguyên:

1. Rerun topology sweep 1/1.5/2/2.5 km, 50 seed × 600 s sau thay đổi power-scaled envelope và A2G model. Số cũ chỉ là historical evidence.
2. Sweep reward: delivery `5/10/20/40`, peer attempt cost, GCS progress scale, safety-correction scale, energy cost.
3. Horizon sensitivity 300/600/900/1200 s.
4. Obstacle count/radius sensitivity; sáu obstacle và 80–220 m là assumptions.
5. So sánh continuous-hybrid encoding với actor hybrid thật nếu đây trở thành đóng góp nghiên cứu.
6. Nếu cần Markov observability chặt hơn, thêm previous recipient identity hoặc recurrent policy—việc này đổi observation/checkpoint.
7. Kiểm tra buffer 3 MB và TTL 300 s bằng workload sensitivity.
8. Không diễn giải max-power topology metrics như realized policy connectivity.

## 8. Paper được dùng

Tất cả paper trực tiếp dùng trong audit này **đã có trong thư mục của bạn**; không cần bạn tải thêm:

| Paper | Vị trí local | Vai trò |
|---|---|---|
| Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network* | `01_active_scenario/` | miền 5 km, target/building, reward/safety nền |
| Liu et al., *Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios* | `01_active_scenario/` | repeated Bayes, altitude sensing, threshold và fine confirmation |
| Wang & Yang, *Joint UAV Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks (JUROR)* | `01_active_scenario/` | finite buffer, TTL, store-carry-forward, flow motion/routing/expiry |
| Zhou et al., *UavNetSim-v1* | `01_active_scenario/` | MAC/PHY/ACK/ARQ, 2 Mbps, payload 1024 B |
| Wang et al., *Multi-UAV Collaborative Maritime Search via Deep Reinforcement Learning* | `01_active_scenario/` | communication gate và attempt penalty |
| Du et al., *A Routing Protocol for UAV-Assisted Vehicular Delay Tolerant Networks* | `02_supporting/` | 1 MB report và TTL 300 s |
| *Joint Trajectory and Communication Design for Buffer-Aided Multi-UAV Relaying Networks* | `02_supporting/` | power 0.1–0.4 W và slot-based relaying |
| *Drone delivery problem with multi-flight level* | `02_supporting/` | mốc độ cao 50/100/150 m |
| *Reinforcement Learning-Based Dynamic Coverage Control of Multi-Rotor UAVs With Safety Priority* | thư mục gốc | safety filter và corrective/buffer reward |

Không đọc thêm full paper/PDF bên ngoài cây `/home/aduy/Documents/NCKH/uav_research_paper` để đưa ra các sửa đổi này. Tra cứu web/plugin chỉ dùng để kiểm tra metadata/DOI và API contract, không dùng một paper ngoài thư mục làm căn cứ mới.

## 9. Kiểm chứng

- Hai lượt `pytest -q` độc lập trước đồng bộ đều đạt **285 passed, 3 skipped** (109.63 s và 115.28 s).
- Regression mới: confirmation tích lũy; idle/active interference; contiguous closed-slot prefix; không có late work; byte-proportional reward; shield correction.
- Stress probe u9 với 9 intent đồng thời (2,250,000 B requested): có đủ 9 outcome, commit 129,024 B; hai lần chạy cùng seed cho kết quả giống hệt; sau busy slot và idle slot đều còn 0 pending completion, 0 MAC bookkeeping entry và 0 late event.
- `python -m py_compile src/uav_search/envs/paper_env.py src/uav_search/envs/network_backends.py`: pass.
- `git diff --check`: pass.
- Không sửa code thuật toán MASAC, MADDPG hoặc MATD3.
- Commit/push chỉ được thực hiện sau khi toàn bộ các cổng kiểm chứng trên đã xanh.

## 10. File thay đổi chính

- `src/uav_search/envs/paper_env.py`
- `src/uav_search/envs/network_backends.py`
- `configs/scenarios/u6.yaml`
- `configs/scenarios/u9.yaml`
- `tests/test_scenario_audit_regressions.py`
- `tests/test_network_backends.py`
- `tests/test_u6_scenario_semantics.py`
- `tests/test_u6_scenario_hardening_v2.py`
- `README.md`
- `docs/U6_PROVENANCE.md`
- `docs/PAPER_FIDELITY.md`

Các thay đổi có sẵn lúc bắt đầu audit được giữ lại, không bị reset hoặc ghi đè.

Lưu ý: một số file trong `docs/superpowers/plans/2026-09-09-*` vẫn ghi lại prototype cũ 5-D/10 MB. Chúng là nhật ký kế hoạch lịch sử, không phải contract runtime hiện tại; `README.md`, `docs/U6_PROVENANCE.md`, `docs/PAPER_FIDELITY.md`, spec hiện hành và báo cáo này mới mô tả kịch bản đang chạy.
