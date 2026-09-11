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

Audit phát hiện và đã sửa mười ba nhóm lỗi/ambiguity có thể làm sai kết quả thí nghiệm:

1. Điều kiện xác nhận mục tiêu bị siết sai thành “posterior và fine-positive phải mới trong cùng step”.
2. UavNetSim phát quá nhiều packet song song, chỉ commit được prefix 1 KiB và để ACK/process của action cũ tràn sang slot sau.
3. Event/MAC bookkeeping của UavNetSim có thể tăng theo thời gian.
4. Analytical backend tính cả UAV không truyền như nguồn nhiễu.
5. Reward truyền GCS trả tối đa cho chỉ một byte và peer hop có thể farm reward.
6. Safety shield đưa state về an toàn rồi reward không còn thấy action danh định nguy hiểm.
7. Belief tuyến tính 8-bit tạo xác suất giả `0/1` và làm lệch prior `0.5`.
8. Nhiều sender đồng thời có thể cùng chiếm một phần dung lượng trống của receiver.
9. Chọn recipient đã cạn pin trở thành free no-op, policy không nhận được feedback thất bại.
10. False confirmation không có hậu quả reward và local evidence không được tiêu thụ.
11. Boundary clip/obstacle rollback có cùng reward với idling dù policy yêu cầu chuyển động bất hợp lệ.
12. UAV cạn pin vẫn nhận team reward và tiếp tục bootstrap như agent sống; runner không dừng được dictionary done trộn.
13. ACK feedback thiếu identity của recipient đã chọn dù own-state có một slot hằng bằng 0.

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
| Own state | 9 | normalized xyz, vxyz, battery, local contact degree, previous recipient-bin center (peer mode); legacy fixed-wing vẫn dùng heading |
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
- `last_tx_success` đi kèm previous recipient-bin center trong own-state slot 9; `last_tx_active` là validity flag. Checkpoint cũ cùng shape nhưng khác semantics, nên run khoa học mới phải retrain.
- Topology info ở max power mô tả connectivity tiềm năng, không phải link đã thực sự được action chọn.
- Centralized critic ghép local observations; không có privileged global state. Đây vẫn là một CTDE variant hợp lệ, nhưng phải mô tả đúng.

## 5. Flow một step sau khi sửa

1. Kiểm tra đủ action cho mọi agent, kiểm tra shape và clip về `[-1,1]`.
2. Tích phân chuyển động 3D từ ba action mobility.
3. Clip boundary; rollback segment đụng obstacle; ghi riêng world-constraint displacement so với candidate danh định.
4. Safety shield chiếu candidate pose lên safe set; ghi riêng pairwise-shield displacement cho từng UAV.
5. Tính propulsion energy từ **chuyển động đã được shield**, cập nhật battery/deactivation.
6. Refresh link/rate snapshot.
7. Thực thi transmission từ dữ liệu đã tồn tại ở đầu slot:
   - gate/power/recipient từ action;
   - frozen slot-start snapshot;
   - report capacity tại receiver được reserve bảo thủ theo thứ tự sender; sync bundle không chiếm application buffer;
   - chọn peer đã depleted tạo failed-attempt feedback nhưng không tạo network intent/RF giả;
   - tối đa một application-level hop trong macro-step;
   - UavNetSim chỉ admit packet kế tiếp sau ACK hoặc terminal ARQ drop;
   - không còn packet của action cũ chạy qua ranh giới slot.
8. Tính radio energy, cập nhật battery/deactivation.
9. Cho report hiện có cơ hội forward cuối cùng, sau đó mới purge report quá TTL.
10. Chạy sensing tại pose mới, cập nhật Bayes, xác nhận target và enqueue/retry report.
11. Tính reward và diagnostics.
12. Tăng time; mission success/all-depleted terminate toàn đội; mỗi UAV depleted terminate riêng và nhận reward 0; horizon chỉ truncate UAV còn sống. Runner dừng khi mọi agent đã `terminated OR truncated`, nhưng replay chỉ cắt bootstrap bởi `terminated`.

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

### 6.7 Belief codec endpoint-safe

**Lỗi:** codec tuyến tính `round(p*255)/255` biến xác suất lớn/nhỏ hữu hạn thành đúng `1/0` và biến prior `0.5` thành khoảng `0.50196`. Với minimum-entropy fusion, các endpoint giả này có thể trở thành bằng chứng gần như không đảo được.

**Sửa:** dùng signed quantized log-odds đối xứng trong một byte. `0.5` có code trung tâm chính xác; decode luôn nằm trong `(0,1)`; code thừa của số level chẵn bị reject.

### 6.8 Concurrent fan-in hữu hạn

**Lỗi:** mọi sender tính receiver room từ cùng queue đầu slot nên có thể cùng admit vượt dung lượng còn lại; application chỉ nhận sender đầu và bỏ byte đã ACK của sender sau.

**Sửa:** reserve report bytes theo sender index từ queue snapshot đầu slot. Reservation không bao gồm 4 KiB sync bundle và không được tái cấp phát sau link loss trong cùng closed slot.

### 6.9 Recipient depleted

**Lỗi:** gate-on tới peer depleted bị bỏ trước telemetry, reward bằng gate-off và policy không học tránh recipient chết.

**Sửa:** lưu recipient/power/distance/report attempt trước liveness check; không tạo network intent hay RF energy, nhưng giữ `last_tx_active=True`, `success=False` để attempt cost và failed-report penalty có hiệu lực.

### 6.10 False confirmation đối xứng

**Lỗi:** xác nhận sai một empty cell chỉ tăng metric, không làm giảm return và không tiêu thụ evidence.

**Sửa:** mỗi false confirmation mới tạo shared task event
`-search_reward_coeff * sensing_target_reward_weight` đúng một lần, đánh dấu cell đã verified, đưa belief local của detector về codec floor và clear fine-positive local. Không sinh false report động.

### 6.11 Boundary/obstacle correction

**Lỗi:** candidate bị clip/rollback có thể kết thúc đúng pose của idle và nhận cùng reward.

**Sửa:** ghi world-constraint correction sau boundary/obstacle và trước pair shield. Safety reward phạt riêng cả world correction và pair correction bằng hệ số safety hiện có; info xuất sum/max riêng.

### 6.12 Agent depletion và runner

**Lỗi:** UAV cạn pin vẫn nhận team reward và có `terminated=False` cho đến khi cả đội cạn; điều kiện runner `all(terminated) OR all(truncated)` không bao phủ trạng thái trộn.

**Sửa:** inactive UAV nhận reward 0 và terminate riêng; horizon chỉ truncate agent chưa terminate. Helper dùng chung dừng khi mọi agent có `terminated OR truncated`. Replay terminal mask vẫn chỉ dùng `terminated`, nên time-limit transition tiếp tục bootstrap.

### 6.13 Previous recipient feedback

**Lỗi:** ACK/rate/power feedback không cho biết action recipient nào tạo ra kết quả, trong khi peer-mode own heading slot luôn bằng 0.

**Sửa:** slot đó chứa tâm bin action của recipient trước, `last_tx_active` làm validity flag. U6/U9 vẫn 158/197 chiều; legacy heading không đổi. Checkpoint cũ tương thích shape nhưng không tương thích nghĩa, nên phải retrain cho run khoa học mới.

## 7. Những việc còn cần hiệu chuẩn/ablation

Các mục sau không phải bug có đáp án duy nhất nên audit giữ nguyên:

1. Rerun topology sweep 1/1.5/2/2.5 km, 50 seed × 600 s sau thay đổi power-scaled envelope và A2G model. Số cũ chỉ là historical evidence.
2. Sweep reward: delivery `5/10/20/40`, peer attempt cost, GCS progress scale, safety-correction scale, energy cost.
3. Horizon sensitivity 300/600/900/1200 s.
4. Obstacle count/radius sensitivity; sáu obstacle và 80–220 m là assumptions.
5. So sánh continuous-hybrid encoding với actor hybrid thật nếu đây trở thành đóng góp nghiên cứu.
6. Nếu cần memory dài hơn một bước cho partial observability, so sánh feed-forward hiện tại với recurrent policy; previous recipient một bước đã có trong observation.
7. Kiểm tra buffer 3 MB và TTL 300 s bằng workload sensitivity.
8. Không diễn giải max-power topology metrics như realized policy connectivity.

## 8. Paper được dùng

Các paper nền tảng dưới đây đã có trong thư mục của bạn:

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

Ba paper ngoài cây local đã được tra cứu và phải được báo riêng:

| Paper ngoài local | Link chính thức/toàn văn | Vai trò trong quyết định |
|---|---|---|
| Khan, Yanmaz, Rinner, *Information Merging in Multi-UAV Cooperative Search*, ICRA 2014 | [PDF tác giả](https://pervasive.uni-klu.ac.at/BR/pubs/2014/Khan_ICRA2014.pdf) | Occupancy-map merging có giới hạn communication và detection error; củng cố yêu cầu không tạo certainty giả. Codec byte cụ thể vẫn là adaptation của dự án. |
| Xiong et al., *Parametrized Deep Q-Networks Learning*, 2018 | [arXiv 1810.06394](https://arxiv.org/abs/1810.06394) · [PDF](https://arxiv.org/pdf/1810.06394) | Phương án actor rời rạc-liên tục cho future work; không được dùng để đổi MASAC/MADDPG/MATD3 trong patch này. |
| Sun et al., *Multi-Agent Reinforcement Learning Based on Hybrid Action Representation for UAV Swarms' Integrated Communication and Control*, IEEE LWC 2026 | [DOI 10.1109/LWC.2026.3663841](https://doi.org/10.1109/LWC.2026.3663841) | Bằng chứng UAV-specific cho hướng hybrid communication/control; chỉ là future-work provenance, trang IEEE có thể yêu cầu quyền truy cập. |

Ngoài ba nguồn này, tra cứu web/plugin được dùng để kiểm tra metadata, paper liên quan và contract Gymnasium; các constant kịch bản không được âm thầm thay bằng số ngoài paper local.

## 9. Kiểm chứng

- Lượt sạch sau residual hardening: `python -m compileall -q src tests && pytest -q` đạt **300 passed, 3 skipped trong 120.68 s**.
- Nhóm regression environment/runner mới đạt **48 passed**; smoke test MASAC/MADDPG/MATD3 đạt **25 passed, 3 skipped**.
- Regression bao phủ: confirmation tích lũy; endpoint-safe belief codec; fan-in reservation; depleted-recipient feedback; false-confirmation penalty; world/pair correction; per-agent depletion; mixed done dictionaries; previous-recipient feature; idle/active interference; contiguous closed-slot prefix; không có late work; byte-proportional reward.
- Stress probe u9 với 9 intent đồng thời (2,250,000 B requested): có đủ 9 outcome, commit 129,024 B; hai lần chạy cùng seed cho kết quả giống hệt; sau busy slot và idle slot đều còn 0 pending completion, 0 MAC bookkeeping entry và 0 late event.
- `git diff --check` và kiểm tra compile đều pass.
- Không sửa file nào trong `src/uav_search/algorithms/`; replay terminal mask vẫn chỉ dùng `terminated`.
- Push chỉ được thực hiện sau khi toàn bộ các cổng kiểm chứng trên xanh.

## 10. File thay đổi chính

- `src/uav_search/envs/paper_env.py`
- `src/uav_search/envs/sensing.py`
- `src/uav_search/envs/network_backends.py`
- `src/uav_search/runner/termination.py`
- `src/uav_search/runner/train.py`
- `src/uav_search/runner/network_calibration.py`
- `configs/scenarios/u6.yaml`
- `configs/scenarios/u9.yaml`
- `tests/test_scenario_audit_regressions.py`
- `tests/test_runner_termination.py`
- `tests/test_network_backends.py`
- `tests/test_u6_scenario_semantics.py`
- `tests/test_u6_scenario_hardening_v2.py`
- `README.md`
- `docs/U6_PROVENANCE.md`
- `docs/PAPER_FIDELITY.md`

Các thay đổi có sẵn lúc bắt đầu audit được giữ lại, không bị reset hoặc ghi đè.

Lưu ý: một số file trong `docs/superpowers/plans/2026-09-09-*` vẫn ghi lại prototype cũ 5-D/10 MB. Chúng là nhật ký kế hoạch lịch sử, không phải contract runtime hiện tại; `README.md`, `docs/U6_PROVENANCE.md`, `docs/PAPER_FIDELITY.md`, spec hiện hành và báo cáo này mới mô tả kịch bản đang chạy.
