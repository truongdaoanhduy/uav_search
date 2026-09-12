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

Audit phát hiện và đã sửa mười lăm nhóm lỗi/ambiguity có thể làm sai kết quả thí nghiệm:

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
14. Topology/rate snapshot có thể stale trong chính transition nếu radio energy làm UAV cạn pin sau lần refresh đầu slot.
15. Dùng team `return_sum` làm reward-facing metric chính làm U9 trông lớn/nhỏ cơ học hơn U6 chỉ vì số agent khác nhau.

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
[a0 acceleration_x,
 a1 acceleration_y,
 a2 acceleration_z,
 a3 transmit_gate,
 a4 transmit_power,
 a5 recipient]
```

**Cartesian action contract:** `a0:a2` are direct normalized acceleration components. `[0,0,0]` means zero commanded acceleration. `a3` (gate) and `a5` (recipient) are hard execution semantics; training canonicalizes them before replay/critic use and uses straight-through gradients, while MASAC entropy covers only `a0,a1,a2,a4`.

| Chiều | Mapping hiện tại | Có cần thiết? | Lưu ý |
|---|---|---|---|
| `a0` | gia tốc Cartesian `a_x ∈ [-8,+8] m/s²` | Có | Zero thật sự là không gia tốc theo x; không còn bias 50% thrust tại action 0 |
| `a1` | gia tốc Cartesian `a_y ∈ [-8,+8] m/s²` | Có | Độc lập với `a_x`, tránh singularity/plateau của polar thrust–azimuth |
| `a2` | gia tốc Cartesian `a_z ∈ [-8,+8] m/s²` | Có | Tạo trade-off altitude–FOV–Pd/Pf–link |
| `a3` | truyền khi `a3 > 0` | Có | Tránh always-on interference/energy/attempt; là quyết định nhị phân qua threshold |
| `a4` | tuyến tính sang 0.1–0.4 W | Có | Điều khiển reach/rate/energy; chỉ có tác dụng khi gate bật |
| `a5` | lượng tử hóa thành một trong `N-1` peer hoặc GCS | Có | Cần cho routing; các khoảng action tạo plateau |

### Hạn chế action còn lại

- `a3` và `a5` vẫn là biến rời rạc được vận chuyển qua continuous `Box`; critic/replay dùng hard canonical values và actor dùng straight-through relaxation, nên đây là hybrid approximation chứ chưa phải categorical actor chính xác.
- Khi gate tắt, power/recipient được canonicalize về neutral values và không truyền gradient vào critic path; nhờ đó mọi lệnh `TX OFF` có cùng semantic representation thay vì tạo alias vô nghĩa.
- GCS được chọn khi queue trống vẫn là no-op ở application layer; đây là conditional relevance của routing action, không phải một transmission attempt có payload.
- Metric `action_saturation` chỉ đếm ba gia tốc continuous và power khi gate bật; gate/recipient hard semantics không còn bị tính như saturation continuous.

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

- Probe động U6/U9 xác nhận bốn summary feature là **thừa thông tin theo nghĩa đại số**: own contact-degree suy ra đúng từ các peer max-power feasibility bit; queue fraction suy ra đúng từ tổng held-fraction với report/buffer size cố định; hai tail bit `min_best_peer`/`max_best_peer` là phép OR/max của các peer feasibility bit. Sai số kiểm tra bằng `0` trên các rollout đã probe. Chúng không phải dead state vì là summary/inductive-bias có thể giúp học; xóa sẽ đổi `obs_dim` và làm checkpoint cũ không còn cùng kiến trúc, nên chỉ nên xóa sau ablation.
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
12. Tăng time; mission success/all-depleted terminate toàn đội. UAV đã inactive từ đầu step không tạo transition mới; UAV cạn pin trong chính step vẫn nhận reward của transition cuối rồi terminate riêng. Với U6/U9, deadline 600-step là finite-horizon terminal cho toàn bộ agent vì remaining time nằm trong observation; legacy paper scenarios vẫn dùng external truncation. Replay dùng `valid` mask để loại row hậu-terminal và chỉ bootstrap khi `terminated=False`.

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

**Sửa:** snapshot liveness ở đầu transition. UAV đã inactive từ đầu step nhận reward 0 và row replay của nó có `valid=0`; UAV cạn pin do action hiện tại vẫn nhận đầy đủ reward của transition đó rồi `terminated=True`. Replay/actor/critic của MADDPG, MATD3 và MASAC dùng per-agent validity mask để không học từ các row hậu-terminal. Với U6/U9, horizon quan sát được là terminal và không bootstrap; legacy paper scenarios giữ truncation để tương thích reproduction cũ.

### 6.13 Previous recipient feedback

**Lỗi:** ACK/rate/power feedback không cho biết action recipient nào tạo ra kết quả, trong khi peer-mode own heading slot luôn bằng 0.

**Sửa:** slot đó chứa tâm bin action của recipient trước, `last_tx_active` làm validity flag. U6/U9 vẫn 158/197 chiều; legacy heading không đổi. Checkpoint cũ tương thích shape nhưng không tương thích nghĩa, nên phải retrain cho run khoa học mới.

### 6.14 Topology stale sau khi cạn pin vì radio

**Lỗi:** topology được refresh trước pha truyền. Nếu radio energy trong chính slot đó làm một UAV cạn pin, `uav_active` chuyển sang false nhưng ma trận pair/GCS-rate và adjacency của snapshot trước truyền vẫn còn giữ link tới UAV vừa chết cho đến slot kế tiếp. Điều này có thể làm next observation và GCS-hop diagnostics nhìn thấy một link không còn tồn tại.

**Sửa:** sau khi trừ radio energy, so sánh liveness trước/sau. Chỉ khi có UAV đổi trạng thái active→inactive mới refresh topology lần nữa trước TTL/sensing/next observation. Regression test ép UAV 0 cạn pin bằng radio và kiểm tra toàn bộ min/max pair-rate, GCS-rate và active count được cập nhật ngay trong cùng transition.

### 6.15 Reward metric U6/U9 không được lấy team sum làm metric chính

**Lỗi:** `return_sum` cộng reward của mọi agent nên tăng/giảm cơ học theo kích thước swarm khi team reward được broadcast. Plot training và W&B trước đây dùng `return_sum`/`paper/reward_total` làm reward-facing metric, dễ làm so sánh U6 với U9 sai nghĩa dù trainer đã dùng `return_mean` cho score.

**Sửa:** plot và paper-comparison ưu tiên `return_mean`; W&B thêm `paper/reward_mean` làm metric chính nhưng giữ `paper/reward_total` để tương thích dashboard cũ. Các metric nhiệm vụ (`targets_found`, search/delivery rate, energy, broken-link) vẫn phải là căn cứ chính cho kết luận khoa học.

### 6.16 Non-finite action có thể làm nhiễm reward/replay

**Lỗi:** `PaperUAVEnv.step()` chỉ kiểm tra shape rồi `clip` action. `NaN` đi qua `np.clip`; trong U6/U9 safety fallback có thể kéo position về hữu hạn nhưng reward và correction diagnostic vẫn thành `NaN`. `Inf` cũng bị im lặng đổi thành biên action. Một policy bị numerical instability vì vậy có thể làm nhiễm replay buffer/critic thay vì fail-fast tại nguồn.

**Sửa:** kiểm tra toàn bộ action bằng `np.isfinite` ngay sau stack/shape-check và trước bất kỳ state mutation nào. `NaN`, `+Inf`, `-Inf` đều raise `ValueError`; regression test đồng thời xác nhận position, velocity, battery và `step_count` chưa đổi sau lỗi.

### 6.17 MATD3 target smoothing phá canonical semantics khi TX OFF

**Lỗi:** target actor đã canonicalize `tx_gate=-1 => tx_power=-1, recipient=neutral`, nhưng TD3 target-policy smoothing sau đó vẫn cộng noise vào `tx_power`. Target critic vì thế nhìn thấy các action như `gate=-1, power=-0.8` mà environment/replay canonical không bao giờ thực thi. Đây là off-manifold target action và làm mất ý nghĩa conditional parameter.

**Sửa:** target smoothing chỉ thêm noise lên continuous controls của agent valid, sau đó canonicalize hybrid action lần nữa trước khi đưa vào target critic. Khi TX OFF, power và recipient trở lại đúng neutral semantics.

### 6.18 MATD3 target noise có thể hồi sinh action của UAV invalid

**Lỗi:** code cũ nhân target action với `next_valid` trước, nhưng cộng smoothing noise sau đó. UAV đã depleted/terminal bị zero action vẫn nhận noise ở `a_x/a_y/a_z/tx_power`, nên centralized target critic thấy một agent hậu-terminal tiếp tục hành động.

**Sửa:** mask noise bằng `next_valid` và áp `next_valid` lần cuối sau canonicalization. Regression test đặt agent 0 invalid và ép deterministic smoothing noise, kiểm tra toàn bộ 6 action coordinates của agent đó vẫn chính xác bằng zero trong joint target action.

### 6.19 `q_gap_abs_mean` của MATD3/MASAC đo sai đại lượng

**Lỗi:** metric tên `q_gap_abs_mean` trước đây tính `abs(mean(critic1_loss) - mean(critic2_loss))`. Hai critic có thể cho `Q1=+1`, `Q2=-1` nhưng cùng MSE, khiến metric báo 0 dù critic disagreement thực là 2. Điều này không đổi policy update nhưng làm sai diagnostic/W&B interpretation.

**Sửa:** metric hiện lấy `mean(|Q1-Q2|)` trên các transition valid cho từng agent rồi trung bình qua agent. Regression test dựng Q1/Q2 đối xứng để phân biệt rõ Q-disagreement với loss-disagreement cho cả MATD3 và MASAC.

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
| Sun et al., *Multi-Agent Reinforcement Learning Based on Hybrid Action Representation for UAV Swarms' Integrated Communication and Control* | thư mục gốc | bằng chứng UAV-specific rằng trajectory/power là continuous trong khi resource/association là discrete; hỗ trợ kết luận action hiện tại là hybrid approximation và nên ablation bằng actor hybrid thật |

Hai paper ngoài cây local đã được tra cứu trong audit trước và phải được báo riêng:

| Paper ngoài local | Link chính thức/toàn văn | Vai trò trong quyết định |
|---|---|---|
| Khan, Yanmaz, Rinner, *Information Merging in Multi-UAV Cooperative Search*, ICRA 2014 | [PDF tác giả](https://pervasive.uni-klu.ac.at/BR/pubs/2014/Khan_ICRA2014.pdf) | Occupancy-map merging có giới hạn communication và detection error; củng cố yêu cầu không tạo certainty giả. Codec byte cụ thể vẫn là adaptation của dự án. |
| Xiong et al., *Parametrized Deep Q-Networks Learning*, 2018 | [arXiv 1810.06394](https://arxiv.org/abs/1810.06394) · [PDF](https://arxiv.org/pdf/1810.06394) | Phương án actor rời rạc-liên tục cho future work; không được dùng để đổi MASAC/MADDPG/MATD3 trong patch này. |

Sun et al. 2026 không còn thuộc nhóm ngoài-local ở lượt follow-up này vì PDF đã được tải vào thư mục gốc ngày 2026-09-12. Tra cứu web/plugin chỉ dùng để kiểm tra metadata/paper liên quan và contract Gymnasium; các constant kịch bản không được âm thầm thay bằng số ngoài paper local.

## 9. Kiểm chứng

- Fresh verification ngày 2026-09-12 sau patch cuối: `./scripts/verify_repo.sh` pass đủ 6/6 cổng (compile, shell syntax, `git diff --check`, Ruff, strict runtime/UavNetSim preflight, full regression suite). Kết quả suite: **349 passed, 3 CUDA-only skipped**; UavNetSim **2.0.0** đúng commit pin `04daafb815eb377409b40b285574eeb62b9a8d58`, SimPy **4.1.1** đúng pin, U6/U9 đều loadable với backend mặc định `uavnetsim`.
- Development `run_all.py` đã chạy trọn train → evaluate → paper-comparison cho **MADDPG/MATD3/MASAC × U6/U9** bằng backend mặc định `uavnetsim`, deterministic CPU; cả sáu run đều tạo checkpoint, evaluation output và bốn figure so sánh.
- Fuzz/invariant probe analytical chạy 12 seed × 120 bước cho mỗi U6/U9 (**2,880 transitions**), kiểm tra finite observation/reward, map/altitude bounds, obstacle exclusion, finite-buffer conservation và zero topology cho UAV inactive; không phát hiện vi phạm invariant.
- Regression bao phủ: confirmation tích lũy; endpoint-safe belief codec; fan-in reservation; depleted-recipient feedback; false-confirmation penalty; world/pair correction; per-agent depletion; mixed done dictionaries; previous-recipient feature; idle/active interference; contiguous closed-slot prefix; không có late work; byte-proportional reward; reject `NaN/±Inf` trước state mutation; MATD3 target smoothing giữ canonical TX-off semantics; target noise không hồi sinh invalid agent; `q_gap_abs_mean` đo đúng `|Q1-Q2|` cho MATD3/MASAC.
- Stress probe u9 với 9 intent đồng thời (2,250,000 B requested): có đủ 9 outcome, commit 129,024 B; hai lần chạy cùng seed cho kết quả giống hệt; sau busy slot và idle slot đều còn 0 pending completion, 0 MAC bookkeeping entry và 0 late event.
- Algorithm layer hiện có per-agent validity masking và hybrid-action canonicalization; checkpoint U6/U9 cũ không được resume cho scientific run mới.
- Push chỉ được thực hiện sau khi toàn bộ các cổng kiểm chứng trên xanh.

## 10. File thay đổi chính

Các file được sửa thêm trong lượt audit cuối này:

- `src/uav_search/envs/paper_env.py`
- `src/uav_search/algorithms/matd3.py`
- `src/uav_search/algorithms/masac.py`
- `tests/test_scenario_audit_regressions.py`
- `tests/test_action_semantics_regressions.py`
- `tests/test_algorithms.py`
- `docs/U6_U9_SCENARIO_AUDIT_2026-09-12.md`

Danh sách file chính của toàn bộ chuỗi hardening trước đó:

- `src/uav_search/envs/paper_env.py`
- `src/uav_search/envs/sensing.py`
- `src/uav_search/envs/network_backends.py`
- `src/uav_search/runner/termination.py`
- `src/uav_search/runner/train.py`
- `src/uav_search/runner/visualize.py`
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

Lưu ý: toàn bộ plan/spec superseded đã được chuyển vào `docs/archive/`. Chúng chỉ là lịch sử thiết kế, không phải contract runtime. `README.md`, `docs/README.md`, `docs/U6_PROVENANCE.md`, `docs/PAPER_FIDELITY.md`, source code và tests hiện hành mới mô tả kịch bản đang chạy.
