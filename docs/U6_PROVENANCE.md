# `u6` Provenance: Paper-Backed Baselines vs Research Adaptations

`u6` is a new homogeneous post-disaster search/networking research scenario built on top of the root-paper simulator. It is **not** claimed to reproduce one paper end-to-end. This file records exactly which numerical choices are directly backed by literature/simulator defaults and which remain adaptations or calibration choices.

## Provenance labels

- `ROOT_PAPER_EXPLICIT`: printed by the Ao et al. root paper.
- `CORE_PAPER_EXPLICIT:<paper>`: printed by a paper used as a core subsystem model.
- `UAVNETSIM_NATIVE`: taken from the pinned UavNetSim implementation/defaults.
- `LITERATURE_BACKED_BASELINE:<paper>`: a published value used as a baseline in a different but relevant UAV scenario.
- `RESEARCH_ADAPTATION`: a combination/mapping of published ideas into this scenario.
- `RESEARCH_ASSUMPTION`: no sufficiently close source was found; the value is kept explicit rather than disguised as paper-backed.
- `RESEARCH_DESIGN_CALIBRATION_REQUIRED`: engineering/reward choice that must be justified by sensitivity/calibration rather than citation alone.

## Literature-backed values currently used by `u6`

| Item | `u6` value | Status | Source / rationale |
|---|---:|---|---|
| Physical flight levels | 50, 100, 150 m | Literature-backed baseline | *Drone delivery problem with multi-flight level: Machine learning based solution approach*, Computers & Industrial Engineering 197 (2024) 110565 explicitly uses these three levels. |
| Sensor FOV sizes | 1, 5, 9 cells | Core-paper explicit | Liu et al., *Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios*, Drones 8 (2024) 378. |
| Detection probabilities | 0.90, 0.80, 0.70 | Core-paper explicit | Liu et al. 2024, low/mid/high sensor table. |
| False-alarm probabilities | 0.10, 0.20, 0.30 | Core-paper explicit | Liu et al. 2024. |
| Initial target belief | 0.5 | Core-paper explicit | Liu et al. 2024 initializes unknown search cells with target probability 0.5. |
| Confirmation threshold | 0.99 | Core-paper explicit | Liu et al. 2024, `tau_p = 0.99`. |
| Target:cognitive reward weights | 1.0 : 0.1 | Core-paper explicit | Liu et al. 2024. The extra overall `search_reward_coeff=20` used by `u6` is **not** from that paper. |
| Safe distance | 141.4 m | Literature-backed baseline | Liu et al. 2024, simulation parameter `d_safe=141.4 m`. |
| 3D continuous acceleration precedent | x/y/z acceleration | Literature-backed concept | GLIDE, Information 15 (2024) 477 uses continuous acceleration in all three axes and all UAVs starting from a base. The exact six-dimensional `u6` hybrid action is still an adaptation. |
| Common-base deployment | yes | Literature-backed concept | GLIDE 2024 and homogeneous simultaneous search/routing literature both initialize all UAVs at a single base. Separate launch pads are a project safety adaptation. |
| Ground station placement | edge of area | Literature-backed concept | Wheeb et al., Electronics 12 (2023) 1334 places the ground base station at the edge of a SAR search area and lets UAVs relay target reports toward it. The exact coordinate `[0,2500,0]` is adapted. |
| Macro step | 1 s | Literature-backed baseline | Buffer-aided multi-UAV relay literature uses 1 s trajectory slots; `u6` makes this scenario-specific so root-paper scenarios are unchanged. |
| Development horizon | 600 s = 600 x 1 s | Literature-backed baseline + calibration required | Wheeb et al. 2023 uses 600 s in a UAV SAR FANET study. Map/team differences mean sensitivity analysis is still required. |
| Local candidate communication radius | 2 km | Literature-backed baseline + calibration required | *Energy–Information–Decision Coupling Optimization for Cooperative Operations of Heterogeneous Maritime Unmanned Systems*, Drones 10 (2026) 234 reports approximately 2 km WLAN radius. `u6` treats it as a fixed candidate radius, not a universal optimum. |
| RF power range | 0.1--0.4 W | Literature-backed baseline | Cao et al., *Joint Trajectory and Communication Design for Buffer-Aided Multi-UAV Relaying Networks*, Applied Sciences 9 (2019) 5524 uses 0.1 W source/initial UAV power and 0.4 W maximum UAV power. |
| UavNetSim default RF power | 0.1 W | UavNetSim native | Pinned UavNetSim `TRANSMITTING_POWER`. |
| UavNetSim data rate | 2 Mbps | UavNetSim native | Pinned UavNetSim 802.11b configuration. |
| UavNetSim SINR threshold | 6 dB | UavNetSim native | Pinned UavNetSim 802.11b configuration. |
| Packet payload | 1024 B | UavNetSim native + literature support | Pinned UavNetSim uses `AVERAGE_PAYLOAD_LENGTH=8192` bits. FANET simulation literature also explicitly uses 1024-byte packets. |
| Target report size | 1 MB | Literature-backed baseline | Du et al., IEEE OJCS 2 (2021), UAV-assisted VDTN message size 500 KB--1 MB. |
| Report TTL | 300 s | Literature-backed baseline | Du et al. 2021 explicitly uses message TTL 300 s. |
| UAV application buffer | 100 MB | Literature-backed baseline | Du et al. 2021 explicitly gives UAV buffer size 100 MB in the isolated-area UAV scenario. |
| Battery capacity | 77 Wh = 277.2 kJ | Literature-backed hardware baseline | Mavic 3 Enterprise value reported in *Energy-Aware Multilingual Vision–Language Models for Drone Smart Sensing*, Drones 10 (2026) 361. This is a hardware reference, not a root-paper constant. |
| Neighbor-cache freshness | 5 s | Research adaptation from UavNetSim native behavior | The pinned UavNetSim virtual-force neighbor table uses a 5 s entry lifetime. `u6` mirrors that lifetime in its actor-visible cache; it does not use UavNetSim's table object directly. |
| Movement + next-hop + transmit-power joint control | yes | Literature-backed concept | MRMG, arXiv:2606.06954, jointly learns UAV movement, next-hop selection and transmit-power control. The exact continuous encoding in `u6` is not copied from MRMG. |

## Deliberate research adaptations

These choices have related literature, but **no source was found that specifies the exact `u6` construction**:

| Item | Current design | Why it is not labeled paper-explicit |
|---|---|---|
| Altitude-to-sensor mapping | Liu low/mid/high profiles mapped to 50/100/150 m | Liu gives abstract levels; the drone-delivery paper gives physical levels. Combining them is a new mapping. |
| Search grid | 50 x 50 cells of 100 m on the 5 km map | Liu uses 2000 m / 20 = 100 m per cell. Preserving that density on a 5 km root-paper map is a scale adaptation. |
| 5-cell FOV geometry | center + four cardinal cells | Liu publishes FOV *size* 5 but does not textually specify this exact footprint. |
| Combined action | `[horizontal_thrust, heading, vertical_accel, tx_gate, tx_power, recipient]` | 3D acceleration and movement/next-hop/power each have precedent, but no source uses this exact six-coordinate hybrid Box action. |
| Vertical acceleration mapping | policy `[-1,1]` mapped to root-paper `a_max` | Reuses a published root constraint with a new control mapping. |
| Common-base launch pads | six separated pads on a 300 m semicircle | Literature supports a common base; the pad geometry/radius is derived to avoid artificial initial collisions. |
| Exact GCS coordinate | `[0,2500,0]` | Literature supports an edge base station, not this exact midpoint. |
| Same 2 km radius for UAV-UAV and UAV-GCS candidate links | yes | A simple project baseline; must be checked by topology calibration. |
| Static victim targets | yes | Retains the post-disaster root-task interpretation; Liu's 3D sensing paper uses moving targets. |
| Overall sensing-task reward scale | existing `search_reward_coeff=20` | Liu supports the 1.0:0.1 relative weights, not this global scale. |
| Belief-confirmation tie break | highest posterior, shortest distance, lowest ID | Deterministic project rule for simultaneous confirmations. |

## Parameters still not sourced closely enough

The following are intentionally **not promoted to paper-backed values**:

| Parameter | Current value/design | Status |
|---|---:|---|
| Number of circular obstacles in `u6` | 6 | `RESEARCH_ASSUMPTION` |
| Obstacle radius distribution | 80--220 m from legacy assumed config | `RESEARCH_ASSUMPTION` for exact numerical distribution |
| Obstacles as vertical no-fly columns | yes | `RESEARCH_ADAPTATION` |
| Launch-pad radius | 300 m | `RESEARCH_DESIGN` derived safe geometry, not literature numeric value |
| Delivery completion reward | 20 | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` |
| Per-joule reward cost | 0.001 | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` |
| Simplified propulsion coefficients (`P0`, blade/frame drag terms) | legacy assumed values | `RESEARCH_ASSUMPTION`; battery capacity is sourced, propulsion coefficients are not |
| Exact 2 km suitability for this 5 km/6-UAV topology | not yet established | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` despite literature precedent |
| 600 s sufficiency for this task | not yet established | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` despite SAR precedent |

A citation to an unrelated scenario is not enough to make these values valid. They should be calibrated/swept, replaced by a more faithful subsystem model, or left visibly assumed.

## Primary references

1. T. Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network*, IEEE Transactions on Vehicular Technology, DOI `10.1109/TVT.2025.3594534`.
2. Liu et al., *Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios*, Drones 8 (2024) 378, DOI `10.3390/drones8080378`.
3. *Drone delivery problem with multi-flight level: Machine learning based solution approach*, Computers & Industrial Engineering 197 (2024) 110565, DOI `10.1016/j.cie.2024.110565`.
4. S. et al., *GLIDE: Multi-Agent Deep Reinforcement Learning for Coordinated UAV Control in Dynamic Military Environments*, Information 15 (2024) 477, DOI `10.3390/info15080477`.
5. A. H. Wheeb et al., *Performance Evaluation of Standard and Modified OLSR Protocols for Uncoordinated UAV Ad-Hoc Networks in Search and Rescue Environments*, Electronics 12 (2023) 1334, DOI `10.3390/electronics12061334`.
6. Z. Du et al., *A Routing Protocol for UAV-Assisted Vehicular Delay Tolerant Networks*, IEEE Open Journal of the Computer Society 2 (2021) 85--98, DOI `10.1109/OJCS.2021.3054759`.
7. *Joint Trajectory and Communication Design for Buffer-Aided Multi-UAV Relaying Networks*, Applied Sciences 9 (2019) 5524, DOI `10.3390/app9245524`.
8. *Energy–Information–Decision Coupling Optimization for Cooperative Operations of Heterogeneous Maritime Unmanned Systems*, Drones 10 (2026) 234, DOI `10.3390/drones10040234`.
9. *A Hybrid Communication Scheme for Efficient and Low-Cost Deployment of Future Flying Ad-Hoc Network (FANET)*, Drones 3 (2019) 16, DOI `10.3390/drones3010016`.
10. Z. Zhou et al., *UavNetSim-v1: A Python-based Simulation Platform for UAV Communication Networks*, arXiv:2507.09852 / ICCC 2025; official repository `Zihao-Felix-Zhou/UavNetSim`.
11. Y. Yuan and S. Gao, *Learn to Access and Backhaul the Sky: Multi-Scale Radio Map Guided Multi-UAV Cooperation*, arXiv:2606.06954.
12. *Energy-Aware Multilingual Vision–Language Models for Drone Smart Sensing*, Drones 10 (2026) 361.
