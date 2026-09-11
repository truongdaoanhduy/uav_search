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
| Inter-UAV safety mechanism | discrete-time CBF-style projection/shield | Literature-backed concept + research adaptation | Safe-RL/CBF UAV literature uses CBF safety filters/shields between learned actions and plant dynamics. This implementation avoids a new QP dependency: it minimally projects unsafe one-step pair candidates onto the safe set, rechecks boundary/obstacle constraints, logs interventions, and keeps rollback only as a conservative fallback. |
| 3D continuous acceleration precedent | x/y/z acceleration | Literature-backed concept | GLIDE, Information 15 (2024) 477 uses continuous acceleration in all three axes and all UAVs starting from a base. The exact six-dimensional `u6` hybrid action is still an adaptation. |
| Common-base deployment | yes | Literature-backed concept | GLIDE 2024 and homogeneous simultaneous search/routing literature both initialize all UAVs at a single base. Separate launch pads are a project safety adaptation. |
| Number of ground targets | 10 in both U6 and U9 | Literature-backed baseline + fixed-workload comparison design | Ao et al. explicitly evaluate several 10-target post-disaster scenarios. Holding workload fixed while changing only team size makes U6→U9 a **team-size ablation**, not a load-scaled scalability claim. |
| Ground station placement | edge of area | Literature-backed concept | Wheeb et al., Electronics 12 (2023) 1334 places the ground base station at the edge of a SAR search area and lets UAVs relay target reports toward it. The exact coordinate `[0,2500,0]` is adapted. |
| Macro step | 1 s | Literature-backed baseline | Buffer-aided multi-UAV relay literature uses 1 s trajectory slots; `u6` makes this scenario-specific so root-paper scenarios are unchanged. |
| Development horizon | 600 s = 600 x 1 s | Literature-backed baseline + calibration required | Wheeb et al. 2023 uses 600 s in a UAV SAR FANET study. Map/team differences mean sensitivity analysis is still required. |
| Reference-power communication envelope | 2 km at 0.1 W | Literature-backed distance baseline + research power scaling; recalibration required | *Energy–Information–Decision Coupling Optimization for Cooperative Operations of Heterogeneous Maritime Unmanned Systems*, Drones 10 (2026) 234 reports approximately 2 km WLAN radius. `u6/u9` interpret that value at the 0.1 W reference power and apply `d_max ∝ P_tx^(1/3)` using the pinned UavNetSim urban-NLoS A2A exponent `n=3.0` as the reference-envelope scaling exponent before PHY/MAC evaluation. |
| RF power range | 0.1--0.4 W | Literature-backed baseline | Cao et al., *Joint Trajectory and Communication Design for Buffer-Aided Multi-UAV Relaying Networks*, Applied Sciences 9 (2019) 5524 uses 0.1 W source/initial UAV power and 0.4 W maximum UAV power. |
| UavNetSim default RF power | 0.1 W | UavNetSim native | Pinned UavNetSim `TRANSMITTING_POWER`. |
| UavNetSim data rate | 2 Mbps | UavNetSim native | Pinned UavNetSim 802.11b configuration. |
| UavNetSim SINR threshold | 6 dB | UavNetSim native | Pinned UavNetSim 802.11b configuration. |
| Packet payload | 1024 B | UavNetSim native + literature support | Pinned UavNetSim uses `AVERAGE_PAYLOAD_LENGTH=8192` bits. FANET simulation literature also explicitly uses 1024-byte packets. |
| Target report size | 1 MB | Literature-backed baseline | Du et al., IEEE OJCS 2 (2021), UAV-assisted VDTN message size 500 KB--1 MB. |
| Report TTL | 300 s | Literature-backed baseline + research lifecycle semantics | Du et al. 2021 explicitly uses message TTL 300 s. In this project, expiry purges the current report generation/copies and is penalized, but does not terminate the whole mission; independently confirmed information may be regenerated with a fresh TTL, following the message-level expiry spirit used in DTN work such as JUROR. |
| UAV application buffer | 3 MB (3 complete reports) | Research design, workload-coupled | Du et al. 2021 reports a 100 MB UAV buffer, but that cannot bind with this scenario's maximum 10 MB report workload. The 3 MB capacity is deliberately **not** presented as a paper value. |
| Battery capacity | 77 Wh = 277.2 kJ | Literature-backed hardware baseline + scenario calibration | Mavic 3 Enterprise value reported in *Energy-Aware Multilingual Vision–Language Models for Drone Smart Sensing*, Drones 10 (2026) 361. In 50 aggressive 3D no-TX episodes of 600 s, mean final swarm battery was 72.68% and no UAV depleted. This is a hardware reference, not a root-paper constant. |
| Neighbor-cache freshness | 5 s | Research adaptation from UavNetSim native behavior | The pinned UavNetSim virtual-force neighbor table uses a 5 s entry lifetime. `u6` mirrors that lifetime in its actor-visible cache; it does not use UavNetSim's table object directly. |
| Movement + next-hop + transmit-power joint control | yes | Literature-backed concept | MRMG, arXiv:2606.06954, jointly learns UAV movement, next-hop selection and transmit-power control. The exact continuous encoding in `u6` is not copied from MRMG. |
| Continuous sensing footprint | exact UAV XY + circular footprint/cell intersection + fractional evidence | Literature-backed geometry + research adaptation | Hu et al. 2014 defines the sensing disk from the UAV planar coordinate. Because this project uses 100 m cells with a 50 m minimum radius, it computes exact circle–cell coverage fraction and scales Bayesian likelihood evidence in log-odds space instead of treating any positive-area overlap as a full-cell scan. |
| Routing actor lifecycle/link context | TTL/age, held fraction, delivery progress, local/cached queue, contact degree, GCS progress, recipient-specific min/max-power feasibility | Literature-backed concept + research encoding | JUROR (arXiv:2608.04590) motivates contact/candidate-aware routing context; MRMG motivates radio-aware next-hop/power decisions. The exact fixed-width feature layout here is project-specific. |

## `u6` topology calibration status

The previous full-3D non-learning topology sweep used real persistent UavNetSim, 50 deterministic seeds per radius and 600 s per seed. It produced the table below **before** the September 2026 networking hardening:

| Fixed candidate radius | Historical direct node-steps | Historical multi-hop node-steps | Historical disconnected node-steps | Historical mean GCS hops | Historical mean neighbor degree |
|---:|---:|---:|---:|---:|---:|
| 1.0 km | 26.2% | 5.1% | 68.6% | 1.20 | 1.97 |
| 1.5 km | 37.1% | 11.3% | 51.6% | 1.33 | 2.60 |
| **2.0 km** | **49.3%** | **21.3%** | **29.4%** | **1.41** | **3.11** |
| 2.5 km | 61.6% | 25.7% | 12.7% | 1.37 | 3.43 |

These fractions are **not current calibration evidence anymore**. The hardened model now uses native UavNetSim A2A propagation for UAV peers, an Al-Hourani-style A2G model for every link involving the GCS, and a maximum-controllable-power topology snapshot. Because those changes alter connectivity semantics, `contact_range_calibration_status` is `RECALIBRATION_REQUIRED:POWER_SCALED_ENVELOPE`. The 2 km value is retained only as the literature-backed **0.1 W reference envelope** until the same 50-seed × 600 s sweep is rerun under the new power-scaled semantics. A one-seed verification probe after hardening was intentionally not promoted to calibration evidence.

Battery calibration used 50 deterministic 600 s episodes with aggressive random 3D motion and TX disabled to isolate propulsion. With the 77 Wh baseline, mean final swarm battery was **72.68%**, the minimum observed UAV battery was **72.58%**, and **0/50 episodes** had a depleted UAV. This is materially more constraining than the previous 1 MJ placeholder, while not forcing battery death during every 10-minute mission. Radio traffic will add further energy usage and is evaluated separately.

## Deliberate research adaptations

These choices have related literature, but **no source was found that specifies the exact `u6` construction**:

| Item | Current design | Why it is not labeled paper-explicit |
|---|---|---|
| Altitude-to-sensor mapping | Liu low/mid/high profiles mapped to 50/100/150 m | Liu gives abstract levels; the drone-delivery paper gives physical levels. Combining them is a new mapping. |
| Search grid | 50 x 50 cells of 100 m on the 5 km map | Liu uses 2000 m / 20 = 100 m per cell. Preserving that density on a 5 km root-paper map is a scale adaptation. |
| Actor belief crop under continuous sensing | fixed 5 × 5 ego crop, zero-masked outside current physical FOV | Liu fixes actor input width around its maximum discrete FOV; the active continuous physical model can intersect cells outside 3 × 3 at 150 m, so 5 × 5 is the smallest fixed crop covering the current maximum footprint. |
| Combined action | `[horizontal_thrust, horizontal_acceleration_azimuth, vertical_accel, tx_gate, tx_power, recipient]` | `action[1]` is the azimuth/direction of the horizontal acceleration vector, not a persistent vehicle heading state. 3D acceleration and movement/next-hop/power each have precedent, but no source uses this exact six-coordinate hybrid Box action. |
| Vertical acceleration mapping | policy `[-1,1]` mapped to root-paper `a_max` | Reuses a published root constraint with a new control mapping. |
| Common-base launch pads | six separated pads on a 300 m semicircle | Literature supports a common base; the pad geometry/radius is derived to avoid artificial initial collisions. For six pads spanning -75°..+75° at 30° increments and a 141.4 m minimum separation, the adjacent-chord bound requires `R >= 141.4/(2 sin 15°) = 273.2 m`; 300 m adds a small geometry margin. |
| Exact GCS coordinate | `[0,2500,0]` | Literature supports an edge base station, not this exact midpoint. |
| Same 2 km **reference-power envelope** for UAV-UAV and UAV-GCS candidate links | yes | A simple project baseline; both use the same `P^(1/3)` envelope scaling based on UavNetSim's urban-NLoS A2A exponent but different A2A/A2G propagation models. Current hardened model requires a fresh topology recalibration. |
| Static victim targets | yes | Retains the post-disaster root-task interpretation; Liu's 3D sensing paper uses moving targets. |
| Overall sensing-task reward scale | existing `search_reward_coeff=20` | Liu supports the 1.0:0.1 relative weights, not this global scale. |
| Belief fusion and cell confirmation | each UAV keeps a local posterior; `0.99` creates a suspicious cell, while final target confirmation requires the UAV's own low/fine-altitude positive evidence. Receiver-only minimum-entropy fusion occurs only after a complete peer synchronization bundle. | Liu 2024 supports high-altitude broad search followed by descent for precise target capture. The exact 50 m final-confirmation gate, 4 KiB transport, 8-bit belief serialization, and receiver-only fusion are deterministic research adaptations. |

## Parameters still not sourced closely enough

The following are intentionally **not promoted to paper-backed values**:

| Parameter | Current value/design | Status |
|---|---:|---|
| Number of circular obstacles in `u6` | 6 | `RESEARCH_ASSUMPTION`; Ao et al. explicitly use circular building abstractions and use 10 obstacles in scaled physical validation, but do not publish six as the main-simulation count |
| Obstacle radius distribution | 80--220 m from legacy assumed config | `RESEARCH_ASSUMPTION` for exact numerical distribution |
| Obstacles as vertical no-fly columns | yes | `RESEARCH_ADAPTATION`; the root paper uses 2-D circular buildings, while 3-D cylindrical obstacles are common in UAV RL literature |
| Launch-pad radius | 300 m | `RESEARCH_DESIGN` derived safe geometry, not literature numeric value |
| Delivery completion reward | 20 | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` |
| Per-joule reward cost | 0.001 | `RESEARCH_DESIGN_CALIBRATION_REQUIRED` |
| Simplified propulsion coefficients (`P0`, blade/frame drag terms) | legacy assumed values | `RESEARCH_ASSUMPTION`; battery capacity is sourced, propulsion coefficients are not |
| Exact 2 km-at-0.1 W suitability after power-scaled/A2G hardening | rerun the 1/1.5/2/2.5 km 50-seed × 600 s sweep and report sensitivity | `RECALIBRATION_REQUIRED`; the old fixed-radius sweep is historical only |
| 600 s sufficiency for learning/evaluation conclusions | topology calibration covers this horizon, but task-learning sensitivity remains advisable | literature-backed development horizon, not a root-paper constant |

A citation to an unrelated scenario is not enough to make these values valid. They should be calibrated/swept, replaced by a more faithful subsystem model, or left visibly assumed.


## Remaining assumptions: why they remain and defensible alternatives

These are intentionally not silently replaced just because a related paper contains a number. Replacing them changes the research problem and therefore requires either a sensitivity study or an explicit subsystem change.

| Current assumption | Why it is kept for now | Literature-backed alternatives | Recommended handling |
|---|---|---|---|
| 6 obstacles, radii 80--220 m | The 5 km root scenario states that buildings/targets are random and models buildings as circles, but does not publish this exact density/radius distribution. Changing density would invalidate the current topology/search calibration. | Ao et al. use 10 ground obstacles in scaled physical validation; other 3-D UAV RL studies use cylindrical obstacles but at scenario-specific radii. | Keep 6 for the current baseline; report an obstacle-density/radius sensitivity (`3/6/10`, small/medium/large radii) rather than claiming a paper-exact value. |
| Vertical no-fly columns | A 2-D root-paper circle must be given 3-D semantics after the altitude rewrite. Infinite-height columns are conservative and prevent an agent from trivially overflying every hazard. | Full finite-height cylinders are used in 3-D UAV obstacle-avoidance literature; a height map/building mesh is another option. | Keep columns as the baseline abstraction; add finite-height cylinders as a later realism ablation if obstacle-overflight is part of the research question. |
| Delivery reward = 20 | No uploaded paper uses this exact reward on the same combination of sensing + DTN + GCS delivery. It was retained to keep final delivery materially more valuable than intermediate information gain. | JUROR uses a strongly delivery-centric team reward (`alpha_d=4.0`) on its own reward scale; Ao et al. use a communication/search/energy/safety decomposition; Liu et al. use target:cognitive weights 1:0.1. | Do not copy `4.0` or another raw number across reward scales. Sweep delivery reward (e.g. 5/10/20/40) and report delivery-rate/search trade-offs. |
| Energy cost = 0.001 per J | The root paper uses a positive residual-energy reward with `tau=0.2`, whereas `u6` has a 600-step DTN mission and an explicit Joule battery. Directly reusing that positive reward would reward simply staying alive every step. | Ao et al. `tau=0.2`; JCAS-MARL models battery/resource trade-offs; Zeng et al. provide a physical rotary-wing propulsion model. | Keep the explicit cost as a transparent baseline until a reward sweep. A stronger alternative is normalized energy fraction (`Delta E / E_cap`) with a separately calibrated weight. |
| Simplified propulsion coefficients | The root paper's form is implemented but its numerical hover/blade/frame coefficients are not published. The sourced 77 Wh Mavic battery does not make those legacy coefficients hardware-faithful. | Zeng--Xu--Zhang derive the standard rotary-wing power model and publish a complete parameter set; however their reference aircraft is about 100 N and is not a Mavic 3. | Do **not** mix Zeng's 100-N aircraft power coefficients with a 77-Wh Mavic battery. Either keep the current abstract model and label it assumed, or switch the whole vehicle/energy subsystem to a single consistent Zeng-style reference platform and recalibrate battery/endurance. |
| Exact GCS coordinate `[0,2500,0]` | Literature supports an edge ground station, not the exact midpoint. Mid-edge is symmetric and avoids favoring a corner. | Edge midpoint, edge corner, or outside-map command vehicle are all used in relay/SAR studies depending on the mission. | Keep midpoint for baseline; use GCS-placement sensitivity if infrastructure geometry becomes a claimed contribution. |
| 300 m launch radius | No paper dictates 300 m. It is a deterministic geometry consequence of six same-base UAVs plus the sourced 141.4 m safety distance. | Same exact launch point (GLIDE-style), random launch disk, separate pads. | Keep 300 m because it prevents an artificial t=0 collision while preserving a common staging site; it is derived, not paper-explicit. |
| Partial-cell continuous sensing evidence | circle–cell covered-area fraction scales log-likelihood evidence | Hu et al. assume cells small relative to the FOV, which is not true at the 50 m radius / 100 m cell minimum used here. | Keep the exact continuous geometry and fractional-evidence adaptation; report it explicitly rather than claiming the Liu discrete-cell Bayes model is reproduced exactly. |
| 6-D continuous hybrid action | No single paper uses exactly `[horizontal thrust, horizontal-acceleration azimuth, vertical accel, TX gate, TX power, recipient]`. | MRMG uses movement + next-hop + power; maritime SAR uses communication gating; GLIDE gives continuous 3-D acceleration. | Keep the synthesis because it lets MASAC/MATD3/MADDPG share one interface; document it as a new adaptation and later compare with a hybrid/discrete actor if necessary. |
| Same 2 km-at-0.1 W reference envelope for UAV-UAV and UAV-GCS | Equality of the reference envelopes remains a simplification. Propagation is separated: native UavNetSim A2A for peers and Al-Hourani-style urban A2G for GCS links, while the project-level envelope scales with selected power. | Separate A2A/A2G reference envelopes, calibrate the path-loss exponent, or remove the project envelope and use pure PHY/rate reachability. | Retain as a transparent development baseline only; rerun topology/power sensitivity before reporting final results. |
| 600 s episode | It has SAR precedent and has been used for full-3D topology/battery calibration, but that does not prove learning conclusions are horizon-invariant. | 300/600/900/1200 s horizons or terminate when all reports are delivered. | Retain 600 s development baseline and include horizon sensitivity for final paper evaluation. |

### Propulsion-model note

Zeng, Xu and Zhang's rotary-wing model is a strong option when a physically self-consistent reference aircraft is desired. Their published example uses a 100 N aircraft, air density 1.225 kg/m^3, rotor radius 0.5 m, disc area 0.79 m^2, tip speed 200 m/s, rotor solidity 0.05, fuselage drag ratio 0.3, induced-power correction 0.1, hover induced velocity 7.2 m/s and profile drag coefficient 0.012. Those values should be adopted **as a complete reference-aircraft set**, not selectively mixed with the current Mavic 3 battery baseline.

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
13. Y. Zeng, J. Xu and R. Zhang, *Energy Minimization for Wireless Communication with Rotary-Wing UAV*, IEEE Transactions on Wireless Communications 18(4), 2019.
14. X. Wang and S.-R. Yang, *Joint UAV Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks* (JUROR), 2026 preprint.

15. *Reinforcement Learning-Based Dynamic Coverage Control of Multi-Rotor UAVs With Safety Priority*, IEEE Transactions on Automation Science and Engineering (2024), DOI `10.1109/TASE.2024.3420094` — external safety-filter/CBF concept source; not a local scenario paper.
16. *Multi-UAV-UGV Collision-Free Tracking Control via Control Barrier Function-Based Reinforcement Learning*, IROS 2025, DOI `10.1109/IROS60139.2025.11247497` — external CBF safety-layer concept source.
17. *Safe reinforcement learning for multi-UAV formation and obstacle avoidance control*, Scientia Sinica Informationis (2026), DOI `10.1360/SSI-2025-0048` — external multi-UAV CBF/RL safety-filter concept source.
