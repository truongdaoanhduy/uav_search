# u6 Persistent DTN + Power-Control Implementation Plan

**Goal:** Harden the homogeneous `u6` research scenario into a finite-battery, post-disaster search + DTN environment where the MARL policy jointly controls movement, transmit gating, transmit power, and next-hop selection, while UavNetSim remains the MAC/PHY transport authority.

**Scope:** `u6` only. Paper-faithful scenarios (`f1_m5`, `f1_m9`, etc.) must retain published behavior.

**Literature grounding:**
- Ao et al. root paper: 5 km map, multirotor motion/search/energy, MASAC/MATD3/MADDPG baselines.
- Du et al. 2021 UAV-assisted VDTN: store-carry-forward, intermittent links, persistent connection time; 500 KB–1 MB messages, 300 s message TTL, finite buffers.
- Miyata & Matsuzawa 2026: disaster DTN/FANET, GS-bound bundles, finite UAV batteries, 2 MB image bundles, explicit buffers, battery-aware mobility.
- Wang & Yang 2026 JUROR: CTDE, local/contact-limited observations, finite buffers, TTL, SCF, joint UAV flight + opportunistic routing.
- Cao et al. 2019 buffer-aided multi-UAV relay: jointly optimize relay selection, UAV transmit power, and trajectory; P_peak=0.4 W and initial 0.1 W in their simulation.
- JTFR / Ding et al.: trajectory-routing coupling, queue/link stability/residual energy, next-hop selection.
- Yuan & Gao 2026 MRMG: MARL jointly controls UAV movement, immediate next-hop selection, and transmit power; this directly grounds the expanded networking action while `tx_gate` remains a project adaptation.
- Zhou et al. UavNetSim-v1: open-source Python UAV-network simulator providing routing/MAC, topology, mobility/energy and performance evaluation; the project uses its persistent CSMA/CA + PHY/channel + native ACK/ARQ components while retaining MARL routing authority.

## Task 1 — Regression tests for mission semantics
Files: `tests/test_homogeneous_paper_env.py`, `tests/test_network_backends.py`, `tests/test_homogeneous_algorithms.py`.
Add failing tests for:
1. launch pads colocated with GCS site but safely separated;
2. swept segment/obstacle collision (no tunneling);
3. battery depletion disables move/sense/TX/RX/relay;
4. episode terminates on all reports delivered or all UAVs dead, otherwise truncates at horizon;
5. actor observation no longer exposes global end-to-end GCS reachability;
6. directional freshness uses only `j -> i`;
7. action dimension remains 5 but coordinate 3 is transmit power rather than byte fraction;
8. higher power increases/maintains link feasibility and costs more TX energy;
9. `u6` propulsion excludes root-paper constant `P_com`, avoiding semantic double-counting;
10. report baseline uses 1 MB, 10 MB buffer, 300 s lifetime.

## Task 2 — Geometry, battery, and termination
Files: `src/uav_search/envs/models.py`, `src/uav_search/envs/paper_env.py`, `configs/scenarios/u6.yaml`.
- Add reusable segment-circle intersection helper and use it for movement collision rejection.
- Replace launch disk with deterministic safe launch pads around the GCS on the inward-facing semicircle.
- Add `active_uavs`; dead UAVs cannot move, sense, transmit, receive, or relay and are removed from network links.
- Preserve dead UAV position for visualization.
- Terminate `u6` on mission completion or all UAVs dead; horizon remains truncation.
- Make report lifetime seconds-based (`report_lifetime_s=300`) and derive steps from `dt`.
- Use 10 MB application report buffer; keep 1 MB reports.

## Task 3 — Energy model separation
Files: `src/uav_search/envs/models.py`, `src/uav_search/envs/paper_env.py`.
- Add an option to exclude root-paper constant communication power from propulsion.
- `u6`: propulsion only + backend-reported radio TX energy.
- Legacy paper scenarios: unchanged root-paper `P_com` behavior.

## Task 4 — Power-aware 5D action
Files: `src/uav_search/envs/network_backends.py`, `src/uav_search/envs/paper_env.py`, `configs/scenarios/u6.yaml`.
New semantics:
`[move_force, move_direction, tx_gate, tx_power, recipient]`.
- Map `tx_power` from [-1,1] to [0.1, 0.4] W (research baseline anchored at the 0.1 W reference and literature-backed 0.4 W peak).
- Remove policy `tx_amount`; backend/application layer sends capacity-limited bytes from queue.
- Extend `TransmissionIntent` with `tx_power_w`.
- Data-link success and TX energy use the chosen per-intent power.
- Peer forwarding remains limited to locally known/current contact candidates; GCS coordinates are infrastructure knowledge.
- Keep MATD3 smoothing mask `[1,1,0,1,0]`.

## Task 5 — Persistent UavNetSim episode instance
Files: `src/uav_search/envs/network_backends.py`, `src/uav_search/envs/paper_env.py`.
- Add backend lifecycle `reset_episode(...)`.
- UavNetSim backend creates one SimPy/channel/node set per episode and advances its clock across RL steps instead of recreating SimPy every step.
- Update radio-node coordinates and airspace geometry each step.
- Keep MARL as next-hop authority; do not enable UavNetSim routing protocols.
- Keep application-level TargetReport/SCF/TTL separate from packet-level MAC/PHY state.
- Use native UavNetSim stop-and-wait ACK/ARQ for per-hop confirmation and retry timing through an ACK-only routing adapter. MARL remains the sole next-hop authority; autonomous UavNetSim routing protocols are not invoked.

## Task 6 — Observation cleanup and diagnostics
Files: `src/uav_search/envs/paper_env.py`.
- Replace peer own `network_state` slot with direct-GCS availability, not global BFS reachability.
- Neighbor freshness is directional (`j -> i`) only.
- Keep true GCS hop count only in critic-time/global metrics/evaluation, not actor observation.
- Expose selected TX power, active/dead UAV count, mission terminal reason, and persistent network clock in `info`.

## Task 7 — Verification and synchronization
1. Run targeted RED/GREEN tests after each task.
2. Run `pytest -q` full suite.
3. Run `python -m compileall -q src scripts tests`.
4. Run `git diff --check` and scan conflict markers.
5. Smoke-train MASAC, MATD3, MADDPG on `u6`, then force a small-batch gradient-update probe.
6. Compare feature branch to `main` and origin for divergence/conflict risk.
7. Commit to `feature/homo-u6-paper-sim`, push, fetch, and verify local HEAD == remote HEAD.

**Known intentional limitations after this plan:** continuous surrogate encoding of `tx_gate` and `recipient`; no charging station; no full native UavNetSim routing protocol; application bundle buffer remains mission-layer state while UavNetSim supplies persistent MAC/PHY timing/channel state.
