# u6 Power-Aware Persistent DTN Design

**Date:** 2026-09-09
**Scope:** `u6` research scenario only. Paper-faithful scenarios remain unchanged.

## Goal

Turn `u6` into a six-peer-UAV post-disaster search-and-delivery environment in which mobility, finite battery, opportunistic multi-hop communication, and transmit-power decisions genuinely affect whether target reports reach the GCS.

## Research semantics

- Six homogeneous multirotor UAVs use CTDE; there is no fixed leader/relay role.
- UAVs deploy from one emergency GCS site on separated launch pads, not arbitrary map-wide initial positions and not collision-inducing identical coordinates.
- Each UAV searches locally. Detecting a target creates local knowledge and an application-layer `TargetReport`; mission success requires reports to reach the GCS.
- Application reports use finite storage and finite lifetime. Store-carry-forward is allowed when no useful contact exists.
- MARL owns movement, transmit gating, transmit power, and immediate next-hop selection. UavNetSim must not overwrite the MARL recipient with its own routing protocol.
- UavNetSim is persistent for the whole episode and owns the per-hop lower-layer communication consequence: packet/chunk queueing, CSMA/CA, PHY/channel/interference, ACK/retry timing, packet success, delay, and transmit energy.
- Application report storage remains in `PaperUAVEnv`; the UavNetSim queue is a lower-layer packet/chunk queue. These are different layers and are intentionally both retained.

## Action

Keep a fixed 5-D Box action so MASAC, MADDPG and MATD3 remain directly comparable:

`[move_force, move_direction, tx_gate, tx_power, recipient]`

- `tx_gate <= 0`: do not admit new report bytes to the lower-layer queue.
- `tx_power`: maps continuously to a configured UAV RF power interval.
- `recipient`: retains the current continuous surrogate/binning to one peer or the GCS. This is a documented baseline limitation, not a claim of an intrinsically continuous routing decision.
- `tx_amount` is removed. The environment/network stack sends as many bytes as allowed by application data, receiver room, packet/chunk size, MAC/PHY service, and the slot.

Power baseline is literature-backed by buffer-aided multi-UAV relaying work that jointly optimizes relay selection, UAV transmit power and trajectory (initial 0.1 W, peak 0.4 W). The exact power-to-reach calibration used by `u6` remains a research parameter and must be sensitivity-tested.

## Data model

- `TargetReport`: 1,000,000 bytes baseline. UAV-assisted VDTN literature uses 500 KB–1 MB messages; the 2026 disaster DTN FANET paper additionally evaluates 2 MB low-resolution image bundles.
- Application report buffer baseline: 10 MB. This is an adaptation chosen to create finite-storage pressure; literature spans 5 MB regular-node buffers, 100 MB UAV buffers, and multi-GB UAV buffers depending on workload. It is not presented as a paper-explicit constant.
- Application report lifetime: 300 seconds, directly aligned with the UAV-assisted VDTN simulation TTL.
- Lower-layer network chunks are smaller than a report. Their size is a simulation-granularity parameter, not a mission-data-size claim.

## Network lifetime

One UavNetSim instance is created at `reset()` and advanced monotonically through the episode. Node coordinates and alive/dead state are synchronized before each network slot. It is destroyed/reinitialized only at the next environment reset.

The persistent adapter uses UavNetSim channel/MAC/packet components and keeps the SimPy clock, MAC RNG/state, ACK/retry processes, channel state, and any in-flight lower-layer work alive across RL steps. Application report queues remain in `PaperUAVEnv`. It does not enable UavNetSim's autonomous application traffic generator, mobility model, or routing policy, because those would conflict with the MARL mission controller.

## Battery and energy

- No charging station and no recharge action.
- `u6` propulsion energy excludes the root paper's constant `P_com` term when UavNetSim radio transmit energy is accounted separately; paper-faithful scenarios retain the root-paper formula.
- Data transmissions use action-selected transmit power. ACK transmissions also consume energy for the UAV that sends the ACK; GCS energy is not charged to the UAV mission battery.
- When battery reaches zero, the UAV becomes disabled: zero velocity, no movement, sensing, transmitting, receiving, or relaying. Its stored application reports remain stranded until expiration/episode termination.

## Obstacles

Circular obstacles are true impassable regions. Collision checking is swept along the complete old-to-new movement segment; endpoint-only collision checking is insufficient because it permits tunneling through an obstacle.

## Observability

- Actor self-state may include direct-to-GCS link information and its own ACK/history state.
- Peer information is available only through directed live contact or stale cached information with age/freshness.
- The actor must not receive exact global end-to-end GCS reachability/hop count. Global hop/reachability remains an evaluation metric and may be used by the centralized critic through training infrastructure if explicitly wired there later.
- A hop ACK/success signal is local feedback; global report-delivery truth is not injected into unrelated actors.

## Termination

For `u6`:
1. terminate successfully when all target reports are delivered to GCS;
2. terminate unsuccessfully when all UAVs are disabled;
3. otherwise truncate at the configured horizon (600 steps baseline).

## Explicit limitations retained

- Recipient remains a continuous surrogate for a discrete next-hop choice to keep the three requested continuous-action baselines comparable.
- The GCS uses the adapted UavNetSim A2A-style channel backend rather than a dedicated independently calibrated A2G stack.
- Power-to-connectivity calibration, application buffer size, chunk size, target sensing/confirmation ranges, target count, obstacle count, and 600-step horizon require sensitivity analysis.
- The application DTN bundle/report layer is custom mission logic; UavNetSim is the persistent lower-layer transport simulator, not the mission generator.
