# U6/U9 Scenario Semantics Hardening Design

**Date:** 2026-09-11
**Scope:** Homogeneous peer scenarios `u6` and `u9` only. Keep MASAC, MADDPG, and MATD3 implementations and the six-dimensional action contract unchanged.

## Goal

Remove information leakage and physically inconsistent sensing, make altitude, DTN lifetime, buffering, routing, communication power, safety observations, and cooperative reward semantically meaningful, while preserving the already-correct one-hop, communication-gated UavNetSim flow.

## Sensing and confirmation

- Peer observations must never expose target coordinates or identities before a target is independently confirmed. The legacy target-distance block is omitted in homogeneous-peer mode; the anonymous local belief patch remains.
- Continuous sensing enumerates all grid cells intersected by the physical circular ground footprint using the UAV's exact planar coordinate.
- Whether a target contributes occupancy evidence is determined by its exact Euclidean distance from the UAV, not merely by sharing an intersected grid cell.
- A posterior threshold alone creates only a suspected cell. A cell may be confirmed only after an active UAV records a positive low-altitude/fine-grained measurement there. A true target report can only originate from that independent fine observation.
- Cognitive shaping is signed entropy-potential change, so an uncertainty oscillation has zero undiscounted net reward instead of repeatedly earning positive reward.

## DTN lifecycle and scheduling

- Report lifetime starts when the target is confirmed, including time spent waiting for buffer space.
- Reports are scheduled earliest-deadline-first, with target index as deterministic tie-breaker.
- An expired report is an irreversible mission failure and produces `terminated=True`; the fixed horizon remains `truncated=True`.
- U6/U9 buffers hold three complete 1 MB reports, making finite-buffer behavior reachable under the current ten-report workload.
- Failed report attempts are tracked separately from delivered bytes and receive the configured communication failure penalty.

## Cooperative reward

All agents receive the same target-confirmation, final-delivery, and expiry terms. Local information gain, motion energy, safety, and actual hop progress remain local shaping signals. This gives sources and relays credit for eventual team delivery without changing any algorithm class.

## Observation contract

Peer-mode observations contain only locally available or onboard-sensor information:

- own normalized pose/velocity/battery/direct-GCS state;
- cached peer pose, battery, freshness, queue occupancy, contact degree, and GCS progress received in a successful sync;
- per-report local knowledge: known flag, held fraction, remaining lifetime, and locally known GCS delivery progress;
- episode time remaining, queue state, current min/max-power link summaries, previous attempted/successful TX and selected power;
- anonymous 3x3 belief patch;
- nearest-obstacle and nearest-active-UAV proximity vectors within the configured sensor range.

Altitude and velocity use vehicle-specific ranges rather than map/fixed-wing scaling. Synchronization transmits a deterministic 8-bit quantized belief map plus metadata, validates that the encoded payload fits the configured 4 KiB budget, and fuses only the decoded snapshot after complete delivery.

## Power/link semantics

Topology diagnostics may continue using the maximum-power reachability graph, but actor observations expose both minimum- and maximum-power link availability so the policy can see when power changes feasibility. Actual transmission success remains determined by UavNetSim using the selected action power.

## Non-goals

- No action dimension is removed.
- No routing action is added for selecting a report.
- No optimizer, network architecture, loss, replay buffer, or algorithm hyperparameter is changed.
- Legacy heterogeneous paper scenarios retain their published observation and reward flow.

## Verification

Each semantic defect gets a regression test that fails on the pre-change code, then passes after the minimal implementation. Run targeted sensing/DTN/environment tests, the complete pytest suite, deterministic smoke training for all three algorithms, and verify the pushed branch SHA equals local HEAD.
