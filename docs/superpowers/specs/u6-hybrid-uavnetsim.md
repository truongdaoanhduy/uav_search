# U6 Hybrid UavNetSim Design

## Goal

Keep `PaperUAVEnv` authoritative for the post-disaster search mission and UAV motion while moving peer-network transmission mechanics to a pluggable backend. The `u6` research scenario must remain leaderless and retain the five-dimensional continuous action `[move_x, move_y, tx_gate, tx_amount, recipient]`.

## Responsibilities

- `PaperUAVEnv`: target/building generation, motion, sensing, target report creation, finite application buffers, rewards, MARL step semantics.
- `AnalyticalNetworkBackend`: preserve the existing root-paper analytical peer channel for regression/ablation.
- `UavNetSimBackend`: use the pinned UavNetSim v2 code for CSMA/CA contention, A2A PHY/SINR/interference, unicast packet delivery, delay/PDR/throughput and communication energy.
- MARL policy: selects whether to transmit, how much data to attempt, and the immediate next-hop recipient. UavNetSim routing protocols must not override `recipient`.

## UavNetSim pin and mode

- Repository: `https://github.com/Zihao-Felix-Zhou/UavNetSim.git`
- Commit: `04daafb815eb377409b40b285574eeb62b9a8d58`
- Distribution version at that commit: `2.0.0`
- Default integration channel mode: `a2a`, which uses UavNetSim's A2A path-gain model without requiring Sionna RT scene compilation during normal MARL training.
- Full `sionna-rt` remains an optional future validation mode; it is not required by the fast training backend.

## Causality

Bytes received at a relay during one MARL macro-step are not eligible to be forwarded until the next macro-step. `PaperUAVEnv` enforces this by freezing application-buffer eligibility at the beginning of every step.

## Fast packet abstraction

For training speed, each transmitting UAV produces at most one aggregate UavNetSim data frame per MARL step. The frame length equals the policy-requested byte budget capped by the current UavNetSim link bit-rate and the macro-step duration. UavNetSim still resolves CSMA/CA backoff, channel occupancy, PHY interference and delivery. Application reports remain byte-addressable in `PaperUAVEnv`.

## Metrics

Expose at minimum: backend name, attempted bytes, delivered bytes, byte PDR, throughput, mean successful-link delay, PHY failures, network communication energy, report delivery count and mission delivery rate.
