# U6/U9 Communication-Gated Research Scenarios

## Goal
Use exactly two active homogeneous research scales: six UAVs (`u6`) and nine UAVs (`u9`). Preserve historical paper configs for regression only, while the default multi-run workflow executes only U6/U9.

## Information-flow contract
- Sensing updates only the sensing UAV's local Bayesian belief map.
- Peer-scenario targets occupy unique binary belief cells; a single cell declaration cannot count multiple targets.
- A local posterior crossing 0.99 may confirm a cell; environment truth is mission bookkeeping, not actor knowledge.
- A topology/link snapshot is only a candidate-connectivity diagnostic. It never refreshes another actor's position, battery, target knowledge, or belief.
- Peer state/belief/target knowledge is transferred only when the policy enables TX, selects a peer, and the configured network backend successfully delivers a synchronization bundle.
- The synchronization bundle is modeled as 4096 bytes (`RESEARCH_ASSUMPTION`). It is complete only when the network ACKs a contiguous payload prefix of at least 4096 bytes; out-of-order later ACKs remain link metrics and cannot commit application state past an earlier gap. After full delivery, the receiver caches the sender state and performs cell-wise minimum-uncertainty fusion into the receiver's map only. Newly received state has freshness 1.0, then ages by completed unsynchronized steps and is fully masked after the cache TTL.
- Slot-start snapshots prevent information received at a relay from being forwarded a second hop in the same macro-step.
- Control-only synchronization receives no immediate communication reward; communication shaping remains tied to mission report payload/delivery.
- A pending mission report can only be materialized by its active detector/source; depleted UAVs cannot generate new report payload after death.

## Scenario scales
`u6` and `u9` share map, sensing, target, obstacle, network, battery, horizon, and reward parameters. `u9` changes only swarm size and uses a 450 m launch-pad radius so nine pads satisfy the same 141.4 m minimum separation.

## Algorithm boundary
MASAC, MATD3, and MADDPG retain the existing 6-D continuous peer action. The continuous surrogate/binning for discrete recipient selection is kept as a documented baseline limitation; changing it would be an algorithm-interface redesign, outside this scenario-only change.

## Verification
Regression tests must prove no topology-only cache refresh, no global belief fusion, no update after a failed peer sync, receiver-only fusion after success, and at most one hop of knowledge propagation per macro-step. Both U6 and U9 must pass smoke training before synchronization with GitHub.
