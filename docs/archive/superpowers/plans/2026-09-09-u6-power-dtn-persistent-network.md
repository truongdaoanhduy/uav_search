# u6 Power-Aware Persistent DTN Implementation Plan

> Execute with TDD. Every behavior change gets a failing regression test before production code.

**Goal:** Implement the approved peer-UAV search + power-aware persistent-DTN design without changing paper-faithful scenarios.

**Architecture:** `PaperUAVEnv` remains mission/application authority; `UavNetSimBackend` becomes episode-persistent lower-layer transport. Action stays 5-D but dimension 3 changes from amount to RF power. Reports remain application bundles, while the backend packetizes/adopts chunks and simulates MAC/PHY/ACK.

---

### Task 1: Provenance/config contract
- Update `configs/scenarios/u6.yaml` with 1 MB report, 10 MB app buffer, 300 s lifetime, power range, network chunk size, same-GCS launch-pad mode.
- Add tests asserting explicit provenance and that legacy scenarios are unchanged.

### Task 2: Same-base launch, swept obstacles, finite UAV life
- Add deterministic same-GCS safe launch pads.
- Add reusable segment-circle collision helper and use it for movement.
- Add active/dead UAV state; dead UAVs cannot move/sense/communicate/relay.
- Add success/all-dead termination while preserving horizon truncation.

### Task 3: Power-aware action and actor-local state
- Replace `tx_amount` semantics with `tx_power` while keeping action dimension 5.
- Remove global GCS-path leak from actor self state.
- Fix live-contact freshness to directed receiver/transmitter semantics.
- Add local previous-hop ACK/success feedback.
- Keep MATD3 smoothing on continuous power, but not gate/recipient.

### Task 4: Episode-persistent UavNetSim transport
- Add backend episode reset/close lifecycle.
- Keep one SimPy/channel/radio state per episode.
- Add lower-layer per-node queues and packet/chunk metadata.
- Add per-packet transmit power, CSMA/PHY, ACK/retry outcome handling, and sender/ACK energy accounting.
- Keep MARL-selected recipient authoritative; do not invoke autonomous UavNetSim routing/mobility/application generation.
- Add pending/in-flight byte accounting so bytes cannot be double scheduled across RL slots.

### Task 5: Energy accounting and report lifecycle
- Use propulsion-only flight energy for `u6`; keep root-paper `P_com` formula for paper-faithful scenarios.
- Charge backend per-node radio/ACK transmit energy once.
- Change report lifetime to seconds.
- Preserve store-carry-forward and application finite buffer semantics.

### Task 6: Documentation/runner compatibility
- Update README/FILES documentation for new action semantics, persistent backend, no charging, battery death, and provenance.
- Ensure MASAC/MADDPG/MATD3 runner paths remain unchanged and all produce `(6, 5)` actions.

### Task 7: Verification and synchronization
- Run focused tests after each task, then full pytest.
- Run `compileall`, `git diff --check`, conflict-marker scan.
- Smoke-train MASAC/MATD3/MADDPG and force real gradient updates.
- Audit topology/power/battery logic with deterministic probes.
- Commit on `feature/homo-u6-paper-sim`, push, fetch, and verify local HEAD equals remote HEAD.
