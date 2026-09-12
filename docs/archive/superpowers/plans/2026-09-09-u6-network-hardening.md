# U6 Network Hardening Implementation Plan

**Goal:** Make the current u6 implementation internally consistent without redesigning the approved scenario or algorithms.

**Rollback baseline:** `0c89c33a45553a540eccdfff925701a4d52a502c`, also preserved as `backup/u6-before-hardening-20260909` locally and on origin.

## Task 1 — Regression tests first
- Add UavNetSim test that an idle positive-duration macro-step advances the persistent SimPy clock.
- Add UavNetSim test that the configured reference contact range is honored at reference power while higher power may extend operational reach.
- Add deterministic-rollout test that evaluation stops on `terminated`, not only `truncated`.
- Audit the 32 KiB application/chunk coupling; do not change it unless evidence justifies a P2 packet-fidelity change.

## Task 2 — Minimal implementation fixes
- Advance persistent UavNetSim time on idle and prefailed-only steps.
- Apply the same power-scaled operational contact-range contract to UavNetSim transmission eligibility as the analytical backend.
- Stop deterministic evaluation on either all-terminated or all-truncated.
- Keep the existing 32 KiB baseline during P1 topology calibration; defer packet-size/throughput sensitivity to the dedicated P2 phase.

## Task 2b — Replay/bootstrap correctness
- Treat Gymnasium `terminated` as the replay terminal mask.
- Let `truncated` stop rollout collection without suppressing Q bootstrapping.
- Regression-test the distinction.

## Task 2c — Calibration reporting correctness
- Topology-only runs must report traffic PDR/delay/throughput as unavailable, not zero.
- Clarify that `--ranges` are reference contact ranges at 0.1 W and `--tx-power-w` affects optional traffic execution.

## Task 3 — Verification
- Run focused tests for all four regressions.
- Run full `pytest -q`.
- Run `python -m compileall -q src scripts tests`, `git diff --check`, conflict-marker scan.
- Run short real-UavNetSim u6 smoke for MASAC/MATD3/MADDPG and same-seed repeatability where existing scripts support it.
- Audit final diff.

## Task 4 — Git synchronization
- Commit only after verification.
- Push `feature/homo-u6-paper-sim`.
- Verify local HEAD, tracking SHA, and remote SHA are identical.
