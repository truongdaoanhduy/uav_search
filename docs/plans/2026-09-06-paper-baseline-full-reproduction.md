# Paper-faithful MADDPG / MATD3 / MASAC reproduction plan

**Goal:** Reproduce the simulation protocol of Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network*, using only MADDPG, MATD3, and MASAC. Hardware validation and GATAC/SA-GATAC are out of scope.

## Ground-truth hierarchy

1. **PAPER_EXPLICIT** — equations, Table I, captions, and prose in the 2026 TVT paper.
2. **PAPER_INFERRED** — values unambiguously implied by a figure/caption (for example target counts in a plotted scenario when the preceding small-scale paragraph omits the count).
3. **REFERENCE_BACKED** — values missing from the TVT paper but numerically published by a reference the TVT paper cites for that exact model/algorithm.
4. **ASSUMED** — still-undisclosed values required to make the simulator executable. These must remain isolated and documented, never relabeled as paper values.

## Implementation tasks

### 1. Reproduce all six simulation scales
- Add scenarios `(1,5)`, `(1,9)`, `(2,10)`, `(2,18)`, `(4,12)`, `(8,24)` from Figs. 7–9.
- Use 10 targets for Fig. 8 scenarios and 20 targets for Fig. 9 scenarios. Preserve the documented inference status for Fig. 7 target count.
- Expose all six through training/evaluation/run-all CLIs.

### 2. Correct A2A channel provenance
- Keep `P_com = 5 W` as the TVT paper's communication-energy term.
- Use target-paper ref. [39]'s `40 dBm = 10 W` as the reference-backed A2A transmit-power fallback.
- Keep `700 MHz`, `1 MHz`, `-100 dBm`, and urban LoS parameters under reference-backed provenance.

### 3. Make the MARL baselines structurally faithful
The TVT paper and ref. [42] assign each UAV an Actor and a centralized Critic. Refactor the current parameter-shared two-actor implementation into:
- **MADDPG:** one deterministic Actor + target Actor and one centralized Critic + target Critic per UAV.
- **MATD3:** one Actor per UAV and two centralized Critics per UAV with clipped double-Q, target policy smoothing, and delayed policy updates.
- **MASAC:** one stochastic Actor per UAV and twin centralized soft Critics per UAV.

### 4. Match ref. [44]'s MASAC entropy mechanism
- Independent trainable temperature coefficient per Actor.
- Initial alpha = 0.01.
- Target entropy = `-dim(action)`.
- Preserve TVT Table I learning rate, gamma, 50k episodes, 50 steps, and its soft-update interpretation over conflicting baseline-reference values.

### 5. Keep large-scale baselines runnable
- Centralized per-agent critics make the replay transition large at `(8,24)`.
- Add a deterministic replay-memory budget cap. The TVT paper does not publish replay capacity, so this is an implementation-resource constraint rather than a paper-value substitution.
- Record requested and effective replay capacities.

### 6. Paper-style experiment outputs
- Train/evaluate all requested algorithms under the same six scenarios.
- Preserve 5,000 random evaluation cases as the paper protocol.
- Produce training curves and aggregate target-search, rotor-energy, and broken-link metrics corresponding to Figs. 7–12, but only for MADDPG/MATD3/MASAC.

### 7. Literature/provenance audit
- Classify every uploaded paper as essential, useful-secondary, or nonessential for this reproduction.
- Record unresolved values explicitly (target radii, reward coefficients, obstacle geometry/count, initial positions/altitudes, timeslot, battery capacity, rotor power coefficients, fixed-wing aerodynamic constants) unless a directly relevant cited source publishes the exact value.
- Do not import numerical values from unrelated UAV models merely because a symbol has the same name.

### 8. Verification and sync
- Run unit tests.
- Smoke-test all three algorithms and instantiate all six scenarios.
- Run at least one short end-to-end train/evaluate/visualize path.
- Review git diff/status.
- Commit and push the verified local branch to `origin/main` so local and GitHub match.
