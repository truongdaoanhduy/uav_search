# Paper / Code Conflict Audit

Authoritative paper: Ao et al., *Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network*, IEEE TVT 2026.

Current active reproduction scope is deliberately limited to `f1_m5` and `f1_m9`, and to the three requested baselines `MASAC`, `MATD3`, and `MADDPG`.

## No current conflict for published scenario items

- 5 km x 5 km task domain.
- Uniform-random targets and buildings; buildings represented as circles.
- One fixed-wing relay/leader with five or nine multi-rotor search UAVs for the active Fig. 7 scenarios.
- Fixed-wing / multi-rotor masses, speed limits, maximum acceleration, 1 Mbps minimum communication rate, 5 W communication-energy power, 10% safe battery level, and energy reward scale 0.2.
- 50 steps per episode, 50,000 paper training rounds, learning rate 0.001, discount 0.99.
- Paper soft-update `epsilon=0.99` is represented as code `tau=0.01` because the two update conventions are algebraically equivalent.
- Star rotor-to-leader topology and fixed-wing mesh topology, with rate-threshold link availability.
- Per-UAV Actor/Critic parameterization for the baselines; MATD3/MASAC use twin centralized critics.

## Deliberate scope differences, not accidental conflicts

- GATAC and SA-GATAC are not implemented because the requested baseline scope is MASAC/MATD3/MADDPG only.
- Only Fig. 7 small-scale scenarios are active. Four larger scenario configs exist for later work but are not default runs.
- `run_all.py` keeps a 10-episode development default. This is intentionally NOT the paper protocol: a paper-scale comparison must pass `--episodes 50000`; 5,000 evaluation cases are then used by default. Running `run_all.py` with no `--episodes` argument is therefore a smoke/development run, not a paper reproduction run.
- W&B aggregate diagnostics and deterministic runtime controls are engineering instrumentation, not paper model variables.

## Remaining paper under-specification / approximations

These prevent an exact numerical reproduction of the authors' curves until author source/supplementary parameters are available:

- Fig. 7 target count is not printed explicitly; current `10` is `PAPER_INFERRED`.
- Time-slot duration `dt` is assumed.
- UAV initial coordinates/distribution beyond the documented implementation margin are assumed.
- Fixed-wing and multi-rotor simulation altitudes are assumed.
- Building counts and radius distribution are assumed.
- `D_detect`, `D_found`, `D_detect-f`, `D_safe`, reward coefficients and distance stabilizer are not numerically published.
- Multi-rotor propulsion coefficients and battery capacity are not numerically published.
- Fixed-wing aerodynamic/yaw constants required for an exact integrator are not published.
- The code is a custom MPE-style/Gymnasium simulator rather than the authors' unpublished OpenAI MPE scenario source.
- Dynamics are 2.5D with fixed type-specific altitude rather than a complete numerical 3D dynamics implementation.
- Fixed-wing actions are mapped to bounded yaw-rate and target-speed controls; the root paper defines `{gamma_dot, F}` and Newton-Euler/yaw dynamics, but does not publish the aerodynamic/yaw constants needed to integrate Eqs. (9)-(10) exactly.
- Multi-rotor actions preserve the paper `{F, theta}` semantics but map them directly to bounded planar acceleration; the exact mass/drag force integration of Eq. (12) is therefore approximated.
- The two paper observation spaces are embedded into one padded common-width vector for the baseline implementations; type-inapplicable fields are zero-masked. This preserves the published information content but is not evidence that the authors used the same tensor layout.
- Baseline-specific replay size, batch size, hidden sizes, warmup, exploration noise, and MATD3/MASAC implementation details not given by the root paper rely on reference-backed or explicit implementation choices.

## Resolved internal conflicts

- `P_com=5 W` and A2A `P_tx=10 W` are now separated: the former is root-paper Table I; the latter is reference-backed from root-paper Ref. [39].
- Old `assumed.n_targets` / `assumed.n_obstacles` generic values were removed; scenario YAML files are the only count source.
- Current docs no longer claim type-shared Actors; each UAV has its own Actor/Critic.
- Current docs no longer claim a fixed MASAC entropy coefficient; automatic temperature tuning is implemented.
- Old design/plan files that describe superseded behavior are explicitly marked historical.
- W&B no longer exposes per-UAV telemetry or a `worst_agent`; only swarm/group aggregates are remote-tracked.
