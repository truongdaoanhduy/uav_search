# Paper-Faithful Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing heterogeneous UAV baseline environment match the original paper's disclosed scenario, communication topology/equations, action semantics, constraints, and rewards as closely as the published information permits.

**Architecture:** Preserve the existing parallel environment API and 2-D normalized action interface so MASAC/MATD3/MADDPG trainers remain compatible. Move physical/channel equations into stateless helpers in `models.py`; keep world state, formation topology, movement, rewards, and metrics in `paper_env.py`; record non-original-paper numeric fallbacks explicitly in YAML.

**Tech Stack:** Python 3, NumPy, Gymnasium, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-paper-faithful-environment-design.md`

## Global Constraints

- Original paper is authoritative for scenario semantics, equations, Table I, and experiment swarm sizes.
- Do not change MASAC/MATD3/MADDPG algorithm hyperparameters in this implementation.
- Preserve `PaperUAVEnv`'s parallel return tuple and per-agent action shape `(2,)`.
- Do not label a value `PAPER_EXPLICIT` unless the original paper directly publishes it.
- Keep undisclosed constants visible as `PAPER_INFERRED`, `REFERENCE_BACKED`, or `ASSUMED`.
- Physical verification/GATAC implementation is out of scope.

---

### Task 1: Configuration provenance and paper scenario sizes

**Files:**
- Modify: `configs/paper.yaml:1-66`
- Modify: `configs/scenarios/f1_m5.yaml:1-6`
- Modify: `configs/scenarios/f1_m9.yaml:1-6`
- Modify: `tests/test_config.py`
- Modify: `tests/test_env.py:12-27`

**Interfaces:**
- Consumes: existing `load_config(algorithm: str, scenario: str) -> dict`.
- Produces: `cfg["reference_backed"]` for channel constants, `scenario.targets_provenance`, and target count 10 in both small scenarios.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_paper_scenarios_use_inferred_ten_target_budget():
    for scenario in ("f1_m5", "f1_m9"):
        cfg = load_config("masac", scenario)
        assert cfg["scenario"]["targets"] == 10
        assert cfg["scenario"]["targets_provenance"] == "PAPER_INFERRED"


def test_channel_fallbacks_are_reference_backed_not_paper_explicit():
    cfg = load_config("masac", "f1_m5")
    ref = cfg["reference_backed"]
    assert ref["provenance"] == "REFERENCE_BACKED"
    assert ref["carrier_hz"] == 700_000_000.0
    assert ref["bandwidth_hz"] == 1_000_000.0
    assert ref["noise_power_w"] == 1e-13
    assert ref["los_a"] == 11.95
    assert ref["los_b"] == 0.14
    assert ref["los_extra_loss_db"] == 1.0
    assert ref["nlos_extra_loss_db"] == 20.0
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_config.py tests/test_env.py -q`
Expected: FAIL because scenarios still use 5 targets and `reference_backed` does not exist.

- [ ] **Step 3: Update YAML provenance**

Use `reference_backed` for [39]-[41] channel constants; retain undisclosed altitude, detection, safety, energy coefficients, reward coefficients and timestep under `assumed`; add `uav_init_margin_m: 300.0`; remove reward-only `boundary_penalty` and `obstacle_penalty` because the paper does not include them in Eqs. (21)-(27).

- [ ] **Step 4: Run focused tests and confirm pass**

Run: `python -m pytest tests/test_config.py tests/test_env.py -q`
Expected: PASS for configuration assertions; any environment failures caused by target-count expectations are updated to 10.

- [ ] **Step 5: Commit**

```bash
git add configs/paper.yaml configs/scenarios tests/test_config.py tests/test_env.py
git commit -m "config: align paper scenario provenance"
```

---

### Task 2: Paper A2A channel, SINR, and formation topology

**Files:**
- Modify: `src/uav_search/envs/models.py:8-24`
- Modify: `src/uav_search/envs/paper_env.py:84-96,240-243,281-300`
- Modify: `tests/test_env.py`

**Interfaces:**
- Produces:
  - `path_gain_linear(horizontal_distance_m, vertical_distance_m, cfg, *, force_los=False) -> float`
  - `communication_sinr(horizontal_distance_m, vertical_distance_m, cfg, *, interference_power_w=0.0, force_los=False) -> float`
  - `communication_rate_bps(horizontal_distance_m, vertical_distance_m, cfg, *, interference_power_w=0.0, force_los=False) -> float`
  - `PaperUAVEnv.last_pair_rates_bps: np.ndarray[N,N]`
  - `PaperUAVEnv.last_adjacency: np.ndarray[N,N]`
  - `PaperUAVEnv.rotor_leaders: np.ndarray[n_rotor]`

- [ ] **Step 1: Write failing channel/topology tests**

```python
def test_interference_reduces_sinr_and_rate():
    cfg = load_config("masac", "f1_m5")
    clean = communication_rate_bps(500.0, 140.0, cfg, interference_power_w=0.0)
    interfered = communication_rate_bps(500.0, 140.0, cfg, interference_power_w=1e-10)
    assert interfered < clean


def test_one_leader_forms_star_and_two_leaders_form_mesh():
    env = make_env()
    assert np.all(env.rotor_leaders == env.fixed_indices[0])
    cfg = load_config("masac", "f1_m5")
    cfg["scenario"]["fixed_wing"] = 2
    cfg["scenario"]["multirotor"] = 4
    env2 = PaperUAVEnv(cfg, seed=1)
    env2.positions[0, :2] = [1000, 1000]
    env2.positions[1, :2] = [1100, 1000]
    env2._refresh_links()
    assert env2.last_adjacency[0, 1] == env2.last_adjacency[1, 0] == 1
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_env.py -q`
Expected: FAIL due to missing keyword arguments and topology attributes.

- [ ] **Step 3: Implement minimal paper channel/topology**

Compute average LoS/NLoS path loss, convert to linear gain, then compute Eq. (6) SINR with other simultaneous transmitters as interference and Eq. (7) rate. Force `P_LoS=1` for fixed-wing-to-fixed-wing links. Build pairwise rates and adjacency using `rate > R_min`; rotor links are restricted to their assigned fixed-wing formation leader; fixed-wing pairs are eligible mesh edges.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_env.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uav_search/envs/models.py src/uav_search/envs/paper_env.py tests/test_env.py
git commit -m "feat: implement paper A2A SINR topology"
```

---

### Task 3: Reset geometry, hard obstacle constraint, and heterogeneous action semantics

**Files:**
- Modify: `src/uav_search/envs/paper_env.py:45-82,185-239`
- Modify: `tests/test_env.py`

**Interfaces:**
- Preserves normalized action `(2,)`.
- Multi-rotor decoding: `[force_scalar, theta]` -> nonnegative acceleration magnitude and planar direction.
- Fixed-wing decoding: `[yaw_rate, thrust]` -> signed heading-rate command and bounded forward acceleration/speed.

- [ ] **Step 1: Write failing behavior tests**

```python
def test_targets_and_building_centers_are_sampled_without_paper_margin():
    env = make_env(seed=11)
    seen_edge = False
    for seed in range(100):
        env.reset(seed=seed)
        if np.any(env.targets < 300) or np.any(env.targets > 4700):
            seen_edge = True
            break
    assert seen_edge


def test_speed_constraints_follow_table_i():
    env = make_env(seed=2)
    actions = {a: np.array([1.0, 1.0], dtype=np.float32) for a in env.agents}
    for _ in range(20):
        env.step(actions)
    fixed_speeds = np.linalg.norm(env.velocities[env.fixed_indices], axis=1)
    rotor_speeds = np.linalg.norm(env.velocities[env.multirotor_indices], axis=1)
    assert np.all((fixed_speeds >= 10.0) & (fixed_speeds <= 40.0))
    assert np.all(rotor_speeds <= 10.0 + 1e-9)


def test_obstacle_domain_is_hard_constraint():
    env = make_env(seed=3)
    rotor = env.multirotor_indices[0]
    env.positions[rotor, :2] = [1000.0, 1000.0]
    env.velocities[rotor] = 0.0
    env.obstacles[0] = [1008.0, 1000.0, 20.0]
    env.step({a: np.array([1.0, 0.0], dtype=np.float32) if i == rotor else np.zeros(2, dtype=np.float32)
              for i, a in enumerate(env.agents)})
    assert not circle_collision(env.positions[rotor, :2], env.obstacles[0])
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m pytest tests/test_env.py -q`
Expected: edge-distribution and obstacle-domain tests fail with current margin/diagnostic-only behavior.

- [ ] **Step 3: Implement reset/action/constraint behavior**

Targets and building centers use `uniform(0, area_size_m)`. UAV initial XY positions continue to use `uav_init_margin_m` because original initialization is undisclosed. Before movement, retain previous XY; after integration clip to task area and revert agents whose candidate position is inside a building circle so C3 is enforced. Keep fixed-wing `[10,40]` m/s and multi-rotor `<=10` m/s.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_env.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uav_search/envs/paper_env.py tests/test_env.py
git commit -m "feat: align paper environment dynamics constraints"
```

---

### Task 4: Paper reward equations (21)-(27)

**Files:**
- Modify: `src/uav_search/envs/paper_env.py:150-183,245-271`
- Modify: `tests/test_env.py`

**Interfaces:**
- `_communication_reward(rotor_idx: int) -> float`
- `_energy_reward(idx: int) -> float`
- `_safety_reward(idx: int) -> tuple[float, int]`
- `_task_reward(idx: int, fixed: bool) -> float`

- [ ] **Step 1: Write failing equation tests**

```python
def test_boundary_and_obstacle_diagnostics_do_not_add_unpublished_reward_penalties():
    env = make_env(seed=4)
    fixed = env.fixed_indices[0]
    env.targets[:] = [4900.0, 4900.0]
    env.positions[fixed, :2] = [0.0, 0.0]
    env.obstacles[:, :2] = [4500.0, 4500.0]
    env._refresh_links()
    expected = env._safety_reward(fixed)[0] + env._task_reward(fixed, fixed=True)
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    assert np.isclose(rewards[env.agents[fixed]], expected)


def test_confirmed_target_awards_exact_zeta_task_term():
    env = make_env(seed=5)
    rotor = env.multirotor_indices[0]
    env.positions[rotor, :2] = env.targets[0]
    assert env._task_reward(rotor, fixed=False) == env.assumed["search_reward_coeff"]
```

- [ ] **Step 2: Run tests and confirm RED where current unpublished shaping changes reward**

Run: `python -m pytest tests/test_env.py -q`
Expected: at least boundary/obstacle reward composition fails under controlled placement.

- [ ] **Step 3: Implement Eqs. (21)-(27)**

Use distance in meters consistently with `reward_distance_epsilon_m`; communication reward is `-r_c`, `r_c/(d+Delta_d)`, or `r_c`; energy reward is `tau*e_i` only above `e_safe`; safety reward is `-eta/(d+Delta_d)` for cooperative UAVs inside `D_safe`; rotor task reward uses Eq. (24); fixed-wing task reward uses Eq. (25). Remove boundary/obstacle reward penalties while retaining diagnostics/hard constraints.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_env.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uav_search/envs/paper_env.py tests/test_env.py
git commit -m "fix: match paper reward equations"
```

---

### Task 5: Full verification and reproduction-gap documentation

**Files:**
- Modify: `README.md` only if current claims imply exact reproduction.
- Create: `docs/PAPER_FIDELITY.md`
- Test: all tests.

**Interfaces:**
- Produces a user-readable audit of exact matches, inferred/reference-backed values, and remaining gaps.

- [ ] **Step 1: Add fidelity document**

Document Table-I exact values, paper-derived world/topology/action/reward semantics, and unresolved numeric gaps: altitude, timestep, detection/found/safety distances, building count/radii, UAV initialization, battery Joules, `P0/mu/d`, and reward coefficients not published by the paper.

- [ ] **Step 2: Run focused environment/config tests**

Run: `python -m pytest tests/test_config.py tests/test_env.py -q`
Expected: PASS.

- [ ] **Step 3: Run full suite**

Run: `python -m pytest -q`
Expected: all tests PASS with zero failures.

- [ ] **Step 4: Run smoke training for all three algorithms without changing algorithm hyperparameters**

Run three short commands using the existing CLI with a very small episode count for `masac`, `matd3`, and `maddpg`; require exit code 0.

- [ ] **Step 5: Inspect final diff and repository status**

Run: `git diff main...HEAD --check && git status --short && git log --oneline main..HEAD`
Expected: no whitespace errors; only intended files changed.

- [ ] **Step 6: Commit final documentation**

```bash
git add docs/PAPER_FIDELITY.md README.md
git commit -m "docs: record paper fidelity and remaining gaps"
```
