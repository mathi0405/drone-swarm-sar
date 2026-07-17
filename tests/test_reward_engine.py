"""Invariants of the redesigned reward (see docs/reward_design.md)."""
import numpy as np

from swarm_sar.config import EnvConfig, RewardConfig
from swarm_sar.environment.reward_engine import RewardEngine, RewardInputs


def _engine(**overrides) -> RewardEngine:
    return RewardEngine(RewardConfig(**overrides))


def test_duplicate_penalty_is_bounded_by_its_weight():
    # Full-footprint overlap must cost exactly the weight, not weight * cells.
    # (The unbounded version reached -19.7/step and out-massed rescue rewards,
    # teaching policies to avoid explored ground harder than to rescue.)
    r = RewardConfig()
    eng = RewardEngine(r)
    full_overlap = RewardInputs(overlap_cells=197, sensor_footprint=197)
    lone_overlap = RewardInputs(overlap_cells=1, sensor_footprint=197)

    assert np.isclose(eng.compute(full_overlap) - r.time_penalty, r.duplicate_explore)
    assert eng.compute(full_overlap) < eng.compute(lone_overlap) < 0


def test_exploration_reward_is_footprint_normalized():
    r = RewardConfig()
    eng = RewardEngine(r)
    all_new = RewardInputs(exploration_cells=197, team_new_cells=197,
                           sensor_footprint=197)
    expected = r.time_penalty + r.explore_new_cell + r.coverage_new_cell
    assert np.isclose(eng.compute(all_new), expected)


def test_dense_drag_is_bounded_and_small_vs_mission_events():
    # The worst dense per-step drag (max overlap + time + hover + idle) must
    # stay a small bounded nudge, and a full-urgency rescue must outweigh
    # dozens of such steps — mission events dominate by construction.
    r = RewardConfig()
    eng = RewardEngine(r)
    worst_step = eng.compute(RewardInputs(overlap_cells=197, sensor_footprint=197,
                                          idle=True, hovering=True))
    rescue = eng.compute(RewardInputs(victim_rescued=True, rescue_urgency=1.0))
    assert worst_step >= -0.5
    assert rescue > 40 * abs(worst_step)


def test_rescue_urgency_scales_reward_and_can_be_disabled():
    on = _engine(time_critical_rescue=True)
    off = _engine(time_critical_rescue=False)
    early = RewardInputs(victim_rescued=True, rescue_urgency=1.0)
    late = RewardInputs(victim_rescued=True, rescue_urgency=0.3)

    assert on.compute(early) > on.compute(late)
    assert np.isclose(off.compute(early), off.compute(late))


def test_sparse_mode_keeps_only_events():
    eng = _engine(mode="sparse")
    shaping_only = RewardInputs(exploration_cells=50, overlap_cells=100,
                                sensor_footprint=197, idle=True, hovering=True)
    assert eng.compute(shaping_only) == 0.0
    assert eng.compute(RewardInputs(victim_rescued=True)) > 0


def test_full_mission_returns_positive_total():
    """Integration: a competent policy must be able to earn a positive return."""
    from swarm_sar.environment.sar_env import SARSwarmEnv
    from swarm_sar.mission.planner import HeuristicSwarmController

    env = SARSwarmEnv(EnvConfig(num_drones=3, max_steps=200), seed=3)
    env.reset(seed=3)
    ctl = HeuristicSwarmController(env, seed=3)
    total = 0.0
    while env.agents:
        _, rew, *_ = env.step(ctl.act())
        total += sum(rew.values())
    summary = env.episode_summary()
    assert summary["rescue_rate"] > 0.5     # the heuristic is competent here
    assert total > 0.0                       # ...and the reward reflects it


def test_worst_case_rescue_beats_travel_drag():
    # Even a floor-urgency, minimum-severity rescue must out-earn ~100 steps
    # of maximum travel drag (full duplicate overlap + time pressure) — the
    # economics that keep "go rescue" a winning gradient late in an episode.
    r = RewardConfig()
    eng = RewardEngine(r)
    min_urgency = r.rescue_urgency_floor * (0.5 + 0.5 * 0.3)   # decay 0, sev 0.3
    worst_rescue = eng.compute(RewardInputs(victim_rescued=True,
                                            rescue_urgency=min_urgency))
    travel_drag_step = abs(r.duplicate_explore) + abs(r.time_penalty)
    assert worst_rescue > 100 * travel_drag_step


def test_hazard_entry_and_dwell_are_separate_events():
    r = RewardConfig()
    eng = RewardEngine(r)
    entry = eng.compute(RewardInputs(hazard_entered=True))
    dwell = eng.compute(RewardInputs(hazard_dwell=True))
    assert np.isclose(entry - r.time_penalty, r.hazard_penalty)
    assert np.isclose(dwell - r.time_penalty, r.hazard_dwell)
    assert abs(r.hazard_dwell) < abs(r.hazard_penalty)   # dwell is the small one
