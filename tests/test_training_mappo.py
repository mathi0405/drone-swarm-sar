from swarm_sar.training.mappo import (
    RolloutBuffer,
    merge_rollout_buffers,
    merged_advantages_returns,
)


def _buf(marker: int) -> RolloutBuffer:
    return RolloutBuffer(
        obs=[f"obs{marker}"],
        actions=[marker],
        logps=[marker + 0.1],
        rewards=[marker + 0.2],
        values=[marker + 0.3],
        dones=[0.0],
        global_state=[f"state{marker}"],
        expert_actions=[marker + 10],
    )


def _multi_step_bufs(n_agents: int = 2, steps: int = 3) -> list:
    """One env group of ``n_agents`` buffers, each transition tagged (agent, t).

    Rewards encode the tag (agent * 100 + t) so, with gamma=0 and zero values,
    the raw returns reproduce the tag and alignment can be checked exactly.
    """
    bufs = []
    for agent in range(n_agents):
        b = RolloutBuffer.empty()
        for t in range(steps):
            b.obs.append(f"a{agent}t{t}")
            b.actions.append(agent * 10 + t)
            b.logps.append(0.0)
            b.rewards.append(float(agent * 100 + t))
            b.values.append(0.0)
            b.dones.append(0.0)
            b.global_state.append(f"a{agent}t{t}")
            b.graphs.append(None)
            b.action_masks.append(None)
            b.expert_actions.append(0)
        bufs.append(b)
    return bufs


def test_shared_policy_update_uses_all_agent_buffers():
    merged = merge_rollout_buffers([_buf(1), _buf(2), _buf(3)])

    assert merged.obs == ["obs1", "obs2", "obs3"]
    assert merged.actions == [1, 2, 3]
    assert merged.rewards == [1.2, 2.2, 3.2]
    assert merged.global_state == ["state1", "state2", "state3"]
    assert merged.expert_actions == [11, 12, 13]


def test_merge_is_time_major_within_env_group():
    # Regression test: the merge was once buffer-major ([a0t0, a0t1, a0t2,
    # a1t0, ...]) while advantages were emitted time-major, silently pairing
    # every PPO transition with another transition's advantage.
    merged = merge_rollout_buffers(_multi_step_bufs(), group_size=2)

    assert merged.obs == ["a0t0", "a1t0", "a0t1", "a1t1", "a0t2", "a1t2"]
    assert merged.rewards == [0.0, 100.0, 1.0, 101.0, 2.0, 102.0]


def test_merged_advantages_align_with_merged_transitions():
    bufs = _multi_step_bufs()
    merged = merge_rollout_buffers(bufs, group_size=2)
    # gamma=0, lam=0, zero values -> raw return k equals reward k, i.e. the
    # (agent, t) tag of the transition the advantage belongs to.
    _, ret = merged_advantages_returns(bufs, gamma=0.0, lam=0.0, group_size=2)

    tags = [f"a{int(v) // 100}t{int(v) % 100}" for v in ret]
    assert tags == merged.obs
    assert len(ret) == len(merged.obs)


def test_merge_handles_multiple_env_groups():
    # Two envs x two agents: env blocks must stay contiguous, time-major inside.
    env0 = _multi_step_bufs(n_agents=2, steps=2)
    env1 = _multi_step_bufs(n_agents=2, steps=2)
    for b in env1:  # retag env1 so the two envs are distinguishable
        b.obs = [f"e1_{o}" for o in b.obs]
    merged = merge_rollout_buffers(env0 + env1, group_size=2)

    assert merged.obs == ["a0t0", "a1t0", "a0t1", "a1t1",
                          "e1_a0t0", "e1_a1t0", "e1_a0t1", "e1_a1t1"]
