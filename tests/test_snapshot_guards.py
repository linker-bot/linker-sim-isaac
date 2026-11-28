"""What a snapshot refuses because Isaac Lab keeps it outside the saved state (CPU)."""

from types import SimpleNamespace

import pytest

from linker_lab import snapshot


def environment(*, term=None, class_instances=(), reset_events=None):
    term = term or SimpleNamespace(history_length=0, delay_max_lag=0)
    reset_events = reset_events or {}
    return SimpleNamespace(
        num_envs=1,
        action_manager=SimpleNamespace(active_terms=[]),
        observation_manager=SimpleNamespace(
            _group_obs_term_cfgs={"policy": [term]}, _group_obs_class_instances=list(class_instances)
        ),
        event_manager=SimpleNamespace(
            available_modes=["reset"],
            active_terms={"reset": list(reset_events)},
            get_term_cfg=reset_events.__getitem__,
        ),
        scene=SimpleNamespace(sensors={}, articulations={}, rigid_objects={}),
    )


@pytest.mark.parametrize(
    "env, message",
    [
        (environment(term=SimpleNamespace(history_length=0, delay_max_lag=2)), "with a delay"),
        (environment(class_instances=[object()]), "modifier or noise model with state"),
        (
            environment(reset_events={"home": SimpleNamespace(min_step_count_between_reset=10)}),
            "'home' has a cooldown",
        ),
    ],
)
def test_state_outside_the_snapshot_is_refused(env, message):
    with pytest.raises(snapshot.SnapshotError, match=message):
        snapshot._refusals(env)


def test_the_workstation_and_a_reset_event_without_cooldown_are_accepted():
    from linker_lab.workstation import LWv0EnvCfg

    cfg = LWv0EnvCfg()
    env = environment(reset_events={"reset_to_home": cfg.events.reset_to_home})
    policy = cfg.observations.policy
    env.observation_manager._group_obs_term_cfgs = {
        "policy": [policy.left_joint_pos, policy.right_joint_pos, policy.left_joint_vel, policy.right_joint_vel]
    }
    snapshot._refusals(env)
