"""Refuse state that a snapshot cannot restore, before inspecting or changing live data."""

from functools import partial
from types import SimpleNamespace

import pytest

from linker_lab import snapshot


def stateless(value, offset=0):
    return value + offset


class Stateful:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return value


def manager(terms, *, event=False):
    result = SimpleNamespace(
        active_terms={"reset": list(terms)} if event else list(terms),
        get_term_cfg=terms.__getitem__,
    )
    if event:
        result.available_modes = ["reset"] if terms else []
    return result


def environment():
    term = SimpleNamespace(
        func=stateless,
        history_length=0,
        delay_max_lag=0,
        modifiers=None,
        noise=None,
    )
    env = SimpleNamespace(
        num_envs=1,
        action_manager=SimpleNamespace(active_terms=[]),
        observation_manager=SimpleNamespace(_group_obs_term_cfgs={"policy": [term]}),
        event_manager=manager({}, event=True),
        reward_manager=manager({}),
        termination_manager=manager({}),
        scene=SimpleNamespace(sensors={}, articulations={}, rigid_objects={}),
    )
    return env, term


def unsupported(case, env, term):
    callback = Stateful()
    if case == "delay":
        term.delay_max_lag = 2
    elif case == "observation":
        term.func = callback
    elif case == "bound_method":
        term.func = callback.__call__
    elif case == "bound_builtin":
        term.func = [].append
    elif case == "partial_stateful":
        term.func = partial(callback)
    elif case == "modifier":
        term.modifiers = [SimpleNamespace(func=callback)]
    elif case == "noise":
        term.noise = SimpleNamespace(func=callback)
    elif case == "cooldown":
        cfg = SimpleNamespace(func=stateless, min_step_count_between_reset=10)
        env.event_manager = manager({"home": cfg}, event=True)
    else:
        cfg = SimpleNamespace(func=callback, min_step_count_between_reset=0)
        setattr(env, f"{case}_manager", manager({"custom": cfg}, event=case == "event"))
    return callback


@pytest.mark.parametrize("operation", ["capture", "restore"])
@pytest.mark.parametrize(
    "case, message",
    [
        ("delay", "observation delay"),
        ("observation", "unsaved state"),
        ("bound_method", "unsaved state"),
        ("bound_builtin", "unsaved state"),
        ("partial_stateful", "unsaved state"),
        ("modifier", "modifier uses a callable"),
        ("noise", "noise uses a callable"),
        ("reward", "reward term 'custom'"),
        ("termination", "termination term 'custom'"),
        ("event", "event term 'custom'"),
        ("cooldown", "reset cooldown"),
    ],
)
def test_unsaved_state_is_refused_before_capture_or_restore(monkeypatch, operation, case, message):
    env, term = environment()
    callback = unsupported(case, env, term)

    def unexpected(*args):
        pytest.fail("an unsupported snapshot reached the state-reading/writing phase")

    for name in ("_header", "_read", "_write"):
        monkeypatch.setattr(snapshot, name, unexpected)
    with pytest.raises(snapshot.SnapshotError, match=message) as error:
        if operation == "capture":
            snapshot.capture(env)
        else:
            snapshot.restore(env, snapshot.Snapshot({}, {}))
    assert not error.value.physics_modified
    assert callback.calls == 0


def test_stateless_function_terms_modifiers_and_noise_remain_supported():
    env, term = environment()
    term.func = partial(stateless, offset=1)
    term.modifiers = [SimpleNamespace(func=partial(stateless, offset=2))]
    term.noise = SimpleNamespace(func=stateless)
    for kind in ("reward", "termination", "event"):
        cfg = SimpleNamespace(func=stateless, min_step_count_between_reset=0)
        setattr(env, f"{kind}_manager", manager({"plain": cfg}, event=kind == "event"))
    snapshot._refusals(env)


def test_default_workstation_observations_and_reset_remain_supported():
    from linker_lab.workstation import LWv0EnvCfg

    cfg = LWv0EnvCfg()
    env, _ = environment()
    policy = cfg.observations.policy
    env.observation_manager._group_obs_term_cfgs = {
        "policy": [policy.left_joint_pos, policy.right_joint_pos,
                   policy.left_joint_vel, policy.right_joint_vel]
    }
    env.event_manager = manager({"reset_to_home": cfg.events.reset_to_home}, event=True)
    snapshot._refusals(env)
