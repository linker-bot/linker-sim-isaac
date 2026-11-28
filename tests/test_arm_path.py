"""Complete interpolated paths: convex position bounds and consistent derivatives."""

from types import SimpleNamespace

import pytest
import torch

from linker_lab import arm
from linker_lab._arm_path import BoundedPlanner

F64 = dict(dtype=torch.float64)
V, A, J = (torch.tensor(values, **F64) for values in (arm.VELOCITY, arm.ACCELERATION, arm.JERK))
CORNERS = torch.tensor(arm.WRIST_OCTAGON, **F64)
LOWER = torch.tensor([-3.1, -2.1, -3.1, -1.0, -3.1, -torch.pi / 3, -torch.pi / 2], **F64)
UPPER = -LOWER
DT, STEPS = 1 / 480, 8


def wrist(j6, j7):
    q = torch.zeros(1, 7, **F64)
    q[0, 5:] = torch.deg2rad(torch.tensor([j6, j7], **F64))
    return q


def planner(q):
    return BoundedPlanner(q, V, A, J, LOWER, UPPER, CORNERS)


def assert_path(history, initial):
    q, v, a = (torch.cat([chunk[i] for chunk in history], dim=1) for i in range(3))
    projected, _ = arm.project(q, LOWER, UPPER)
    torch.testing.assert_close(q, projected, atol=1e-11, rtol=0)
    assert (v.abs() <= V * (1 + 1e-9)).all()
    assert (a.abs() <= A * (1 + 1e-9)).all()
    previous_a = torch.cat((torch.zeros_like(initial)[:, None], a[:, :-1]), dim=1)
    assert ((a - previous_a).abs() <= J * DT * (1 + 1e-6)).all()
    previous_q = torch.cat((initial[:, None], q[:, :-1]), dim=1)
    previous_v = torch.cat((torch.zeros_like(initial)[:, None], v[:, :-1]), dim=1)
    # Integrate the reported velocity. The trapezoidal error is bounded by jerk;
    # projecting q while retaining the original v fails this check.
    error = q - previous_q - 0.5 * DT * (v + previous_v)
    assert (error.abs() <= J * DT**3 / 12 + 1e-11).all()
    return q, v, a


@pytest.mark.parametrize("reverse", [False, True])
def test_motion_between_wrist_vertices_stays_on_the_edge_with_consistent_derivatives(reverse):
    start, goal = wrist(23, 90), wrist(60, 49)
    if reverse:
        start, goal = goal, start
    p = planner(start)
    state = [start, torch.zeros_like(start), torch.zeros_like(start)]
    history = []
    for _ in range(150):
        out = p.sample(*state, goal, DT, STEPS)
        history.append(out)
        state = [x[:, -1].clone() for x in out[:3]]
    assert history[0][3].item()  # the independent-joint proposal leaves the edge
    q, v, _ = assert_path(history, start)
    normal = torch.tensor([41.0, 37.0], **F64)
    assert (v[..., 5:] @ normal).abs().max() < 1e-10
    torch.testing.assert_close(q[:, -1], goal, atol=1e-10, rtol=0)


def test_an_interior_goal_retains_the_existing_profile():
    start = torch.zeros(1, 7, **F64)
    goal = torch.tensor([[0.2, -0.1, 0.15, 0.08, -0.12, 0.1, -0.2]], **F64)
    p = planner(start)
    state = [start, torch.zeros_like(start), torch.zeros_like(start)]
    expected = arm.plan(*state, goal, V, A, J, DT, STEPS * 120)
    history = []
    for _ in range(120):
        out = p.sample(*state, goal, DT, STEPS)
        assert not out[3].item()
        history.append(out)
        state = [x[:, -1].clone() for x in out[:3]]
    for actual, wanted in zip(assert_path(history, start), expected):
        torch.testing.assert_close(actual, wanted, atol=1e-10, rtol=0)


def test_rejected_retargets_keep_moving_and_eventually_use_the_latest_goal():
    start, original = wrist(23, 90), wrist(60, 49)
    # Points close behind the moving reference on the same edge cannot be
    # reached by independently braking the two joints without leaving the edge.
    pending = wrist(24, 90 - 41 / 37)
    latest = wrist(26, 90 - 3 * 41 / 37)
    p = planner(start)
    state = [start, torch.zeros_like(start), torch.zeros_like(start)]
    history = []
    rejected_while_moving = False
    for decision in range(480):
        goal = original if decision < 12 else (pending if decision < 20 else latest)
        before = state[0].clone()
        out = p.sample(*state, goal, DT, STEPS)
        if decision >= 12 and out[3].item() and state[1].abs().max() > 0:
            rejected_while_moving = True
            assert not torch.equal(out[0][:, -1], before)
        history.append(out)
        state = [x[:, -1].clone() for x in out[:3]]
    assert rejected_while_moving
    q, v, a = assert_path(history, start)
    torch.testing.assert_close(q[:, -1], latest, atol=1e-10, rtol=0)
    assert not v[:, -1].any() and not a[:, -1].any()
    torch.testing.assert_close(p.state["path_goal"], latest, atol=0, rtol=0)


def test_goals_updated_at_60_hz_do_not_restart_a_stop():
    start = torch.zeros(1, 7, **F64)
    p = planner(start)
    state = [start, torch.zeros_like(start), torch.zeros_like(start)]
    history = []
    for decision in range(240):
        goal = start.clone()
        goal[0, 0] = min(0.3, 0.1 + decision / 600)
        out = p.sample(*state, goal, DT, STEPS)
        history.append(out)
        state = [x[:, -1].clone() for x in out[:3]]
    q, _, _ = assert_path(history, start)
    assert q[0, 60 * STEPS - 1, 0] > 0.1
    assert q[0, -1, 0] == pytest.approx(0.3, abs=1e-10)


def test_partial_reset_and_copied_path_state_resume_independently():
    start = wrist(23, 90).expand(2, -1).clone()
    goal = wrist(60, 49).expand(2, -1).clone()
    p = planner(start)
    state = [start, torch.zeros_like(start), torch.zeros_like(start)]
    for _ in range(20):
        out = p.sample(*state, goal, DT, STEPS)
        state = [x[:, -1].clone() for x in out[:3]]
    restored = planner(start)
    for name, value in p.state.items():
        restored.state[name].copy_(value)
    expected = p.sample(*state, goal, DT, STEPS)
    actual = restored.sample(*state, goal, DT, STEPS)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    rows = torch.tensor([0])
    before = {name: value[1].clone() for name, value in restored.state.items()}
    restored.reset(start, rows)
    for name, value in restored.state.items():
        torch.testing.assert_close(value[1], before[name], atol=0, rtol=0)
    torch.testing.assert_close(restored.state["path_start"][0, :, 0], start[0])
    assert not restored.state["path_duration"][0].any()


def test_action_saves_the_same_position_and_velocity_it_sends_to_the_servo():
    start, goal = wrist(23, 90), wrist(60, 49)
    data = SimpleNamespace(
        joint_stiffness=SimpleNamespace(torch=torch.full((1, 7), 16000.0)),
        joint_damping=SimpleNamespace(torch=torch.full((1, 7), 600.0)),
    )
    action = SimpleNamespace(
        cfg=SimpleNamespace(limits="real", interpolation=True, stream="points", command="joint", asset_name="left"),
        _raw=start.float(), _lower=LOWER, _upper=UPPER,
        _joint_command=lambda raw: raw.double(),
        _plan=[start, torch.zeros_like(start), torch.zeros_like(start)],
        _bounded=planner(start), _dt=DT, _steps=STEPS,
        _point=[start.clone(), torch.zeros_like(start)],
        _last=torch.zeros(1, 7, 3, **F64), _target=torch.zeros(1, 7),
        _asset=SimpleNamespace(data=data), _joint_ids=list(range(7)),
        _env=SimpleNamespace(extras={}), limit_hits=torch.zeros(1, dtype=torch.long),
    )
    for _ in range(30):
        arm.ArmAction.process_actions(action, goal.float())
        torch.testing.assert_close(action._plan[0], action._ref_q[:, -1], atol=0, rtol=0)
        torch.testing.assert_close(action._plan[1], action._ref_v[:, -1], atol=0, rtol=0)
    assert action.limit_hits.item() > 0
    action.get_state = lambda: arm.ArmAction.get_state(action)
    saved = action.get_state()
    assert set(action._bounded.state) <= set(saved)
    arm.ArmAction.process_actions(action, goal.float())
    expected_q, expected_v = action._ref_q.clone(), action._ref_v.clone()
    arm.ArmAction.set_state(action, saved)
    arm.ArmAction.process_actions(action, goal.float())
    torch.testing.assert_close(action._ref_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(action._ref_v, expected_v, atol=0, rtol=0)
