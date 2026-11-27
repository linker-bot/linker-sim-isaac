"""The arm's trajectory generator against Ruckig, its limits, and the servo model (CPU)."""

import numpy as np
import pytest
import torch

from linker_lab import arm

ruckig = pytest.importorskip("ruckig")

F64 = dict(dtype=torch.float64)
V = torch.tensor(arm.VELOCITY, **F64)
A = torch.tensor(arm.ACCELERATION, **F64)
J = torch.tensor(arm.JERK, **F64)
DT = 1e-3  # sampling for the comparison [s]
TOL = 1e-9  # numerical tolerance on the limits (relative)


def ruckig_duration(q, v, a, goal, synchronization=None):
    inp = ruckig.InputParameter(7)
    inp.current_position, inp.current_velocity, inp.current_acceleration = q, v, a
    inp.target_position = goal
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = V.tolist(), A.tolist(), J.tolist()
    inp.synchronization = synchronization or ruckig.Synchronization.Time
    trajectory = ruckig.Trajectory(7)
    result = ruckig.Ruckig(7).calculate(inp, trajectory)
    assert result in (ruckig.Result.Working, ruckig.Result.Finished)
    return trajectory.duration, trajectory


def one_plan(q, v, a, goal, seconds=8.0):
    """The generator's whole trajectory from one plan, sampled every DT."""
    t = lambda x: torch.tensor(x, **F64)[None]  # noqa: E731
    qs, vs, as_ = arm.plan(t(q), t(v), t(a), t(goal), V, A, J, DT, round(seconds / DT))
    return qs[0], vs[0], as_[0]


def arrival(qs, vs, goal):
    at_goal = ((qs - torch.tensor(goal, **F64)).abs().amax(-1) < 1e-9) & (vs.abs().amax(-1) < 1e-9)
    assert at_goal[-1], "never arrived"
    first = len(at_goal) - int(at_goal.flip(0).cumprod(0).sum())  # first sample from which it stays
    return (first + 1) * DT


def assert_within_limits(q0, v0, a0, qs, vs, as_):
    vs = torch.cat((torch.tensor(v0, **F64)[None], vs))
    as_ = torch.cat((torch.tensor(a0, **F64)[None], as_))
    assert (vs.abs() <= V * (1 + TOL)).all()
    assert (as_.abs() <= A * (1 + TOL)).all()
    # jerk is piecewise constant, so differences of the sampled acceleration bound it
    assert ((as_[1:] - as_[:-1]).abs() <= J * DT * (1 + 1e-6)).all()


def random_cases(seed, count, moving):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        q = rng.uniform(-2, 2, 7)
        goal = q + rng.uniform(-1.5, 1.5, 7) * rng.integers(0, 2, 7)  # some joints stay
        if not moving:
            yield q.tolist(), [0.0] * 7, [0.0] * 7, goal.tolist()
            continue
        # a state some plan passes through: start a plan at rest, take its state at a random time
        start = rng.uniform(-2, 2, 7)
        qs, vs, as_ = one_plan(start.tolist(), [0.0] * 7, [0.0] * 7, (start + rng.uniform(-1, 1, 7)).tolist(), 3.0)
        k = int(rng.integers(0, 1500))
        yield qs[k].tolist(), vs[k].tolist(), as_[k].tolist(), goal.tolist()


@pytest.mark.parametrize("moving", [False, True], ids=["from_rest", "moving"])
def test_generator_arrives_when_ruckig_does_within_the_limits(moving):
    for q, v, a, goal in random_cases(20261117 + moving, 40, moving):
        expected, _ = ruckig_duration(q, v, a, goal)
        qs, vs, as_ = one_plan(q, v, a, goal, expected + 1)
        assert arrival(qs, vs, goal) == pytest.approx(expected, abs=1.5 * DT)
        assert_within_limits(q, v, a, qs, vs, as_)


def test_each_joint_alone_is_as_fast_as_ruckig():
    """Without synchronization, every joint's own time-optimal profile has Ruckig's duration."""
    for q, v, a, goal in random_cases(7, 20, moving=True):
        duration, trajectory = ruckig_duration(q, v, a, goal, ruckig.Synchronization.No)
        for j in range(7):
            single_goal = list(q)
            single_goal[j] = goal[j]
            qs, vs, _ = one_plan(q, [v[j] if i == j else 0.0 for i in range(7)],
                                 [a[j] if i == j else 0.0 for i in range(7)], single_goal, duration + 1)
            # the other joints are at rest at their goal: only joint j moves
            assert arrival(qs, vs, single_goal) == pytest.approx(
                trajectory.independent_min_durations[j], abs=1.5 * DT
            )


def test_no_overshoot_from_rest_and_straight_line_in_joint_space():
    for q, v, a, goal in random_cases(3, 20, moving=False):
        qs, _, _ = one_plan(q, v, a, goal)
        q0, g = torch.tensor(q, **F64), torch.tensor(goal, **F64)
        low, high = torch.minimum(q0, g), torch.maximum(q0, g)
        assert ((qs >= low - 1e-12) & (qs <= high + 1e-12)).all()
        progress = (qs - q0) / torch.where((g - q0).abs() > 1e-9, g - q0, 1.0)
        moving = (g - q0).abs() > 1e-9
        # all joints arrive together; from rest they also travel nearly along the line
        assert (progress[:, moving].amax(-1) - progress[:, moving].amin(-1)).max() < 0.1


def test_replanning_every_decision_follows_the_same_trajectory():
    """The action replans from its own state every 8 steps; that is the same motion."""
    q, v, a, goal = next(random_cases(11, 1, moving=True))
    dt, steps = 1 / 480, 8
    whole = arm.plan(*(torch.tensor(x, **F64)[None] for x in (q, v, a, goal)), V, A, J, dt, 8 * 400)
    state = [torch.tensor(x, **F64)[None] for x in (q, v, a)]
    pieces = []
    for _ in range(400):
        out = arm.plan(*state, torch.tensor(goal, **F64)[None], V, A, J, dt, steps)
        pieces.append(out[0])
        state = [x[:, -1] for x in out]
    torch.testing.assert_close(torch.cat(pieces, 1), whole[0], rtol=0, atol=1e-9)


def test_retargeting_while_moving_stays_within_the_limits():
    rng = np.random.default_rng(5)
    dt = 1 / 480
    state = [torch.zeros(4, 7, **F64) for _ in range(3)]
    history = [[x.clone() for x in state]]
    for decision in range(600):
        if decision % 20 == 0:  # a new goal every third of a second
            goal = torch.tensor(rng.uniform(-1, 1, (4, 7)), **F64)
        out = arm.plan(*state, goal, V, A, J, dt, 8)
        history += [[x[:, k] for x in out] for k in range(8)]
        state = [x[:, -1] for x in out]
    q, v, a = (torch.stack([h[i] for h in history]) for i in range(3))
    assert (v.abs() <= V * (1 + TOL)).all() and (a.abs() <= A * (1 + TOL)).all()
    assert ((a[1:] - a[:-1]).abs() <= J * dt * (1 + 1e-6)).all()


# ------------------------------------------------------------------- limits


WIDE = torch.full((7,), 10.0, **F64)  # joint ranges that do not matter


def wrist(j6, j7):
    """Project wrist positions only: joints J1..J5 at 0, a box wider than the octagon."""
    q = torch.zeros(len(j6), 7, **F64)
    q[:, 5], q[:, 6] = j6, j7
    out, changed = arm.project(q, -WIDE, WIDE)
    return out[:, 5], out[:, 6], changed


def test_wrist_octagon_projection():
    deg = lambda *x: torch.tensor(np.radians(x), **F64)  # noqa: E731
    inside = deg(0, 0, 50, 80, -59, -48)  # J6 values, then J7 values (3 points)
    j6, j7, changed = wrist(inside[:3], inside[3:])
    torch.testing.assert_close(torch.cat((j6, j7)), inside)
    assert not changed.any()
    # box corners are outside; each goes to the octagon's nearest edge point
    j6, j7, changed = wrist(deg(60, -60, 60), deg(90, -90, 0))
    corner = np.radians([60, 90])
    a, b = np.radians([23, 90]), np.radians([60, 49])
    t = np.dot(corner - a, b - a) / np.dot(b - a, b - a)
    expected = a + t * (b - a)
    assert np.allclose([j6[0], j7[0]], expected) and np.allclose([j6[1], j7[1]], -expected)
    assert np.allclose([j6[2], j7[2]], np.radians([60, 0]))  # on the edge: unchanged
    assert changed.tolist() == [True, True, False]
    # every vertex is a fixed point
    v = torch.tensor(arm.WRIST_OCTAGON, **F64)
    j6, j7, changed = wrist(v[:, 0], v[:, 1])
    torch.testing.assert_close(torch.stack((j6, j7), -1), v)
    assert not changed.any()


def test_project_clamps_to_the_joint_ranges():
    lower = torch.tensor([-3.1, -2.1, -3.1, -1.0, -3.1, -1.05, -1.57], **F64)
    upper = -lower
    q = torch.tensor([[5.0, -5.0, 0.0, 2.0, 0.0, 0.0, 0.0]], **F64)
    out, changed = arm.project(q, lower, upper)
    assert out[0, :4].tolist() == [3.1, -2.1, 0.0, 1.0] and changed.tolist() == [True]
    out, changed = arm.project(q[:, None].expand(1, 8, 7), lower, upper)  # per-step samples
    assert out.shape == (1, 8, 7) and changed.shape == (1, 8)


# ------------------------------------------------------------------- servo


def test_servo_feed_forward_and_torque_guard():
    kp, kd = torch.full((1, 7), 16000.0), torch.full((1, 7), 600.0)
    cap = torch.tensor([[120.0] * 4 + [100.0] * 3])
    joints = torch.arange(7, dtype=torch.int32)
    q = torch.zeros(1, 7)
    requested = arm.feed_forward(torch.full((1, 7), 1e-3, **F64), torch.full((1, 7), 0.01, **F64), kp, kd)
    target = arm.torque_guard(requested, q, q, kp, kd, cap, joints)
    torch.testing.assert_close(target, torch.full((1, 7), 1e-3 + 600 / 16000 * 0.01))  # 16 + 6 N·m < guard
    requested = arm.feed_forward(torch.full((1, 7), 0.1, **F64), torch.zeros(1, 7, **F64), kp, kd)
    target = arm.torque_guard(requested, q, q, kp, kd, cap, joints)
    torch.testing.assert_close(kp * target, cap - 1)  # 1600 N·m requested: held 1 N·m below the cap


def test_rate_limit_holds_velocity_and_acceleration_and_counts_jerk():
    dt = 1 / 480
    last = torch.zeros(2, 7, 3, **F64)
    q = torch.zeros(2, 8, 7, **F64)
    q[0] = 0.1  # env 0: a 0.1 rad jump; env 1: stays
    v = torch.zeros_like(q)
    hit = arm.rate_limit(q, v, last, V, A, J, dt)
    assert hit.tolist() == [True, False]
    fd = torch.diff(torch.cat((torch.zeros(1, 7, **F64), q[0])), dim=0) / dt
    assert (fd.abs() <= V * (1 + TOL)).all()
    assert (torch.diff(torch.cat((torch.zeros(1, 7, **F64), fd)), dim=0).abs() <= A * dt * (1 + TOL)).all()
    torch.testing.assert_close(v[0], fd)  # the reference velocity is the limited one
    torch.testing.assert_close(last[0, :, 0], q[0, -1])
    assert (q[1] == 0).all()
    # a motion within the limits passes unchanged: from rest at jerk 20 rad/s³
    last = torch.zeros(1, 7, 3, **F64)
    t = torch.arange(1, 9, **F64)[:, None] * dt
    ramp = (20 / 6 * t**3).expand(8, 7)[None].clone()
    assert not arm.rate_limit(ramp.clone(), torch.zeros_like(ramp), last, V, A, J, dt).item()


def test_sample_feed_forward_speed_is_limited_even_when_positions_do_not_move():
    last = torch.zeros(2, 7, 3, **F64)
    q = torch.zeros(2, 8, 7, **F64)
    v = torch.zeros_like(q)
    v[0, :, ::2] = 1e6
    v[0, :, 1::2] = -1e6
    hit = arm.rate_limit(q, v, last, V, A, J, 1 / 480)
    assert hit.tolist() == [True, False]
    assert (v.abs() <= V).all()
    assert (q == 0).all() and (last == 0).all()
    torch.testing.assert_close(v[0, :, ::2], V[::2].expand(8, -1))
    torch.testing.assert_close(v[0, :, 1::2], -V[1::2].expand(8, -1))


@pytest.mark.parametrize("wrist_edge", [False, True], ids=["joint_range", "wrist_octagon"])
def test_stream_reaches_a_boundary_from_rest_without_leaving_the_feasible_region(wrist_edge):
    """Continuous accepted samples reach a boundary; stored derivatives match actual motion."""
    dt = 1 / 480
    lower, upper = -WIDE.clone(), WIDE.clone()
    start = torch.zeros(1, 7, **F64)
    goal = torch.zeros(1, 7, **F64)
    if wrist_edge:
        start[0, 5:] = torch.tensor(np.radians([20.0, 80.0]), **F64)
        goal[0, 5:] = torch.tensor(np.radians([60.0, 90.0]), **F64)
    else:
        upper[0] = 1.0
        start[0, 0] = 0.99
        goal[0, 0] = 1.0
    last = torch.zeros(1, 7, 3, **F64)
    last[:, :, 0] = start
    boundary_stop = False
    for _ in range(120):
        q = goal[:, None, :].expand(1, 8, 7).clone()
        q, _ = arm.project(q, lower, upper)
        v = torch.zeros_like(q)
        before = last.clone()
        arm.rate_limit(q, v, last, V, A, J, dt, lower, upper)
        projected, _ = arm.project(q, lower, upper)
        torch.testing.assert_close(q, projected, rtol=0, atol=1e-12)
        fd = torch.diff(torch.cat((before[:, :, 0][:, None], q), dim=1), dim=1) / dt
        acc = torch.diff(torch.cat((before[:, :, 1][:, None], fd), dim=1), dim=1) / dt
        assert (fd.abs() <= V * (1 + TOL)).all()
        boundary_stop |= bool((acc.abs() > A * (1 + TOL)).any())
        torch.testing.assert_close(last[:, :, 0], q[:, -1])
        torch.testing.assert_close(last[:, :, 1], fd[:, -1])
        torch.testing.assert_close(last[:, :, 2], acc[:, -1])
    assert boundary_stop, "the regression must exercise a position-priority boundary stop"
    # The next command starts from the final feasible state, not an outward candidate.
    q = start[:, None, :].expand(1, 8, 7).clone()
    v = torch.zeros_like(q)
    arm.rate_limit(q, v, last, V, A, J, dt, lower, upper)
    projected, _ = arm.project(q, lower, upper)
    torch.testing.assert_close(q, projected, rtol=0, atol=1e-12)


def test_reset_projects_an_outside_measured_pose_before_starting_the_stream():
    """Solver drift past a limit must not leave the limiter stuck outside the region."""
    from types import SimpleNamespace

    measured = torch.zeros(1, 7, **F64)
    measured[0, 0] = 1.001
    lower, upper = -WIDE.clone(), WIDE.clone()
    upper[0] = 1.0
    action = SimpleNamespace(
        cfg=SimpleNamespace(limits="real", command="joint", stream="points"),
        _asset=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured))),
        _joint_ids=list(range(7)), _lower=lower, _upper=upper,
        _plan=[torch.zeros(1, 7, **F64) for _ in range(3)],
        _point=[torch.zeros(1, 7, **F64) for _ in range(2)],
        _last=torch.zeros(1, 7, 3, **F64),
        _ref_q=torch.zeros(1, 8, 7, **F64), _ref_v=torch.zeros(1, 8, 7, **F64),
        _requested=torch.zeros(8, 1, 7), _raw=torch.zeros(1, 7),
        limit_hits=torch.zeros(1, dtype=torch.long),
    )
    arm.ArmAction.reset(action)
    assert action._last[0, 0, 0] == 1.0
    assert action._plan[0][0, 0] == 1.0 and measured[0, 0] == 1.001
    q = torch.zeros(1, 8, 7, **F64)
    v = torch.zeros_like(q)
    arm.rate_limit(q, v, action._last, V, A, J, 1 / 480, lower, upper)
    assert q[0, 0, 0] < 1.0  # starts moving inward, rather than locking outside
    projected, _ = arm.project(q, lower, upper)
    torch.testing.assert_close(q, projected, rtol=0, atol=1e-12)
