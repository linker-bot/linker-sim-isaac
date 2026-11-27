"""The O6 motor model: self-locking shafts, the torque budget, the speed limit (CPU)."""

import pytest
import torch

from linker_lab import assets, hand
from linker_lab.hand import HandMotor

KP = (2500.0, 600.0, 1250.0, 1250.0, 1250.0, 1250.0)  # the bundle's drive stiffness
LOWER = (0.0,) * 6
UPPER = (0.58, 1.36, 1.6, 1.6, 1.6, 1.6)
DT = 1 / 60


def motor(q):
    return HandMotor(torch.as_tensor(q, dtype=torch.float32), KP, LOWER, UPPER, assets.motor("left"), 8, DT)


def test_a_pushed_finger_does_not_move_its_shaft_back():
    """Holding a pose, the fingers are pushed back by an external load: the shafts stay."""
    held = torch.tensor([[0.4, 1.0, 1.2, 1.2, 1.2, 1.2]])
    m = motor(held)
    for push in torch.linspace(0, 0.5, 30):  # deflect the fingers further and further
        q = (held - push).clamp(min=0)
        shaft = m.decide(q.float(), held)
        torch.testing.assert_close(shaft, held.double(), rtol=0, atol=1e-12)


def test_a_blocked_finger_stalls_and_never_backs_off():
    """Closing on an obstacle at 0.9 rad: the shaft stops just past it and holds while the
    load grows. (It stops up to one decision's travel past the obstacle: the motor
    predicts the finger one decision ahead, and the finger stopped abruptly.)"""
    target = torch.tensor([[0.0, 0.0, 1.5, 1.5, 1.5, 1.5]])
    q = torch.zeros(1, 6)
    m = motor(q)
    shafts = []
    for step in range(300):
        q = m.shaft.float().clamp(max=0.9)  # the fingers stop at the obstacle
        if step > 200:
            q = q - 0.002 * (step - 200)  # and are then pushed back
        shafts.append(m.decide(q, target)[0, 2:].clone())
    shafts = torch.stack(shafts)
    assert (shafts[1:] >= shafts[:-1] - 1e-12).all(), "a shaft moved away from its target"
    one_decision = 5.0 / (4.2 + 4.31684) * DT  # free travel per decision at the torque budget
    assert ((shafts[-1] > 0.9) & (shafts[-1] <= 0.9 + one_decision + 1e-9)).all()
    torch.testing.assert_close(shafts[-1], shafts[150], rtol=0, atol=0)  # stalled, then held


def test_free_shafts_move_at_most_at_the_speed_limit_and_stop_at_the_target():
    """Unloaded, viscosity limits the speed further: fingers about 0.6 rad/s, thumb yaw
    about 0.13 rad/s (advancing torque / (viscosity + passive damping))."""
    target = torch.tensor([[0.5, 1.3, 1.5, 1.5, 1.5, 1.5]])
    m = motor(torch.zeros(1, 6))
    previous = m.shaft.clone()
    for _ in range(1500):
        shaft = m.decide(m.shaft.float(), target)  # unloaded: the joint follows the shaft
        assert ((shaft - previous).abs() <= 1.0 * DT + 1e-12).all()
        previous = shaft.clone()
    torch.testing.assert_close(previous, target.double(), rtol=0, atol=1e-6)


def test_opening_under_load_moves_toward_the_target():
    m = motor(torch.full((1, 6), 0.5))
    shaft = m.decide(torch.full((1, 6), 0.45), torch.zeros(1, 6))
    assert (shaft < 0.5).all()


def test_targets_are_clamped_into_the_joint_range():
    m = motor(torch.zeros(1, 6))
    for _ in range(1500):
        shaft = m.decide(m.shaft.float(), torch.full((1, 6), 9.0))
    torch.testing.assert_close(shaft, torch.tensor([UPPER], dtype=torch.float64), rtol=0, atol=1e-6)


def test_per_step_targets_interpolate_the_shaft_and_end_on_the_feed_forward_target():
    m = motor(torch.zeros(1, 6))
    m.decide(torch.zeros(1, 6), torch.full((1, 6), 1.0))
    targets = torch.stack([m.step_target() for _ in range(8)])
    assert (targets[1:] >= targets[:-1]).all()
    assert (targets[-1] > m.shaft.float()).all()  # velocity feed-forward while moving


def test_presets():
    assert set(hand.PRESETS) == {"open", "close", "pinch"}
    assert hand.PRESETS["pinch"][:3] == (0.37, 1.04, 0.85)
    current = torch.tensor([hand.PRESETS["open"]])
    opposed = hand.thumb_oppose(current)
    assert opposed[0, 1] == pytest.approx(1.04) and (opposed[0, [0, 2, 3, 4, 5]] == 0).all()


@pytest.mark.parametrize("normalized", [False, True])
def test_action_nonfinite_values_keep_previous_targets(normalized):
    """NaN and either infinity hold their shaft; finite normalized values still clamp."""
    from types import SimpleNamespace

    previous = torch.full((1, 6), 0.4)
    measured = torch.zeros(1, 6)
    calls = []
    action = SimpleNamespace(
        cfg=SimpleNamespace(normalized=normalized),
        _raw=torch.zeros(1, 6),
        _target=previous.clone(),
        _lower=torch.tensor(LOWER),
        _upper=torch.tensor(UPPER),
        _joint_ids=list(range(6)),
        _asset=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured))),
        motor=SimpleNamespace(decide=lambda q, target: calls.append(target.clone())),
    )
    command = torch.tensor([[float("nan"), float("inf"), -float("inf"), -2.0, 0.0, 2.0]])
    hand.HandAction.process_actions(action, command)
    expected = previous.clone()
    expected[:, 3:] = torch.tensor([[0.0, 0.8, 1.6]] if normalized else [[-2.0, 0.0, 2.0]])
    torch.testing.assert_close(action._target, expected)
    torch.testing.assert_close(calls[0], expected)
