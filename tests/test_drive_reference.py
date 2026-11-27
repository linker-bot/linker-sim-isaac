"""The arm and hand drive code reproduces the recorded reference trajectories (CPU).

``tests/data/drive_reference.npz`` was recorded on the GPU in the LW-v0 environment
(2 environments, 6 s, both arms and hands) with this repository's ``ArmAction`` and
``HandAction`` (LW-v0 0.6.0, hand motor ``geared-shaft-v1``). The script: arm
joints following sinusoids near home as Hermite points with velocities, then a 0.1 rad step
of J1 that drives the PD torque into the arm's torque guard; the hands closing (the thumb
loads against the index finger), opening, pinching, opposing the thumb, then closing again
and pressing until the thumb-yaw drive reaches the hand's torque guard.

Per physics step it holds what the drives read (measured joint positions and velocities)
and wrote (the arm reference, the requested and the guarded position targets of both); the
drives' gains and force limits as the engine holds them. Feeding the recorded inputs to the
code must give the recorded outputs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from linker_lab import arm, assets
from linker_lab.hand import HandMotor

REFERENCE = np.load(Path(__file__).with_name("data") / "drive_reference.npz")
SIDES = ("left", "right")
STEPS = 8
DECISION_DT = 1 / 60


def ref(side, name):
    return torch.from_numpy(REFERENCE[f"{side}/{name}"])


@pytest.mark.parametrize("side", SIDES)
def test_hermite_stream_reproduces_the_arm_reference(side):
    points_q, points_v = ref(side, "arm_endpoint_q"), ref(side, "arm_endpoint_v")
    q_ref, v_ref = ref(side, "arm_q_ref"), ref(side, "arm_v_ref")
    u = torch.arange(1, STEPS + 1, dtype=torch.float64)[:, None] / STEPS
    q0 = ref(side, "arm_q")[:, 0].double()  # the reset state
    v0 = torch.zeros_like(q0)
    for k in range(points_q.shape[1]):
        q, v = arm.hermite(q0, points_q[:, k], v0, points_v[:, k], u, DECISION_DT)
        # recorded on the GPU, whose fused multiply-adds round differently in the last bit
        torch.testing.assert_close(q, q_ref[:, k * STEPS : (k + 1) * STEPS], rtol=0, atol=1e-15)
        torch.testing.assert_close(v, v_ref[:, k * STEPS : (k + 1) * STEPS], rtol=0, atol=1e-15)
        q0, v0 = points_q[:, k], points_v[:, k]


@pytest.mark.parametrize("side", SIDES)
def test_servo_model_reproduces_the_arm_targets(side):
    kp = torch.from_numpy(REFERENCE["arm_kp"]).float().expand(2, 7).contiguous()
    kd = torch.from_numpy(REFERENCE["arm_kd"]).float().expand(2, 7).contiguous()
    cap = (torch.from_numpy(REFERENCE["arm_guard_torque"]).float() + arm.TORQUE_GUARD_MARGIN).expand(2, 7)
    cap, joints = cap.contiguous(), torch.arange(7, dtype=torch.int32)
    q_ref, v_ref = ref(side, "arm_q_ref"), ref(side, "arm_v_ref")
    q, qd, expected = ref(side, "arm_q"), ref(side, "arm_qd"), ref(side, "arm_target")
    requested = arm.feed_forward(q_ref, v_ref, kp[:, None], kd[:, None])
    targets = torch.stack(
        [
            arm.torque_guard(requested[:, t].contiguous(), q[:, t].contiguous(), qd[:, t].contiguous(), kp, kd, cap, joints)
            for t in range(q.shape[1])
        ],
        1,
    )
    torch.testing.assert_close(targets, expected, rtol=0, atol=0)
    assert ref(side, "arm_guard").any(), "the reference must exercise the torque guard"


@pytest.mark.parametrize("side", SIDES)
def test_hand_motor_reproduces_the_hand_requests(side):
    commands, q_at_decision = ref(side, "hand_command"), ref(side, "hand_q_at_decision")
    expected = ref(side, "hand_request")
    motor = HandMotor(
        ref(side, "hand_q")[:, 0],
        REFERENCE["hand_kp"],
        REFERENCE["hand_lower"],
        REFERENCE["hand_upper"],
        assets.motor(side),
        steps_per_decision=STEPS,
        decision_dt=DECISION_DT,
    )
    targets = []
    for k in range(commands.shape[1]):
        motor.decide(q_at_decision[:, k], commands[:, k])
        targets += [motor.step_target() for _ in range(STEPS)]
    torch.testing.assert_close(torch.stack(targets, 1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("side", SIDES)
def test_torque_guard_reproduces_the_hand_targets(side):
    """The hand's drive targets are the motor's requests through the arm's torque guard."""
    kp, kd, cap = (torch.from_numpy(REFERENCE[f"hand_{k}"]).expand(2, 6).contiguous() for k in ("kp", "kd", "cap"))
    joints = torch.arange(6, dtype=torch.int32)
    q, qd, requested, expected = (ref(side, f"hand_{k}") for k in ("q", "qd", "request", "target"))
    targets = torch.stack(
        [
            arm.torque_guard(requested[:, t].contiguous(), q[:, t].contiguous(), qd[:, t].contiguous(), kp, kd, cap, joints)
            for t in range(q.shape[1])
        ],
        1,
    )
    torch.testing.assert_close(targets, expected, rtol=0, atol=0)
    assert ref(side, "hand_guard").any(), "the reference must exercise the hand's torque guard"


def test_reference_gains_are_lw_v0s():
    """The recorded gains and force limits are the ones in LW-v0's files (the hand's drive
    damping is the MJCF ``kv``; the thumb yaw's passive damping is on the joint)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(assets.path("left.xml")))
    for part, joints in (("hand", assets.motor("left")["shafts"]), ("arm", assets.controller("left")["joints"])):
        kp, kd, cap = [], [], []
        for name in joints:
            j = model.joint(name).id
            (u,) = [u for u in range(model.nu) if model.actuator_trnid[u, 0] == j]
            kp.append(model.actuator_gainprm[u][0])
            kd.append(-model.actuator_biasprm[u][2])
            cap.append(model.jnt_actfrcrange[j][1])
        np.testing.assert_allclose(REFERENCE[f"{part}_kp"], kp, rtol=1e-6)
        np.testing.assert_allclose(REFERENCE[f"{part}_kd"], kd, rtol=1e-6)
        np.testing.assert_allclose(REFERENCE[f"{part}_cap"], cap, rtol=1e-6)
