"""Native CPU/CUDA checks for command timing, bounded shafts and exact restore."""

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.newton.compliant_drive import (
    DriveLayout,
    NewtonCompliantDrive,
)


wp = pytest.importorskip("warp")


@pytest.fixture(params=["cpu", "cuda:0"])
def rig(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = wp.get_device(request.param)
    with wp.ScopedDevice(device):

        def control():
            return SimpleNamespace(
                joint_f=wp.zeros(2),
                joint_target_q=wp.zeros(2),
                joint_target_qd=wp.zeros(2),
                joint_act=None,
            )

        model = SimpleNamespace(device=device, control=control)
        state = SimpleNamespace(joint_q=wp.zeros(2), joint_qd=wp.zeros(2))
        layout = DriveLayout(
            ((0,), (1,)),
            ((0,), (1,)),
            ((1, 1250, 40, 40, 600, 5, 4.2, 8.51684, 1, 0, 1.6, 0),),
        )
        drive = NewtonCompliantDrive(
            model, state, layout, command_dt=1 / 60, substeps=8, stream=None
        )
        yield drive, state, control()


def test_motor_runs_once_per_command_and_native_reference_interpolates(rig):
    drive, state, command = rig
    command.joint_target_q.assign(np.array([0.5, 0.5], np.float32))
    targets = []
    for substep in range(8):
        drive.prepare(state, command)
        drive.apply(state, substep)
        targets.append(drive.control.joint_target_q.numpy().copy())
    expected = 5 / (1250 + (40 + 4.2) * 60)
    np.testing.assert_allclose(drive.history.numpy()[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(np.diff(targets, axis=0), expected / 8, atol=1e-9)
    np.testing.assert_allclose(targets[-1], expected * (1 + 40 / 1250 * 60), atol=1e-9)
    np.testing.assert_array_equal(command.joint_target_q.numpy(), [np.float32(0.5)] * 2)
    np.testing.assert_array_equal(state.joint_q.numpy(), [0, 0])


def test_blocked_output_does_not_wind_up_and_reversal_releases(rig):
    drive, state, command = rig
    command.joint_target_q.assign(np.array([0.5, 0.5], np.float32))
    for _ in range(80):
        drive.prepare(state, command)
        for substep in range(8):
            drive.apply(state, substep)
    holding = drive.history.numpy()[:, 0].copy()
    np.testing.assert_allclose(holding, 5 / 1250, atol=1e-10)
    command.joint_target_q.zero_()
    drive.prepare(state, command)
    assert np.all(drive.history.numpy()[:, 0] < holding)


def test_selected_mid_interval_restore_preserves_other_world(rig):
    drive, state, command = rig
    command.joint_target_q.assign(np.array([0.5, 0.3], np.float32))
    drive.prepare(state, command)
    drive.apply(state, 0)
    saved = wp.clone(drive.history)
    drive.prepare(state, command)
    drive.apply(state, 1)
    expected = drive.history.numpy().copy()
    expected_target = drive.control.joint_target_q.numpy().copy()
    mask = wp.array([True, False], dtype=wp.bool, device=drive.model.device)
    drive.restore(saved, mask)
    np.testing.assert_array_equal(drive.history.numpy()[1], expected[1])
    drive.prepare(state, command)
    drive.apply(state, 1)
    np.testing.assert_array_equal(drive.history.numpy()[0], expected[0])
    assert drive.control.joint_target_q.numpy()[0] == expected_target[0]


def test_graph_capture_retains_the_virtual_clock(rig):
    drive, state, command = rig
    if not drive.model.device.is_cuda:
        pytest.skip("CPU uses eager kernels")
    command.joint_target_q.assign(np.array([0.5, 0.5], np.float32))
    baseline = wp.clone(drive.history)
    with wp.ScopedCapture(device=drive.model.device) as capture:
        drive.prepare(state, command)
        drive.apply(state, 0)
    for _ in range(8):
        wp.capture_launch(capture.graph)
    captured = drive.history.numpy().copy()
    drive.restore(baseline)
    drive.prepare(state, command)
    for substep in range(8):
        drive.apply(state, substep)
    np.testing.assert_array_equal(drive.history.numpy(), captured)


def test_arm_interpolation_feedforward_and_preintegration_guard(rig):
    drive, state, command = rig
    parameters = np.array(
        [(0, 16000, 600, 0, 0, 0, 0, 0, 0, 0, 0, 199)], dtype=np.float64
    )
    drive.parameters.assign(parameters)
    command.joint_target_q.assign(np.array([0.001, 1.0], dtype=np.float32))
    command.joint_target_qd.assign(np.array([0.02, 0], dtype=np.float32))
    drive.prepare(state, command)
    for substep in range(8):
        drive.apply(state, substep)
        target = drive.control.joint_target_q.numpy()
        # The first joint stays below the guard; the other is deliberately saturated.
        assert abs(16000 * target[0]) < 199
        np.testing.assert_allclose(16000 * target[1], 199, atol=2e-5)
    np.testing.assert_allclose(target[0], 0.001 + 600 / 16000 * 0.02, atol=1e-9)
    np.testing.assert_array_equal(drive.control.joint_target_qd.numpy(), [0, 0])


@pytest.mark.parametrize("substeps", [4, 8, 12])
def test_hand_command_rate_does_not_change_with_physics_rate(rig, substeps):
    drive, state, command = rig
    drive.substeps = substeps
    command.joint_target_q.assign(np.array([0.5, 0.5], np.float32))
    for substep in range(substeps):
        drive.prepare(state, command)
        drive.apply(state, substep)
    expected = 5 / (1250 + (40 + 4.2) * 60)
    np.testing.assert_allclose(drive.history.numpy()[:, 0], expected, rtol=1e-12)
    np.testing.assert_array_equal(drive.history.numpy()[:, 8], [0, 0])
