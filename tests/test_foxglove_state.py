from __future__ import annotations

import numpy as np

from linkerbot_sim.telemetry.foxglove_state import FoxgloveStateSink
from linkerbot_sim.telemetry.state_snapshot import (
    ObjectPoseSnapshot,
    RobotJointStateSnapshot,
    StateSnapshot,
)


class _FakeFoxgloveLogger:
    def __init__(self) -> None:
        self.joint_states = []
        self.state_json = []
        self.scene_spheres = []
        self.closed = False

    def log_joint_state(self, **kwargs) -> None:
        self.joint_states.append(kwargs)

    def log_state_json(self, state, *, time_s=None) -> None:
        self.state_json.append((state, time_s))

    def log_scene_spheres(self, **kwargs) -> None:
        self.scene_spheres.append(kwargs)

    def close(self) -> None:
        self.closed = True


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        step=7,
        time_s=0.8,
        phase="push",
        robots=(
            RobotJointStateSnapshot(
                side="left",
                joint_names=("j0", "j1"),
                positions_rad=np.asarray([0.1, 0.2]),
                velocities_rad_s=np.asarray([1.0, 2.0]),
                accelerations_rad_s2=np.asarray([10.0, 20.0]),
                commanded_efforts=np.asarray([0.3, 0.4]),
                measured_efforts=np.asarray([0.5, 0.6]),
                applied_efforts=np.asarray([0.7, 0.8]),
            ),
            RobotJointStateSnapshot(
                side="right",
                joint_names=("j0",),
                positions_rad=np.asarray([-0.1]),
                velocities_rad_s=np.asarray([-1.0]),
                accelerations_rad_s2=np.asarray([-10.0]),
                commanded_efforts=np.asarray([-0.3]),
                measured_efforts=np.asarray([-0.5]),
                applied_efforts=np.asarray([-0.7]),
            ),
        ),
        objects=(
            ObjectPoseSnapshot(
                name="Tblock",
                prim_path="/World/TBlock",
                position_m=np.asarray([0.1, 0.2, 0.3]),
                orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
        ),
    )


def test_foxglove_state_sink_publishes_joint_state_and_full_json() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(logger, joint_effort_field="measured")

    sink.publish(_snapshot())

    joint_state = logger.joint_states[0]
    assert joint_state["joint_names"] == ["left/j0", "left/j1", "right/j0"]
    np.testing.assert_allclose(joint_state["positions"], [0.1, 0.2, -0.1])
    np.testing.assert_allclose(joint_state["efforts"], [0.5, 0.6, -0.5])
    state_json, time_s = logger.state_json[0]
    assert state_json["phase"] == "push"
    assert state_json["objects"]["Tblock"]["prim_path"] == "/World/TBlock"
    assert time_s == 0.8
    np.testing.assert_allclose(
        logger.scene_spheres[0]["positions"], [[0.1, 0.2, 0.3]]
    )


def test_foxglove_state_sink_can_omit_joint_effort_field() -> None:
    logger = _FakeFoxgloveLogger()
    sink = FoxgloveStateSink(logger, joint_effort_field="none")

    sink.publish(_snapshot())

    assert logger.joint_states[0]["efforts"] is None
    sink.close()
    assert logger.closed is True
