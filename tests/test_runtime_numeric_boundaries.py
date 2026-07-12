from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.utils.json import strict_json_loads
from linkerbot_sim.app.interactive.single_scene.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.single_scene import (
    transports as single_scene_transports,
)
from linkerbot_sim.app.interactive.tiled_scene import transport as tiled_scene_transport
from linkerbot_sim.controllers.types import ControlTargets, resolve_joint_parameter
from linkerbot_sim.planning.requests import IKRequest, MotionRequest
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SnapshotMetadata,
)
from linkerbot_sim.tiled.control.types import (
    TiledCommandAction,
    TiledCommandTarget,
    TiledCommandTrajectory,
)
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningResult,
)
from linkerbot_sim.trajectories.retiming import trajectory_sample_times
from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.timing import differentiate_samples


@pytest.mark.parametrize(
    "payload",
    (
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e309}',
        '{"outer": {"value": 1, "value": 2}}',
    ),
)
def test_strict_json_decoder_rejects_ambiguous_or_non_finite_input(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


def test_strict_json_decoder_accepts_canonical_json() -> None:
    assert strict_json_loads('{"value": 1.25, "nested": {"ok": true}}') == {
        "value": 1.25,
        "nested": {"ok": True},
    }


@pytest.mark.parametrize(
    "payload",
    (
        '{"type": "status", "value": NaN}',
        '{"type": "status", "type": "reset"}',
        '{"type": "status", "value": 1e309}',
    ),
)
def test_single_scene_and_tiled_scene_transports_share_strict_json_contract(
    payload: str,
) -> None:
    single_scene_response = single_scene_transports._handle_json_line(
        payload,
        queue=InteractiveMotionQueue(),
    )
    tiled_response = tiled_scene_transport._handle_json_line(payload, object())

    assert single_scene_response["event"] == "rejected"
    assert tiled_response["event"] == "rejected"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_command_data_types_reject_non_finite_targets(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TiledCommandAction("joint_position_target", np.asarray([[value]]))
    with pytest.raises(ValueError, match="finite"):
        TiledCommandTarget(np.asarray([[value]]))
    with pytest.raises(ValueError, match="finite"):
        TiledCommandTrajectory(np.asarray([[[value]]]))
    with pytest.raises(ValueError, match="finite"):
        ControlTargets(
            positions=np.asarray([value]),
            velocities=np.zeros(1),
            efforts=np.zeros(1),
        )


def test_planning_requests_reject_non_finite_vectors_and_timing() -> None:
    with pytest.raises(ValueError, match="finite"):
        IKRequest(target_position=np.asarray([0.0, np.nan, 0.0])).validate_structure()
    with pytest.raises(ValueError, match="finite"):
        MotionRequest(
            current_q=np.asarray([0.0]),
            goal_q=np.asarray([1.0]),
            duration_s=float("inf"),
        ).validate_structure()
    with pytest.raises(ValueError, match="finite"):
        TiledPlanningRequest(
            request_id="invalid",
            robot_name="robot",
            env_ids=(0,),
            current_positions=np.asarray([[np.nan]]),
            goal_positions=np.asarray([[1.0]]),
            joint_names=("j0",),
            sample_dt_s=0.1,
        )


def test_trajectory_boundaries_require_finite_strictly_increasing_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        JointTrajectory.from_samples(
            times=np.asarray([0.0, 0.0]),
            positions=np.asarray([[0.0], [1.0]]),
            joint_names=("j0",),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        differentiate_samples(
            np.asarray([[0.0], [1.0]]),
            np.asarray([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        trajectory_sample_times(duration_s=float("inf"), sample_dt_s=0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        TiledPlanningResult(
            request_id="invalid",
            robot_name="robot",
            env_ids=(0,),
            success=True,
            status="success",
            message="",
            times=np.asarray([0.1, 0.1]),
            positions=np.asarray([[[0.0], [1.0]]]),
            joint_names=("j0",),
        )


def test_snapshot_schema_rejects_non_finite_state() -> None:
    with pytest.raises(ValueError, match="finite"):
        RobotSnapshot(
            label="robot",
            robot_id=0,
            joint_names=("j0",),
            joint_positions=np.asarray([np.nan]),
            joint_velocities=np.zeros(1),
        )
    with pytest.raises(ValueError, match="finite"):
        ObjectSnapshot(
            name="object",
            positions_local=np.asarray([0.0, 0.0, np.inf]),
            orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        SnapshotMetadata.from_mapping({"time_s": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        SnapshotMetadata(time_s=float("inf"))


def test_controller_parameters_reject_non_finite_gains() -> None:
    with pytest.raises(ValueError, match="finite"):
        resolve_joint_parameter(float("nan"), ("j0",), label="stiffness")
