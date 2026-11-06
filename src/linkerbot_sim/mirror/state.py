"""Mirror 低频状态读写用例。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from linkerbot_sim.snapshots.runtime_objects import (
    _command_target_modes,
    _robot_snapshot_from_execution,
    _runtime_object_snapshots,
)


@dataclass
class MirrorStateService:
    """把物理状态 adapter 包装成不泄漏内部可写引用的产品服务。

    setter 的事务/回滚由 engine-aware adapter 实现；本层负责只接受 mapping，并在边界
    深复制输入输出，避免 transport 调用方在校验后并发修改 payload。
    """

    getter: Callable[[], Mapping[str, object]]
    setter: Callable[..., object]
    observation_getter: Callable[[], Mapping[str, object]] | None = None

    def get_observation(self) -> dict[str, object]:
        if self.observation_getter is None:
            raise RuntimeError("Mirror assembly does not provide observations")
        value = self.observation_getter()
        if not isinstance(value, Mapping):
            raise RuntimeError("Mirror observation adapter must return a mapping")
        return deepcopy(dict(value))

    def get_state(self) -> dict[str, object]:
        value = self.getter()
        if not isinstance(value, Mapping):
            raise RuntimeError("Mirror state adapter must return a mapping")
        return deepcopy(dict(value))

    def set_state(
        self,
        state: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> object:
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")
        return self.setter(deepcopy(dict(state)), strict=bool(strict))


def read_mirror_observation(resources: object) -> dict[str, object]:
    """Read only root body/joint fields; skip SceneSnapshot and restore metadata.

    Reuse the physical field readers/validators so command modes, root frames and
    velocity semantics agree with snapshots. No asset/config fingerprints, solver
    integration, controller history or generalized/body restore payload is read.
    """
    robots = []
    for robot_id, robot in resources.robots_by_id.items():
        values = _robot_snapshot_from_execution(
            label=robot.label,
            robot_id=robot_id,
            execution=robot.execution,
            robot_profile=None,
            asset_fingerprint=None,
        ).as_dict()
        values["command_target_modes"] = list(
            _command_target_modes(
                robot.execution.joint_controller,
                command_count=len(values["command_joint_names"]),
            )
        )
        robots.append(values)
    objects = _runtime_object_snapshots(
        stage=resources.session.stage,
        handles=resources.object_handles,
        state_views=resources.object_state_views,
        include_articulation_state=False,
    )
    return {
        "simulation_time_s": float(resources.session.physics_runtime.simulation_time),
        "coordinate_frame": "scene-local",
        "robots": robots,
        "objects": {
            name: {
                field: item
                for field, item in value.as_dict().items()
                if field
                in {
                    "name",
                    "positions_local",
                    "orientations_wxyz",
                    "linear_velocities",
                    "angular_velocities",
                }
            }
            for name, value in objects.items()
        },
    }


__all__ = ["MirrorStateService"]
