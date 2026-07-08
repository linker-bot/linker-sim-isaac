"""Realtime simulation state snapshots.

本模块定义交互实时模式的状态快照和线程安全转交通道。Isaac/PhysX 状态采样必须在
仿真主线程完成；后台 publisher 只能消费这里生成的纯 Python/numpy 快照。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Condition

import numpy as np

from linkerbot_sim.logging.effort_logger import read_joint_efforts
from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


@dataclass(frozen=True)
class RobotJointStateSnapshot:
    """单侧 articulation 的一帧关节状态。"""

    side: str
    joint_names: tuple[str, ...]
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray
    accelerations_rad_s2: np.ndarray
    commanded_efforts: np.ndarray | None = None
    measured_efforts: np.ndarray | None = None
    applied_efforts: np.ndarray | None = None

    def effort_values(self, field: str) -> np.ndarray | None:
        """按 Foxglove `JointStates.effort` 语义选择一类 effort。"""

        if field == "none":
            return None
        if field == "commanded":
            return self.commanded_efforts
        if field == "measured":
            return self.measured_efforts
        if field == "applied":
            return self.applied_efforts
        raise ValueError(f"Unsupported effort field: {field!r}")

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的字典，供 Foxglove JSON topic 使用。"""

        return {
            "joint_names": list(self.joint_names),
            "positions_rad": _json_vector(self.positions_rad),
            "velocities_rad_s": _json_vector(self.velocities_rad_s),
            "accelerations_rad_s2": _json_vector(self.accelerations_rad_s2),
            "commanded_efforts": _optional_json_vector(self.commanded_efforts),
            "measured_efforts": _optional_json_vector(self.measured_efforts),
            "applied_efforts": _optional_json_vector(self.applied_efforts),
        }


@dataclass(frozen=True)
class ObjectPoseSnapshot:
    """一个 runtime object 的世界位姿。"""

    name: str
    prim_path: str
    position_m: np.ndarray
    orientation_wxyz: np.ndarray

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的对象位姿。"""

        return {
            "prim_path": self.prim_path,
            "position_m": _json_vector(self.position_m),
            "orientation_wxyz": _json_vector(self.orientation_wxyz),
        }


@dataclass(frozen=True)
class StateSnapshot:
    """交互实时模式的一帧完整状态快照。"""

    step: int
    time_s: float
    robots: tuple[RobotJointStateSnapshot, ...]
    objects: tuple[ObjectPoseSnapshot, ...] = ()
    phase: str | None = None

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的完整快照。"""

        result: dict[str, object] = {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "robots": {robot.side: robot.as_dict() for robot in self.robots},
            "objects": {obj.name: obj.as_dict() for obj in self.objects},
        }
        if self.phase is not None:
            result["phase"] = self.phase
        return result


class StateStream:
    """保存最新状态快照的线程安全通道。

    该通道只保留最新帧。publisher 慢于采样时会自然丢旧帧，从而避免反压仿真
    主循环。
    """

    def __init__(self) -> None:
        """初始化快照通道。"""

        self._condition = Condition()
        self._sequence = 0
        self._snapshot: StateSnapshot | None = None
        self._closed = False

    def publish(self, snapshot: StateSnapshot) -> int:
        """发布一帧快照，并返回递增序号。"""

        with self._condition:
            if self._closed:
                return self._sequence
            self._sequence += 1
            self._snapshot = snapshot
            self._condition.notify_all()
            return self._sequence

    def latest(self) -> tuple[int, StateSnapshot] | None:
        """返回最新快照和序号；还没有快照时返回 None。"""

        with self._condition:
            if self._snapshot is None:
                return None
            return self._sequence, self._snapshot

    def wait_next(
        self, after_sequence: int = 0, *, timeout_s: float | None = None
    ) -> tuple[int, StateSnapshot] | None:
        """等待并返回比 `after_sequence` 更新的快照。"""

        with self._condition:
            if not self._condition.wait_for(
                lambda: self._closed or self._sequence > int(after_sequence),
                timeout=timeout_s,
            ):
                return None
            if self._closed or self._snapshot is None:
                return None
            if self._sequence <= int(after_sequence):
                return None
            return self._sequence, self._snapshot

    def close(self) -> None:
        """关闭通道并唤醒等待中的消费者。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()


class DualRobotStateSampler:
    """从双 articulation runtime 采样状态快照。

    本对象应只在仿真主线程调用。它内部维护上一帧速度，用于计算差分加速度。
    """

    def __init__(
        self,
        *,
        stage,
        object_handles: Sequence[object] = (),
        rate_hz: float = 60.0,
        include_efforts: bool = False,
        include_objects: bool = False,
    ) -> None:
        """创建双臂状态采样器。"""

        self.stage = stage
        self.object_handles = tuple(object_handles)
        self.rate_hz = float(rate_hz)
        self.include_efforts = bool(include_efforts)
        self.include_objects = bool(include_objects)
        self._previous_velocities: dict[str, tuple[float, np.ndarray]] = {}

    def should_sample(self, *, step: int, physics_dt: float) -> bool:
        """按配置频率判断当前 physics step 是否采样。"""

        if self.rate_hz <= 0.0:
            return False
        if physics_dt <= 0.0:
            return True
        interval_steps = max(1, int(round(1.0 / (physics_dt * self.rate_hz))))
        return int(step) % interval_steps == 0

    def reset(self) -> None:
        """清理 reset 前的差分速度缓存。"""

        self._previous_velocities.clear()

    def sample(self, runtime, *, step: int, phase: str | None = None) -> StateSnapshot:
        """采样一帧双臂和对象状态。"""

        physics_dt = float(runtime.simulation_world.get_physics_dt())
        time_s = (int(step) + 1) * physics_dt
        robots = (
            self._sample_side(runtime.left, time_s=time_s),
            self._sample_side(runtime.right, time_s=time_s),
        )
        objects = (
            self._sample_objects()
            if self.include_objects and self.stage is not None
            else ()
        )
        return StateSnapshot(
            step=int(step),
            time_s=time_s,
            phase=phase,
            robots=robots,
            objects=objects,
        )

    def _sample_side(self, side_runtime, *, time_s: float) -> RobotJointStateSnapshot:
        """采样单侧 articulation 的关节状态。"""

        robot = side_runtime.articulation
        positions = _vector_from_method(robot, "get_joint_positions", _num_dof(robot))
        velocities = _vector_from_method(robot, "get_joint_velocities", positions.size)
        accelerations = self._acceleration_from_velocity(
            side_runtime.side, velocities, time_s=time_s
        )
        commanded = measured = applied = None
        if self.include_efforts:
            commanded = _commanded_efforts(side_runtime.joint_controller, positions.size)
            efforts = read_joint_efforts(robot, None, measured=True, applied=True)
            measured = efforts.measured
            applied = efforts.applied
        return RobotJointStateSnapshot(
            side=str(side_runtime.side),
            joint_names=tuple(str(name) for name in getattr(robot, "dof_names", ())),
            positions_rad=positions,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            commanded_efforts=commanded,
            measured_efforts=measured,
            applied_efforts=applied,
        )

    def _acceleration_from_velocity(
        self, side: str, velocities: np.ndarray, *, time_s: float
    ) -> np.ndarray:
        """用上一帧速度计算差分加速度。"""

        previous = self._previous_velocities.get(side)
        self._previous_velocities[side] = (float(time_s), velocities.copy())
        if previous is None:
            return np.full(velocities.size, np.nan, dtype=float)
        previous_time, previous_velocity = previous
        dt = float(time_s) - float(previous_time)
        if dt <= 0.0 or previous_velocity.size != velocities.size:
            return np.full(velocities.size, np.nan, dtype=float)
        return (velocities - previous_velocity) / dt

    def _sample_objects(self) -> tuple[ObjectPoseSnapshot, ...]:
        """采样 runtime objects 的 root prim 世界位姿。"""

        return _sample_runtime_objects(self.stage, self.object_handles)


class DualRobotStateObserver:
    """执行层 step 后调用的状态观察器。"""

    def __init__(self, *, sampler: DualRobotStateSampler, stream: StateStream) -> None:
        """保存采样器和快照通道。"""

        self.sampler = sampler
        self.stream = stream

    def observe(self, runtime, *, step: int, phase: str | None = None) -> None:
        """必要时采样并发布一帧状态。"""

        physics_dt = float(runtime.simulation_world.get_physics_dt())
        if not self.sampler.should_sample(step=step, physics_dt=physics_dt):
            return
        self.stream.publish(self.sampler.sample(runtime, step=step, phase=phase))

    def reset(self) -> None:
        """清理状态采样器内部派生缓存。"""

        self.sampler.reset()


class SingleRobotStateSampler:
    """从单 articulation runtime 采样状态快照。"""

    def __init__(
        self,
        *,
        stage,
        object_handles: Sequence[object] = (),
        rate_hz: float = 60.0,
        include_efforts: bool = False,
        include_objects: bool = False,
        side_label: str = "single",
    ) -> None:
        """创建单臂状态采样器。"""

        self.stage = stage
        self.object_handles = tuple(object_handles)
        self.rate_hz = float(rate_hz)
        self.include_efforts = bool(include_efforts)
        self.include_objects = bool(include_objects)
        self.side_label = str(side_label)
        self._previous_velocity: tuple[float, np.ndarray] | None = None

    def should_sample(self, *, step: int, physics_dt: float) -> bool:
        """按配置频率判断当前 physics step 是否采样。"""

        if self.rate_hz <= 0.0:
            return False
        if physics_dt <= 0.0:
            return True
        interval_steps = max(1, int(round(1.0 / (physics_dt * self.rate_hz))))
        return int(step) % interval_steps == 0

    def reset(self) -> None:
        """清理 reset 前的差分速度缓存。"""

        self._previous_velocity = None

    def sample(self, runtime, *, step: int, phase: str | None = None) -> StateSnapshot:
        """采样一帧单臂和对象状态。"""

        physics_dt = float(runtime.simulation_world.get_physics_dt())
        time_s = (int(step) + 1) * physics_dt
        objects = (
            _sample_runtime_objects(self.stage, self.object_handles)
            if self.include_objects and self.stage is not None
            else ()
        )
        return StateSnapshot(
            step=int(step),
            time_s=time_s,
            phase=phase,
            robots=(self._sample_robot(runtime, time_s=time_s),),
            objects=objects,
        )

    def _sample_robot(self, runtime, *, time_s: float) -> RobotJointStateSnapshot:
        """采样单 articulation 的关节状态。"""

        robot = runtime.articulation
        positions = _vector_from_method(robot, "get_joint_positions", _num_dof(robot))
        velocities = _vector_from_method(robot, "get_joint_velocities", positions.size)
        accelerations = self._acceleration_from_velocity(velocities, time_s=time_s)
        commanded = measured = applied = None
        if self.include_efforts:
            commanded = _commanded_efforts(runtime.joint_controller, positions.size)
            efforts = read_joint_efforts(robot, None, measured=True, applied=True)
            measured = efforts.measured
            applied = efforts.applied
        return RobotJointStateSnapshot(
            side=self.side_label,
            joint_names=tuple(str(name) for name in getattr(robot, "dof_names", ())),
            positions_rad=positions,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            commanded_efforts=commanded,
            measured_efforts=measured,
            applied_efforts=applied,
        )

    def _acceleration_from_velocity(
        self,
        velocities: np.ndarray,
        *,
        time_s: float,
    ) -> np.ndarray:
        """用上一帧速度计算差分加速度。"""

        previous = self._previous_velocity
        self._previous_velocity = (float(time_s), velocities.copy())
        if previous is None:
            return np.full(velocities.size, np.nan, dtype=float)
        previous_time, previous_velocity = previous
        dt = float(time_s) - float(previous_time)
        if dt <= 0.0 or previous_velocity.size != velocities.size:
            return np.full(velocities.size, np.nan, dtype=float)
        return (velocities - previous_velocity) / dt


class SingleRobotStateObserver:
    """单臂执行层 step 后调用的状态观察器。"""

    def __init__(
        self,
        *,
        runtime,
        sampler: SingleRobotStateSampler,
        stream: StateStream,
    ) -> None:
        """保存 app runtime、采样器和快照通道。"""

        self.runtime = runtime
        self.sampler = sampler
        self.stream = stream

    def observe(self, _world, *, step: int, phase: str | None = None) -> None:
        """必要时采样并发布一帧状态。"""

        execution = self.runtime.execution
        physics_dt = float(execution.simulation_world.get_physics_dt())
        if not self.sampler.should_sample(step=step, physics_dt=physics_dt):
            return
        self.stream.publish(self.sampler.sample(execution, step=step, phase=phase))

    def reset(self) -> None:
        """清理状态采样器内部派生缓存。"""

        self.sampler.reset()


def _num_dof(robot) -> int:
    """读取 articulation DOF 数。"""

    if hasattr(robot, "num_dof"):
        return int(robot.num_dof)
    return len(getattr(robot, "dof_names", ()))


def _sample_runtime_objects(
    stage,
    object_handles: Sequence[object],
) -> tuple[ObjectPoseSnapshot, ...]:
    """采样 runtime objects 的 root prim 世界位姿。"""

    snapshots: list[ObjectPoseSnapshot] = []
    for handle in object_handles:
        prim_path = _runtime_object_prim_path(handle)
        if prim_path is None:
            continue
        pose = _read_prim_world_pose(stage, prim_path)
        if pose is None:
            continue
        position, orientation = pose
        snapshots.append(
            ObjectPoseSnapshot(
                name=str(getattr(handle, "name", prim_path)),
                prim_path=prim_path,
                position_m=position,
                orientation_wxyz=orientation,
            )
        )
    return tuple(snapshots)


def _vector_from_method(source, method_name: str, expected_size: int) -> np.ndarray:
    """调用一个无参状态读取方法；缺失或失败时返回 nan。"""

    method = getattr(source, method_name, None)
    if method is None:
        return np.full(int(expected_size), np.nan, dtype=float)
    try:
        values = np.asarray(method(), dtype=float).reshape(-1)
    except Exception:
        return np.full(int(expected_size), np.nan, dtype=float)
    if values.size != int(expected_size):
        return np.full(int(expected_size), np.nan, dtype=float)
    return values


def _commanded_efforts(controller, expected_size: int) -> np.ndarray:
    """读取控制器缓存的 Python 侧 commanded effort。"""

    values = getattr(controller, "last_commanded_efforts", None)
    if values is None:
        return np.full(int(expected_size), np.nan, dtype=float)
    try:
        efforts = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.full(int(expected_size), np.nan, dtype=float)
    if efforts.size != int(expected_size):
        return np.full(int(expected_size), np.nan, dtype=float)
    return efforts.copy()


def _runtime_object_prim_path(handle: object) -> str | None:
    """从 RuntimeObjectHandle 或兼容替身中读取对象 root prim path。"""

    model = getattr(handle, "model", None)
    for source in (model, getattr(handle, "config", None)):
        if source is None:
            continue
        prim_path = getattr(source, "prim_path", None)
        if prim_path is not None:
            return str(prim_path)
        if isinstance(source, Mapping):
            root = source.get("root")
            if root is not None and hasattr(root, "GetPath"):
                return str(root.GetPath())
            prim_path = source.get("prim_path")
            if prim_path is not None:
                return str(prim_path)
    return None


def _read_prim_world_pose(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 USD prim 的世界位姿；只能在仿真主线程调用。"""

    from pxr import Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid():
        return None
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    position = np.asarray(
        [translation[0], translation[1], translation[2]], dtype=float
    )
    rotation_matrix = _matrix3_to_numpy(matrix.ExtractRotationMatrix())
    return position, matrix_to_quat_wxyz(rotation_matrix)


def _matrix3_to_numpy(matrix) -> np.ndarray:
    """把 USD/Gf 3x3 matrix 转成 numpy。"""

    return np.asarray(
        [[float(matrix[row][col]) for col in range(3)] for row in range(3)],
        dtype=float,
    )


def _optional_json_vector(values: np.ndarray | None) -> list[float | None] | None:
    """可选向量转 JSON；None 保持 None。"""

    if values is None:
        return None
    return _json_vector(values)


def _json_vector(values) -> list[float | None]:
    """把 numpy 向量转成 JSON 友好的 list，非有限值用 null 表示。"""

    vector = np.asarray(values, dtype=float).reshape(-1)
    result: list[float | None] = []
    for value in vector:
        numeric = float(value)
        result.append(numeric if np.isfinite(numeric) else None)
    return result
