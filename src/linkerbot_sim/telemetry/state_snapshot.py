"""实时仿真状态快照、主线程采样器和线程安全转交通道。

Isaac/PhysX 对象只能在仿真主线程读取。本模块先把关节状态、力矩和 USD 世界位姿冻结为
只含 Python 标量与 numpy 数组的快照，再交给后台 publisher；后台线程不得持有或访问
articulation、stage 等仿真对象。``StateStream`` 通过有界队列明确处理背压，使遥测消费者
变慢时只丢弃快照，而不会阻塞 physics step。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from threading import Condition

import numpy as np

from linkerbot_sim.isaac.scene.pose import read_prim_world_pose
from linkerbot_sim.logging.effort_logger import read_joint_efforts
from linkerbot_sim.objects.runtime import runtime_object_prim_path
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


@dataclass(frozen=True)
class RobotJointStateSnapshot:
    """一个已注册机器人 articulation 的一帧关节状态。

    所有向量均按 ``joint_names`` 排列，位置、速度和加速度分别使用 rad、rad/s 和
    rad/s²。某类 effort 无法读取时保留 ``None``；API 存在但本帧读取失败时则用同长度
    ``NaN`` 向量表示，序列化时转换为 JSON ``null``。
    """

    robot_id: int
    label: str
    joint_names: tuple[str, ...]
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray
    accelerations_rad_s2: np.ndarray
    commanded_efforts: np.ndarray | None = None
    measured_efforts: np.ndarray | None = None
    applied_efforts: np.ndarray | None = None

    def effort_values(self, field: str) -> np.ndarray | None:
        """按 Foxglove ``JointStates.effort`` 配置选择一类力矩向量。

        ``none`` 明确关闭该字段；其余值返回对应采样结果。未知名称属于配置错误，会抛出
        ``ValueError``，避免默默发布语义错误的力矩。
        """

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

        result = {
            "joint_names": list(self.joint_names),
            "positions_rad": _json_vector(self.positions_rad),
            "velocities_rad_s": _json_vector(self.velocities_rad_s),
            "accelerations_rad_s2": _json_vector(self.accelerations_rad_s2),
            "commanded_efforts": _optional_json_vector(self.commanded_efforts),
            "measured_efforts": _optional_json_vector(self.measured_efforts),
            "applied_efforts": _optional_json_vector(self.applied_efforts),
        }
        result["robot_id"] = int(self.robot_id)
        result["label"] = self.label
        return result


@dataclass(frozen=True)
class ObjectPoseSnapshot:
    """一个 runtime object 的 USD 世界位姿。

    ``position_m`` 使用米，``orientation_wxyz`` 使用 wxyz 四元数顺序；对象名只作为
    快照键，实际 USD 身份由 ``prim_path`` 保留。
    """

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
    """交互实时模式的一帧完整、可跨线程传递的状态快照。

    ``step`` 是完成本次 physics step 前使用的零基序号，``time_s`` 是该 step 完成后的
    仿真时间。``phase`` 用于区分 motion/hold 等执行阶段，不存在时不写入 JSON。
    """

    step: int
    time_s: float
    robots: tuple[RobotJointStateSnapshot, ...]
    objects: tuple[ObjectPoseSnapshot, ...] = ()
    phase: str | None = None
    hybrid_control: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的完整快照。"""

        result: dict[str, object] = {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "robots": [robot.as_dict() for robot in self.robots],
            "objects": {obj.name: obj.as_dict() for obj in self.objects},
        }
        if self.phase is not None:
            result["phase"] = self.phase
        if self.hybrid_control is not None:
            result["hybrid_control"] = deepcopy(dict(self.hybrid_control))
        return result


class StateStream:
    """有界状态快照通道；生产者发布时永不等待后台 telemetry。

    ``latest`` 每次发布只保留最新帧；``drop_oldest`` 在满载时淘汰最早帧；
    ``drop_newest`` 在满载时拒绝新帧。序号对每次被处理的 ``publish`` 调用递增，即使
    新帧因 ``drop_newest`` 被丢弃也不会复用序号。``wait_next`` 会消费返回的队列项，因而
    该通道按单消费者语义设计。
    """

    def __init__(
        self,
        *,
        capacity: int = 1,
        drop_policy: str = "latest",
    ) -> None:
        """初始化快照通道并校验背压策略。"""

        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if drop_policy not in {"latest", "drop_oldest", "drop_newest"}:
            raise ValueError(
                "drop_policy must be one of: latest, drop_oldest, drop_newest"
            )
        self.capacity = capacity
        self.drop_policy = drop_policy
        self._condition = Condition()
        self._sequence = 0
        self._items: deque[tuple[int, StateSnapshot]] = deque()
        self._dropped = 0
        self._closed = False

    def publish(self, snapshot: StateSnapshot) -> int:
        """非阻塞发布一帧快照，并返回当前递增序号。

        通道关闭后的发布不修改队列或序号，直接返回最后序号；丢帧策略的累计结果可通过
        ``status`` 观察。
        """

        with self._condition:
            if self._closed:
                return self._sequence
            self._sequence += 1
            item = (self._sequence, snapshot)
            if self.drop_policy == "latest":
                self._dropped += len(self._items)
                self._items.clear()
                self._items.append(item)
            elif len(self._items) >= self.capacity:
                self._dropped += 1
                if self.drop_policy == "drop_oldest":
                    self._items.popleft()
                    self._items.append(item)
            else:
                self._items.append(item)
            self._condition.notify_all()
            return self._sequence

    def latest(self) -> tuple[int, StateSnapshot] | None:
        """返回最新快照和序号；还没有快照时返回 None。"""

        with self._condition:
            if not self._items:
                return None
            return self._items[-1]

    def wait_next(
        self, after_sequence: int = 0, *, timeout_s: float | None = None
    ) -> tuple[int, StateSnapshot] | None:
        """等待并消费一个比 ``after_sequence`` 更新的快照。

        超时，或通道关闭且没有待消费快照时返回 ``None``。调用期间会清除不新于指定序号
        的队列项，调用方应把上次返回的序号继续传入，避免重复处理。
        """

        with self._condition:
            if not self._condition.wait_for(
                lambda: (
                    self._closed
                    or any(
                        sequence > int(after_sequence) for sequence, _ in self._items
                    )
                ),
                timeout=timeout_s,
            ):
                return None
            if self._closed and not self._items:
                return None
            while self._items and self._items[0][0] <= int(after_sequence):
                self._items.popleft()
            if not self._items:
                return None
            return self._items.popleft()

    def is_closed(self) -> bool:
        """返回通道是否已拒绝后续生产者发布。"""

        with self._condition:
            return self._closed

    def status(self) -> dict[str, object]:
        """返回有界队列深度、容量和累计丢帧数。"""

        with self._condition:
            return {
                "buffer_depth": len(self._items),
                "buffer_capacity": self.capacity,
                "drop_policy": self.drop_policy,
                "dropped_snapshots": self._dropped,
                "last_sampled_sequence": self._sequence,
                "closed": self._closed,
            }

    def close(self, *, discard_pending: bool = False) -> None:
        """停止接收新快照，并唤醒所有等待者。

        默认保留已经进入队列的快照供消费者排空；``discard_pending=True`` 会立即丢弃并
        计入累计丢帧数。重复关闭是幂等的。
        """

        with self._condition:
            if discard_pending:
                self._dropped += len(self._items)
                self._items.clear()
            self._closed = True
            self._condition.notify_all()


class SceneRobotStateSampler:
    """按 registry 顺序在主线程采样全部机器人和可选场景对象。

    采样器不按机器人侧别或数量分派逻辑；每个机器人的关节维度来自 articulation。速度
    历史只用于有限差分加速度，reset 后首帧因没有前值而明确输出 ``NaN``。
    """

    def __init__(
        self,
        *,
        stage,
        object_handles: Sequence[object] = (),
        rate_hz: float = 60.0,
        include_efforts: bool = False,
        include_objects: bool = False,
        include_hybrid_control: bool = False,
        hybrid_diagnostics_provider: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        """保存采样依赖和频率设置，不读取任何 Isaac 状态。

        ``stage`` 与 ``object_handles`` 仅在 ``include_objects`` 为真时使用；实际读取推迟到
        ``sample``，以确保调用发生在仿真主线程。
        """

        self.stage = stage
        self.object_handles = tuple(object_handles)
        self.rate_hz = float(rate_hz)
        self.include_efforts = bool(include_efforts)
        self.include_objects = bool(include_objects)
        self.include_hybrid_control = bool(include_hybrid_control)
        self.hybrid_diagnostics_provider = hybrid_diagnostics_provider
        self._previous_velocities: dict[int, tuple[float, np.ndarray]] = {}

    def should_sample(self, *, step: int, physics_dt: float) -> bool:
        """把目标 Hz 量化为整数 physics-step 间隔并判断当前全局 step。

        采样频率高于 physics 频率时钳制为每步一次；非正 ``rate_hz`` 表示禁用采样。
        """

        if self.rate_hz <= 0:
            return False
        interval = max(1, int(round(1.0 / (physics_dt * self.rate_hz))))
        return int(step) % interval == 0

    def reset(self) -> None:
        """清除 velocity history，使 reset 后第一帧 acceleration 返回 NaN。"""

        self._previous_velocities.clear()

    def sample(
        self,
        runtime,
        *,
        step: int,
        phase: str | None = None,
    ) -> StateSnapshot:
        """在主线程冻结本 step 的机器人、可选力矩与对象位姿。

        返回值不再引用 Isaac 对象，可由后台线程安全消费。调用方必须保证当前线程拥有
        runtime/stage 的读取权。
        """

        physics_dt = _runtime_physics_dt(runtime)
        time_s = (int(step) + 1) * physics_dt
        robots = tuple(
            self._sample_robot(robot, time_s=time_s)
            for robot in runtime.robots_by_id.values()
        )
        objects = (
            _sample_runtime_objects(self.stage, self.object_handles)
            if self.include_objects and self.stage is not None
            else ()
        )
        hybrid_control = self._sample_hybrid_diagnostics()
        return StateSnapshot(
            step=int(step),
            time_s=time_s,
            phase=phase,
            robots=robots,
            objects=objects,
            hybrid_control=hybrid_control,
        )

    def _sample_hybrid_diagnostics(self) -> dict[str, object] | None:
        """冻结 owner 缓存，不从 telemetry 路径读取 articulation 或 PhysX。"""

        if not self.include_hybrid_control:
            return None
        if self.hybrid_diagnostics_provider is None:
            return {"active": False}
        payload = self.hybrid_diagnostics_provider()
        if not isinstance(payload, Mapping) or type(payload.get("active")) is not bool:
            raise ValueError("hybrid diagnostics provider returned an invalid payload")
        if payload["active"] is False:
            return {"active": False}
        return deepcopy(dict(payload))

    def _sample_robot(self, runtime, *, time_s: float) -> RobotJointStateSnapshot:
        """采样单 robot，并用相邻 velocity frame 估算 acceleration。"""

        execution = runtime.execution
        articulation = execution.articulation
        positions = _vector_from_method(
            articulation, "get_joint_positions", _num_dof(articulation)
        )
        velocities = _vector_from_method(
            articulation, "get_joint_velocities", positions.size
        )
        previous = self._previous_velocities.get(runtime.robot_id)
        self._previous_velocities[runtime.robot_id] = (time_s, velocities.copy())
        if previous is None or time_s <= previous[0]:
            accelerations = np.full(velocities.size, np.nan, dtype=float)
        else:
            accelerations = (velocities - previous[1]) / (time_s - previous[0])
        commanded = measured = applied = None
        if self.include_efforts:
            commanded = _commanded_efforts(execution.joint_controller, positions.size)
            efforts = read_joint_efforts(
                articulation, None, measured=True, applied=True
            )
            measured = efforts.measured
            applied = efforts.applied
        return RobotJointStateSnapshot(
            robot_id=runtime.robot_id,
            label=runtime.label,
            joint_names=tuple(str(name) for name in articulation.dof_names),
            positions_rad=positions,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            commanded_efforts=commanded,
            measured_efforts=measured,
            applied_efforts=applied,
        )


class SceneRobotStateObserver:
    """在共享 World step 后按频率发布与后端对象解耦的场景快照。"""

    def __init__(self, *, sampler: SceneRobotStateSampler, stream: StateStream) -> None:
        """绑定主线程采样器与单一输出通道。"""

        self.sampler = sampler
        self.stream = stream

    def observe(self, runtime, *, step: int, phase: str | None = None) -> None:
        """命中 sampling interval 时冻结 snapshot 并写入 ``StateStream``。"""

        dt = _runtime_physics_dt(runtime)
        if self.sampler.should_sample(step=step, physics_dt=dt):
            self.stream.publish(self.sampler.sample(runtime, step=step, phase=phase))

    def reset(self) -> None:
        """把 reset 传播给 sampler 的 derivative history。"""

        self.sampler.reset()


def _runtime_physics_dt(runtime: object) -> float:
    """从产品 runtime 的显式 physics port 读取步长。

    MirrorSceneResources 暴露 ``physics``，ExecutionRuntime 暴露 ``simulation_world``；
    telemetry 不再要求已经删除的 ``IsaacSession.world`` facade。保留最后的 ``world``
    structural 分支仅服务该产品无关采样器的轻量调用方，不会在 Mirror composition 中使用。
    """

    for name in ("physics", "simulation_world", "world"):
        physics = getattr(runtime, name, None)
        get_dt = getattr(physics, "get_physics_dt", None)
        if callable(get_dt):
            dt = float(get_dt())
            if np.isfinite(dt) and dt > 0.0:
                return dt
            raise ValueError(f"runtime.{name}.get_physics_dt() 必须返回有限正数")
    raise RuntimeError("state sampler runtime 缺少 physics time-step port")


def _num_dof(robot) -> int:
    """读取 articulation DOF 数。"""

    if hasattr(robot, "num_dof"):
        return int(robot.num_dof)
    return len(getattr(robot, "dof_names", ()))


def _sample_runtime_objects(
    stage,
    object_handles: Sequence[object],
) -> tuple[ObjectPoseSnapshot, ...]:
    """采样 runtime objects 的 root prim 世界位姿。

    没有可解析 prim path、prim 已失效或暂时无法读取位姿的对象会跳过；单个对象缺失不应
    阻断同一帧其余机器人遥测。
    """

    snapshots: list[ObjectPoseSnapshot] = []
    for handle in object_handles:
        prim_path = runtime_object_prim_path(handle)
        if prim_path is None:
            continue
        pose = read_prim_world_pose(stage, prim_path)
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
    """调用无参状态读取方法；缺失、异常或维度错误时返回 ``NaN`` 向量。

    遥测采样不得因某个可选 Isaac getter 的瞬时失败中断仿真主循环，同时必须保持关节
    列数稳定，因此失败被编码为固定形状的缺失值。
    """

    method = getattr(source, method_name, None)
    if method is None:
        return np.full(int(expected_size), np.nan, dtype=float)
    try:
        values = tensor_like_to_numpy(method(), dtype=float).reshape(-1)
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
