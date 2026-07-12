"""机器人会话身份 registry 与 per-consumer planning context 池。

``RobotRegistry`` 只管理稠密 session ID 和稳定 label；``RobotPlanningRegistry`` 才拥有
cuRobo context。context key 包含 ``(robot_id, consumer_role, worker_slot)``，避免交互线程、
IK worker 和 planner worker 共享 cuRobo 内部可变 solver/cache。共享的是 immutable scene
snapshot，GPU checker 仍由各 context 独占。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Literal

from linkerbot_sim.app.runtime.collision.registry import PlanningSceneSnapshot
from linkerbot_sim.robots.capabilities import PlanningCapability, RobotKind
from linkerbot_sim.robots.joint_groups import JointGroupLayout


ConsumerRole = Literal["interactive", "ik", "planner", "mpc"]
CoordinationPolicy = Literal["independent", "static_others", "coupled"]


@dataclass
class RobotRuntime:
    """一个 articulation 的 simulation handle、控制器、分组和 planning 元数据。"""

    robot_id: int
    label: str
    kind: RobotKind
    profile_name: str
    controller_profile: str
    profile_config: Mapping[str, object]
    scene_instance: object
    imported: object
    prepared: object
    execution: object
    joint_groups: JointGroupLayout
    planning_capability: PlanningCapability
    curobo_config: object | None = None

    @property
    def supports_planning(self) -> bool:
        """返回 robot binding 是否满足创建 planning context 的静态条件。"""

        return self.planning_capability.supports_planning

    @property
    def articulation(self):
        """暴露 execution 持有的 Isaac articulation handle。"""

        return self.execution.articulation

    @property
    def controller(self):
        """暴露 execution 持有的 command-space joint controller。"""

        return self.execution.joint_controller

    def status(self) -> dict[str, object]:
        """生成不触发 GPU/context 分配的 robot discovery 摘要。"""

        return {
            "robot_id": int(self.robot_id),
            "label": self.label,
            "robot_profile": self.profile_name,
            "controller_profile": self.controller_profile,
            "profile_fingerprint": self.profile_fingerprint,
            "kind": self.kind.value,
            "supports_planning": self.supports_planning,
            # Collision-aware capability is context- and snapshot-specific.
            # SingleSceneRuntime.status replaces this conservative value when a
            # context has actually been materialized and synchronized.
            "supports_collision_aware_planning": False,
            "planning_joint_group": self.planning_capability.planning_joint_group,
            "joint_groups": {
                "arm": list(self.joint_groups.arm),
                "hand": list(self.joint_groups.hand),
                "passive": list(self.joint_groups.passive),
            },
        }

    @property
    def profile_fingerprint(self) -> str:
        """计算稳定 profile 内容指纹，供 status 与 snapshot 兼容性校验使用。"""

        payload = json.dumps(
            self.profile_config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class RobotRegistry:
    """稠密 ID 主索引与稳定 label 精确反向索引。

    ID 由当前 env robots list 顺序生成，只用于本次会话；跨会话 snapshot 恢复依赖 label 和
    profile fingerprint，不能把 ID 当持久身份。
    """

    def __init__(self, robots: Mapping[int, RobotRuntime] | tuple[RobotRuntime, ...]):
        values = (
            tuple(robots.values()) if isinstance(robots, Mapping) else tuple(robots)
        )
        by_id = {int(robot.robot_id): robot for robot in values}
        expected = list(range(len(values)))
        if sorted(by_id) != expected:
            raise ValueError(
                f"robot IDs must be dense and ordered, expected {expected}, got {sorted(by_id)}"
            )
        by_label = {robot.label: robot.robot_id for robot in values}
        if len(by_label) != len(values):
            raise ValueError("robot labels must be unique")
        self.robots_by_id = by_id
        self.robot_id_by_label = by_label

    def robot(self, robot_id: int) -> RobotRuntime:
        """按本次 session 的稠密 ID 查找 robot，并在错误中列出可用身份。"""

        try:
            return self.robots_by_id[int(robot_id)]
        except (KeyError, TypeError, ValueError) as exc:
            available = [
                {"robot_id": item.robot_id, "label": item.label}
                for item in self.robots_by_id.values()
            ]
            raise KeyError(
                f"unknown robot_id {robot_id!r}; available={available}"
            ) from exc

    def robot_by_label(self, label: str) -> RobotRuntime:
        """按稳定 label 精确查找 robot；该接口主要供内部匹配而非 public selector。"""

        try:
            return self.robot(self.robot_id_by_label[str(label)])
        except KeyError as exc:
            raise KeyError(
                f"unknown robot label {label!r}; available={list(self.robot_id_by_label)}"
            ) from exc

    def resolve(
        self,
        robot_id: int,
        *,
        robot_label: str | None = None,
    ) -> RobotRuntime:
        """按 ID 解析 robot，并可同时校验客户端携带的 label assertion。"""

        robot = self.robot(robot_id)
        if robot_label is not None and str(robot_label) != robot.label:
            raise ValueError(
                f"robot_id {robot_id} is label {robot.label!r}, not {robot_label!r}"
            )
        return robot

    def status(self) -> tuple[dict[str, object], ...]:
        """按 robot ID 顺序返回整个 registry 的 discovery rows。"""

        return tuple(robot.status() for robot in self.robots_by_id.values())

    def __len__(self) -> int:
        return len(self.robots_by_id)


@dataclass
class _ContextEntry:
    """一个独占 mutable solver context 及其 collision sync 缓存。"""

    context: object
    lock: RLock = field(default_factory=RLock)
    synced_scene_version: int | None = None
    materialized_view_fingerprint: str | None = None
    sync_duration_s: float = 0.0
    obstacle_count: int = 0


class RobotPlanningRegistry:
    """按 robot、consumer role 和 worker slot 延迟创建 context 的资源池。

    cuRobo solver、CUDA graph 和 collision checker 都包含可变缓存，因此同一个 key 由
    ``lease`` 串行使用，不同 worker slot 则获得物理隔离的 context。
    """

    _VALID_ROLES = frozenset({"interactive", "ik", "planner", "mpc"})

    def __init__(
        self,
        robots: RobotRegistry,
        *,
        context_factory: Callable[[RobotRuntime], object] | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self.robots = robots
        self._context_factory = (
            context_factory
            if context_factory is not None
            else lambda robot: _default_context_factory(robot, cache_root=cache_root)
        )
        self._entries: dict[tuple[int, str, int], _ContextEntry] = {}
        self._lock = RLock()
        self._closed = False

    def context(
        self,
        robot_id: int,
        *,
        consumer_role: ConsumerRole = "interactive",
        worker_slot: int = 0,
    ) -> object:
        """返回指定 key 的 context；首次访问会完成 lazy materialization。"""

        return self._entry(robot_id, consumer_role, worker_slot).context

    @contextmanager
    def lease(
        self,
        robot_id: int,
        *,
        consumer_role: ConsumerRole = "interactive",
        worker_slot: int = 0,
    ) -> Iterator[object]:
        """租用并串行保护一个 mutable solver context。"""

        entry = self._entry(robot_id, consumer_role, worker_slot)
        with entry.lock:
            yield entry.context

    def sync_before_plan(
        self,
        robot_id: int,
        snapshot: PlanningSceneSnapshot,
        *,
        consumer_role: ConsumerRole = "interactive",
        worker_slot: int = 0,
        force: bool = False,
        coordination: CoordinationPolicy = "static_others",
    ) -> object:
        """按目标 robot 过滤 scene view，并只在 fingerprint 变化时同步到 context。

        ``static_others`` 会包含其它 robot 几何，``independent`` 只包含公共障碍物；当前没有
        ``coupled`` backend。scene version 与 materialized view fingerprint 同时命中时直接
        复用 collision world，避免重复上传 GPU cache。
        """

        if coordination == "coupled":
            raise RuntimeError(
                "coordination='coupled' requires a coupled planning backend; none is configured"
            )
        entry = self._entry(robot_id, consumer_role, worker_slot)
        robot = self.robots.robot(robot_id)
        include_robots = coordination == "static_others"
        model_fingerprint = _model_fingerprint(robot)
        view_fingerprint = snapshot.view_fingerprint(
            robot_id,
            include_other_robots=include_robots,
            model_fingerprint=model_fingerprint,
        )
        with entry.lock:
            if (
                not force
                and entry.synced_scene_version == snapshot.version
                and entry.materialized_view_fingerprint == view_fingerprint
            ):
                return getattr(entry.context, "collision_world", None)
            objects = snapshot.collision_objects_for(
                robot_id,
                include_other_robots=include_robots,
            )
            started = perf_counter()
            sync = getattr(entry.context, "sync_collision_world", None)
            if not callable(sync):
                raise RuntimeError(
                    "planning context cannot synchronize collision world"
                )
            world = sync(objects)
            entry.sync_duration_s = perf_counter() - started
            entry.obstacle_count = len(objects)
            entry.synced_scene_version = snapshot.version
            entry.materialized_view_fingerprint = view_fingerprint
            record = getattr(entry.context, "record_collision_sync", None)
            if callable(record):
                record(snapshot.version, view_fingerprint)
            return world

    def close(
        self,
        robot_id: int | None = None,
        *,
        consumer_role: str | None = None,
    ) -> None:
        """关闭匹配的 context；失败项保留所有权，供后续 ``close`` 重试。"""

        with self._lock:
            keys = [
                key
                for key in self._entries
                if (robot_id is None or key[0] == int(robot_id))
                and (consumer_role is None or key[1] == str(consumer_role))
            ]
            entries = [(key, self._entries[key]) for key in keys]
            if robot_id is None and consumer_role is None:
                self._closed = True
        first_error: BaseException | None = None
        for key, entry in entries:
            try:
                with entry.lock:
                    close = getattr(entry.context, "close", None)
                    if callable(close):
                        close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                continue
            with self._lock:
                if self._entries.get(key) is entry:
                    self._entries.pop(key)
        if first_error is not None:
            raise first_error

    def metrics(self) -> dict[str, object]:
        """返回 context 数量、collision sync 状态和可获取的 CUDA memory 指标。"""

        with self._lock:
            entries = tuple(self._entries.items())
        return {
            "context_count": len(entries),
            "contexts": [
                {
                    "robot_id": key[0],
                    "consumer_role": key[1],
                    "worker_slot": key[2],
                    "synced_scene_version": entry.synced_scene_version,
                    "materialized_view_fingerprint": entry.materialized_view_fingerprint,
                    "sync_duration_s": entry.sync_duration_s,
                    "obstacle_count": entry.obstacle_count,
                }
                for key, entry in entries
            ],
            **_cuda_memory_metrics(entries),
        }

    def collision_capability(
        self,
        robot_id: int,
        *,
        consumer_role: ConsumerRole = "interactive",
        worker_slot: int = 0,
    ) -> object | None:
        """检查已存在 context 的 collision capability，不触发 GPU allocation。"""

        key = (int(robot_id), str(consumer_role), int(worker_slot))
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        with entry.lock:
            inspect = getattr(entry.context, "collision_capability", None)
            return inspect() if callable(inspect) else None

    def _entry(self, robot_id: int, role: str, worker_slot: int) -> _ContextEntry:
        """校验 context key，并在 registry lock 内原子地 lazy-create entry。"""

        normalized_role = str(role).lower()
        if normalized_role not in self._VALID_ROLES:
            raise ValueError(
                f"consumer_role must be one of {sorted(self._VALID_ROLES)}"
            )
        slot = int(worker_slot)
        if slot < 0:
            raise ValueError("worker_slot must be non-negative")
        robot = self.robots.robot(robot_id)
        robot.planning_capability.require()
        key = (robot.robot_id, normalized_role, slot)
        with self._lock:
            if self._closed:
                raise RuntimeError("RobotPlanningRegistry is closed")
            entry = self._entries.get(key)
            if entry is None:
                entry = _ContextEntry(self._context_factory(robot))
                self._entries[key] = entry
            return entry


def _default_context_factory(
    robot: RobotRuntime,
    *,
    cache_root: str | Path | None = None,
) -> object:
    """从 RobotRuntime 的 merged cuRobo config 创建默认 ``CuroboContext``。"""

    if robot.curobo_config is None:
        raise RuntimeError(f"robot {robot.label!r} has no cuRobo planning config")
    from linkerbot_sim.backends.curobo.context import CuroboContext

    return CuroboContext(robot.curobo_config, cache_root=cache_root)


def _model_fingerprint(robot: RobotRuntime) -> str:
    """对 robot model 路径、文件内容和 arm joint order 计算 collision view 指纹。"""

    config = robot.curobo_config
    model = getattr(config, "robot", None)
    values = (
        getattr(model, "robot_config_path", None),
        getattr(model, "urdf_path", None),
        tuple(robot.joint_groups.arm),
    )
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _cuda_memory_metrics(entries) -> dict[str, int | None]:
    """从第一个可用 Torch CUDA context 读取 process-level memory 指标。"""

    for _, entry in entries:
        torch = getattr(entry.context, "torch", None)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            try:
                return {
                    "cuda_allocated_bytes": int(cuda.memory_allocated()),
                    "cuda_reserved_bytes": int(cuda.memory_reserved()),
                }
            except Exception:
                break
    return {"cuda_allocated_bytes": None, "cuda_reserved_bytes": None}


__all__ = [
    "ConsumerRole",
    "CoordinationPolicy",
    "RobotPlanningRegistry",
    "RobotRegistry",
    "RobotRuntime",
]
