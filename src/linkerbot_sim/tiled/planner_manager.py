"""tiled 场景外部的异步规划管理器。

planner manager 是 tiled runtime 外围的生产者：它接收已经从主线程复制出的状态快照，
在线程池里计算 ``JointTrajectory``，再把 ready result 交给 trajectory buffer 回放。worker
不能访问 Isaac ``World``、stage 或 articulation view，这样 PhysX 仍保持单 scene 同步推进。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from linkerbot_sim.tiled.trajectory import TiledTrajectoryOverlay


@dataclass(frozen=True)
class TiledPlanningSegment:
    """一次 tiled 规划请求中的一个逻辑运动段。

    ``goal_positions`` 表示关节空间目标，shape 为 ``(len(env_ids), command_dim)``；
    ``path`` 表示 specified-path 几何对象，例如 ``TaskSpacePath`` 或 ``CSpaceWaypointPath``。
    这里保持纯数据边界，不导入 cuMotion，也不访问 Isaac runtime。
    """

    kind: str
    duration_s: float | None = None
    sample_dt_s: float | None = None
    goal_positions: np.ndarray | None = None
    path: object | None = None
    tcp_frame_name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结并校验单段规划的结构性约束。"""

        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("planning segment kind cannot be empty")
        if self.duration_s is not None and float(self.duration_s) <= 0.0:
            raise ValueError("planning segment duration_s must be positive")
        if self.sample_dt_s is not None and float(self.sample_dt_s) <= 0.0:
            raise ValueError("planning segment sample_dt_s must be positive")
        goal = None
        if self.goal_positions is not None:
            goal = np.asarray(self.goal_positions, dtype=float)
            if goal.ndim != 2:
                raise ValueError("planning segment goal_positions must have shape (E, D)")
        tcp_frame_name = (
            None
            if self.tcp_frame_name is None
            else str(self.tcp_frame_name).strip()
        )
        if tcp_frame_name == "":
            raise ValueError("planning segment tcp_frame_name cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "goal_positions", None if goal is None else goal.copy())
        object.__setattr__(self, "tcp_frame_name", tcp_frame_name)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TiledPlanningRequest:
    """一次 tiled 规划请求。

    ``current_positions`` 已经由主线程按 robot command-space 补齐，shape 固定为
    ``(len(env_ids), command_dim)``。worker 只消费这些 numpy 副本，不读取 runtime 当前
    状态。

    旧的单段关节目标使用 ``goal_positions``；新版 MoveSpec/路径队列使用 ``segments``，
    每段可以是关节目标，也可以携带 specified-path 几何给支持该能力的后端。
    """

    request_id: str
    robot_name: str
    env_ids: tuple[int, ...]
    current_positions: np.ndarray
    joint_names: tuple[str, ...]
    goal_positions: np.ndarray | None = None
    duration_s: float = 1.0
    sample_dt_s: float = 0.02
    source: str = "interactive"
    load_on_success: bool = True
    replace: bool = True
    segments: tuple[TiledPlanningSegment, ...] = ()
    trajectory_overlays: tuple[TiledTrajectoryOverlay, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结并校验请求的结构性约束。"""

        env_ids = tuple(int(env_id) for env_id in self.env_ids)
        if not env_ids:
            raise ValueError("env_ids cannot be empty")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("env_ids cannot contain duplicates")
        current = np.asarray(self.current_positions, dtype=float)
        if current.ndim != 2:
            raise ValueError("current_positions must have shape (E, D)")
        if current.shape[0] != len(env_ids):
            raise ValueError("current_positions first dimension must match env_ids")
        if current.shape[1] != len(self.joint_names):
            raise ValueError("joint_names length must match command dimension")
        if float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive")
        if float(self.sample_dt_s) <= 0.0:
            raise ValueError("sample_dt_s must be positive")
        goal = None if self.goal_positions is None else np.asarray(self.goal_positions, dtype=float)
        segments = tuple(self.segments)
        if not segments and goal is None:
            raise ValueError("planning request requires goal_positions or segments")
        if goal is not None and goal.shape != current.shape:
            raise ValueError("goal_positions must match current_positions shape")
        for index, segment in enumerate(segments):
            if segment.goal_positions is not None and segment.goal_positions.shape != current.shape:
                raise ValueError(
                    f"segments[{index}].goal_positions must match current_positions shape"
                )
        object.__setattr__(self, "env_ids", env_ids)
        object.__setattr__(self, "current_positions", current.copy())
        object.__setattr__(self, "goal_positions", None if goal is None else goal.copy())
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "trajectory_overlays", tuple(self.trajectory_overlays))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TiledPlanningResult:
    """一次异步规划的结果。"""

    request_id: str
    robot_name: str
    env_ids: tuple[int, ...]
    success: bool
    status: str
    message: str
    times: np.ndarray
    positions: np.ndarray
    joint_names: tuple[str, ...]
    source: str = "planner"
    load_on_success: bool = True
    replace: bool = True
    trajectory_overlays: tuple[TiledTrajectoryOverlay, ...] = ()

    def __post_init__(self) -> None:
        """校验成功结果的轨迹矩阵形状。"""

        times = np.asarray(self.times, dtype=float).reshape(-1)
        positions = np.asarray(self.positions, dtype=float)
        if self.success:
            if times.size == 0:
                raise ValueError("successful planning result requires times")
            if positions.ndim != 3:
                raise ValueError("successful planning result positions must be (E,T,D)")
            if positions.shape[0] != len(self.env_ids):
                raise ValueError("positions first dimension must match env_ids")
            if positions.shape[1] != times.size:
                raise ValueError("positions sample dimension must match times")
            if positions.shape[2] != len(self.joint_names):
                raise ValueError("joint_names length must match positions width")
        object.__setattr__(self, "env_ids", tuple(int(env_id) for env_id in self.env_ids))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        object.__setattr__(self, "trajectory_overlays", tuple(self.trajectory_overlays))

    @classmethod
    def failed(
        cls,
        request: TiledPlanningRequest,
        *,
        status: str,
        message: str,
    ) -> "TiledPlanningResult":
        """构造失败结果，保留 request 元数据便于交互层回报。"""

        return cls(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=False,
            status=str(status),
            message=str(message),
            times=np.asarray([], dtype=float),
            positions=np.empty((0, 0, 0), dtype=float),
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=False,
            replace=request.replace,
            trajectory_overlays=(),
        )

    def to_json(self) -> dict[str, object]:
        """返回不包含大矩阵的状态摘要。"""

        return {
            "request_id": self.request_id,
            "robot": self.robot_name,
            "env_ids": list(self.env_ids),
            "success": bool(self.success),
            "status": self.status,
            "message": self.message,
            "samples": int(self.times.size),
            "joint_names": list(self.joint_names),
            "source": self.source,
            "load_on_success": bool(self.load_on_success),
            "overlay_count": len(self.trajectory_overlays),
        }


class TiledPlannerBackend(Protocol):
    """异步 manager 可调用的 planner backend 协议。"""

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """执行一次规划并返回项目侧统一结果。"""


class LinearJointPlannerBackend:
    """可测试的关节空间线性 planner。

    它不做碰撞检查或速度优化，但提供真实的异步规划/回放数据流：从 snapshot current 到
    goal 生成带时间戳的 batched C-space trajectory。真实 cuMotion backend 可以替换同一接口。
    """

    def plan(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """线性插值 current -> goal。"""

        if request.segments:
            return _plan_linear_segments(request)
        if request.goal_positions is None:
            return TiledPlanningResult.failed(
                request,
                status="UNSUPPORTED",
                message="linear planner requires joint-space goal_positions",
            )
        steps = max(1, int(np.ceil(float(request.duration_s) / float(request.sample_dt_s))))
        times = np.linspace(0.0, float(request.duration_s), steps + 1)
        alpha = (times / float(request.duration_s)).reshape(1, -1, 1)
        positions = (
            request.current_positions[:, None, :]
            + (request.goal_positions - request.current_positions)[:, None, :] * alpha
        )
        return TiledPlanningResult(
            request_id=request.request_id,
            robot_name=request.robot_name,
            env_ids=request.env_ids,
            success=True,
            status="SUCCESS",
            message="linear joint trajectory generated",
            times=times,
            positions=positions,
            joint_names=request.joint_names,
            source=request.source,
            load_on_success=request.load_on_success,
            replace=request.replace,
            trajectory_overlays=request.trajectory_overlays,
        )


def _plan_linear_segments(request: TiledPlanningRequest) -> TiledPlanningResult:
    """用默认 linear backend 拼接多段关节空间规划。"""

    current = request.current_positions.copy()
    global_times: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    elapsed_s = 0.0
    for index, segment in enumerate(request.segments):
        if segment.goal_positions is None:
            return TiledPlanningResult.failed(
                request,
                status="UNSUPPORTED",
                message=(
                    "linear planner only supports joint-space segments; "
                    f"segment {index} kind={segment.kind!r} has no goal_positions"
                ),
            )
        duration_s = _segment_duration_s(request, segment)
        sample_dt_s = _segment_sample_dt_s(request, segment)
        local_times, local_positions = _linear_segment_samples(
            current=current,
            goal=segment.goal_positions,
            duration_s=duration_s,
            sample_dt_s=sample_dt_s,
        )
        if position_parts:
            global_times.append(elapsed_s + local_times[1:])
            position_parts.append(local_positions[:, 1:, :])
        else:
            global_times.append(elapsed_s + local_times)
            position_parts.append(local_positions)
        elapsed_s += duration_s
        current = segment.goal_positions.copy()
    return TiledPlanningResult(
        request_id=request.request_id,
        robot_name=request.robot_name,
        env_ids=request.env_ids,
        success=True,
        status="SUCCESS",
        message="linear joint plan queue generated",
        times=np.concatenate(global_times, axis=0),
        positions=np.concatenate(position_parts, axis=1),
        joint_names=request.joint_names,
        source=request.source,
        load_on_success=request.load_on_success,
        replace=request.replace,
        trajectory_overlays=request.trajectory_overlays,
    )


def _linear_segment_samples(
    *,
    current: np.ndarray,
    goal: np.ndarray,
    duration_s: float,
    sample_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一段 batched linear joint trajectory。"""

    steps = max(1, int(np.ceil(float(duration_s) / float(sample_dt_s))))
    times = np.linspace(0.0, float(duration_s), steps + 1)
    alpha = (times / float(duration_s)).reshape(1, -1, 1)
    positions = current[:, None, :] + (goal - current)[:, None, :] * alpha
    return times, positions


def _segment_duration_s(
    request: TiledPlanningRequest,
    segment: TiledPlanningSegment,
) -> float:
    """返回单段有效 duration。"""

    return float(request.duration_s if segment.duration_s is None else segment.duration_s)


def _segment_sample_dt_s(
    request: TiledPlanningRequest,
    segment: TiledPlanningSegment,
) -> float:
    """返回单段有效采样间隔。"""

    return float(request.sample_dt_s if segment.sample_dt_s is None else segment.sample_dt_s)


class TiledPlannerManager:
    """多线程 tiled planner 请求管理器。"""

    def __init__(
        self,
        *,
        backend: TiledPlannerBackend | None = None,
        max_workers: int = 2,
        max_pending_requests: int | None = 64,
        max_completed_results: int | None = 256,
    ) -> None:
        """创建 manager。"""

        if int(max_workers) < 1:
            raise ValueError("max_workers must be positive")
        if max_pending_requests is not None and int(max_pending_requests) < 1:
            raise ValueError("max_pending_requests must be positive or None")
        if max_completed_results is not None and int(max_completed_results) < 0:
            raise ValueError("max_completed_results must be non-negative or None")
        self.backend = backend or LinearJointPlannerBackend()
        self.max_pending_requests = (
            None if max_pending_requests is None else int(max_pending_requests)
        )
        self.max_completed_results = (
            None if max_completed_results is None else int(max_completed_results)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="tiled-planner",
        )
        self._futures: dict[str, Future[TiledPlanningResult]] = {}
        self._requests: dict[str, TiledPlanningRequest] = {}
        self._completed: dict[str, TiledPlanningResult] = {}
        self._cancelled: set[str] = set()

    def submit(self, request: TiledPlanningRequest) -> str:
        """提交规划请求并返回 request id。"""

        request_id = str(request.request_id)
        if request_id in self._futures or request_id in self._completed:
            raise ValueError(f"duplicate planning request_id: {request_id}")
        if (
            self.max_pending_requests is not None
            and len(self._futures) >= self.max_pending_requests
        ):
            raise RuntimeError(
                "too many pending planning requests: "
                f"{len(self._futures)} >= {self.max_pending_requests}"
            )
        future = self._executor.submit(self._run_backend, request)
        self._futures[request_id] = future
        self._requests[request_id] = request
        return request_id

    def collect_ready(self, *, timeout_s: float = 0.0) -> tuple[TiledPlanningResult, ...]:
        """收集已经完成的 planner results。"""

        if timeout_s > 0.0 and self._futures:
            wait(
                tuple(self._futures.values()),
                timeout=float(timeout_s),
                return_when=FIRST_COMPLETED,
            )
        ready: list[TiledPlanningResult] = []
        for request_id, future in list(self._futures.items()):
            if not future.done():
                continue
            request = self._requests.pop(request_id)
            self._futures.pop(request_id)
            if request_id in self._cancelled:
                result = TiledPlanningResult.failed(
                    request,
                    status="CANCELLED",
                    message="planning request was cancelled",
                )
                self._cancelled.discard(request_id)
            elif future.cancelled():
                result = TiledPlanningResult.failed(
                    request,
                    status="CANCELLED",
                    message="planning future was cancelled",
                )
            else:
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - 防御 backend 意外异常
                    result = TiledPlanningResult.failed(
                        request,
                        status=type(exc).__name__,
                        message=str(exc),
                    )
            self._store_completed(request_id, result)
            ready.append(result)
        return tuple(ready)

    def cancel(self, request_id: str) -> dict[str, object]:
        """取消单个规划请求。"""

        key = str(request_id)
        future = self._futures.get(key)
        if future is None:
            return {
                "request_id": key,
                "accepted": False,
                "status": "not_found" if key not in self._completed else "completed",
            }
        self._cancelled.add(key)
        return {
            "request_id": key,
            "accepted": True,
            "status": "cancel_requested",
            "future_cancelled": bool(future.cancel()),
        }

    def cancel_matching(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        """按 robot/env 取消 in-flight 请求，用于 reset/set_state 防止 stale result。"""

        env_set = (
            None
            if env_ids is None
            else {int(env_id) for env_id in np.asarray(env_ids, dtype=int).reshape(-1)}
        )
        cancelled = []
        for request_id, request in list(self._requests.items()):
            if robot_name is not None and request.robot_name != str(robot_name):
                continue
            if env_set is not None and not (set(request.env_ids) & env_set):
                continue
            cancelled.append(self.cancel(request_id))
        return cancelled

    def status(self) -> dict[str, object]:
        """返回 manager 当前请求状态。"""

        self.collect_ready()
        pending = []
        for request_id, future in sorted(self._futures.items()):
            request = self._requests[request_id]
            pending.append(
                {
                    "request_id": request_id,
                    "robot": request.robot_name,
                    "env_ids": list(request.env_ids),
                    "running": bool(future.running()),
                    "cancel_requested": request_id in self._cancelled,
                }
            )
        return {
            "pending": pending,
            "pending_count": len(pending),
            "completed_count": len(self._completed),
            "max_pending_requests": self.max_pending_requests,
            "max_completed_results": self.max_completed_results,
            "completed": [
                result.to_json()
                for _, result in sorted(self._completed.items())
            ],
        }

    def clear_completed(
        self,
        request_ids: str | Sequence[str] | None = None,
    ) -> dict[str, object]:
        """清理 completed result 缓存，返回已清理和未找到的 request id。"""

        if request_ids is None:
            cleared = list(self._completed)
            self._completed.clear()
            return {"cleared": cleared, "missing": [], "count": len(cleared)}
        if isinstance(request_ids, str):
            keys = (request_ids,)
        else:
            keys = tuple(str(item) for item in request_ids)
        cleared: list[str] = []
        missing: list[str] = []
        for key in keys:
            if key in self._completed:
                self._completed.pop(key, None)
                cleared.append(key)
            else:
                missing.append(key)
        return {"cleared": cleared, "missing": missing, "count": len(cleared)}

    def shutdown(self) -> None:
        """关闭线程池。"""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_backend(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """线程池入口。"""

        return self.backend.plan(request)

    def _store_completed(
        self,
        request_id: str,
        result: TiledPlanningResult,
    ) -> None:
        """写入 completed 缓存，并按上限裁掉最旧结果。"""

        if self.max_completed_results == 0:
            return
        self._completed[request_id] = result
        if self.max_completed_results is None:
            return
        while len(self._completed) > self.max_completed_results:
            oldest = next(iter(self._completed))
            self._completed.pop(oldest, None)
