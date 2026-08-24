"""Mirror timeline 编译、规划与主线程执行的正式后端。"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from linkerbot_sim.configuration.modes.mirror import MirrorConfig
from linkerbot_sim.controllers.control_mode import (
    ControlModeIncompatibleError,
    require_control_mode,
)
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.mirror.lifecycle import close_result_stopped
from linkerbot_sim.mirror.motion.request_parser import parse_mirror_motion_request
from linkerbot_sim.mirror.motion.hybrid_executor import (
    HybridExecutionError,
    MirrorHybridExecutor,
)
from linkerbot_sim.mirror.motion.timeline.compiler import TimelinePlanningSession
from linkerbot_sim.mirror.motion.timeline.executor import (
    TimelineExecutionInterrupted,
    TimelinePostStepError,
    execute_robot_timeline,
)
from linkerbot_sim.mirror.motion.timeline.model import RobotTimeline
from linkerbot_sim.mirror.motion.timeline.requests import (
    JointGroupTrackRequest,
    RobotMotionUnitRequest,
    RobotTimelineRequest,
    RobotTrackRequest,
    TimelineSegmentRequest,
)
from linkerbot_sim.mirror.timing import WallClockStepSynchronizer


class MirrorTimelineBackend:
    """把 Mirror v1 motion operation 落到同一个整数 tick timeline。

    该对象唯一拥有 ``TimelinePlanningSession`` 所使用的按机器人 cuRobo planning
    contexts。它只借用场景资源与物理 session；关闭时释放 planning registry，但绝不
    关闭 IsaacSession，后者始终由 :class:`MirrorRuntime` 负责。
    """

    resource_name = "mirror_timeline_backend"

    def __init__(self, resources: object, *, config: MirrorConfig) -> None:
        self._resources = resources
        self._config = config
        self._planner = TimelinePlanningSession(
            resources,
            # Mirror mode 已通过唯一 profiles.curobo 选择数值后端；这里不再读取第二个
            # backend selector。
            planner_backend="curobo",
        )
        self._step = 0
        self._render_frame: Callable[[], object] | None = None
        self._step_synchronizer: WallClockStepSynchronizer | None = None
        self._control_mode_provider: Callable[[], ControlMode] = lambda: (
            config.control.mode
        )
        self._control_mode_provider_bound = False
        self._hybrid = MirrorHybridExecutor(
            resources,
            settings=config.hybrid_control,
            physics_engine=config.physics.engine,
            physics_execution=config.physics.execution,
        )
        self._render_required = bool(
            config.outputs.render.enabled
            or config.outputs.camera.enabled
            or config.scene.cameras
        )
        self._closed = False

    def bind_render_frame(self, callback: Callable[[], object]) -> None:
        """绑定产品根拥有的唯一 render transaction，且只允许绑定一次。"""

        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        if not callable(callback):
            raise TypeError("timeline render callback must be callable")
        if self._render_frame is not None:
            raise RuntimeError("timeline render callback is already bound")
        self._render_frame = callback
        self._hybrid.bind_render_frame(callback)

    def bind_step_synchronizer(self, synchronizer: WallClockStepSynchronizer) -> None:
        """绑定产品根拥有的唯一 physics tick 墙钟。"""

        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        if not isinstance(synchronizer, WallClockStepSynchronizer):
            raise TypeError("timeline step synchronizer has an invalid type")
        if self._step_synchronizer is not None:
            raise RuntimeError("timeline step synchronizer is already bound")
        self._step_synchronizer = synchronizer
        self._hybrid.bind_before_step(synchronizer.before_step)

    def bind_control_mode_provider(
        self,
        provider: Callable[[], ControlMode],
    ) -> None:
        """Bind the runtime-owned logical mode query exactly once."""

        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        if not callable(provider):
            raise TypeError("timeline control-mode provider must be callable")
        if self._control_mode_provider_bound:
            raise RuntimeError("timeline control-mode provider is already bound")
        self._control_mode_provider = provider
        self._control_mode_provider_bound = True
        self._hybrid.bind_control_mode_provider(provider)

    def bind_hybrid_parameter_provider(self, provider: Callable[[], object]) -> None:
        """Bind the runtime-owned immutable gain snapshot provider."""

        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        if not callable(provider):
            raise TypeError("hybrid parameter provider must be callable")
        self._hybrid.bind_parameter_provider(provider)  # type: ignore[arg-type]

    def bind_runtime_owner(self, runtime: object) -> None:
        """Bind fail-stop ownership after the MirrorRuntime root exists."""

        self._hybrid.bind_runtime_owner(runtime)

    @property
    def step_count(self) -> int:
        """返回最近一次完整执行后的全局 physics tick。"""

        return self._step

    def hybrid_status(self) -> dict[str, object]:
        """Return cached capability/tare state without sampling physics."""

        return self._hybrid.status()

    def hybrid_diagnostics(self) -> dict[str, object]:
        """Return the latest frozen control sample without sampling physics."""

        return self._hybrid.diagnostics()

    def execute(
        self,
        operation: str,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: Callable[[], bool],
        protocol: str = "linkerbot.mirror.v1",
    ) -> dict[str, object]:
        """严格解析、原子编译并在 owner thread 执行一次 motion 请求。"""

        self._require_ready()
        if operation == "motion.hybrid_force_position":
            if protocol != "linkerbot.mirror.v3":
                raise ValueError(
                    "motion.hybrid_force_position requires linkerbot.mirror.v3"
                )
            try:
                result, completed_step = self._hybrid.execute(
                    arguments,
                    start_step=self._step,
                    should_cancel=should_cancel,
                    request_id=request_id,
                )
            except HybridExecutionError as exc:
                self._step = exc.step
                raise
            self._step = completed_step
            return result
        request = parse_mirror_motion_request(
            operation,
            arguments,
            request_id=request_id,
            config=self._config,
            allow_effort=protocol in {"linkerbot.mirror.v2", "linkerbot.mirror.v3"},
        )
        _require_mode_compatible(
            request,
            active_mode=require_control_mode(self._control_mode_provider()),
            operation=operation,
        )
        timeline = self._planner.compile(request)
        self._execute_and_commit_step(
            timeline,
            start_step=self._step,
            should_stop=should_cancel,
        )
        return {
            "event": "motion_completed",
            "operation": operation,
            "steps": self._step,
        }

    def tare_wrench(
        self,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: Callable[[], bool],
    ) -> dict[str, object]:
        """Run a v3 tare transaction and commit its shared physics clock."""

        del request_id
        self._require_ready()
        try:
            result, completed_step = self._hybrid.tare_wrench(
                arguments,
                start_step=self._step,
                should_cancel=should_cancel,
            )
        except HybridExecutionError as exc:
            self._step = exc.step
            raise
        self._step = completed_step
        return result

    def after_scene_reset(self, *, hold_duration_s: float | None) -> int:
        """重置 timeline 时钟，并按需为全部机器人执行一个同步 hold。

        Physics reset 已把 engine 状态恢复到初值，因此旧的累计 step 不能穿过 reset。
        hold 使用与普通 motion 相同的 planner、整数 tick executor 和 render coordinator，
        不在 controller 旁边维护第二套逐机器人写入循环。
        """

        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        self._step = 0
        self._hybrid.invalidate_tare()
        if self._step_synchronizer is not None:
            self._step_synchronizer.rebase()
        if hold_duration_s is None:
            return self._step
        self._require_ready()
        request = _all_robot_hold_request(
            self._resources,
            duration_s=float(hold_duration_s),
        )
        self._execute_and_commit_step(
            self._planner.compile(request),
            start_step=0,
        )
        return self._step

    def _execute_and_commit_step(
        self,
        timeline: RobotTimeline,
        *,
        start_step: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """提交 executor 已实际推进的 tick，包括携带 committed step 的异常路径。"""

        try:
            completed_step = execute_robot_timeline(
                self._resources,
                timeline,
                start_step=start_step,
                should_stop=should_stop,
                render_frame=self._render_frame,
                before_step=(
                    None
                    if self._step_synchronizer is None
                    else self._step_synchronizer.before_step
                ),
            )
        except (TimelineExecutionInterrupted, TimelinePostStepError) as exc:
            # 这两类异常都可能发生在若干 world.step 已提交之后。先吸收异常携带的
            # authoritative step 再原样抛出，避免下一条 motion 从过期时钟继续执行。
            self._step = exc.step
            raise
        self._step = completed_step

    def _require_ready(self) -> None:
        if self._closed:
            raise RuntimeError("Mirror timeline backend is closed")
        if self._render_required and self._render_frame is None:
            raise RuntimeError(
                "Mirror rendering is enabled, but the timeline has not bound a RenderCoordinator yet"
            )

    def close(self) -> object:
        """幂等释放 planner contexts；关闭失败时保留 owner 供下一次重试。"""

        if self._closed:
            return True
        registry = getattr(self._resources, "planning_registry", None)
        callback = getattr(registry, "close", None)
        result = True if not callable(callback) else callback()
        if close_result_stopped(result):
            self._hybrid.close()
            self._render_frame = None
            self._step_synchronizer = None
            self._control_mode_provider = lambda: self._config.control.mode
            self._closed = True
        return result


def _all_robot_hold_request(
    resources: object,
    *,
    duration_s: float,
) -> RobotTimelineRequest:
    """按当前 robot/group 拓扑构造 reset 后共享 tick 的内部 hold 请求。"""

    tracks: list[RobotTrackRequest] = []
    for robot in resources.robots_by_id.values():
        groups = tuple(
            JointGroupTrackRequest(
                group=group,  # type: ignore[arg-type]
                segments=(
                    TimelineSegmentRequest(
                        kind="hold",
                        duration_s=duration_s,
                        phase="reset_hold",
                    ),
                ),
            )
            for group in ("arm", "hand")
            if robot.joint_groups.names(group)
        )
        if not groups:
            raise RuntimeError(
                f"Mirror robot {robot.label!r} has no arm/hand group that can execute a reset hold"
            )
        tracks.append(
            RobotTrackRequest(
                robot_id=robot.robot_id,
                robot_label=robot.label,
                units=(RobotMotionUnitRequest(groups),),
            )
        )
    if not tracks:
        raise RuntimeError("Mirror reset hold requires at least one robot")
    return RobotTimelineRequest(tracks=tuple(tracks))


def _require_mode_compatible(
    request: RobotTimelineRequest,
    *,
    active_mode: ControlMode,
    operation: str,
) -> None:
    """Reject incompatible segments before compilation can read collision state."""

    for track_index, track in enumerate(request.tracks):
        for unit_index, unit in enumerate(track.units):
            for group_track in unit.group_tracks:
                for segment_index, segment in enumerate(group_track.segments):
                    compatible = (
                        segment.kind != "joint_effort"
                        if active_mode in {"position", "velocity"}
                        else segment.kind in {"hold", "joint_effort"}
                    )
                    if compatible:
                        continue
                    location = {
                        "track_index": track_index,
                        "unit_index": unit_index,
                        "robot_id": track.robot_id,
                        "robot_label": track.robot_label,
                        "group": group_track.group,
                        "segment_index": segment_index,
                        "segment_kind": segment.kind,
                    }
                    raise ControlModeIncompatibleError(
                        f"operation {operation!r} segment {segment.kind!r} is "
                        f"incompatible with active control mode {active_mode!r}",
                        active_mode=active_mode,
                        operation=operation,
                        location=location,
                    )


__all__ = ["MirrorTimelineBackend"]
