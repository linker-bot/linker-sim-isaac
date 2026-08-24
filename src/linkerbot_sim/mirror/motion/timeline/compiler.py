"""把 Mirror 多机器人逻辑请求原子编译为可执行整数 tick timeline。

编译器只读取 articulation；包含 planning segment 时额外冻结 collision snapshot。所有
track 成功后才返回 timeline，规划失败不会向 PhysX 写入部分机器人状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from linkerbot_sim.mirror.motion.timeline.builders import (
    _direct_trajectory,
    _expand_full_command_unit,
    _joint_goal,
    make_goal_segment,
    make_effort_segment,
    make_hold_segment,
    make_trajectory_segment,
    sequential_group_track,
    sequential_robot_track,
)
from linkerbot_sim.mirror.motion.timeline.model import (
    HoldSegment,
    JointGroupTrack,
    MotionSegment,
    RobotMotionUnit,
    RobotTimeline,
)
from linkerbot_sim.mirror.motion.timeline.requests import (
    JointGroupTrackRequest,
    RobotTimelineRequest,
    TimelineSegmentRequest,
)
from linkerbot_sim.backends.curobo.call_guard import call_curobo
from linkerbot_sim.backends.curobo.trajectory_adapter import (
    joint_trajectory_from_motion_result,
)
from linkerbot_sim.planning.backend import (
    PlannerBackend,
    PlanningRequest,
    normalize_planner_backend,
)
from linkerbot_sim.planning.frames import FrameTransformer
from linkerbot_sim.planning.linear_backend import LinearPlannerBackend
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    MotionRequest,
    PoseTarget,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.trajectories.types import JointTrajectory
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


@dataclass(frozen=True)
class TimelinePlanningLocation:
    """定位嵌套请求中失败 segment 的稳定坐标。"""

    robot_id: int
    robot_label: str
    track_index: int
    unit_index: int
    group: str
    segment_index: int


class TimelinePlanningError(RuntimeError):
    """携带 robot/track/unit/group/segment 位置的原子规划失败。"""

    def __init__(
        self,
        message: str,
        *,
        location: TimelinePlanningLocation,
        cause: BaseException | None = None,
    ) -> None:
        cause_detail = (
            "" if cause is None else f"; cause={type(cause).__name__}: {cause}"
        )
        super().__init__(
            f"{message}; robot_id={location.robot_id} "
            f"label={location.robot_label!r} track={location.track_index} "
            f"unit={location.unit_index} group={location.group} "
            f"segment={location.segment_index}{cause_detail}"
        )
        self.location = location
        self.__cause__ = cause

    def as_dict(self) -> dict[str, object]:
        """把嵌套失败位置转换为可直接放入 JSON diagnostics 的 mapping。"""

        return {
            "message": str(self),
            "robot_id": self.location.robot_id,
            "label": self.location.robot_label,
            "track_index": self.location.track_index,
            "unit_index": self.location.unit_index,
            "group": self.location.group,
            "segment_index": self.location.segment_index,
        }


class TimelinePlanningSession:
    """原子编译全部 robot tracks，并为 planning segments 共享一个 collision snapshot。

    同一 session 复用 runtime registry 和 physics dt；每次 compile 都重新读取机器人
    command state，避免 reset 或外部写状态后继续使用陈旧 seed。
    """

    def __init__(
        self,
        runtime: object,
        *,
        consumer_role: str = "interactive",
        worker_slot: int = 0,
        planner_backend: str = "curobo",
    ) -> None:
        self.runtime = runtime
        self.consumer_role = str(consumer_role)
        self.worker_slot = int(worker_slot)
        physics = getattr(runtime, "physics", None)
        if physics is None:
            physics = getattr(runtime, "simulation_world", None)
        if physics is None:
            raise RuntimeError("timeline runtime is missing a physics adapter")
        self.physics_dt = float(physics.get_physics_dt())
        self.planner_backend = normalize_planner_backend(planner_backend)

    def compile(self, request: RobotTimelineRequest) -> RobotTimeline:
        """全部 segment 成功后才返回可执行 timeline。"""

        collision_registry = self.runtime.collision_registry
        snapshot = (
            collision_registry.snapshot()
            if _request_requires_planning_snapshot(request)
            else None
        )
        tracks = []
        for track_index, track_request in enumerate(request.tracks):
            robot = self.runtime.robot_registry.resolve(
                track_request.robot_id,
                robot_label=track_request.robot_label,
            )
            command = _current_command(robot)
            units = []
            for unit_index, raw_unit in enumerate(track_request.units):
                unit_request = _expand_full_command_unit(robot, raw_unit)
                baseline = command.copy()
                group_tracks = []
                group_final: dict[str, float] = {}
                for group_request in unit_request.group_tracks:
                    try:
                        group_track, final = self._compile_group_track(
                            robot=robot,
                            request=group_request,
                            command_baseline=baseline,
                            snapshot=snapshot,
                            coordination=request.coordination,
                            force_collision_refresh=(request.force_collision_refresh),
                            track_index=track_index,
                            unit_index=unit_index,
                        )
                    except TimelinePlanningError:
                        raise
                    except Exception as exc:
                        location = TimelinePlanningLocation(
                            robot.robot_id,
                            robot.label,
                            track_index,
                            unit_index,
                            group_request.group,
                            -1,
                        )
                        raise TimelinePlanningError(
                            "failed to compile joint group",
                            location=location,
                            cause=exc,
                        ) from exc
                    overlap = set(group_final) & set(final)
                    if overlap:
                        raise ValueError(
                            f"motion unit has multiple final writers: {sorted(overlap)}"
                        )
                    group_final.update(final)
                    group_tracks.append(group_track)
                for name, value in group_final.items():
                    command[robot.joint_groups.command_joint_names.index(name)] = value
                units.append(RobotMotionUnit(tuple(group_tracks)))
            tracks.append(
                sequential_robot_track(
                    robot.robot_id,
                    units,
                    robot_label=robot.label,
                )
            )
        metadata: dict[str, object] = {"command_id": request.command_id}
        if snapshot is not None:
            metadata["scene_fingerprint"] = snapshot.fingerprint
        return RobotTimeline(
            tracks=tuple(tracks),
            physics_dt=self.physics_dt,
            coordination=request.coordination,
            scene_version=(
                snapshot.version if snapshot is not None else collision_registry.version
            ),
            metadata=metadata,
        )

    def _compile_group_track(
        self,
        *,
        robot: object,
        request: JointGroupTrackRequest,
        command_baseline: np.ndarray,
        snapshot: object | None,
        coordination: str,
        force_collision_refresh: bool,
        track_index: int,
        unit_index: int,
    ) -> tuple[JointGroupTrack, dict[str, float]]:
        """串行编译一个 arm/hand group，并返回 track 与 terminal command state。"""

        names = robot.joint_groups.names(request.group)
        if not names:
            raise ValueError(
                f"robot kind {robot.kind.value!r} has no {request.group} group"
            )
        command_names = robot.joint_groups.command_joint_names
        command_by_name = dict(zip(command_names, command_baseline, strict=True))
        current = np.asarray([command_by_name[name] for name in names], dtype=float)
        segments = []
        for segment_index, segment_request in enumerate(request.segments):
            location = TimelinePlanningLocation(
                robot.robot_id,
                robot.label,
                track_index,
                unit_index,
                request.group,
                segment_index,
            )
            try:
                segment = self._compile_segment(
                    robot=robot,
                    group=request.group,
                    group_names=names,
                    current=current,
                    request=segment_request,
                    snapshot=snapshot,
                    coordination=coordination,
                    force_collision_refresh=(
                        force_collision_refresh
                        or segment_request.force_collision_refresh
                    ),
                )
            except TimelinePlanningError:
                raise
            except Exception as exc:
                raise TimelinePlanningError(
                    "segment planning failed",
                    location=location,
                    cause=exc,
                ) from exc
            segments.append(segment)
            terminal = segment.terminal_sample()
            if terminal is not None:
                index_by_name = {name: index for index, name in enumerate(names)}
                for column, name in enumerate(segment.joint_names):
                    current[index_by_name[name]] = terminal.positions[column]
        return (
            sequential_group_track(request.group, segments),
            dict(zip(names, current, strict=True)),
        )

    def _compile_segment(
        self,
        *,
        robot: object,
        group: str,
        group_names: tuple[str, ...],
        current: np.ndarray,
        request: TimelineSegmentRequest,
        snapshot: object | None,
        coordination: str,
        force_collision_refresh: bool,
    ) -> MotionSegment | HoldSegment:
        """把 request segment 编译到 physics grid，并推进该 group 的 command baseline。"""

        phase = request.phase or request.kind
        if request.kind == "hold":
            return make_hold_segment(
                joint_names=group_names,
                positions=current,
                duration_s=request.duration_s,
                physics_dt=self.physics_dt,
                phase=phase,
            )
        if request.kind == "joint_effort":
            efforts = _joint_goal(
                request.joint_efforts,
                group_names,
                np.zeros_like(current),
            )
            return make_effort_segment(
                joint_names=group_names,
                hold_positions=current,
                joint_efforts=efforts,
                duration_s=request.duration_s,
                physics_dt=self.physics_dt,
                phase=phase,
                metadata={"backend": "direct", **dict(request.metadata)},
            )
        if request.kind in {"joint_goal", "joint_delta"}:
            target = _joint_goal(
                request.joint_positions,
                group_names,
                np.zeros_like(current) if request.kind == "joint_delta" else current,
            )
            if request.kind == "joint_delta":
                target = current + target
            return make_goal_segment(
                joint_names=group_names,
                start_positions=current,
                target_positions=target,
                duration_s=request.duration_s,
                physics_dt=self.physics_dt,
                phase=phase,
                interpolation=request.interpolation,
                metadata={
                    "backend": "direct",
                    "interpolation": request.interpolation,
                    **dict(request.metadata),
                },
            )
        if request.kind == "joint_trajectory":
            trajectory = _direct_trajectory(
                request,
                joint_names=group_names,
                current=current,
                physics_dt=self.physics_dt,
                phase=phase,
            )
            return make_trajectory_segment(
                trajectory,
                physics_dt=self.physics_dt,
                requested_duration_s=request.duration_s,
                start_positions=current,
                phase=phase,
                metadata={"backend": "direct", **dict(request.metadata)},
            )
        if group != "arm":
            raise RuntimeError("planning can only target an arm joint group")
        trajectory = self._plan_segment_with_backend(
            robot=robot,
            group_names=group_names,
            request=request,
            current=current,
            snapshot=snapshot,
            coordination=coordination,
            force_collision_refresh=force_collision_refresh,
        )
        return make_trajectory_segment(
            trajectory,
            physics_dt=self.physics_dt,
            requested_duration_s=request.duration_s,
            start_positions=current,
            phase=phase,
            metadata={
                "backend": self.planner_backend,
                "collision_aware": request.avoid_collisions,
                "sample_dt_s": self._sample_dt_s(request),
                **dict(request.metadata),
            },
        )

    def _plan_segment_with_backend(
        self,
        *,
        robot: object,
        group_names: tuple[str, ...],
        request: TimelineSegmentRequest,
        current: np.ndarray,
        snapshot: object | None,
        coordination: str,
        force_collision_refresh: bool,
    ) -> JointTrajectory:
        """按 backend 名称分派 joint linear 或 cuRobo planning。"""

        if self.planner_backend == "linear":
            if request.kind not in {"plan_cspace_goal", "plan_cspace_delta"}:
                raise RuntimeError(
                    "linear planner backend only supports plan_cspace_goal and "
                    "plan_cspace_delta in mirror mode"
                )
            backend_request = _joint_backend_planning_request(
                request=request,
                current=current,
                joint_names=group_names,
                sample_dt_s=self._sample_dt_s(request),
            )
            return _plan_with_backend(
                LinearPlannerBackend(
                    group_names,
                    default_duration_s=request.duration_s,
                    default_sample_dt_s=self.physics_dt,
                ),
                backend_request,
                joint_names=group_names,
                duration_s=request.duration_s,
                sample_dt_s=self._sample_dt_s(request),
                phase=request.phase or request.kind,
                timeout_s=request.timeout_s,
            )
        if snapshot is None:
            raise RuntimeError("planning segment requires a collision snapshot")
        return self._plan_segment_with_curobo(
            robot=robot,
            request=request,
            current=current,
            snapshot=snapshot,
            coordination=coordination,
            force_collision_refresh=force_collision_refresh,
        )

    def _plan_segment_with_curobo(
        self,
        *,
        robot: object,
        request: TimelineSegmentRequest,
        current: np.ndarray,
        snapshot: object,
        coordination: str,
        force_collision_refresh: bool,
    ) -> JointTrajectory:
        """租用独占 cuRobo context，同步 collision view，并执行 canonical request。"""

        robot.planning_capability.require(request.kind)
        planning_registry = self.runtime.planning_registry
        with planning_registry.lease(
            robot.robot_id,
            consumer_role=self.consumer_role,
            worker_slot=self.worker_slot,
        ) as context:
            planning_registry.sync_before_plan(
                robot.robot_id,
                snapshot,
                consumer_role=self.consumer_role,
                worker_slot=self.worker_slot,
                force=force_collision_refresh,
                coordination=coordination,
            )
            joint_names = tuple(context.joint_names())
            robot.joint_groups.validate_planning_joints(joint_names)
            backend_request, tcp_frame_name = _backend_planning_request(
                robot=robot,
                context=context,
                request=request,
                current=current,
                sample_dt_s=self._sample_dt_s(request),
            )
            if request.avoid_collisions:
                capability = context.ensure_collision_checker("planner")
                if not capability.available:
                    missing = ", ".join(capability.missing_requirements)
                    raise RuntimeError(
                        "collision-aware planning is unavailable; "
                        f"missing={missing}; capability={capability}"
                    )
            planner = context.make_motion_planner(tcp_frame_name=tcp_frame_name)
            return _plan_with_backend(
                planner,
                backend_request,
                joint_names=joint_names,
                duration_s=request.duration_s,
                sample_dt_s=self._sample_dt_s(request),
                phase=request.phase or request.kind,
                guard_curobo=True,
                timeout_s=request.timeout_s,
            )

    def _sample_dt_s(self, request: TimelineSegmentRequest) -> float:
        """返回 segment 的显式 planner period，省略时使用 physics dt。"""

        return (
            self.physics_dt
            if request.sample_dt_s is None
            else float(request.sample_dt_s)
        )


def _request_requires_planning_snapshot(request: RobotTimelineRequest) -> bool:
    """判断请求中是否存在会读取规划场景几何的 segment。

    只有规划与 IK 类 segment 需要在编译前冻结场景；纯关节插值、保持等轨迹不触发
    不必要的 planning snapshot。
    """

    planning_kinds = {
        "plan_cspace_goal",
        "plan_cspace_delta",
        "ik_pose",
        "ik_offset",
        "plan_linear_pose_path",
    }
    return any(
        segment.kind in planning_kinds
        for track in request.tracks
        for unit in track.units
        for group_track in unit.group_tracks
        for segment in group_track.segments
    )


def _plan_with_backend(
    backend: PlannerBackend,
    request: PlanningRequest,
    *,
    joint_names: tuple[str, ...],
    duration_s: float,
    sample_dt_s: float,
    phase: str,
    guard_curobo: bool = False,
    timeout_s: float | None = None,
) -> JointTrajectory:
    """执行 canonical planning request，并把成功结果归一化为 ``JointTrajectory``。

    cuRobo 0.8 的同步 API 不提供可安全中断 GPU solver 的 deadline；这里的 timeout 是
    结果接纳上限：超时结果不会进入执行 timeline。它不能回收已经花费的求解时间，但能
    防止过期轨迹继续写入真实映像场景。
    """

    if not isinstance(backend, PlannerBackend):
        raise TypeError("planner backend does not implement the shared contract")
    started = monotonic()
    result = (
        call_curobo("motion planner", backend.plan, request)
        if guard_curobo
        else backend.plan(request)
    )
    elapsed = monotonic() - started
    if timeout_s is not None and elapsed > float(timeout_s):
        raise TimeoutError(
            "planning result exceeded acceptance timeout: "
            f"elapsed_s={elapsed:.6f} timeout_s={float(timeout_s):.6f}"
        )
    if not result.success:
        backend_name = str(getattr(backend, "backend_name", type(backend).__name__))
        raise RuntimeError(
            f"{backend_name} planning failed: status={result.status} "
            f"diagnostics={result.diagnostics.message}"
        )
    return joint_trajectory_from_motion_result(
        result,
        joint_names=joint_names,
        duration_s=duration_s,
        sample_dt=sample_dt_s,
        phase=phase,
    )


def _backend_planning_request(
    *,
    robot: object,
    context: object,
    request: TimelineSegmentRequest,
    current: np.ndarray,
    sample_dt_s: float,
) -> tuple[MotionRequest | LinearPosePathRequest, str]:
    """把 timeline segment 转成 cuRobo facade 接受的 canonical request。"""

    tcp = request.tcp_frame_name or context.default_tcp_frame
    if request.kind in {"plan_cspace_goal", "plan_cspace_delta"}:
        result = _joint_backend_planning_request(
            request=request,
            current=current,
            joint_names=tuple(context.joint_names()),
            sample_dt_s=sample_dt_s,
            tcp_frame_name=tcp,
        )
        return result, tcp

    tcp_positions, tcp_orientations = context.compute_tcp_poses(
        current.reshape(1, -1), tcp_frame_name=tcp
    )
    current_position = np.asarray(tcp_positions, dtype=float).reshape(-1, 3)[0]
    current_orientation = np.asarray(tcp_orientations, dtype=float).reshape(-1, 4)[0]
    transformer = FrameTransformer.from_root_pose(
        robot.scene_instance.root_pose,
        tcp_position_in_base=current_position,
        tcp_orientation_wxyz_in_base=current_orientation,
    )
    if request.kind == "ik_pose":
        target = transformer.pose_to_robot_base(
            position=np.asarray(request.target_position, dtype=float),
            orientation_wxyz=(
                None
                if request.target_orientation_wxyz is None
                else np.asarray(request.target_orientation_wxyz, dtype=float)
            ),
            reference_frame=str(request.reference_frame),
        )
        result = MotionRequest(
            current_q=current,
            goal_pose=PoseTarget(target.position, target.orientation_wxyz),
            tcp_frame_name=tcp,
            duration_s=request.duration_s,
            sample_dt_s=sample_dt_s,
            avoid_collisions=request.avoid_collisions,
        )
        result.validate_structure()
        return result, tcp
    if request.kind == "ik_offset":
        offset = transformer.offset_to_robot_base(
            np.asarray(request.offset, dtype=float),
            offset_frame=str(request.offset_frame),
        )
        result = MotionRequest(
            current_q=current,
            goal_pose=PoseTarget(
                current_position + offset,
                (
                    None
                    if request.target_orientation_wxyz is None
                    else np.asarray(request.target_orientation_wxyz, dtype=float)
                ),
            ),
            tcp_frame_name=tcp,
            duration_s=request.duration_s,
            sample_dt_s=sample_dt_s,
            avoid_collisions=request.avoid_collisions,
        )
        result.validate_structure()
        return result, tcp
    if request.kind == "plan_linear_pose_path":
        if request.target_position is not None:
            target_orientation = (
                request.target_orientation_wxyz
                if request.orientation_mode == "target"
                else None
            )
            target = transformer.pose_to_robot_base(
                position=np.asarray(request.target_position, dtype=float),
                orientation_wxyz=(
                    None
                    if target_orientation is None
                    else np.asarray(target_orientation, dtype=float)
                ),
                reference_frame=str(request.reference_frame),
            )
            line = TcpLineSegment(
                target_position=target.position,
                orientation_mode=request.orientation_mode,
                target_orientation=target.orientation_wxyz,
            )
        else:
            offset = transformer.offset_to_robot_base(
                np.asarray(request.offset, dtype=float),
                offset_frame=str(request.offset_frame),
            )
            line = TcpLineSegment(
                target_offset=offset,
                orientation_mode=request.orientation_mode,
                target_orientation=(
                    None
                    if request.orientation_mode != "target"
                    else np.asarray(request.target_orientation_wxyz, dtype=float)
                ),
            )
        result = LinearPosePathRequest(
            current_q=current,
            path=TaskSpacePath((line,)),
            tcp_frame_name=tcp,
            duration_s=request.duration_s,
            sample_dt_s=sample_dt_s,
            avoid_collisions=request.avoid_collisions,
        )
        result.validate_structure()
        return result, tcp
    raise ValueError(f"unsupported planning request kind: {request.kind}")


def _joint_backend_planning_request(
    *,
    request: TimelineSegmentRequest,
    current: np.ndarray,
    joint_names: tuple[str, ...],
    sample_dt_s: float,
    tcp_frame_name: str | None = None,
) -> MotionRequest:
    """构造不携带 backend 私有状态的 canonical joint-space request。"""

    if request.kind not in {"plan_cspace_goal", "plan_cspace_delta"}:
        raise ValueError(f"{request.kind} is not a joint-space planning request")
    goal = _joint_goal(
        request.joint_positions,
        joint_names,
        np.zeros_like(current) if request.kind == "plan_cspace_delta" else current,
    )
    if request.kind == "plan_cspace_delta":
        goal = current + goal
    result = MotionRequest(
        current_q=current,
        goal_q=goal,
        tcp_frame_name=tcp_frame_name,
        duration_s=request.duration_s,
        sample_dt_s=sample_dt_s,
        avoid_collisions=request.avoid_collisions,
    )
    result.validate_structure()
    return result


def _current_command(robot: object) -> np.ndarray:
    """按 controller command indices 读取当前 command-space joint positions。"""

    articulation = robot.execution.articulation
    controller = robot.execution.joint_controller
    full = tensor_like_to_numpy(
        articulation.get_joint_positions(), dtype=float
    ).reshape(-1)
    return full[np.asarray(controller.command_indices, dtype=int)].copy()


__all__ = [
    "TimelinePlanningError",
    "TimelinePlanningLocation",
    "TimelinePlanningSession",
]
