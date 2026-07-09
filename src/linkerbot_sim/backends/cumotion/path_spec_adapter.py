"""specified-path 请求到 cuMotion PathSpec 的适配层。

本模块只负责把项目侧 ``CSpaceWaypointPath`` / ``TaskSpacePath`` /
``CompositePath`` 转成 cuMotion 官方 PathSpec/conversion API，并把返回的
``LinearCSpacePath`` 归一化成 numpy ``joint_path``。pipeline 分发和
``MotionResult`` 构造留在 ``specified_path_planner.py``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from linkerbot_sim.backends.cumotion.context import validate_cumotion_frame
from linkerbot_sim.backends.cumotion.motion_planner_utils import (
    attr,
    validate_cspace_width,
)
from linkerbot_sim.backends.cumotion.pose_adapter import (
    pose_from_position_quat_wxyz,
    pose_from_rotation_translation,
    rotation_from_quat_wxyz,
)
from linkerbot_sim.planning.requests import (
    CSpaceWaypointPath,
    CompositePathPart,
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
    TcpPoseSequenceSegment,
    TcpRotationSegment,
)


def cspace_waypoints_to_joint_path(
    context,
    request,
    config: MotionPlannerBackendConfig,
) -> np.ndarray:
    """把 ``CSpaceWaypointPath`` 经官方 ``CSpacePathSpec`` 转为 joint path。

    C-space 指定路径是三类 specified-path 中最直接的一类：调用方已经给出完整的关节
    waypoint，所以 adapter 只做项目侧的结构校验，再把 waypoint 原样喂给 cuMotion 的
    ``create_cspace_path_spec`` / ``add_cspace_waypoint`` /
    ``create_linear_cspace_path``。这里不额外插入 ``current_q``，避免悄悄改变调用方给出的
    路径几何。
    """

    current = np.asarray(request.current_q, dtype=float).reshape(-1)
    waypoints = _validated_cspace_waypoints(context, request.path)
    # 默认要求第一个 waypoint 和请求起点一致。这个检查属于后端策略，因为容差来自
    # MotionPlannerBackendConfig；planning/requests.py 只做不依赖机器人模型的形状校验。
    _validate_first_waypoint_matches_current(current, waypoints[0], config)
    path_spec = _cspace_path_spec_from_waypoints(context.cumotion, waypoints)
    linear_path = context.cumotion.create_linear_cspace_path(path_spec)
    return _joint_path_from_linear_cspace_path(linear_path)


def task_space_path_to_joint_path(
    context,
    request,
    config: MotionPlannerBackendConfig,
    *,
    tcp_frame_name: str,
) -> np.ndarray:
    """把 ``TaskSpacePath`` 经官方 path conversion 转为 joint path。

    Task-space 路径的输入是 TCP 几何段，而最终的轨迹生成器消费 C-space waypoint。
    cuMotion官方流程是先构造 ``TaskSpacePathSpec``，再通过
    ``convert_task_space_path_spec_to_cspace`` 做 IK/path conversion。

    注意这里是“先计算后发送”：本函数在规划阶段一次性完成 task-space 到 C-space 的转换，
    返回的 joint path 随后会被 trajectory generator 时间参数化；执行阶段只按时间采样关节
    轨迹，不会实时调用 task-space conversion。
    """

    current = np.asarray(request.current_q, dtype=float).reshape(-1)
    validate_cspace_width(context, current, "current_q")
    _validate_frame(context, tcp_frame_name)
    path_spec = _task_space_path_spec_from_segments(
        context,
        current,
        request.path,
        tcp_frame_name,
    )
    conversion_config = _task_space_conversion_config(context.cumotion, config)
    ik_config = _ik_config_for_path_conversion(context, current, config)
    # 官方 conversion 需要 robot kinematics、控制 frame、路径转换数值配置和 IK 配置。真实
    # cuMotion 可能在 conversion 内部抛出不可达/数值失败异常；本层不吞掉异常，以便调用方看到
    # 更贴近后端的错误。
    linear_path = context.cumotion.convert_task_space_path_spec_to_cspace(
        path_spec,
        context.kinematics,
        tcp_frame_name,
        conversion_config,
        ik_config,
    )
    return _joint_path_from_linear_cspace_path(linear_path)


def composite_path_to_joint_path(
    context,
    request,
    config: MotionPlannerBackendConfig,
    *,
    tcp_frame_name: str,
) -> np.ndarray:
    """把 ``CompositePath`` 经官方 composite conversion 转为 joint path。

    Composite path 允许调用方把 C-space 子路径和 task-space 子路径拼成一条指定路径。项目侧
    只负责把每个子段转换成对应的 PathSpec 并指定段间 transition；最终的 C-space 化仍交给
    cuMotion ``convert_composite_path_spec_to_cspace``，这样 task-space 子段和过渡段都由同一套
    官方 path conversion 处理。

    和 ``TaskSpacePath`` 一样，composite conversion 是规划期动作。执行层拿到的是已经
    时间参数化的关节轨迹，而不是每帧混合解析 C-space/task-space 子段。
    """

    current = np.asarray(request.current_q, dtype=float).reshape(-1)
    validate_cspace_width(context, current, "current_q")
    _validate_frame(context, tcp_frame_name)
    composite_spec = context.cumotion.create_composite_path_spec(current)
    # tracked_current 只在“上一个已知末端仍然是 C-space waypoint”时可靠。task-space 子段的
    # conversion 结果由 cuMotion 后续统一求解，项目侧此时不知道该段末尾对应的精确 C-space。
    tracked_current = current
    can_validate_cspace_start = True
    for index, part in enumerate(request.path.parts):
        nested_path, transition_mode = _composite_part_and_transition(part, config)
        transition = _transition_mode(context.cumotion, transition_mode)
        if isinstance(nested_path, CSpaceWaypointPath):
            waypoints = _validated_cspace_waypoints(context, nested_path)
            if can_validate_cspace_start:
                # 对连续 C-space 子段保留“首点必须等于上一段末点”的约束。若上一段是
                # task-space，则不猜测 conversion 后的关节解，只让 cuMotion 负责段间连接。
                _validate_first_waypoint_matches_current(
                    tracked_current, waypoints[0], config
                )
            path_spec = _cspace_path_spec_from_waypoints(context.cumotion, waypoints)
            ok = composite_spec.add_cspace_path_spec(path_spec, transition)
            _ensure_added(ok, f"composite cspace part {index}")
            tracked_current = waypoints[-1]
            can_validate_cspace_start = True
        elif isinstance(nested_path, TaskSpacePath):
            path_spec = _task_space_path_spec_from_segments(
                context,
                tracked_current,
                nested_path,
                tcp_frame_name,
            )
            ok = composite_spec.add_task_space_path_spec(path_spec, transition)
            _ensure_added(ok, f"composite task-space part {index}")
            # task-space 子段之后，下一段若是 C-space，项目侧没有官方 conversion 的末端 C-space
            # 结果可用，因此不能继续做首点匹配检查。
            can_validate_cspace_start = False
        else:
            raise ValueError(
                f"Unsupported composite path part type: {type(nested_path).__name__}"
            )
    conversion_config = _task_space_conversion_config(context.cumotion, config)
    ik_config = _ik_config_for_path_conversion(context, current, config)
    linear_path = context.cumotion.convert_composite_path_spec_to_cspace(
        composite_spec,
        context.kinematics,
        tcp_frame_name,
        conversion_config,
        ik_config,
    )
    return _joint_path_from_linear_cspace_path(linear_path)


def _validated_cspace_waypoints(context, path: CSpaceWaypointPath) -> list[np.ndarray]:
    """校验并返回 C-space waypoint 列表。

    这里校验的是后端相关宽度：每个 waypoint 必须等于当前 cuMotion C-space 维度。请求层无法
    知道具体 robot description 暴露了哪些 C-space 坐标，所以只检查非空/基本形状。
    """

    waypoints = [
        np.asarray(waypoint, dtype=float).reshape(-1) for waypoint in path.waypoints
    ]
    if len(waypoints) < 2:
        raise ValueError("CSpaceWaypointPath requires at least 2 waypoints")
    for index, waypoint in enumerate(waypoints):
        validate_cspace_width(context, waypoint, f"path.waypoints[{index}]")
    return waypoints


def _validate_first_waypoint_matches_current(
    current: np.ndarray,
    first_waypoint: np.ndarray,
    config: MotionPlannerBackendConfig,
) -> None:
    """按配置检查 path 首点是否等于请求起点。

    specified-path 的语义是“调用方给出了完整路径”。如果首点和当前关节状态不一致，自动插入
    起点会改变调用方的几何意图，因此默认直接报错。确实需要接入已有离线路径时，可以通过
    ``specified_path.cspace_waypoints.require_start_match`` 显式关闭该保护。
    """

    settings = _mapping(config.specified_path.cspace_waypoints)
    require_match = bool(settings.get("require_start_match", True))
    if not require_match:
        return
    tolerance = float(settings.get("start_match_tolerance", 1.0e-9))
    if not np.allclose(first_waypoint, current, atol=tolerance, rtol=0.0):
        raise ValueError("CSpaceWaypointPath first waypoint must match current_q")


def _cspace_path_spec_from_waypoints(cumotion, waypoints: list[np.ndarray]):
    """构造 cuMotion ``CSpacePathSpec``。

    cuMotion 的 C-space PathSpec 以第一个 waypoint 作为 initial position，后续 waypoint
    逐个通过 ``add_cspace_waypoint`` 追加。``add_*`` 返回 ``False`` 时说明 pybind 后端拒绝了
    该段，本层立即把它提升成清晰的 ``ValueError``。
    """

    path_spec = cumotion.create_cspace_path_spec(waypoints[0])
    for index, waypoint in enumerate(waypoints[1:], start=1):
        ok = path_spec.add_cspace_waypoint(waypoint)
        _ensure_added(ok, f"cspace waypoint {index}")
    return path_spec


def _task_space_path_spec_from_segments(
    context,
    current_q: np.ndarray,
    path: TaskSpacePath,
    tcp_frame_name: str,
):
    """构造 cuMotion ``TaskSpacePathSpec``。

    TaskSpacePathSpec 的初始 pose 必须来自当前关节状态的 FK，而不是来自第一个 segment 的
    start_position。segment.start_position 在项目侧只作为一致性声明，用来帮助调用方尽早发现
    “我以为路径从 A 开始，但机器人当前 TCP 实际在 B”的错误。
    """

    current_pose = _current_pose(context, current_q, tcp_frame_name)
    path_spec = context.cumotion.create_task_space_path_spec(current_pose)
    for index, segment in enumerate(path.segments):
        current_pose = _add_task_space_segment(
            context,
            path_spec,
            current_pose,
            segment,
            f"task-space segment {index}",
        )
    return path_spec


def _add_task_space_segment(context, path_spec, current_pose, segment, label: str):
    """把一个项目侧 task-space segment 追加到 cuMotion ``TaskSpacePathSpec``。

    返回值是追加该段后的 tracked TCP pose，供下一个相对 segment 解析目标位置。这个 tracked
    pose 是项目侧的几何跟踪，不代表 IK 已经成功；真正的可达性仍在后续 path conversion 中由
    cuMotion 判断。
    """

    cumotion = context.cumotion
    if isinstance(segment, TcpLineSegment):
        _validate_line_start_position(current_pose, segment, label)
        target = _line_target_position(current_pose, segment)
        blend_radius = float(getattr(segment, "blend_radius", 0.0))
        if segment.orientation_mode == "none":
            # add_translation 只约束 TCP 平移，不要求 orientation。适合调用方明确表示姿态可自由
            # 漂移的路径段。
            ok = path_spec.add_translation(target, blend_radius)
            next_pose = pose_from_rotation_translation(
                cumotion, _pose_rotation(current_pose), target
            )
        elif segment.orientation_mode == "current":
            # add_linear_path 需要完整目标 Pose3；orientation_mode=current 表示用当前 tracked
            # rotation 生成终点 pose，相当于沿直线保持当前 TCP 姿态。
            next_pose = pose_from_rotation_translation(
                cumotion, _pose_rotation(current_pose), target
            )
            ok = path_spec.add_linear_path(
                next_pose,
                blend_radius,
            )
        elif segment.orientation_mode == "target":
            # target 模式把调用方给出的 wxyz 四元数转成 cuMotion Rotation3，再构造终点 Pose3。
            # 四元数归一化集中在 pose_adapter.py。
            next_pose = pose_from_position_quat_wxyz(
                cumotion, target, segment.target_orientation
            )
            ok = path_spec.add_linear_path(
                next_pose,
                blend_radius,
            )
        else:
            raise ValueError(
                f"{label}.orientation_mode must be one of: current, target, none"
            )
        _ensure_added(ok, label)
        return next_pose

    if isinstance(segment, TcpRotationSegment):
        # 原地旋转段只改变 tracked rotation，translation 保持上一个 pose 的位置。
        target_rotation = rotation_from_quat_wxyz(cumotion, segment.target_orientation)
        ok = path_spec.add_rotation(target_rotation)
        _ensure_added(ok, label)
        return pose_from_rotation_translation(
            cumotion, target_rotation, _pose_translation(current_pose)
        )

    if isinstance(segment, TcpArcSegment):
        target = _arc_target_position(current_pose, segment)
        if segment.target_orientation is None:
            # 不带 orientation target 的圆弧使用 cuMotion 的位置圆弧 API。constant_orientation
            # 直接透传给官方接口，由 cuMotion 决定沿弧线是否保持姿态。
            next_pose = pose_from_rotation_translation(
                cumotion, _pose_rotation(current_pose), target
            )
            if segment.arc_mode == "tangent":
                ok = path_spec.add_tangent_arc(
                    target, bool(segment.constant_orientation)
                )
            elif segment.arc_mode == "three_point":
                ok = path_spec.add_three_point_arc(
                    target,
                    _arc_intermediate_position(current_pose, segment),
                    bool(segment.constant_orientation),
                )
            else:
                raise ValueError(f"{label}.arc_mode must be tangent or three_point")
        else:
            # 带 orientation target 的圆弧使用 *_with_orientation_target API；终点姿态由调用方
            # 给定，中间 orientation 插值/约束由 cuMotion path conversion 处理。
            target_pose = pose_from_position_quat_wxyz(
                cumotion, target, segment.target_orientation
            )
            next_pose = target_pose
            if segment.arc_mode == "tangent":
                ok = path_spec.add_tangent_arc_with_orientation_target(target_pose)
            elif segment.arc_mode == "three_point":
                ok = path_spec.add_three_point_arc_with_orientation_target(
                    target_pose,
                    _arc_intermediate_position(current_pose, segment),
                )
            else:
                raise ValueError(f"{label}.arc_mode must be tangent or three_point")
        _ensure_added(ok, label)
        return next_pose

    if isinstance(segment, TcpPoseSequenceSegment):
        # Pose 序列是多段完整 pose 直线的简写。请求层已经要求每个 PoseTarget 都携带
        # orientation，因此这里始终走 add_linear_path，而不是 position-only translation。
        next_pose = current_pose
        for pose_index, pose in enumerate(segment.poses):
            next_pose = pose_from_position_quat_wxyz(
                cumotion, pose.position, pose.orientation
            )
            ok = path_spec.add_linear_path(
                next_pose,
                float(segment.blend_radius),
            )
            _ensure_added(ok, f"{label} pose {pose_index}")
        return next_pose

    raise ValueError(f"Unsupported task-space segment type: {type(segment).__name__}")


def _joint_path_from_linear_cspace_path(linear_path) -> np.ndarray:
    """从真实或 fake ``LinearCSpacePath`` 读取 waypoints 并堆叠。

    真实 cuMotion pybind 可能把 ``waypoints`` 暴露成零参方法，测试 fake 也可能暴露成属性；
    ``attr`` helper 统一处理这两种形态。返回值统一成 ``(N, dof)`` 的 numpy 矩阵，供
    ``CSpaceTrajectoryGenerator`` 和 diagnostics 共用。
    """

    waypoints = attr(linear_path, "waypoints", default=None)
    if waypoints is None:
        raise ValueError("LinearCSpacePath did not expose waypoints")
    path = [np.asarray(waypoint, dtype=float).reshape(-1) for waypoint in waypoints]
    if len(path) < 2:
        raise ValueError("LinearCSpacePath requires at least 2 waypoints")
    return np.vstack([waypoint.reshape(1, -1) for waypoint in path])


def _task_space_conversion_config(cumotion, config: MotionPlannerBackendConfig):
    """构造并填充 cuMotion ``TaskSpacePathConversionConfig``。

    config 中只接受项目侧白名单字段；字段合法性在 ``motion_planner_config.py`` 中提前校验。
    这里按 cuMotion 字段类型写入：``max_iterations`` 是整数，其余转换参数是浮点数。
    """

    conversion_config = cumotion.TaskSpacePathConversionConfig()
    settings = _mapping(config.specified_path.task_space_segments).get("conversion", {})
    for name, value in _mapping(settings).items():
        if name == "max_iterations":
            setattr(conversion_config, str(name), int(value))
        else:
            setattr(conversion_config, str(name), float(value))
    return conversion_config


def _ik_config_for_path_conversion(
    context,
    current_q: np.ndarray,
    config: MotionPlannerBackendConfig,
):
    """构造 path conversion 使用的 cuMotion ``IkConfig``。

    Task-space conversion 内部需要连续求 IK。默认把当前 C-space 作为 seed，能让第一段的解更
    接近实际关节状态；同时复用 ``CuMotionConfig`` 上已有的 IK 容差和迭代配置，避免指定路径
    pipeline 与普通 IK pipeline 的数值边界不一致。
    """

    ik_config = context.cumotion.IkConfig()
    ik_settings = _mapping(
        _mapping(config.specified_path.task_space_segments).get("ik", {})
    )
    if bool(ik_settings.get("use_current_q_as_seed", True)):
        ik_config.cspace_seeds = [np.asarray(current_q, dtype=float).reshape(-1)]
    backend_config = context.config.kinematics.ik
    _set_if_present(ik_config, "position_tolerance", backend_config)
    _set_if_present(ik_config, "orientation_tolerance", backend_config)
    _set_if_present(ik_config, "ccd_max_iterations", backend_config, cast=int)
    _set_if_present(ik_config, "bfgs_max_iterations", backend_config, cast=int)
    if hasattr(backend_config, "orientation_weight"):
        weight = float(backend_config.orientation_weight)
        ik_config.ccd_orientation_weight = weight
        ik_config.bfgs_orientation_weight = weight
    return ik_config


def _transition_mode(cumotion, value: str):
    """把项目字符串映射成 cuMotion ``CompositePathSpec.TransitionMode``。

    请求层和配置层只暴露稳定的小写字符串；adapter 在唯一边界处转换成 pybind enum，避免把
    cuMotion 类型泄漏到动作脚本层和测试请求模型里。
    """

    enum = cumotion.CompositePathSpec.TransitionMode
    if value == "skip":
        return enum.SKIP
    if value == "free":
        return enum.FREE
    if value == "linear_task_space":
        return enum.LINEAR_TASK_SPACE
    raise ValueError("transition_mode must be one of: skip, free, linear_task_space")


def _composite_part_and_transition(
    part: CSpaceWaypointPath | TaskSpacePath | CompositePathPart,
    config: MotionPlannerBackendConfig,
) -> tuple[CSpaceWaypointPath | TaskSpacePath, str]:
    """返回 composite 子路径和有效 transition mode。

    ``CompositePath.parts`` 允许直接放子路径，也允许用 ``CompositePathPart`` 覆盖单段
    transition。直接放子路径时使用 ``specified_path.composite.default_transition_mode``。
    """

    default_transition = str(
        _mapping(config.specified_path.composite).get("default_transition_mode", "free")
    )
    if isinstance(part, CompositePathPart):
        return part.path, str(part.transition_mode or default_transition)
    return part, default_transition


def _current_pose(context, current_q: np.ndarray, tcp_frame_name: str):
    """读取当前 TCP pose。

    项目侧统一使用 ``kinematics.pose(q, frame)`` 这个调用形态，和 FK/IK wrapper 以及测试替身
    保持一致。若后续真实 cuMotion 版本需要显式 base frame，应在统一适配层集中处理，而不是
    在路径转换主流程中用 ``TypeError`` 猜测签名。
    """

    return context.kinematics.pose(current_q, tcp_frame_name)


def _line_target_position(current_pose, segment: TcpLineSegment) -> np.ndarray:
    """解析 ``TcpLineSegment`` 的目标位置。

    ``target_position`` 是绝对终点；``target_offset`` 是相对当前 tracked TCP 位置的位移。
    两者互斥关系已在请求层校验。
    """

    if segment.target_position is not None:
        return np.asarray(segment.target_position, dtype=float).reshape(3)
    current_translation = _pose_translation(current_pose)
    return current_translation + np.asarray(segment.target_offset, dtype=float).reshape(
        3
    )


def _arc_target_position(current_pose, segment: TcpArcSegment) -> np.ndarray:
    """解析 ``TcpArcSegment`` 的目标位置。"""

    if segment.target_position is not None:
        return np.asarray(segment.target_position, dtype=float).reshape(3)
    current_translation = _pose_translation(current_pose)
    return current_translation + np.asarray(segment.target_offset, dtype=float).reshape(
        3
    )


def _arc_intermediate_position(current_pose, segment: TcpArcSegment) -> np.ndarray:
    """解析三点圆弧的中间点。"""

    if segment.intermediate_position is not None:
        return np.asarray(segment.intermediate_position, dtype=float).reshape(3)
    current_translation = _pose_translation(current_pose)
    return current_translation + np.asarray(
        segment.intermediate_offset, dtype=float
    ).reshape(3)


def _validate_line_start_position(
    current_pose, segment: TcpLineSegment, label: str
) -> None:
    """如果调用方声明了线段起点，检查它和当前 tracked pose 一致。

    start_position 不参与 cuMotion PathSpec 的构造，只作为调用方的防错断言。若它与当前 FK/
    tracked pose 不一致，说明动作脚本层的路径几何和机器人实际状态已经错位，应在进入 conversion
    前失败。
    """

    if segment.start_position is None:
        return
    start_position = np.asarray(segment.start_position, dtype=float).reshape(3)
    current_translation = _pose_translation(current_pose)
    if not np.allclose(start_position, current_translation, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{label}.start_position does not match current TCP pose")


def _pose_translation(pose) -> np.ndarray:
    """读取真实/fake ``Pose3`` 的 translation。

    真实 Pose3 优先使用 translation 字段；若某个替身只暴露齐次矩阵，则退回到矩阵最后一列。
    """

    translation = attr(pose, "translation", default=None)
    if translation is None:
        matrix = attr(pose, "matrix", default=None)
        if matrix is None:
            raise ValueError("Pose3 did not expose translation")
        return np.asarray(matrix, dtype=float).reshape(4, 4)[:3, 3]
    return np.asarray(translation, dtype=float).reshape(3)


def _pose_rotation(pose):
    """读取真实/fake ``Pose3`` 的 rotation。

    orientation_mode=current、rotation segment 和 tracked pose 更新都依赖该 rotation。若对象
    没有暴露 rotation，说明不能可靠构造后续 Pose3，直接报错比静默忽略姿态更安全。
    """

    rotation = attr(pose, "rotation", default=None)
    if rotation is None:
        raise ValueError("Pose3 did not expose rotation")
    return rotation


def _validate_frame(context, tcp_frame_name: str) -> None:
    """如果 context 支持 frame 查询，提前检查 frame 是否存在。

    不是所有 fake/context 都提供 frame 查询；有这个能力时提前失败，可以把拼写错误停在项目层，
    而不是等 pybind 在 conversion 深处抛出更难定位的异常。
    """

    validate_cumotion_frame(context, tcp_frame_name, label="tcp_frame_name")


def _set_if_present(target, name: str, source, *, cast=float) -> None:
    """如果 source 有字段，则设置到 target 上。

    这用于把 CuMotionConfig 上已有的 IK 容差/迭代字段复制进 ``IkConfig``。字段缺失时保持
    cuMotion 默认值，兼容较小的测试 fake 和不同版本配置对象。
    """

    if hasattr(source, name):
        setattr(target, name, cast(getattr(source, name)))


def _ensure_added(ok, label: str) -> None:
    """cuMotion PathSpec add 方法返回 False 时立即报错。

    cuMotion 的 ``add_*`` 接口用 ``False`` 表示该段没有被接受；如果继续 conversion，错误会变成
    “路径为空/不连续”等间接症状。这里保留原始 segment label，便于定位。
    """

    if ok is False:
        raise ValueError(f"cuMotion rejected {label}")


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """把可选 mapping 归一化为空 mapping。

    specified_path 的细分配置保持 mapping 形态，方便 YAML 直接覆盖官方 conversion 字段；但
    adapter 仍要求它们是 mapping，避免字符串/list 被误当成可迭代键值。
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("specified_path settings must be mappings")
    return value
