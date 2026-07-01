#!/usr/bin/env python3
"""运行 AR5 + LinkerHand L6 的绳端夹捏抓取动作脚本。

本文件是一个完整可运行动作入口：抓取动作参数直接写在脚本内，不再通过
外部 trajectory YAML 或任务包间接提供。外部 YAML 只保留机器人、控制器、环境、对象、
日志和 cuMotion profile 这些可复用系统配置。也就是说，本文件表达的是“这一次动作怎么做”，
配置文件表达的是“这套仿真系统和机器人怎么运行”。

执行流程：
    1. 通过 profile 名称创建通用单机器人 runtime。
    2. runtime 导入 env objects 和 AR5+L6 组合机器人。
    3. runtime 创建 JointController 和可选日志器。
    4. 根据内置闭合手型计算 pinch TCP。
    5. 规划 approach、grasp、lift、wiggle 等阶段。
    6. 把 cuMotion 输出的轨迹函数按 physics dt 离散成逐物理帧命令。
    7. execution 逐样本播放主动关节命令，controller 在执行边界展开 mimic follower。

数组/坐标约定：
    动作阶段和轨迹统一使用 ``JointController.command_joint_names`` 定义的主动关节命令空间。
    mimic follower 不出现在动作脚本目标里，而是在 controller 执行边界按实际 master 状态补齐。
    只有给 cuMotion 请求当前 C-space seed 时才从 Isaac articulation 完整 DOF 读取实际机械臂关节。
    笛卡尔目标使用当前 demo 的 world/base 对齐坐标，单位 m；关节角和 RPY 使用 rad。

分层约定：
    - cuMotion backend 只处理机械臂模型、IK、C-space motion planning 和指定 TCP 路径规划。
    - 夹爪/手指的主动关节目标在本脚本中直接写成 command-space 向量。
    - mimic follower 的展开、Isaac ArticulationAction 的构造和逐物理帧下发都属于 controller/execution。
    - 本脚本只把这些模块按线性动作流程串起来，不再定义可复用 task 配置层。
"""

from __future__ import annotations

# 标准库只负责三件事：
# - argparse：把少量“运行方式”开关暴露到命令行；
# - sys/pathlib：处理脚本直接运行时的导入路径和配置文件路径；
# - pathlib.Path：让所有路径参数在 CLI、配置加载和日志输出之间保持一致的类型。
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# numpy 在本脚本中承担“控制/规划数值数组”的角色。这里所有目标关节、TCP 位置、
# 方向向量都会尽早转换成 np.ndarray，以便显式控制 shape、dtype 和索引行为。
import numpy as np

# 仓库采用 src-layout。pinch_grasp.py 位于 scripts/ 下，如果直接用 Python 执行，
# 当前目录不会自动把 src/ 当成包根。因此脚本启动时先根据自身位置推导仓库根目录。
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
# 这个脚本通常会用 ``python scripts/pinch_grasp.py`` 或 Isaac 的 Python 解释器直接运行。
# 直接运行脚本时，Python 默认只把 scripts/ 放进 sys.path，并不会自动把仓库的 src/
# 当作包根目录。这里显式插入 src/，保证导入的是当前工作区里的 linkerbot_sim 源码，
# 而不是系统环境中可能残留的已安装包。
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# 以下导入按系统分层组织：
# 1. app/env/assets：启动 Isaac、读取环境参数、把机器人/场景导入 USD stage；
# 2. backends.cumotion：构造 IK/motion-planning 后端和采样后端轨迹；
# 3. controllers/execution：把“主动命令空间”轨迹下发到 Isaac articulation；
# 4. tcp/robots：pinch TCP、mimic follower、关节分组等领域工具；
# 5. utils/logging：配置、路径、旋转和 CSV 日志。
# 这些层次本身比具体类名更重要：本脚本只做 orchestration，不把某一层的细节泄漏到其它层。
from linkerbot_sim.app.runtime.single_robot import (
    LoggingRuntimeOverrides,
    create_single_robot_runtime,
)
from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
)
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.backends.cumotion.tcp_frame import TcpTransform
from linkerbot_sim.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)
from linkerbot_sim.execution.hold import hold_current_pose
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    CommandPositionTrajectoryStep,
    HoldCommandPositionTargetStep,
    SmoothCommandPositionTargetStep,
)
from linkerbot_sim.planning.requests import (
    IKRequest,
    MotionRequest,
    SpecifiedPathRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.trajectories.command_trajectory import (
    command_trajectory_from_arm_trajectory,
)
from linkerbot_sim.robots.joint_groups import target_vector_from_mapping
from linkerbot_sim.robots.mimic import expand_targets_with_mjcf_equalities
from linkerbot_sim.utils.math_utils import (
    axis_angle_to_matrix,
    make_transform,
    quat_wxyz_to_matrix,
)
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


DEFAULT_PINCH_TCP_FRAME = "pinch_tcp"


# 预夹手型：机械臂移动到 scripted 抓取点上方之前，先把手调整到接近夹捏的形状。
# 这里故意只列主动关节：
# - thumb_cmc_roll/pitch 控制拇指朝食指方向转过去；
# - index_mcp_pitch 让食指略弯；
# - middle/ring/pinky 轻微弯曲，减少移动过程中的展开空间。
# 数值单位是 rad，名字暂时不带 L/R 前缀，后面由 _hand_targets_for_side 补成完整关节名。
_PRE_PINCH_HAND_TARGET_VALUES = (
    ("thumb_cmc_roll", 0.95),
    ("thumb_cmc_pitch", 0.28),
    ("index_mcp_pitch", 0.25),
    ("middle_mcp_pitch", 0.15),
    ("ring_mcp_pitch", 0.15),
    ("pinky_mcp_pitch", 0.12),
)

# 闭合夹捏手型：真正夹住目标物时的主动关节目标。它同时服务两件事：
# 1. 计算“闭合状态下拇指和食指指尖中点”的虚拟 TCP；
# 2. 执行 close_fingers 阶段时作为手指最终命令。
# 因此这里的数值如果被调大/调小，既会改变 TCP 几何，又会改变实际闭合动作。
_CLOSED_PINCH_HAND_TARGET_VALUES = (
    ("thumb_cmc_roll", 0.95),
    ("thumb_cmc_pitch", 0.7),
    ("index_mcp_pitch", 0.85),
    ("middle_mcp_pitch", 0.45),
    ("ring_mcp_pitch", 0.4),
    ("pinky_mcp_pitch", 0.35),
)


def default_pre_pinch_hand_targets(side: str = "left") -> dict[str, float]:
    """返回 pinch_grasp 任务的预夹手型主动关节目标。

    返回值是 ``{完整关节名: 目标角度}`` 的稀疏映射，只包含这个手势需要设置的
    主动手指关节。机械臂关节、未列出的手指关节和 mimic follower 都不会出现在这里。
    """

    return _hand_targets_for_side(side, _PRE_PINCH_HAND_TARGET_VALUES)


def default_closed_pinch_hand_targets(side: str = "left") -> dict[str, float]:
    """返回 pinch_grasp 任务的闭合夹捏手型主动关节目标。

    和预夹手型一样，这里只返回主动关节的稀疏目标。完整 command-space 向量会在
    ``target_vector_from_mapping`` 中基于上一阶段目标补齐，避免未参与闭合的关节被重置。
    """

    return _hand_targets_for_side(side, _CLOSED_PINCH_HAND_TARGET_VALUES)


def _hand_targets_for_side(
    side: str, values: tuple[tuple[str, float], ...]
) -> dict[str, float]:
    """把不带侧别的 L6 关节短名转换成 controller 使用的完整关节名。

    L6 手的完整命名格式形如 ``L6V1_L_hand_index_mcp_pitch``。动作参数为了可读性只维护
    ``index_mcp_pitch`` 这样的短名；这个 helper 负责根据 side 补上 ``L`` 或 ``R``。
    """

    # _hand_side_token 会做输入校验，并把 left/l/right/r 统一成资产命名使用的 L/R。
    side_token = _hand_side_token(side)
    # float(value) 让手型表即使未来从 YAML/CLI 读取 Decimal、np.float32 等类型，也在这里
    # 归一成普通 Python float，便于日志打印和 np.asarray 后续处理。
    return {f"L6V1_{side_token}_hand_{joint}": float(value) for joint, value in values}


def _hand_side_token(side: str) -> str:
    """把用户侧别输入规范化成资产关节名里的 ``L`` 或 ``R``。"""

    # 这里先 str(side).lower()，是为了给 CLI/配置层一点容错空间。例如传入 Path-like 或
    # enum-like 对象时，只要字符串化后是 left/right/l/r，就能继续运行。
    normalized = str(side).lower()
    if normalized in {"left", "l"}:
        return "L"
    if normalized in {"right", "r"}:
        return "R"
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def grasp_target_position(base_position, *, lift_height: float = 0.0) -> np.ndarray:
    """计算夹捏 TCP 的世界坐标目标位置。

    参数:
        base_position: scripted world 抓取点，单位 m。
        lift_height: 额外 z 方向抬升高度，单位 m。
    返回:
        shape 为 ``(3,)`` 的世界坐标位置数组，单位 m。
    """

    return (
        np.asarray(base_position, dtype=float).reshape(3)
        + np.asarray([0.0, 0.0, lift_height], dtype=float)
    )


def infer_hand_body_names(hand_targets: dict[str, float]) -> tuple[str, str, str]:
    """按 L6 关节名前缀推导 hand base 和 thumb/index tip body 名。"""

    for joint_name in hand_targets:
        name = str(joint_name)
        if "_hand_" not in name:
            continue
        system_name = name.split("_hand_", 1)[0]
        return (
            f"{system_name}_hand_base_link",
            f"{system_name}_hand_thumb_tip",
            f"{system_name}_hand_index_tip",
        )
    raise ValueError(
        "Cannot infer hand body names because no target joint name contains '_hand_'"
    )


def parse_mjcf_vec3(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    """解析 MJCF vec3 属性。"""

    if not text:
        return np.asarray(default, dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return np.asarray(values, dtype=float)


def parse_mjcf_quat_wxyz(text: str | None) -> np.ndarray:
    """解析 MJCF wxyz 四元数属性。"""

    if not text:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != 4:
        raise ValueError(f"Expected 4 quaternion values, got {text!r}")
    return np.asarray(values, dtype=float)


def mjcf_parent_map(root: ET.Element) -> dict[int, ET.Element]:
    """建立 MJCF XML child id 到 parent element 的映射。"""

    parent_by_child_id: dict[int, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parent_by_child_id[id(child)] = parent
    return parent_by_child_id


def find_mjcf_body(root: ET.Element, body_name: str) -> ET.Element:
    """按名称查找 MJCF body。"""

    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise ValueError(f"MJCF body not found: {body_name}")


def body_chain_between(
    root: ET.Element, base_body_name: str, tip_body_name: str
) -> list[ET.Element]:
    """返回从 hand base body 到 tip body 的 body 链。"""

    base = find_mjcf_body(root, base_body_name)
    tip = find_mjcf_body(root, tip_body_name)
    parent_by_child_id = mjcf_parent_map(root)
    chain = []
    node = tip
    while node.tag == "body":
        chain.append(node)
        if node is base:
            return list(reversed(chain))
        parent = parent_by_child_id.get(id(node))
        if parent is None:
            break
        node = parent
    raise ValueError(f"{tip_body_name} is not under {base_body_name} in MJCF")


def body_chain_local_transform(
    body_chain: list[ET.Element], joint_positions: dict[str, float]
) -> np.ndarray:
    """沿 MJCF body 链计算 tip 相对 base 的局部齐次变换。"""

    transform = np.eye(4, dtype=float)
    for body in body_chain[1:]:
        body_pos = parse_mjcf_vec3(body.get("pos"))
        body_rot = quat_wxyz_to_matrix(parse_mjcf_quat_wxyz(body.get("quat")))
        transform = transform @ make_transform(body_pos, body_rot)
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            joint_pos = parse_mjcf_vec3(joint.get("pos"))
            joint_axis = parse_mjcf_vec3(joint.get("axis"), default=(1.0, 0.0, 0.0))
            joint_value = float(joint_positions.get(joint_name, 0.0))
            transform = transform @ make_transform(
                joint_pos, axis_angle_to_matrix(joint_axis, joint_value)
            )
    return transform


def fingertip_pinch_local_offset(
    mjcf_path: str | Path,
    hand_targets: dict[str, float],
    *,
    hand_base_body: str | None = None,
    thumb_tip_body: str | None = None,
    index_tip_body: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算闭合手型下 thumb/index 指尖中点相对 hand base 的偏移。"""

    path = Path(mjcf_path)
    expanded_targets = expand_targets_with_mjcf_equalities(hand_targets, path)
    inferred_base, inferred_thumb, inferred_index = infer_hand_body_names(hand_targets)
    hand_base_body = hand_base_body or inferred_base
    thumb_tip_body = thumb_tip_body or inferred_thumb
    index_tip_body = index_tip_body or inferred_index
    root = ET.parse(path).getroot()
    thumb_chain = body_chain_between(root, hand_base_body, thumb_tip_body)
    index_chain = body_chain_between(root, hand_base_body, index_tip_body)
    thumb_tip = body_chain_local_transform(thumb_chain, expanded_targets)[:3, 3]
    index_tip = body_chain_local_transform(index_chain, expanded_targets)[:3, 3]
    pinch_center = 0.5 * (thumb_tip + index_tip)
    return pinch_center, thumb_tip, index_tip


def make_pinch_tcp_transform(
    mjcf_path: str | Path,
    hand_targets: dict[str, float],
    *,
    frame_name: str = DEFAULT_PINCH_TCP_FRAME,
    hand_base_body: str | None = None,
    thumb_tip_body: str | None = None,
    index_tip_body: str | None = None,
) -> TcpTransform:
    """构造 pinch_grasp 脚本专用的末端相对 pinch TCP 变换。"""

    path = Path(mjcf_path)
    inferred_base, inferred_thumb, inferred_index = infer_hand_body_names(hand_targets)
    hand_base_body = hand_base_body or inferred_base
    thumb_tip_body = thumb_tip_body or inferred_thumb
    index_tip_body = index_tip_body or inferred_index
    pinch_center, _thumb_tip, _index_tip = fingertip_pinch_local_offset(
        path,
        hand_targets,
        hand_base_body=hand_base_body,
        thumb_tip_body=thumb_tip_body,
        index_tip_body=index_tip_body,
    )
    root = ET.parse(path).getroot()
    hand_base = find_mjcf_body(root, hand_base_body)
    endpoint_from_hand_base = make_transform(
        parse_mjcf_vec3(hand_base.get("pos")),
        quat_wxyz_to_matrix(parse_mjcf_quat_wxyz(hand_base.get("quat"))),
    )
    endpoint_pinch_center = (
        endpoint_from_hand_base
        @ np.asarray([*pinch_center, 1.0], dtype=float).reshape(4, 1)
    )[:3, 0]
    return TcpTransform.from_xyz_rpy(
        frame_name=frame_name,
        xyz=endpoint_pinch_center,
    )


def build_planned_joint_motion_trajectory(
    *,
    motion_planner,
    command_joint_names,
    arm_command_indices: np.ndarray,
    start_command: np.ndarray,
    target_command: np.ndarray,
    duration_s: float,
    phase: str,
    sample_dt: float = 0.01,
):
    """规划一段 C-space 关节运动并嵌入 controller command-space。

    这个函数用于“已知起点/终点机械臂关节角”的阶段，例如 move_to_approach、lift、
    wiggle 和末尾 sweep。输入的 start_command/target_command 是完整主动命令空间，
    但 cuMotion 只规划机械臂 C-space，所以函数内部会：

    1. 用 arm_command_indices 从 command-space 中抽出机械臂列；
    2. 调 cuMotion motion planner 生成机械臂轨迹；
    3. 按 sample_dt 采样成离散 JointTrajectory；
    4. 再把机械臂轨迹嵌回 command-space，并保留手指目标。

    这样 execution 层永远只看到 controller command-space，不需要理解 cuMotion 的关节顺序。
    """

    # 防御性地把输入变成一维 float/int 数组。调用方通常已经给的是 np.ndarray，但这里统一
    # reshape(-1) 可以避免 list、tuple 或 shape=(N,1) 的数组在高级索引时产生意外维度。
    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_command_indices, dtype=int).reshape(-1)

    # MotionRequest 的 current_q/goal_q 必须是 cuMotion 机械臂模型的关节顺序。由于前面已经
    # 构造了 arm_command_indices，这里可以直接从 command-space 中抽取对应列。
    # 手指关节不会传入 cuMotion，也不会参与这段关节空间规划。
    result = motion_planner.plan(
        MotionRequest(
            current_q=start[arm_indices],
            goal_q=target[arm_indices],
            duration_s=duration_s,
        )
    )
    if not result.success:
        # 失败时把 phase 写进错误，方便从长动作序列里快速定位是哪一段规划失败。
        raise RuntimeError(
            f"cuMotion joint motion planning failed for {phase}: "
            f"status={result.status}"
        )
    if result.trajectory is None:
        # success=False 和 trajectory=None 是两类不同异常：后者说明后端状态不符合本脚本假设，
        # 即使 status 文字看起来正常，也不能继续交给 execution。
        raise RuntimeError(
            f"cuMotion joint motion planning returned no trajectory for {phase}: "
            f"status={result.status}"
        )

    # cuMotion 返回的 trajectory 是后端对象/函数，不适合 execution 逐帧播放。sampler 会按
    # sample_dt 生成项目自己的 JointTrajectory，并保存 phase，供日志和调试使用。
    arm_trajectory = joint_trajectory_from_cumotion(
        result.trajectory,
        joint_names=tuple(motion_planner.joint_names()),
        sample_dt=sample_dt,
        phase=phase,
    )
    # 嵌回 command-space 时，arm_trajectory 只覆盖机械臂列；其它主动关节会从 start/target
    # 端点推导并保持阶段语义，例如 lift 阶段手指应保持闭合。
    return command_trajectory_from_arm_trajectory(
        arm_trajectory=arm_trajectory,
        command_joint_names=command_joint_names,
        arm_command_indices=arm_indices,
        start_command=start,
        target_command=target,
        phase=phase,
    )


def build_specified_tcp_line_trajectory(
    *,
    context,
    tcp_frame_name: str,
    command_joint_names,
    arm_command_indices: np.ndarray,
    start_command: np.ndarray,
    target_position: np.ndarray,
    duration_s: float,
    phase: str,
    motion_planner_config: MotionPlannerBackendConfig | None = None,
    sample_dt: float = 0.01,
):
    """规划一段指定 TCP 直线并嵌入 controller command-space。

    这个函数用于接近 scripted 抓取点的最后一小段：机械臂不是简单做关节插值，而是要求 pinch TCP
    从 approach_world 沿一条任务空间线段移动到 pinch_world。指定路径规划的输入仍然是
    当前机械臂 C-space，输出也是机械臂轨迹；函数末尾同样会嵌回 controller command-space。

    这里的 orientation_mode="current" 表示直线段只指定目标位置，姿态沿用该段起点的 TCP
    姿态。这样可以避免“向下接近端块”的小段里额外发生手腕旋转。
    """

    # start 是整条主动命令向量；specified_path 规划只需要其中机械臂列。
    start = np.asarray(start_command, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_command_indices, dtype=int).reshape(-1)

    # 如果调用方没有显式传 planner config，就使用默认 backend config。实际 run_pinch_grasp_action
    # 会传入 motion_planner_config，这个 fallback 主要让函数更容易单独测试。
    base_config = motion_planner_config or MotionPlannerBackendConfig.from_mapping(None)

    # 为指定 TCP 直线临时构造一个 planning_pipeline="specified_path" 的配置。
    # 其它子配置沿用 profile 中的 graph_search/trajectory_generation/trajectory_optimization，
    # 这样只改变“路径来源”，不改变速度、碰撞验证等通用规划策略。
    specified_path_config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        graph_search=base_config.graph_search,
        trajectory_generation=base_config.trajectory_generation,
        trajectory_optimization=base_config.trajectory_optimization,
        specified_path=SpecifiedPathConfig(
            family="task_space_segments",
            validate_collision_after_generation=(
                base_config.specified_path.validate_collision_after_generation
            ),
            cspace_waypoints=base_config.specified_path.cspace_waypoints,
            task_space_segments=base_config.specified_path.task_space_segments,
            composite=base_config.specified_path.composite,
        ),
    )

    # 这里创建的是“指定路径” planner，不复用外层 C-space motion_planner。二者绑定同一个 TCP
    # frame，但 pipeline 不同：一个处理关节空间起终点，一个处理任务空间线段。
    specified_planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=specified_path_config,
    )

    # SpecifiedPathRequest 中的 path 是任务空间路径描述。当前只有一个 TcpLineSegment，所以整段
    # 接近动作是一条直线；如果以后要绕障，可以在这里追加多个 segment。
    result = specified_planner.plan(
        SpecifiedPathRequest(
            current_q=start[arm_indices],
            tcp_frame_name=tcp_frame_name,
            duration_s=duration_s,
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray(target_position, dtype=float).reshape(3),
                        orientation_mode="current",
                    ),
                )
            ),
        )
    )
    if not result.success:
        # 指定路径失败常见原因包括 TCP 直线穿过障碍、目标超出 IK 可达范围、或 profile
        # 中的碰撞验证过严。phase/status 会一起暴露给调用方。
        raise RuntimeError(
            f"cuMotion specified TCP line planning failed for {phase}: "
            f"status={result.status}"
        )
    if result.trajectory is None:
        # 没有 trajectory 就无法按 physics_dt 离散播放，因此立即停止。
        raise RuntimeError(
            f"cuMotion specified TCP line returned no trajectory for {phase}: "
            f"status={result.status}"
        )
    if result.path is None or np.asarray(result.path).shape[0] == 0:
        # result.path 保存指定路径规划得到的 C-space 路径点。末点会被用来构造 target_command，
        # 供后续 lift IK 和 execution 阶段共享，因此空 path 不能继续。
        raise RuntimeError(
            "cuMotion specified TCP line returned trajectory without path "
            f"for {phase}: status={result.status}"
        )

    # specified_path 返回的是机械臂路径；我们取最后一个 C-space 点写入 command-space，
    # 形成“下沉到抓取点后的主动命令姿态”。手指部分仍沿用 start_command。
    target_command = start.copy()
    target_command[arm_indices] = np.asarray(result.path, dtype=float)[-1]

    # 后端轨迹采样成机械臂 JointTrajectory，再嵌回 command-space，和普通 C-space 规划保持相同
    # execution 接口。
    arm_trajectory = joint_trajectory_from_cumotion(
        result.trajectory,
        joint_names=tuple(specified_planner.joint_names()),
        sample_dt=sample_dt,
        phase=phase,
    )
    return command_trajectory_from_arm_trajectory(
        arm_trajectory=arm_trajectory,
        command_joint_names=command_joint_names,
        arm_command_indices=arm_indices,
        start_command=start,
        target_command=target_command,
        phase=phase,
    )


def run_pinch_grasp_action(
    *,
    robot,
    world,
    articulation_action_type,
    controller,
    simulation_app,
    render: bool,
    mjcf_path: str | Path,
    cumotion_config: CuMotionConfig,
    motion_planner_config: MotionPlannerBackendConfig,
    grasp_world,
    drive_logger=None,
) -> dict[str, object]:
    """按脚本顺序规划并执行完整夹捏抓取动作。

    这个函数故意保持“动作脚本”的形态：参数在函数开头集中赋值，随后按
    “构造目标 -> IK/规划 -> 生成 command-space 轨迹 -> execution 播放”的顺序向下读。
    它不是通用 task 框架，因此不做大量配置对象和 validate 包装；如果某个动作参数需要调试，
    直接改这里的局部变量即可。

    参数中的几个对象来自 ``main`` 中的 Isaac 初始化流程：
        robot: Isaac ``SingleArticulation``，提供完整 DOF 名称、位置和速度读写。
        world: Isaac ``World``，提供 physics dt，并由 execution 每帧推进仿真。
        articulation_action_type: Isaac ``ArticulationAction`` 类型，用于 controller 构造 action。
        controller: ``JointController``，负责 command-space 到实际受驱动 DOF 的转换。
        simulation_app: GUI/Kit 应用对象，execution 可用它判断窗口是否仍在运行。
        render: 是否在 ``world.step`` 时渲染画面。
        mjcf_path: 当前机器人资产路径；pinch TCP 和 mimic follower 都需要读取 MJCF equality。
        cumotion_config: cuMotion 机器人模型、kinematics 和默认规划配置。
        motion_planner_config: 本动作使用的 cuMotion motion planner profile。
        grasp_world: 动作脚本使用的 world 抓取点，不从 rope 段块位置推导。
        drive_logger: 可选逐步关节跟踪日志器。
    """

    mjcf_path = Path(mjcf_path)

    # TCP frame 名称必须与脚本侧创建的虚拟 frame 一致。cuMotion 的 IK 和 motion planner
    # 都是对这个 frame 做目标约束，而不是对法兰盘 frame 直接做约束。
    tcp_frame_name = DEFAULT_PINCH_TCP_FRAME

    # physics_dt 是本脚本离散化轨迹的唯一时间基准：
    # cuMotion 返回的是连续/可评估的 trajectory function，trajectory_sampler 会按 sample_dt
    # 直接取样成“每个 physics step 一行”的 JointTrajectory。execution 后续只逐样本播放，
    # 不再按时间二次插值，也不在项目侧拉长或压缩轨迹。
    physics_dt = float(world.get_physics_dt())

    # 下面这一组是动作本身的局部参数。它们不再放成模块级 DEFAULT_* 常量，也不再放进
    # trajectory YAML；这样读脚本时可以从这里开始看到完整动作意图。
    #
    # target_rpy 描述夹捏 TCP 在抓取目标物时希望保持的姿态，采用 xyz 欧拉角，单位 rad。
    # use_orientation=True 表示 IK 同时约束位置和姿态；如果未来只想验证位置可达性，可以在这里改
    # 成 False，让 IK 只对 TCP 位置收敛。
    target_rpy = (0.0, 2.007128639793479, -1.5707963267948966)
    use_orientation = True

    # approach_distance 是接近阶段在抓取点正上方预留的高度。先到上方接近点，再沿 TCP 直线下沉，
    # 比直接关节空间插值到抓取点更容易得到稳定、直观的接近动作。
    approach_distance = 0.10

    # wiggle 是抓取后的扰动测试：抬起后沿 wiggle_axis_xyz 往返移动，用来观察端块是否被夹住。
    # axis 会在下面归一化，所以这里写方向即可；amplitude 单位是 m。
    wiggle_amplitude = 0.2
    wiggle_axis_xyz = (1.0, 0.0, 0.0)

    lift_height = 0.4

    # 各阶段时长交给 cuMotion 处理：C-space trajectory 的时间参数化由 cuMotion 根据 duration_s
    # 生成。项目侧只负责按 physics_dt 采样，不在采样后重新 retime。
    prep_duration = 2.0
    move_duration = 6.0
    approach_duration = 2.2
    close_duration = 2.0
    lift_duration = 4.0
    wiggle_cycles = 2
    wiggle_duration = 4.0
    final_hold_duration = 3.0
    post_joint_sweep_duration = 5.0
    post_joint_sweep_target_values = (2.1, -2.1)

    # 手指目标只写主动关节。L6 手里由 MJCF equality 表达的 DIP 等 follower 不出现在这里；
    # controller 会在每个 execution step 里根据实际 master 关节位置计算 follower 目标。
    #
    # pre_pinch_hand_targets 是“接近抓取点前”的预成型手势：拇指已经转向食指，食指和其它手指
    # 轻微弯曲，避免机械臂移动时手完全张开造成碰撞或穿插。
    pre_pinch_hand_targets = default_pre_pinch_hand_targets("left")

    # closed_pinch_hand_targets 是真正闭合后的夹捏手势。这个手势有两个用途：
    # 1. make_pinch_tcp_transform 用它计算“闭合时两指尖中点”相对于末端的 TCP；
    # 2. execution 的 close_fingers 阶段会把主动手指关节平滑移动到这些目标。
    # 因此 TCP 几何和实际闭合动作使用同一组主动关节角，避免规划目标和最终手型不一致。
    closed_pinch_hand_targets = default_closed_pinch_hand_targets("left")

    # make_pinch_tcp_transform 的职责是脚本侧离线构造一个“夹捏 TCP”：
    # - 输入闭合手势的主动关节角；
    # - 根据 MJCF equality 展开 mimic follower；
    # - 沿手掌根部到两指尖的 kinematic chain 做一次局部 FK；
    # - 取拇指/食指指尖几何中点作为 TCP；
    # - 只把相对末端的 xyz/rpy 变换交给 src/cumotion 后端。
    #
    # 注意：这里的 mimic 展开只服务于 TCP 几何推导，不表示动作脚本从此开始控制 follower。
    # 运行时 follower 仍然只由 controller 在执行边界根据实际 master 状态补齐。
    tcp = make_pinch_tcp_transform(
        mjcf_path,
        closed_pinch_hand_targets,
        frame_name=tcp_frame_name,
    )

    # 三个核心笛卡尔目标：
    # - pinch_world：真正希望 pinch TCP 到达的 scripted world 抓取点；
    # - approach_world：pinch_world 正上方的预接近点；
    # - lifted_world：保持水平位置不变，只把抓取点向上抬。
    #
    # pinch_grasp 不读取 rope 内部段块位置；目的是让动作目标和资产生成细节解耦。
    pinch_world = grasp_target_position(grasp_world)
    approach_world = pinch_world + np.asarray(
        [0.0, 0.0, approach_distance], dtype=float
    )
    lifted_world = grasp_target_position(grasp_world, lift_height=lift_height)
    wiggle_axis = np.asarray(wiggle_axis_xyz, dtype=float)

    # axis 归一化后再乘 amplitude，保证 wiggle_amplitude 始终表示真实位移长度，而不是受
    # wiggle_axis_xyz 向量模长影响。
    wiggle_axis = wiggle_axis / np.linalg.norm(wiggle_axis)
    wiggle_worlds: list[np.ndarray] = []
    for _cycle_index in range(wiggle_cycles):
        wiggle_worlds.append(lifted_world - wiggle_axis * wiggle_amplitude)
        wiggle_worlds.append(lifted_world + wiggle_axis * wiggle_amplitude)

    # cuMotion 的 IKRequest 使用四元数姿态。rpy_xyz_to_quat_wxyz 输出 wxyz 顺序，和项目内部
    # request 约定一致。
    target_orientation = rpy_xyz_to_quat_wxyz(target_rpy)
    ik_orientation = target_orientation if use_orientation else None

    # IK 后端只认识机器人描述里的 frame。make_cumotion_context 会把上面算出的 pinch TCP
    # 临时装配进 cuMotion 使用的机器人模型/context，让 motion planner 可以直接规划
    # DEFAULT_PINCH_TCP_FRAME。context 退出时临时文件和 backend 资源会被清理。
    with make_cumotion_context(cumotion_config, tcp=tcp) as context:
        ik_defaults = context.config.kinematics.ik

        # ik_joint_names 是 cuMotion 机械臂 C-space 的关节顺序。这个顺序通常只包含机械臂，
        # 不包含 LinkerHand 手指，也不包含 mimic follower。
        ik_joint_names = context.joint_names()

        # dof_names 是 Isaac articulation 的完整 DOF 顺序，包含机械臂、主动手指和 follower。
        # command_joint_names 是 JointController 暴露给动作脚本的主动命令空间顺序，包含机械臂
        # 和主动手指，但不包含 mimic follower。
        dof_names = list(robot.dof_names)
        dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
        command_joint_names = controller.command_joint_names
        command_index_by_name = {
            name: index for index, name in enumerate(command_joint_names)
        }

        # cuMotion 模型和 Isaac articulation 可能来自不同资产文件。这里按名称检查能尽早
        # 发现 URDF/MJCF 关节名不一致，而不是在写目标数组时静默错位。
        missing_ik_joints = [
            name for name in ik_joint_names if name not in dof_index_by_name
        ]
        if missing_ik_joints:
            raise ValueError(
                f"cuMotion joints not found in articulation: {missing_ik_joints}"
            )
        missing_command_joints = [
            name for name in ik_joint_names if name not in command_index_by_name
        ]
        if missing_command_joints:
            raise ValueError(
                "cuMotion joints not found in controller command space: "
                f"{missing_command_joints}"
            )

        # 两组索引描述同一批机械臂关节在不同数组中的位置：
        # - arm_dof_indices：从 Isaac 完整 DOF 数组中抽取 cuMotion 机械臂 C-space；
        # - arm_command_indices：在 controller command-space 中定位这些机械臂关节。
        #
        # 规划阶段用 cuMotion C-space，执行阶段用 command-space，因此这个映射是动作脚本和
        # controller/execution 之间最重要的边界。
        arm_dof_indices = np.asarray(
            [dof_index_by_name[name] for name in ik_joint_names], dtype=int
        )
        arm_command_indices = np.asarray(
            [command_index_by_name[name] for name in ik_joint_names], dtype=int
        )

        # 当前 articulation 完整 DOF 里可能还有手指和 follower；给 IK warm start 只取 cuMotion
        # 认识的机械臂关节，且顺序严格按 context.joint_names() 对齐。
        current_cspace = np.asarray(robot.get_joint_positions(), dtype=float).reshape(
            -1
        )[arm_dof_indices]

        # solver 用于单点 IK，motion_planner 用于 C-space 轨迹规划。二者都绑定到 pinch TCP frame，
        # 因此 target_position 表达的是“夹捏中心”应该到达的位置。
        solver = context.make_inverse_kinematics(tcp_frame_name=tcp_frame_name)
        motion_planner = context.make_motion_planner(
            tcp_frame_name=tcp_frame_name,
            config=motion_planner_config,
        )

        # 强约束：传给 execution 的 JointTrajectory 必须已经是逐 physics step 离散样本。
        # 因此 sample_dt 直接等于 physics_dt。首样本是否去除、末样本如何补齐等细节由
        # trajectory_sampler 统一处理，动作脚本不手写采样时间轴。
        sample_dt = physics_dt

        # 第一次 IK 用当前 articulation C-space 热启动，后续阶段用上一阶段解热启动。
        # 这样做有两个目的：
        # - 让 IK 解尽量落在当前姿态附近，避免机械臂突然选择另一组等价但很远的关节解；
        # - 让后续 cuMotion C-space 规划的起点/终点更连续，减少不必要的大幅摆动。
        approach = solver.solve(
            IKRequest(
                target_position=approach_world,
                target_orientation=ik_orientation,
                warm_start_ik_cspace_seed=current_cspace,
                position_tolerance=ik_defaults.position_tolerance,
                orientation_tolerance=ik_defaults.orientation_tolerance,
            )
        )

        # initial_full 是 Isaac articulation 的完整 DOF 状态。动作脚本不直接向完整 DOF 写目标，
        # 所以这里只把完整 DOF 通过 controller.command_indices 投影到 command-space。
        # 从 initial_command 往后，本脚本里的目标向量都只包含“主动命令关节”。
        initial_full = np.asarray(robot.get_joint_positions(), dtype=float)
        initial_command = initial_full[controller.command_indices]

        # target_vector_from_mapping 用稀疏关节名映射覆盖 base 向量：
        # pre_pinch_hand_targets 只列出要改变的手指主动关节，机械臂和其它主动关节保持当前值。
        # 这比手写完整数组更不容易因为关节顺序变化而错位。
        pre_pinch_command = target_vector_from_mapping(
            command_joint_names,
            pre_pinch_hand_targets,
            base=initial_command,
        )

        # approach.joint_positions 是 cuMotion C-space 顺序，只能写回 command-space 中对应的
        # arm_command_indices。手指部分仍保持 pre_pinch_command 的预成型姿态。
        approach_command = pre_pinch_command.copy()
        approach_command[arm_command_indices] = np.asarray(
            approach.joint_positions, dtype=float
        )
        # approach_command 是接近点的主动命令姿态；从这里开始构建一条短 TCP 直线下沉轨迹，
        # 比直接 IK 到抓取点再关节插值更接近“沿竖直方向靠近端块”的动作意图。TCP 直线现在
        # 直接使用 specified_path 的 TaskSpacePath/TcpLineSegment，在规划期一次性完成路径转换。
        grasp_line_trajectory = build_specified_tcp_line_trajectory(
            context=context,
            tcp_frame_name=tcp_frame_name,
            command_joint_names=command_joint_names,
            arm_command_indices=arm_command_indices,
            start_command=approach_command,
            target_position=pinch_world,
            duration_s=approach_duration,
            phase="approach_box",
            motion_planner_config=motion_planner_config,
            sample_dt=sample_dt,
        )

        # grasp_line_trajectory 已经是 command-space 轨迹，因此取最后一行再用 arm_command_indices
        # 抽出机械臂部分，作为 lift IK 的 warm start。这样 lift 从实际下沉完成后的机械臂姿态继续解。
        grasp_joint_positions = np.asarray(
            grasp_line_trajectory.positions[-1], dtype=float
        )[arm_command_indices]
        lift = solver.solve(
            IKRequest(
                target_position=lifted_world,
                target_orientation=ik_orientation,
                warm_start_ik_cspace_seed=grasp_joint_positions,
                position_tolerance=ik_defaults.position_tolerance,
                orientation_tolerance=ik_defaults.orientation_tolerance,
            )
        )
        # wiggle 阶段每个目标都用上一目标热启动，减少在冗余机械臂上突然换解的概率。
        wiggles = []
        warm = lift.joint_positions
        for target in wiggle_worlds:
            result = solver.solve(
                IKRequest(
                    target_position=target,
                    target_orientation=ik_orientation,
                    warm_start_ik_cspace_seed=warm,
                    position_tolerance=ik_defaults.position_tolerance,
                    orientation_tolerance=ik_defaults.orientation_tolerance,
                )
            )
            wiggles.append((target, result))
            warm = result.joint_positions

        # 把 cuMotion IK 解写回 command-space 目标。这里可以把每个目标理解成“阶段结束姿态”：
        # - grasp_open_command：TCP 已经下沉到端块附近，但手指仍保持预夹姿态；
        # - grasp_closed_command：机械臂不动，手指闭合成夹捏手势；
        # - lifted_command：手指保持闭合，机械臂移动到抬升后的 IK 解。
        #
        # 手部关节仍然通过稀疏映射覆盖，其它主动关节沿用上一阶段目标，保证未参与阶段切换的
        # 关节不被意外归零。
        grasp_open_command = np.asarray(
            grasp_line_trajectory.positions[-1], dtype=float
        ).copy()
        grasp_closed_command = target_vector_from_mapping(
            command_joint_names,
            closed_pinch_hand_targets,
            base=grasp_open_command,
        )
        lifted_command = grasp_closed_command.copy()
        lifted_command[arm_command_indices] = np.asarray(
            lift.joint_positions, dtype=float
        )

        # 每个 wiggle command 都从闭合夹捏姿态开始复制，只替换机械臂关节。也就是说 wiggle
        # 阶段不会改变手指目标，夹持力/手型保持和 close_fingers 后一致。
        wiggle_command_targets = []
        for _world, result in wiggles:
            wiggle_command = grasp_closed_command.copy()
            wiggle_command[arm_command_indices] = np.asarray(
                result.joint_positions, dtype=float
            )
            wiggle_command_targets.append(wiggle_command)

        # 末尾扫动第 1 个机械臂关节是 scripted demo 的额外扰动，用于观察夹持是否稳固。
        post_joint_sweep_targets = []
        for joint_1_target in post_joint_sweep_target_values:
            sweep_command = lifted_command.copy()
            sweep_command[arm_command_indices[0]] = float(joint_1_target)
            post_joint_sweep_targets.append(sweep_command)

        # 下面这些阶段都是“机械臂关节角 -> 机械臂关节角”的运动：
        # - cuMotion MotionPlanner 在机械臂 C-space 中生成 trajectory function；
        # - trajectory_sampler 按 physics_dt 离散成机械臂 JointTrajectory；
        # - command_trajectory_from_arm_trajectory 把机械臂列嵌入 controller command-space；
        # - execution 后续只逐样本播放 command-space，不再调用 cuMotion。
        #
        # 手指开合阶段不走 cuMotion，因为手部 DOF 不属于后端机械臂模型；它们用
        # SmoothCommandPositionTargetStep 在 command-space 中做平滑目标过渡。
        move_to_approach_trajectory = build_planned_joint_motion_trajectory(
            motion_planner=motion_planner,
            command_joint_names=command_joint_names,
            arm_command_indices=arm_command_indices,
            start_command=pre_pinch_command,
            target_command=approach_command,
            duration_s=move_duration,
            phase="move_to_approach",
            sample_dt=sample_dt,
        )
        lift_trajectory = build_planned_joint_motion_trajectory(
            motion_planner=motion_planner,
            command_joint_names=command_joint_names,
            arm_command_indices=arm_command_indices,
            start_command=grasp_closed_command,
            target_command=lifted_command,
            duration_s=lift_duration,
            phase="lift",
            sample_dt=sample_dt,
        )
        wiggle_trajectories = []
        previous_target = lifted_command

        # wiggle 轨迹按目标列表顺序串接。每段都以前一段的终点作为起点，避免每段规划都从
        # lifted_command 重新出发造成轨迹断裂。
        for index, wiggle_command in enumerate(wiggle_command_targets, start=1):
            trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                command_joint_names=command_joint_names,
                arm_command_indices=arm_command_indices,
                start_command=previous_target,
                target_command=wiggle_command,
                duration_s=wiggle_duration,
                phase=f"wiggle_{index}",
                sample_dt=sample_dt,
            )
            wiggle_trajectories.append(trajectory)
            previous_target = wiggle_command
        wiggle_return_trajectory = None
        if wiggle_command_targets:
            # 做完左右扰动后回到 lifted_command 中心姿态，方便 final hold 和后续 sweep 有稳定起点。
            wiggle_return_trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                command_joint_names=command_joint_names,
                arm_command_indices=arm_command_indices,
                start_command=previous_target,
                target_command=lifted_command,
                duration_s=wiggle_duration,
                phase="wiggle_return_center",
                sample_dt=sample_dt,
            )
        post_joint_sweep_trajectories = []
        previous_target = lifted_command

        # post sweep 同样逐段串接；这里的目标只改第一个机械臂关节，其余机械臂关节和手指目标
        # 保持 lifted_command。它是动作末尾的观察动作，不参与真正抓取规划。
        for index, sweep_command in enumerate(post_joint_sweep_targets, start=1):
            trajectory = build_planned_joint_motion_trajectory(
                motion_planner=motion_planner,
                command_joint_names=command_joint_names,
                arm_command_indices=arm_command_indices,
                start_command=previous_target,
                target_command=sweep_command,
                duration_s=post_joint_sweep_duration,
                phase=f"post_joint_1_sweep_{index}",
                sample_dt=sample_dt,
            )
            post_joint_sweep_trajectories.append(trajectory)
            previous_target = sweep_command

    # 退出 cuMotion context 后，所有规划和采样都已经完成。execution 阶段不需要 cuMotion backend；
    # 它只需要逐 physics step 把 command-space 位置目标交给 controller。
    runtime = ExecutionRuntime(
        articulation=robot,
        simulation_world=world,
        articulation_action_type=articulation_action_type,
        joint_controller=controller,
        simulation_app=simulation_app,
        render_enabled=render,
        drive_logger=drive_logger,
    )

    # execution_steps 是真正发送到仿真环境的动作序列。这里统一使用 CommandPosition*
    # 命名的 step，表示它们都下发“主动关节位置命令”：
    # - SmoothCommandPositionTargetStep：在 execution 中按 smoothstep 生成逐帧 command 目标；
    # - CommandPositionTrajectoryStep：播放已经 materialize 好的逐 physics step command 轨迹；
    # - HoldCommandPositionTargetStep：保持最后目标若干秒。
    #
    # 每个 step 内部都会调用 controller。controller 再根据当前实际 master 关节状态展开 mimic
    # follower，并构造 Isaac ArticulationAction；因此这里仍然不出现完整 DOF 目标。
    execution_steps = [
        SmoothCommandPositionTargetStep(
            start_command=initial_command,
            target_command=pre_pinch_command,
            duration=prep_duration,
            phase="pre_pinch",
        ),
        CommandPositionTrajectoryStep(
            trajectory=move_to_approach_trajectory,
        ),
        CommandPositionTrajectoryStep(
            trajectory=grasp_line_trajectory,
        ),
        SmoothCommandPositionTargetStep(
            start_command=grasp_open_command,
            target_command=grasp_closed_command,
            duration=close_duration,
            phase="close_fingers",
        ),
        CommandPositionTrajectoryStep(
            trajectory=lift_trajectory,
        ),
    ]
    for trajectory in wiggle_trajectories:
        execution_steps.append(CommandPositionTrajectoryStep(trajectory=trajectory))
    if wiggle_return_trajectory is not None:
        execution_steps.append(
            CommandPositionTrajectoryStep(trajectory=wiggle_return_trajectory)
        )
    execution_steps.append(
        HoldCommandPositionTargetStep(
            target_command=lifted_command,
            duration=final_hold_duration,
            phase="final",
        )
    )
    for trajectory in post_joint_sweep_trajectories:
        execution_steps.append(CommandPositionTrajectoryStep(trajectory=trajectory))

    # step 是全局 physics step 计数，用于日志时间戳和 phase 连续记录。每个 execution_step.run
    # 返回执行完该阶段后的 step，下一个阶段从这个计数继续。
    step = 0
    for execution_step in execution_steps:
        step = execution_step.run(runtime, step)

    # 返回值主要用于命令行打印和测试诊断，不作为下游控制接口。ik 字段保留关键目标和误差，
    # 方便快速判断失败是目标不可达、姿态约束太严，还是后续执行/接触问题。
    return {
        "steps": step,
        "ik": {
            "pinch_world": pinch_world,
            "approach_world": approach_world,
            "lifted_world": lifted_world,
            "tcp_xyz": tcp.xyz,
            "approach_success": approach.success,
            "approach_error": approach.position_error,
            "lift_success": lift.success,
            "lift_error": lift.position_error,
            "wiggles": [
                (world_target, result.success, result.position_error)
                for world_target, result in wiggles
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    profile 名称默认指向标准抓绳 demo：
    - robot profile 选择 AR5V2_L + L6V1_L 组合机器人；
    - env scene1 提供物理步频、重力、solver iteration 和对象列表；
    - controller/cumotion/logging profile 提供控制、规划和日志默认参数；
    - pinch grasp 动作参数固定在本脚本内，命令行只覆盖少量常用开关。

    这里刻意不暴露每个动作阶段的距离、时长、手指角度等参数。那些值属于这个动作脚本本身，
    直接在 ``run_pinch_grasp_action`` 开头修改更清楚；命令行只保留“换系统配置/切换运行模式”
    这类外部运行时参数。
    """

    # argparse 的 description 直接复用模块 docstring，这样 ``--help`` 能看到脚本整体说明、
    # 执行流程和坐标/数组约定。代价是 help 比普通脚本长一些，但这个入口本来就是 demo/debug
    # 脚本，可解释性优先。
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--env", default="scene1")

    # cuMotion 配置拆成两层使用：profile 中的 cumotion 字段会被 merge 到 robot_config，
    # motion_planner 子配置则在本脚本中直接传给 planner。这样同一份 profile 能同时控制
    # robot-level backend 资源和 action-level 规划策略。
    parser.add_argument(
        "--cumotion-profile",
        default="default",
        help="cuMotion profile name",
    )

    # logging profile 给默认日志行为；后面若干 --log-* 参数只做局部覆盖。这样 CI、调试和 GUI
    # 运行可以共用同一份日志 schema，但在命令行上快速打开/关闭某些 effort 列。
    parser.add_argument(
        "--logging-profile",
        default="default_logger",
    )
    parser.add_argument(
        "--log", type=Path, default=None, help="覆盖关节跟踪 CSV 输出路径"
    )
    parser.add_argument(
        "--log-interval-steps", type=int, default=None, help="覆盖日志采样步长"
    )
    parser.add_argument(
        "--log-measured-effort",
        action="store_true",
        help="记录 PhysX measured joint effort",
    )
    parser.add_argument(
        "--log-applied-effort",
        action="store_true",
        help="记录 Isaac applied joint effort",
    )
    parser.add_argument(
        "--log-action-effort",
        action="store_true",
        help="记录控制器实际下发的 effort action",
    )
    parser.add_argument(
        "--no-log-effort-command",
        action="store_true",
        help="不记录语义 effort command 列",
    )

    # GUI 和 hold 控制仿真生命周期：
    # - --gui 会创建带渲染窗口的 Isaac SimulationApp；
    # - --hold 只在 --no-grasp 分支中让 hold_current_pose 跟随 GUI 窗口生命周期持续
    #   下发初始姿态，方便人工检查资产姿态、碰撞体、关节命名和 follower。
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")

    # --no-grasp 是最小导入检查路径：导入场景和机器人，创建 controller/logger，然后保持初始姿态。
    # 它用于排除“资产导入/控制器初始化”问题；如果这个模式失败，通常还没必要看 cuMotion。
    parser.add_argument(
        "--no-grasp", action="store_true", help="只导入机器人和绳体，并短暂保持初始姿态"
    )

    # 动作层只接受 scripted world 抓取点，不读取 rope 内部段块/端块位置。
    parser.add_argument(
        "--grasp-world",
        type=float,
        nargs=3,
        default=(0.025, -0.55, 0.08),
        metavar=("X", "Y", "Z"),
        help="pinch TCP scripted world target position in meters",
    )

    # control-mode 选择默认 controller 配置中的驱动策略。动作脚本始终生成 command position 目标，
    # 但 controller 可以把这些语义目标转换为 position/velocity/effort 模式下的 ArticulationAction。
    parser.add_argument(
        "--control-mode", choices=("position", "velocity", "effort"), default="position"
    )

    return parser.parse_args()

def main() -> None:
    """脚本主入口：创建通用 runtime，然后执行 pinch grasp 动作。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    runtime = create_single_robot_runtime(
        env=args.env,
        cumotion_profile=args.cumotion_profile,
        logging_profile=args.logging_profile,
        control_mode=args.control_mode,
        gui=args.gui,
        status_prefix="RUN_PINCH_GRASP",
        logging_overrides=LoggingRuntimeOverrides(
            joint_tracking_path=args.log,
            interval_steps=args.log_interval_steps,
            log_measured_effort=True if args.log_measured_effort else None,
            log_applied_effort=True if args.log_applied_effort else None,
            log_action_effort=True if args.log_action_effort else None,
            log_command_effort=False if args.no_log_effort_command else None,
        ),
    )
    try:
        if args.no_grasp:
            hold_current_pose(
                runtime.execution,
                hold_until_app_closed=args.hold,
            )
            print("RUN_PINCH_GRASP_HOLD_OK", flush=True)
            return

        result = run_pinch_grasp_action(
            robot=runtime.robot,
            world=runtime.world,
            articulation_action_type=runtime.session.articulation_action_type,
            controller=runtime.controller,
            simulation_app=runtime.session.app,
            render=args.gui,
            mjcf_path=runtime.mjcf_path_required("pinch grasp"),
            cumotion_config=runtime.robot_cumotion,
            motion_planner_config=runtime.motion_planner_config,
            grasp_world=np.asarray(args.grasp_world, dtype=float),
            drive_logger=runtime.logger,
        )
        print(
            "RUN_PINCH_GRASP_OK "
            f"steps={result['steps']} ik={result['ik']} log={runtime.log_path}",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    # 作为脚本执行时直接进入 main；作为测试/工具模块导入时不会启动 Isaac。
    main()
