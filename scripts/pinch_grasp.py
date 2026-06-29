#!/usr/bin/env python3
"""运行 AR5 + LinkerHand L6 的绳端夹捏抓取动作脚本。

本文件是一个完整可运行动作入口：抓取动作参数直接写在脚本内，不再通过
外部 trajectory YAML 或任务包间接提供。外部 YAML 只保留机器人、控制器、环境、绳体、
日志和 cuMotion profile 这些可复用系统配置。也就是说，本文件表达的是“这一次动作怎么做”，
配置文件表达的是“这套仿真系统和机器人怎么运行”。

执行流程：
    1. 读取系统配置并启动 Isaac Sim。
    2. 导入 capsule rope 和 AR5+L6 组合机器人。
    3. 创建 JointController 和可选日志器。
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

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
# 这个脚本通常会用 ``python scripts/pinch_grasp.py`` 或 Isaac 的 Python 解释器直接运行。
# 直接运行脚本时，Python 默认只把 scripts/ 放进 sys.path，并不会自动把仓库的 src/
# 当作包根目录。这里显式插入 src/，保证导入的是当前工作区里的 linkerbot_sim 源码，
# 而不是系统环境中可能残留的已安装包。
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.runtime_settings import EnvRuntimeSettings
from linkerbot_sim.app.simulation_session import create_simulation_session
from linkerbot_sim.assets.robot_loader import (
    RobotExecutionConfig,
)
from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.backends.cumotion.profile_config import (
    merged_robot_config_with_cumotion_profile,
    motion_planner_config_from_profile,
    robot_cumotion_config,
)
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
)
from linkerbot_sim.backends.cumotion.tcp_context import make_cumotion_context
from linkerbot_sim.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)
from linkerbot_sim.controllers.config import (
    load_controller_profiles,
)
from linkerbot_sim.envs.scene_objects import (
    add_scene_objects,
    scene_objects_from_env_config,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.steps import (
    CommandPositionTrajectoryStep,
    HoldCommandPositionTargetStep,
    SmoothCommandPositionTargetStep,
)
from linkerbot_sim.logging.config import (
    joint_logging_config_from_mapping,
    override_logging_config,
)
from linkerbot_sim.logging.joint_logger import JointTrackingLogger
from linkerbot_sim.objects.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    endpoint_center,
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
from linkerbot_sim.robots.mimic import mjcf_equality_follower_joint_names
from linkerbot_sim.tcp.pinch_tcp import DEFAULT_PINCH_TCP_FRAME, make_pinch_tcp
from linkerbot_sim.execution.setup import (
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


DEFAULT_ROBOT_CONFIG = Path("configs/robots/ar5v2_l6v1_l.yaml")

_PRE_PINCH_HAND_TARGET_VALUES = (
    ("thumb_cmc_roll", 0.95),
    ("thumb_cmc_pitch", 0.28),
    ("index_mcp_pitch", 0.25),
    ("middle_mcp_pitch", 0.15),
    ("ring_mcp_pitch", 0.15),
    ("pinky_mcp_pitch", 0.12),
)
_CLOSED_PINCH_HAND_TARGET_VALUES = (
    ("thumb_cmc_roll", 0.95),
    ("thumb_cmc_pitch", 0.7),
    ("index_mcp_pitch", 0.85),
    ("middle_mcp_pitch", 0.45),
    ("ring_mcp_pitch", 0.4),
    ("pinky_mcp_pitch", 0.35),
)


def default_pre_pinch_hand_targets(side: str = "left") -> dict[str, float]:
    """返回 pinch_grasp 任务的预夹手型主动关节目标。"""

    return _hand_targets_for_side(side, _PRE_PINCH_HAND_TARGET_VALUES)


def default_closed_pinch_hand_targets(side: str = "left") -> dict[str, float]:
    """返回 pinch_grasp 任务的闭合夹捏手型主动关节目标。"""

    return _hand_targets_for_side(side, _CLOSED_PINCH_HAND_TARGET_VALUES)


def _hand_targets_for_side(
    side: str, values: tuple[tuple[str, float], ...]
) -> dict[str, float]:
    side_token = _hand_side_token(side)
    return {f"L6V1_{side_token}_hand_{joint}": float(value) for joint, value in values}


def _hand_side_token(side: str) -> str:
    normalized = str(side).lower()
    if normalized in {"left", "l"}:
        return "L"
    if normalized in {"right", "r"}:
        return "R"
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def grasp_target_position(
    rope_config: CapsuleRopeConfig, *, endpoint: str, lift_height: float = 0.0
) -> np.ndarray:
    """计算夹捏 TCP 的世界坐标目标位置。

    参数:
        rope_config: rope 对象配置，提供端点 cuboid 的几何位置。
        endpoint: 抓取左端或右端。
        lift_height: 额外 z 方向抬升高度，单位 m。
    返回:
        shape 为 ``(3,)`` 的世界坐标位置数组，单位 m。
    """

    # endpoint_center 使用 rope 配置中的几何定义计算端块中心。它不是实时读取仿真中变形后的绳体
    # 状态，而是给 scripted demo 提供一个稳定、可复现的抓取目标。
    #
    # target_world_offset 是针对当前 rope endpoint cuboid 和 pinch TCP 形状调过的偏置：
    # - x 方向略微偏移，让两指尖中心落在更容易夹住端块的位置；
    # - z 方向抬高一点，避免 TCP 目标落在端块几何中心过低处导致手指插入碰撞体；
    # - y 方向保持 0，说明当前 demo 默认从端块中线夹取。
    #
    # lift_height 只额外叠加在 z 方向，用于复用同一水平抓取点生成“已经夹住后向上抬”的目标。
    target_world_offset = (0.02, 0.0, 0.03)
    return (
        np.asarray(endpoint_center(rope_config, endpoint), dtype=float)
        + np.asarray(target_world_offset, dtype=float)
        + np.asarray([0.0, 0.0, lift_height], dtype=float)
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
    """规划一段 C-space 关节运动并嵌入 controller command-space。"""

    start = np.asarray(start_command, dtype=float).reshape(-1)
    target = np.asarray(target_command, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_command_indices, dtype=int).reshape(-1)
    result = motion_planner.plan(
        MotionRequest(
            current_q=start[arm_indices],
            goal_q=target[arm_indices],
            duration_s=duration_s,
        )
    )
    if not result.success:
        raise RuntimeError(
            f"cuMotion joint motion planning failed for {phase}: "
            f"status={result.status}"
        )
    if result.trajectory is None:
        raise RuntimeError(
            f"cuMotion joint motion planning returned no trajectory for {phase}: "
            f"status={result.status}"
        )
    arm_trajectory = joint_trajectory_from_cumotion(
        result.trajectory,
        joint_names=tuple(motion_planner.joint_names()),
        sample_dt=sample_dt,
        phase=phase,
    )
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
    """规划一段指定 TCP 直线并嵌入 controller command-space。"""

    start = np.asarray(start_command, dtype=float).reshape(-1)
    arm_indices = np.asarray(arm_command_indices, dtype=int).reshape(-1)
    base_config = motion_planner_config or MotionPlannerBackendConfig.from_mapping(None)
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
    specified_planner = context.make_motion_planner(
        tcp_frame_name=tcp_frame_name,
        config=specified_path_config,
    )
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
        raise RuntimeError(
            f"cuMotion specified TCP line planning failed for {phase}: "
            f"status={result.status}"
        )
    if result.trajectory is None:
        raise RuntimeError(
            f"cuMotion specified TCP line returned no trajectory for {phase}: "
            f"status={result.status}"
        )
    if result.path is None or np.asarray(result.path).shape[0] == 0:
        raise RuntimeError(
            "cuMotion specified TCP line returned trajectory without path "
            f"for {phase}: status={result.status}"
        )
    target_command = start.copy()
    target_command[arm_indices] = np.asarray(result.path, dtype=float)[-1]
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
    rope_config: CapsuleRopeConfig,
    mjcf_path: str | Path,
    cumotion_config: CuMotionConfig,
    motion_planner_config: MotionPlannerBackendConfig,
    endpoint: str,
    short_smoke: bool,
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
        rope_config: capsule rope 的几何和资产配置，用于计算 scripted 抓取点。
        mjcf_path: 当前机器人资产路径；pinch TCP 和 mimic follower 都需要读取 MJCF equality。
        cumotion_config: cuMotion 机器人模型、kinematics 和默认规划配置。
        motion_planner_config: 本动作使用的 cuMotion motion planner profile。
        endpoint: 抓取 rope 的 left 或 right 端。
        short_smoke: 快速 smoke 模式，压缩时长、关闭 wiggle，用于 CI/headless 检查。
        drive_logger: 可选逐步关节跟踪日志器。
    """

    mjcf_path = Path(mjcf_path)

    # TCP frame 名称必须与 make_pinch_tcp 创建的虚拟 frame 一致。cuMotion 的 IK 和 motion planner
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
    # target_rpy 描述夹捏 TCP 在抓取 rope 端块时希望保持的姿态，采用 xyz 欧拉角，单位 rad。
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

    # short_smoke 模式用于快速验证脚本能跑通导入、规划、采样和 execution 链路。它不是正常 demo
    # 参数：所有阶段时长缩短，抬升高度降低，并且关闭 wiggle/末尾 sweep，避免 headless 测试太慢。
    short_smoke_duration = 0.02
    short_smoke_lift_height = 0.05
    lift_height = short_smoke_lift_height if short_smoke else 0.4

    # 各阶段时长交给 cuMotion 处理：C-space trajectory 的时间参数化由 cuMotion 根据 duration_s
    # 生成。项目侧只负责按 physics_dt 采样，不在采样后重新 retime。
    prep_duration = short_smoke_duration if short_smoke else 2.0
    move_duration = short_smoke_duration if short_smoke else 6.0
    approach_duration = short_smoke_duration if short_smoke else 2.2
    close_duration = short_smoke_duration if short_smoke else 2.0
    lift_duration = short_smoke_duration if short_smoke else 4.0
    wiggle_cycles = 0 if short_smoke else 2
    wiggle_duration = short_smoke_duration if short_smoke else 4.0
    final_hold_duration = short_smoke_duration if short_smoke else 3.0
    post_joint_sweep_duration = (
        short_smoke_duration if short_smoke else 5.0
    )
    post_joint_sweep_target_values = () if short_smoke else (2.1, -2.1)

    # 手指目标只写主动关节。L6 手里由 MJCF equality 表达的 DIP 等 follower 不出现在这里；
    # controller 会在每个 execution step 里根据实际 master 关节位置计算 follower 目标。
    #
    # pre_pinch_hand_targets 是“接近抓取点前”的预成型手势：拇指已经转向食指，食指和其它手指
    # 轻微弯曲，避免机械臂移动时手完全张开造成碰撞或穿插。
    pre_pinch_hand_targets = default_pre_pinch_hand_targets("left")

    # closed_pinch_hand_targets 是真正闭合后的夹捏手势。这个手势有两个用途：
    # 1. make_pinch_tcp 用它计算“闭合时两指尖中点”相对于法兰盘的 TCP；
    # 2. execution 的 close_fingers 阶段会把主动手指关节平滑移动到这些目标。
    # 因此 TCP 几何和实际闭合动作使用同一组主动关节角，避免规划目标和最终手型不一致。
    closed_pinch_hand_targets = default_closed_pinch_hand_targets("left")

    # make_pinch_tcp 的职责是离线构造一个“夹捏 TCP”：
    # - 输入闭合手势的主动关节角；
    # - 根据 MJCF equality 展开 mimic follower；
    # - 沿手掌根部到两指尖的 kinematic chain 做一次局部 FK；
    # - 取拇指/食指指尖几何中点作为 TCP；
    # - 把这个 TCP 挂到 cuMotion 使用的 flange_frame 下。
    #
    # 注意：这里的 mimic 展开只服务于 TCP 几何推导，不表示动作脚本从此开始控制 follower。
    # 运行时 follower 仍然只由 controller 在执行边界根据实际 master 状态补齐。
    tcp = make_pinch_tcp(
        mjcf_path,
        closed_pinch_hand_targets,
        parent_frame=cumotion_config.flange_frame,
        frame_name=tcp_frame_name,
    )

    # 三个核心笛卡尔目标：
    # - pinch_world：真正希望 pinch TCP 到达的 rope 端块抓取点；
    # - approach_world：pinch_world 正上方的预接近点；
    # - lifted_world：保持水平位置不变，只把抓住后的端块向上抬。
    #
    # 这里使用 scripted world 坐标，而不是实时感知 rope 姿态；目的是让 demo 可复现，后续如果接入
    # 视觉/状态估计，只需要把 pinch_world 的来源替换掉。
    pinch_world = grasp_target_position(rope_config, endpoint=endpoint)
    approach_world = pinch_world + np.asarray(
        [0.0, 0.0, approach_distance], dtype=float
    )
    lifted_world = grasp_target_position(
        rope_config, endpoint=endpoint, lift_height=lift_height
    )
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
        # short_smoke 模式下 post_joint_sweep_target_values 为空，因此不会生成这些扰动轨迹。
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

    各配置文件默认指向仓库内的标准抓绳 demo：
    - robot config 选择 AR5V2_L + L6V1_L 组合机器人；
    - controller config 目录提供按部件分组的位置、速度和 effort 控制参数；
    - env config 提供物理步频、重力和 solver iteration；
    - rope config 提供 capsule rope 资产路径和 prim 路径；
    - pinch grasp 动作参数固定在本脚本内，命令行只覆盖少量常用开关。

    这里刻意不暴露每个动作阶段的距离、时长、手指角度等参数。那些值属于这个动作脚本本身，
    直接在 ``run_pinch_grasp_action`` 开头修改更清楚；命令行只保留“换系统配置/切换运行模式”
    这类外部运行时参数。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument(
        "--controller-config", type=Path, default=Path("configs/controllers")
    )
    parser.add_argument(
        "--env-config", type=Path, default=Path("configs/envs/rope_scene.yaml")
    )
    parser.add_argument(
        "--rope-config", type=Path, default=Path("configs/objects/capsule_rope.yaml")
    )
    parser.add_argument(
        "--cumotion-config",
        type=Path,
        default=Path("configs/cumotion/default.yaml"),
        help=(
            "cuMotion profile YAML. Its cumotion section is used as robot-level "
            "defaults, and cumotion.motion_planner is used as action planner defaults."
        ),
    )
    parser.add_argument(
        "--logging-config",
        type=Path,
        default=Path("configs/logging/default_logger.yaml"),
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
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")
    parser.add_argument(
        "--no-grasp", action="store_true", help="只导入机器人和绳体，并短暂保持初始姿态"
    )
    parser.add_argument(
        "--short-smoke",
        action="store_true",
        help="覆盖阶段时长，用于快速 headless smoke",
    )
    parser.add_argument("--endpoint", choices=("left", "right"), default=None)
    parser.add_argument(
        "--control-mode", choices=("position", "velocity", "effort"), default="position"
    )
    parser.add_argument("--physics-frequency", type=float, default=None)
    parser.add_argument("--render-frequency", type=float, default=None)
    parser.add_argument("--gravity-z", type=float, default=None)
    return parser.parse_args()


def hold_initial_pose(
    robot,
    world,
    articulation_action_type,
    controller,
    simulation_app,
    render: bool,
    logger,
) -> None:
    """保持当前姿态几步，用于 import smoke。

    ``--no-grasp`` 会走这个分支。它不执行抓取动作，只把当前机器人关节位置作为目标反复下发，
    用于确认机器人资产、驱动参数、mimic follower 和日志系统是否能正常初始化。
    如果同时传入 ``--hold`` 和 ``--gui``，会持续保持到 Isaac 窗口关闭。

    这个函数使用 full_state 入口是因为它刻意绕过动作脚本和 cuMotion，只测试“当前导入的
    articulation 是否能被 controller 稳定驱动”。controller.targets_from_full_state 会把完整
    DOF 状态转换成受驱动目标，其中 follower 仍按 mimic 关系处理。
    """

    full_target = np.asarray(robot.get_joint_positions(), dtype=float)
    full_velocity = np.zeros(robot.num_dof, dtype=float)
    step = 0
    while step < 3 or (simulation_app is not None and simulation_app.is_running()):
        # --no-grasp 分支不生成新轨迹，每一帧都把当前姿态作为 hold 目标下发。这样如果资产导入、
        # drive 参数或 mimic 映射有问题，通常会在最短路径上暴露出来。
        targets = controller.targets_from_full_state(full_target, full_velocity)
        controller.apply_targets(articulation_action_type, targets)
        world.step(render=render)
        if logger is not None:
            driven_indices = controller.driven_indices
            if logger.should_write(step):
                # 日志记录的是 controller 实际驱动的关节集合，而不是所有 articulation DOF。
                # 这样 CSV 中既能看到主动关节，也能看到 follower 是否按 mimic 关系跟随。
                log_values = logger.collect_step_values(
                    robot, controller, targets, driven_indices
                )
                logger.write(
                    step=step,
                    time_s=(step + 1) * float(world.get_physics_dt()),
                    phase="initial_hold",
                    drive_update=True,
                    **log_values,
                )
        step += 1
        if simulation_app is None and step >= 3:
            break


def main() -> None:
    """脚本主入口。

    main 负责准备“运行环境”：读取配置、启动 Isaac、导入资产、创建 controller/logger，
    然后把这些对象交给 ``run_pinch_grasp_action``。真正的动作语义不写在 main 里，避免把
    仿真启动细节和夹捏动作流程混在一起。
    """

    # Isaac/Kit 日志很多，开启行缓冲可以保证 RUN_PINCH_GRASP_* 状态行尽快刷出，
    # 方便 live log、调试脚本和外部监控程序读取。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    # 先加载所有 YAML 配置。这里还没有启动 Isaac Sim，尽量把纯 Python 的配置错误提前暴露，
    # 避免启动 GUI 后才因为路径或字段缺失失败。
    #
    # 配置职责划分：
    # - robot_config：资产路径、prim path、受控关节、机器人级 cuMotion 资源；
    # - controller_profiles：不同控制模式下的 drive/effort 参数；
    # - env_config：physics/render frequency、gravity、solver iteration；
    # - rope_config_data：rope USD 资产和几何端点；
    # - cumotion_profile：后端通用默认值和 motion planner profile；
    # - logging_config：CSV 输出位置、采样间隔和记录列。
    cumotion_profile = load_yaml(args.cumotion_config)
    robot_config = merged_robot_config_with_cumotion_profile(
        load_yaml(args.robot_config), cumotion_profile
    )
    controller_profiles = load_controller_profiles(args.controller_config)
    env_config = load_yaml(args.env_config)
    rope_config_data = load_yaml(args.rope_config)
    logging_config = joint_logging_config_from_mapping(load_yaml(args.logging_config))
    logging_config = override_logging_config(
        logging_config,
        joint_tracking_path=args.log,
        interval_steps=args.log_interval_steps,
        log_measured_effort=True if args.log_measured_effort else None,
        log_applied_effort=True if args.log_applied_effort else None,
        log_action_effort=True if args.log_action_effort else None,
        log_command_effort=False if args.no_log_effort_command else None,
    )

    runtime_settings = EnvRuntimeSettings.from_env_config(
        env_config,
        physics_frequency_override=args.physics_frequency,
        render_frequency_override=args.render_frequency,
        gravity_z_override=args.gravity_z,
    )
    scene_objects = scene_objects_from_env_config(env_config)

    robot_execution = RobotExecutionConfig.from_mapping(robot_config)

    # robot_cumotion 来自已经合并 profile 默认值后的 robot_config。此处解析的是 cuMotion context
    # 所需的机器人模型配置；motion_planner_config 则单独从 profile 读取，二者传入不同层。
    robot_cumotion = robot_cumotion_config(robot_config)

    rope_config = CapsuleRopeConfig.from_mapping(rope_config_data)
    motion_planner_config = motion_planner_config_from_profile(cumotion_profile)

    # endpoint 默认左端；脚本内不再保留 DEFAULT_ENDPOINT 常量，简单动作默认值直接写在使用处。
    endpoint = args.endpoint or "left"

    session = create_simulation_session(gui=args.gui, settings=runtime_settings)
    try:
        world = session.world
        stage = session.stage
        added_scene_objects = add_scene_objects(stage, scene_objects)
        for scene_object in added_scene_objects:
            print(
                "RUN_PINCH_GRASP_SCENE_OBJECT "
                f"name={scene_object.name} type={scene_object.asset_type} "
                f"asset={scene_object.asset_path} prim_path={scene_object.prim_path} "
                f"imported_path={scene_object.imported_path} "
                f"static={scene_object.static}",
                flush=True,
            )
        rope_model = add_capsule_rope_reference(stage, rope_config)
        print(
            "RUN_PINCH_GRASP_ROPE "
            f"asset={rope_config.asset_file()} prim_path={rope_config.prim_path} "
            f"segments={rope_config.segments} shape={rope_config.shape} "
            f"bodies={len(rope_model['bodies'])} joints={len(rope_model['joints'])}",
            flush=True,
        )

        imported = import_execution_robot_to_stage(
            world=world,
            stage=stage,
            single_articulation_type=session.single_articulation_type,
            robot_execution=robot_execution,
            controller_profiles=controller_profiles,
            env_config=env_config,
        )
        print(
            f"RUN_PINCH_GRASP_GRAVITY {imported.gravity_counts}",
            flush=True,
        )
        print(f"RUN_PINCH_GRASP_SOLVER {imported.solver_counts}", flush=True)

        world.reset()
        world.get_physics_context().set_gravity(runtime_settings.gravity_z)
        prepared = finalize_robot_controller(
            imported=imported,
            controller_profiles=controller_profiles,
            control_mode=args.control_mode,
        )
        robot = prepared.articulation
        controller = prepared.joint_controller
        asset_path = prepared.asset_path
        mjcf_path = prepared.mjcf_path

        # L6 手的 DIP 等 follower 关节由 MJCF equality 描述。运行时根据实际 master 关节状态
        # 更新 follower 目标，避免 follower 跟随“命令目标”而不是“实际主动关节”导致超前。
        mimic_names = mjcf_equality_follower_joint_names(mjcf_path)

        # 日志只记录实际受驱动的 DOF，即主动关节 + mimic follower。flush_interval_steps 控制
        # CSV 刷盘频率，避免每个 physics step 都 flush 造成 I/O 开销过大。
        # 如果 logging_config.enabled=False 或 joint_tracking_path=None，JointTrackingLogger 会成为
        # 空 logger，不会实际写文件，但调用路径保持一致。
        driven_joint_names = [
            list(robot.dof_names)[int(index)] for index in controller.driven_indices
        ]
        flush_interval_steps = logging_config.flush_interval_steps(
            float(world.get_physics_dt())
        )
        log_path = (
            None
            if not logging_config.enabled or logging_config.joint_tracking_path is None
            else repo_path(logging_config.joint_tracking_path)
        )
        logger = JointTrackingLogger(
            log_path,
            driven_joint_names,
            flush_interval_steps=flush_interval_steps,
            config=logging_config,
        )
        print(
            "RUN_PINCH_GRASP_IMPORTED "
            f"asset={asset_path} prim_path={imported.articulation_path} num_dof={robot.num_dof} "
            f"control_mode={args.control_mode} mimic_joint_names={sorted(mimic_names)} "
            f"follower_relations={controller.follower_mapper.relations}",
            flush=True,
        )
        print(
            "RUN_PINCH_GRASP_DOF_NAMES " + ", ".join(list(robot.dof_names)), flush=True
        )

        try:
            if args.no_grasp:
                # 仅做导入和控制器 smoke test，不构造 pinch TCP，也不调用 cuMotion。
                # 如果同时传 --hold --gui，这个分支会一直保持初始姿态，方便在 GUI 中检查资产。
                hold_initial_pose(
                    robot,
                    world,
                    session.articulation_action_type,
                    controller,
                    session.app if args.hold else None,
                    args.gui,
                    logger,
                )
                print("RUN_PINCH_GRASP_HOLD_OK", flush=True)
            else:
                # 正式执行夹捏动作。run_pinch_grasp_action 会在内部完成 TCP 构造、cuMotion 规划、
                # trajectory_sampler 离散采样和 execution 逐帧播放。
                if mjcf_path is None:
                    raise ValueError("pinch grasp requires an MJCF robot asset")
                result = run_pinch_grasp_action(
                    robot=robot,
                    world=world,
                    articulation_action_type=session.articulation_action_type,
                    controller=controller,
                    simulation_app=session.app,
                    render=args.gui,
                    rope_config=rope_config,
                    mjcf_path=mjcf_path,
                    cumotion_config=robot_cumotion,
                    motion_planner_config=motion_planner_config,
                    endpoint=endpoint,
                    short_smoke=args.short_smoke,
                    drive_logger=logger,
                )
                print(
                    "RUN_PINCH_GRASP_OK "
                    f"steps={result['steps']} ik={result['ik']} log={log_path}",
                    flush=True,
                )
        finally:
            # 无论动作成功、失败还是用户 Ctrl+C，都尽量关闭 CSV 文件，避免最后几行日志丢失。
            logger.close()
    finally:
        # 必须关闭 SimulationApp，否则 Kit/Isaac 进程和扩展资源可能残留。
        session.app.close()


if __name__ == "__main__":
    main()
