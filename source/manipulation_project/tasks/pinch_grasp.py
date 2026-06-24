"""夹捏抓取任务流程。

该任务面向机械臂 + 灵巧手 + rope endpoint box 的 scripted demo：
先根据闭合手型从 MJCF 运动链计算 thumb/index 夹捏中心 TCP，再把这个 TCP 写入
临时 URDF 供 IK 后端求解，最后把 approach、grasp、lift、wiggle 等阶段合成为
完整 articulation DOF 目标并在 Isaac 中执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from manipulation_project.backends.cumotion.context import CuMotionConfig, CuMotionContext
from manipulation_project.backends.cumotion.tcp_urdf_builder import write_tcp_urdf
from manipulation_project.planning.requests import IKRequest
from manipulation_project.objects.capsule_rope import CapsuleRopeConfig, endpoint_center
from manipulation_project.robots.joint_groups import target_vector_from_mapping
from manipulation_project.robots.mimic import MimicFollowerTargetMapper, expand_targets_with_mjcf_equalities
from manipulation_project.tcp.pinch_tcp import DEFAULT_PINCH_TCP_FRAME, make_pinch_tcp
from manipulation_project.utils.rotations import rpy_xyz_deg_to_quat_wxyz


@dataclass(frozen=True)
class PinchGraspConfig:
    """夹捏抓取脚本的配置集合。

    输入字段:
        endpoint: 选择 rope 的 ``left`` 或 ``right`` 端点作为抓取目标。
        target_world_offset: 在端点中心上叠加的世界坐标偏移，单位 m。
        target_rpy_deg: 目标 TCP 姿态，固定轴 XYZ 顺序（外旋 XYZ 顺序）的 RPY，单位 degree。
        use_orientation: IK 是否约束 TCP 姿态；为假时只约束位置。
        approach_distance: 抓取前从目标正上方接近的高度差，单位 m。
        lift_height: 抓住后抬升高度，单位 m。
        prep/move/approach/close/lift/wiggle/final/post...: 各阶段持续时间，单位 s。
        wiggle_axis: 抬升后摆动方向，世界坐标向量。
        ik_*: 传给 IK 求解器的容差、迭代次数和种子。
        tcp_frame_name: 写入临时 URDF 的 pinch TCP frame 名。
        pre_pinch_hand_targets: 预夹捏手型的稀疏关节目标，单位 rad。
        closed_pinch_hand_targets: 闭合夹捏手型的稀疏关节目标，单位 rad。
    输出:
        该 dataclass 作为 ``PinchGraspTask`` 的输入；``pre_targets`` 和
        ``closed_targets`` 属性会返回可修改的字典副本。
    """

    endpoint: str = "left"
    target_world_offset: tuple[float, float, float] = (0.02, 0.0, 0.03)
    target_rpy_deg: tuple[float, float, float] = (0.0, 115.0, -90.0)
    use_orientation: bool = True
    approach_distance: float = 0.10
    lift_height: float = 0.4
    prep_duration: float = 1.0
    move_duration: float = 3.0
    approach_duration: float = 1.2
    close_duration: float = 1.0
    lift_duration: float = 2.0
    wiggle_cycles: int = 2
    wiggle_amplitude: float = 0.2
    wiggle_duration: float = 2.0
    wiggle_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    final_hold_duration: float = 5.0
    post_joint_sweep_duration: float = 2.0
    post_joint_sweep_targets: tuple[float, ...] = (2.1, -2.1)
    ik_position_tolerance: float = 0.005
    ik_orientation_tolerance: float = 0.75
    ik_max_iterations: int = 180
    ik_bfgs_max_iterations: int = 80
    ik_orientation_weight: float = 0.25
    ik_seeds: tuple[tuple[float, ...], ...] = ((-1.57, 0.8, 0.0, 0.8, 0.0, 0.0, 0.0),)
    tcp_frame_name: str = DEFAULT_PINCH_TCP_FRAME
    pre_pinch_hand_targets: dict[str, float] | None = None
    closed_pinch_hand_targets: dict[str, float] | None = None

    @classmethod
    def from_mapping(cls, data: dict) -> "PinchGraspConfig":
        """从 YAML 字典构造配置。

        参数:
            data: 完整任务配置，必须包含 ``grasp`` 子字典。
        返回:
            ``PinchGraspConfig``，缺失字段会使用类默认值；手型目标必须由配置提供。
        """

        if "grasp" not in data:
            raise ValueError("Pinch grasp config must contain top-level grasp section")
        grasp = data["grasp"]
        return cls(
            endpoint=str(grasp.get("endpoint", cls.endpoint)),
            target_world_offset=tuple(float(value) for value in grasp.get("target_world_offset", cls.target_world_offset)),
            target_rpy_deg=tuple(float(value) for value in grasp.get("target_rpy_deg", cls.target_rpy_deg)),
            use_orientation=bool(grasp.get("use_orientation", cls.use_orientation)),
            approach_distance=float(grasp.get("approach_distance", cls.approach_distance)),
            lift_height=float(grasp.get("lift_height", cls.lift_height)),
            prep_duration=float(grasp.get("prep_duration", cls.prep_duration)),
            move_duration=float(grasp.get("move_duration", cls.move_duration)),
            approach_duration=float(grasp.get("approach_duration", cls.approach_duration)),
            close_duration=float(grasp.get("close_duration", cls.close_duration)),
            lift_duration=float(grasp.get("lift_duration", cls.lift_duration)),
            wiggle_cycles=int(grasp.get("wiggle_cycles", cls.wiggle_cycles)),
            wiggle_amplitude=float(grasp.get("wiggle_amplitude", cls.wiggle_amplitude)),
            wiggle_duration=float(grasp.get("wiggle_duration", cls.wiggle_duration)),
            wiggle_axis=tuple(float(value) for value in grasp.get("wiggle_axis", cls.wiggle_axis)),
            final_hold_duration=float(grasp.get("final_hold_duration", cls.final_hold_duration)),
            post_joint_sweep_duration=float(grasp.get("post_joint_sweep_duration", cls.post_joint_sweep_duration)),
            post_joint_sweep_targets=tuple(float(value) for value in grasp.get("post_joint_sweep_targets", cls.post_joint_sweep_targets)),
            ik_position_tolerance=float(grasp.get("ik_position_tolerance", cls.ik_position_tolerance)),
            ik_orientation_tolerance=float(grasp.get("ik_orientation_tolerance", cls.ik_orientation_tolerance)),
            ik_max_iterations=int(grasp.get("ik_max_iterations", cls.ik_max_iterations)),
            ik_bfgs_max_iterations=int(grasp.get("ik_bfgs_max_iterations", cls.ik_bfgs_max_iterations)),
            ik_orientation_weight=float(grasp.get("ik_orientation_weight", cls.ik_orientation_weight)),
            ik_seeds=tuple(tuple(float(v) for v in seed) for seed in grasp.get("ik_seeds", cls.ik_seeds)),
            tcp_frame_name=str(grasp.get("tcp_frame_name", cls.tcp_frame_name)),
            pre_pinch_hand_targets=dict(grasp["pre_pinch_hand_targets"]) if "pre_pinch_hand_targets" in grasp else None,
            closed_pinch_hand_targets=dict(grasp["closed_pinch_hand_targets"]) if "closed_pinch_hand_targets" in grasp else None,
        )

    @property
    def pre_targets(self) -> dict[str, float]:
        """返回预夹捏手型目标。

        返回:
            ``关节名 -> 目标位置(rad)`` 的新字典，调用方可安全修改。
        """

        if self.pre_pinch_hand_targets is None:
            raise ValueError("pre_pinch_hand_targets must be provided for the selected hand")
        return dict(self.pre_pinch_hand_targets)

    @property
    def closed_targets(self) -> dict[str, float]:
        """返回闭合夹捏手型目标。

        返回:
            ``关节名 -> 目标位置(rad)`` 的新字典，调用方可安全修改。
        """

        if self.closed_pinch_hand_targets is None:
            raise ValueError("closed_pinch_hand_targets must be provided for the selected hand")
        return dict(self.closed_pinch_hand_targets)

    def validate(self) -> None:
        """检查配置取值是否满足任务执行要求。

        输入:
            使用 dataclass 当前字段值。
        返回:
            无返回值；发现非法端点、负时长或无效 IK 参数时抛出 ``ValueError``。
        """

        if self.endpoint not in {"left", "right"}:
            raise ValueError("endpoint must be left or right")
        nonnegative = {
            "approach_distance": self.approach_distance,
            "lift_height": self.lift_height,
            "prep_duration": self.prep_duration,
            "move_duration": self.move_duration,
            "approach_duration": self.approach_duration,
            "close_duration": self.close_duration,
            "lift_duration": self.lift_duration,
            "wiggle_amplitude": self.wiggle_amplitude,
            "wiggle_duration": self.wiggle_duration,
            "final_hold_duration": self.final_hold_duration,
            "post_joint_sweep_duration": self.post_joint_sweep_duration,
            "ik_position_tolerance": self.ik_position_tolerance,
            "ik_orientation_tolerance": self.ik_orientation_tolerance,
            "ik_orientation_weight": self.ik_orientation_weight,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.wiggle_cycles < 0:
            raise ValueError("wiggle_cycles cannot be negative")
        if self.ik_max_iterations <= 0 or self.ik_bfgs_max_iterations <= 0:
            raise ValueError("IK iteration counts must be positive")
        if not self.tcp_frame_name:
            raise ValueError("tcp_frame_name cannot be empty")
        if not self.pre_pinch_hand_targets:
            raise ValueError("pre_pinch_hand_targets must be provided for the selected hand")
        if not self.closed_pinch_hand_targets:
            raise ValueError("closed_pinch_hand_targets must be provided for the selected hand")
        if np.linalg.norm(np.asarray(self.wiggle_axis, dtype=float)) <= 0.0:
            raise ValueError("wiggle_axis must be non-zero")


def set_joint_targets_by_indices(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    """按索引原地写入一组关节目标。

    参数:
        target: 完整 DOF 目标数组，会被原地修改。
        indices: 要写入的 DOF 索引数组。
        values: 与 ``indices`` 等长的位置值数组，单位 rad。
    返回:
        无返回值；结果写回 ``target``。
    """

    for index, value in zip(indices, values, strict=True):
        target[int(index)] = float(value)


def grasp_target_position(config: PinchGraspConfig, rope_config: CapsuleRopeConfig, *, lift_height: float = 0.0) -> np.ndarray:
    """计算夹捏 TCP 的世界坐标目标位置。

    参数:
        config: 抓取配置，提供端点选择和目标偏移。
        rope_config: rope 对象配置，提供端点 box 的几何位置。
        lift_height: 额外 z 方向抬升高度，单位 m。
    返回:
        shape 为 ``(3,)`` 的世界坐标位置数组，单位 m。
    """

    return (
        np.asarray(endpoint_center(rope_config, config.endpoint), dtype=float)
        + np.asarray(config.target_world_offset, dtype=float)
        + np.asarray([0.0, 0.0, lift_height], dtype=float)
    )


def step_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    target_all: np.ndarray,
    render: bool,
    step: int,
    phase: str,
    target_velocity_all: np.ndarray | None = None,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """下发一帧完整 DOF 目标，并推进一个 physics step。

    参数:
        robot: Isaac articulation 对象。
        world: Isaac world，用于 ``step`` 和读取 physics dt。
        articulation_action_type: Isaac action 类型构造器。
        driven_indices: 实际受驱动的 DOF 索引。
        target_all: 完整 DOF 位置目标，单位 rad。
        render: 是否渲染当前仿真步。
        step: 全局日志步号。
        phase: 当前任务阶段名。
        target_velocity_all: 可选完整 DOF 速度目标，单位 rad/s。
        drive_logger: 可选关节跟踪日志器。
        follower_mapper: 可选 mimic follower 映射器，会用实际 master 状态更新 follower。
    返回:
        下一帧的全局步号，即 ``step + 1``。
    """

    command_target_all = np.asarray(target_all, dtype=float).copy()
    if target_velocity_all is None:
        command_velocity_all = np.zeros(robot.num_dof, dtype=float)
    else:
        command_velocity_all = np.asarray(target_velocity_all, dtype=float).copy()
    if follower_mapper is not None:
        follower_mapper.apply_from_actual(
            command_target_all,
            command_velocity_all,
            np.asarray(robot.get_joint_positions(), dtype=float),
            np.asarray(robot.get_joint_velocities(), dtype=float),
        )
    driven_position = command_target_all[driven_indices]
    driven_velocity = command_velocity_all[driven_indices]
    robot.apply_action(
        articulation_action_type(
            joint_positions=driven_position,
            joint_velocities=driven_velocity,
            joint_indices=driven_indices,
        )
    )
    world.step(render=render)
    if drive_logger is not None:
        actual_position = np.asarray(robot.get_joint_positions(), dtype=float)[driven_indices]
        actual_velocity = np.asarray(robot.get_joint_velocities(), dtype=float)[driven_indices]
        drive_logger.write(
            step=step,
            time_s=(step + 1) * float(world.get_physics_dt()),
            phase=phase,
            drive_update=True,
            desired_position=driven_position,
            actual_position=actual_position,
            desired_velocity=driven_velocity,
            actual_velocity=actual_velocity,
        )
    return step + 1


def move_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    start_all: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """用 smoothstep 在两个完整 DOF 目标之间平滑移动。

    参数:
        robot/world/articulation_action_type: Isaac 执行所需对象。
        driven_indices: 下发 action 时包含的 DOF 索引。
        start_all: 起始完整 DOF 目标，单位 rad。
        target_all: 终止完整 DOF 目标，单位 rad。
        duration: 移动时长，单位 s；会按 world physics dt 离散化。
        phase: 写入日志的阶段名。
        simulation_app: 可选 Isaac app，用于检测窗口是否仍在运行。
        render: 是否渲染。
        step: 输入的全局步号。
        drive_logger: 可选关节跟踪日志器。
        follower_mapper: 可选 mimic follower 映射器。
    返回:
        执行完本阶段后的全局步号。
    """

    physics_dt = float(world.get_physics_dt())
    steps = max(1, int(round(duration / physics_dt)))
    delta = target_all - start_all
    for local_step in range(steps):
        if simulation_app is not None and not simulation_app.is_running():
            break
        alpha = (local_step + 1) / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        smooth_rate = (6.0 * alpha * (1.0 - alpha) / duration) if duration > 0 else 0.0
        command = start_all + smooth * delta
        velocity = smooth_rate * delta
        step = step_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=command,
            target_velocity_all=velocity,
            render=render,
            step=step,
            phase=phase,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))
    return step


def hold_joint_target(
    *,
    robot,
    world,
    articulation_action_type,
    driven_indices: np.ndarray,
    target_all: np.ndarray,
    duration: float,
    phase: str,
    simulation_app,
    render: bool,
    step: int,
    drive_logger=None,
    follower_mapper: MimicFollowerTargetMapper | None = None,
) -> int:
    """保持一个完整 DOF 目标一段时间。

    参数:
        robot/world/articulation_action_type: Isaac 执行所需对象。
        driven_indices: 下发 action 时包含的 DOF 索引。
        target_all: 需要保持的完整 DOF 位置目标，单位 rad。
        duration: 保持时长，单位 s；为 0 时一直保持到 app 结束。
        phase: 写入日志的阶段名。
        simulation_app: 可选 Isaac app。
        render: 是否渲染。
        step: 输入的全局步号。
        drive_logger: 可选关节跟踪日志器。
        follower_mapper: 可选 mimic follower 映射器。
    返回:
        保持阶段结束后的全局步号。
    """

    physics_dt = float(world.get_physics_dt())
    total_steps = max(1, int(round(duration / physics_dt))) if duration > 0 else None
    local_step = 0
    while total_steps is None or local_step < total_steps:
        if simulation_app is not None and not simulation_app.is_running():
            break
        step = step_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=target_all,
            render=render,
            step=step,
            phase=phase,
            target_velocity_all=np.zeros(robot.num_dof, dtype=float),
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        local_step += 1
    return step


class PinchGraspTask:
    """机械臂+灵巧手对 rope 端点 box 的脚本化夹捏任务。

    输入:
        初始化时传入抓取配置、rope 场景配置、MJCF 路径、IK 机器人描述和基础 URDF。
    输出:
        ``plan`` 返回可执行的目标数组和 IK 诊断信息；``run`` 会实际推进仿真并返回同一份
        plan 字典，额外带 ``steps``。
    """

    def __init__(
        self,
        *,
        config: PinchGraspConfig,
        rope_config: CapsuleRopeConfig,
        mjcf_path: str | Path,
        parent_frame: str,
        cumotion_xrdf_path: str | Path,
        cumotion_urdf_path: str | Path,
        tcp_frame_name: str | None = None,
    ) -> None:
        """保存任务配置和 IK/TCP 资源路径。

        参数:
            config: 夹捏抓取配置。
            rope_config: rope 对象配置，用于定位端点。
            mjcf_path: 组合 MJCF 文件路径，用于计算 pinch TCP 和 mimic 关系。
            parent_frame: pinch TCP 固连到的父 link 名称，通常是手掌基座。
            cumotion_xrdf_path: cuMotion XRDF 文件。
            cumotion_urdf_path: 未附加 TCP 的基础 URDF。
            tcp_frame_name: 写入临时 URDF 的 TCP frame 名称。
        返回:
            无返回值。
        """

        self.config = config
        self.rope_config = rope_config
        self.mjcf_path = Path(mjcf_path)
        self.cumotion_xrdf_path = Path(cumotion_xrdf_path)
        self.cumotion_urdf_path = Path(cumotion_urdf_path)
        self.parent_frame = parent_frame
        self.tcp_frame_name = tcp_frame_name or config.tcp_frame_name

    def plan(self, robot) -> dict[str, object]:
        """规划抓取各阶段的 IK 解和完整 DOF 目标。

        参数:
            robot: Isaac articulation，需提供 ``dof_names`` 和当前关节位置。
        返回:
            字典，包含:
            ``arm_indices``: cuMotion C-space 关节在完整 DOF 中的索引；
            ``*_all``: 各阶段完整 DOF 位置目标；
            ``wiggle_all_targets``/``post_joint_sweep_targets``: 后续阶段目标列表；
            ``ik``: TCP 位置、求解后端、各阶段误差和成功标志。
        """

        self.config.validate()
        # 先用闭合手型计算 thumb/index 的几何夹捏中心。这里需要展开 mimic follower，
        # 否则 MJCF 运动链里从动关节会停在 0，TCP 会偏离实际闭合指尖中心。
        closed_geometry_targets = expand_targets_with_mjcf_equalities(self.config.closed_targets, self.mjcf_path)
        tcp = make_pinch_tcp(
            self.mjcf_path,
            closed_geometry_targets,
            parent_frame=self.parent_frame,
            frame_name=self.tcp_frame_name,
        )

        # 三个核心笛卡尔目标：接近点、真正抓取点、抬升点。wiggle 目标在抬升点附近
        # 沿配置轴线来回偏移，用来验证抓取是否稳定。
        pinch_world = grasp_target_position(self.config, self.rope_config)
        approach_world = pinch_world + np.asarray([0.0, 0.0, self.config.approach_distance], dtype=float)
        lifted_world = grasp_target_position(self.config, self.rope_config, lift_height=self.config.lift_height)
        wiggle_axis = np.asarray(self.config.wiggle_axis, dtype=float)
        wiggle_axis = wiggle_axis / np.linalg.norm(wiggle_axis)
        wiggle_worlds: list[np.ndarray] = []
        for _cycle_index in range(self.config.wiggle_cycles):
            wiggle_worlds.append(lifted_world - wiggle_axis * self.config.wiggle_amplitude)
            wiggle_worlds.append(lifted_world + wiggle_axis * self.config.wiggle_amplitude)

        target_orientation = rpy_xyz_deg_to_quat_wxyz(self.config.target_rpy_deg)
        ik_orientation = target_orientation if self.config.use_orientation else None
        # IK 后端只认识机器人描述里的 frame。这里临时写一个“附加 pinch TCP”的 URDF，
        # 避免改动仓库里的基础 URDF，同时让求解器直接以夹捏中心作为末端。
        with tempfile.TemporaryDirectory(prefix="pinch_ik_tcp_") as temp_dir:
            tcp_urdf = Path(temp_dir) / f"{self.cumotion_urdf_path.stem}_{self.tcp_frame_name}.urdf"
            write_tcp_urdf(self.cumotion_urdf_path, tcp_urdf, tcp)
            context = CuMotionContext(
                CuMotionConfig(
                    xrdf_path=self.cumotion_xrdf_path,
                    urdf_path=tcp_urdf,
                    flange_frame=self.parent_frame,
                    default_tcp_frame=self.tcp_frame_name,
                    cspace_seeds=np.asarray(self.config.ik_seeds, dtype=float),
                    ccd_max_iterations=self.config.ik_max_iterations,
                    bfgs_max_iterations=self.config.ik_bfgs_max_iterations,
                    orientation_weight=self.config.ik_orientation_weight,
                    position_tolerance=self.config.ik_position_tolerance,
                    orientation_tolerance=self.config.ik_orientation_tolerance,
                )
            )
            ik_joint_names = context.joint_names()
            dof_names = list(robot.dof_names)
            dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
            missing_ik_joints = [name for name in ik_joint_names if name not in dof_index_by_name]
            if missing_ik_joints:
                raise ValueError(f"cuMotion joints not found in articulation: {missing_ik_joints}")
            arm_indices = np.asarray([dof_index_by_name[name] for name in ik_joint_names], dtype=int)
            current_cspace = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)[arm_indices]
            solver = context.make_inverse_kinematics(
                tcp_frame_name=self.tcp_frame_name,
            )
            # 第一次 IK 用当前 articulation C-space 热启动，后续阶段用上一阶段解热启动，
            # 保持关节轨迹连续，也减少求解器跳解概率。
            approach = solver.solve(
                IKRequest(
                    target_position=approach_world,
                    target_orientation=ik_orientation,
                    warm_start=current_cspace,
                    position_tolerance=self.config.ik_position_tolerance,
                    orientation_tolerance=self.config.ik_orientation_tolerance,
                )
            )
            grasp = solver.solve(
                IKRequest(
                    target_position=pinch_world,
                    target_orientation=ik_orientation,
                    warm_start=approach.joint_positions,
                    position_tolerance=self.config.ik_position_tolerance,
                    orientation_tolerance=self.config.ik_orientation_tolerance,
                )
            )
            lift = solver.solve(
                IKRequest(
                    target_position=lifted_world,
                    target_orientation=ik_orientation,
                    warm_start=grasp.joint_positions,
                    position_tolerance=self.config.ik_position_tolerance,
                    orientation_tolerance=self.config.ik_orientation_tolerance,
                )
            )
            wiggles = []
            warm = lift.joint_positions
            for target in wiggle_worlds:
                result = solver.solve(
                    IKRequest(
                        target_position=target,
                        target_orientation=ik_orientation,
                        warm_start=warm,
                        position_tolerance=self.config.ik_position_tolerance,
                        orientation_tolerance=self.config.ik_orientation_tolerance,
                    )
                )
                wiggles.append((target, result))
                warm = result.joint_positions

        # 把 cuMotion IK 解写回完整 articulation 目标。手部关节用稀疏映射覆盖，其它 DOF
        # 沿用上一阶段目标，保证未参与阶段切换的关节不被意外归零。
        initial_all = np.asarray(robot.get_joint_positions(), dtype=float)
        pre_pinch_all = target_vector_from_mapping(dof_names, self.config.pre_targets, base=initial_all)
        approach_all = pre_pinch_all.copy()
        set_joint_targets_by_indices(approach_all, arm_indices, approach.joint_positions)
        grasp_open_all = pre_pinch_all.copy()
        set_joint_targets_by_indices(grasp_open_all, arm_indices, grasp.joint_positions)
        grasp_closed_all = target_vector_from_mapping(dof_names, self.config.closed_targets, base=grasp_open_all)
        lifted_all = grasp_closed_all.copy()
        set_joint_targets_by_indices(lifted_all, arm_indices, lift.joint_positions)

        wiggle_all_targets = []
        for _world, result in wiggles:
            wiggle_all = grasp_closed_all.copy()
            set_joint_targets_by_indices(wiggle_all, arm_indices, result.joint_positions)
            wiggle_all_targets.append(wiggle_all)

        post_joint_sweep_targets = []
        for joint_1_target in self.config.post_joint_sweep_targets:
            sweep_all = lifted_all.copy()
            sweep_all[arm_indices[0]] = float(joint_1_target)
            post_joint_sweep_targets.append(sweep_all)

        return {
            "arm_indices": arm_indices,
            "initial_all": initial_all,
            "pre_pinch_all": pre_pinch_all,
            "approach_all": approach_all,
            "grasp_open_all": grasp_open_all,
            "grasp_closed_all": grasp_closed_all,
            "lifted_all": lifted_all,
            "wiggle_all_targets": wiggle_all_targets,
            "post_joint_sweep_targets": post_joint_sweep_targets,
            "ik": {
                "pinch_world": pinch_world,
                "approach_world": approach_world,
                "lifted_world": lifted_world,
                "tcp_xyz": tcp.xyz,
                "approach_success": approach.success,
                "approach_error": approach.position_error,
                "grasp_success": grasp.success,
                "grasp_error": grasp.position_error,
                "lift_success": lift.success,
                "lift_error": lift.position_error,
                "wiggles": [(world_target, result.success, result.position_error) for world_target, result in wiggles],
            },
        }

    def run(
        self,
        *,
        robot,
        world,
        articulation_action_type,
        driven_indices: np.ndarray,
        simulation_app,
        render: bool,
        drive_logger=None,
        follower_mapper: MimicFollowerTargetMapper | None = None,
    ) -> dict[str, object]:
        """规划并执行完整夹捏抓取脚本。

        参数:
            robot: Isaac articulation。
            world: Isaac world。
            articulation_action_type: Isaac action 类型构造器。
            driven_indices: 控制器/驱动实际控制的 DOF 索引。
            simulation_app: Isaac app，用于检测仿真窗口是否仍运行。
            render: 是否渲染每个仿真步。
            drive_logger: 可选关节跟踪日志器。
            follower_mapper: 可选 mimic follower 映射器。
        返回:
            ``plan`` 字典，额外写入 ``steps`` 表示实际执行的 physics step 数。
        """

        plan = self.plan(robot)
        step = 0
        step = move_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            start_all=plan["initial_all"],
            target_all=plan["pre_pinch_all"],
            duration=self.config.prep_duration,
            phase="pre_pinch",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        step = move_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            start_all=plan["pre_pinch_all"],
            target_all=plan["approach_all"],
            duration=self.config.move_duration,
            phase="move_to_approach",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        step = move_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            start_all=plan["approach_all"],
            target_all=plan["grasp_open_all"],
            duration=self.config.approach_duration,
            phase="approach_box",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        step = move_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            start_all=plan["grasp_open_all"],
            target_all=plan["grasp_closed_all"],
            duration=self.config.close_duration,
            phase="close_fingers",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        step = move_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            start_all=plan["grasp_closed_all"],
            target_all=plan["lifted_all"],
            duration=self.config.lift_duration,
            phase="lift",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        previous_target = plan["lifted_all"]
        for index, wiggle_all in enumerate(plan["wiggle_all_targets"], start=1):
            step = move_joint_target(
                robot=robot,
                world=world,
                articulation_action_type=articulation_action_type,
                driven_indices=driven_indices,
                start_all=previous_target,
                target_all=wiggle_all,
                duration=self.config.wiggle_duration,
                phase=f"wiggle_{index}",
                simulation_app=simulation_app,
                render=render,
                step=step,
                drive_logger=drive_logger,
                follower_mapper=follower_mapper,
            )
            previous_target = wiggle_all
        if plan["wiggle_all_targets"]:
            step = move_joint_target(
                robot=robot,
                world=world,
                articulation_action_type=articulation_action_type,
                driven_indices=driven_indices,
                start_all=previous_target,
                target_all=plan["lifted_all"],
                duration=self.config.wiggle_duration,
                phase="wiggle_return_center",
                simulation_app=simulation_app,
                render=render,
                step=step,
                drive_logger=drive_logger,
                follower_mapper=follower_mapper,
            )
        step = hold_joint_target(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            driven_indices=driven_indices,
            target_all=plan["lifted_all"],
            duration=self.config.final_hold_duration,
            phase="final",
            simulation_app=simulation_app,
            render=render,
            step=step,
            drive_logger=drive_logger,
            follower_mapper=follower_mapper,
        )
        previous_target = plan["lifted_all"]
        for index, sweep_all in enumerate(plan["post_joint_sweep_targets"], start=1):
            step = move_joint_target(
                robot=robot,
                world=world,
                articulation_action_type=articulation_action_type,
                driven_indices=driven_indices,
                start_all=previous_target,
                target_all=sweep_all,
                duration=self.config.post_joint_sweep_duration,
                phase=f"post_joint_1_sweep_{index}",
                simulation_app=simulation_app,
                render=render,
                step=step,
                drive_logger=drive_logger,
                follower_mapper=follower_mapper,
            )
            previous_target = sweep_all
        plan["steps"] = step
        return plan

    pass
