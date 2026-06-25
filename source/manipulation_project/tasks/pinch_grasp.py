"""夹捏抓取任务流程。

该任务面向机械臂 + 灵巧手 + rope endpoint box 的 scripted demo：先根据闭合手型从 MJCF
运动链计算 thumb/index 夹捏中心 TCP，再把这个 TCP 写入临时 URDF 供 IK 后端求解，最后把
approach、grasp、lift、wiggle 等阶段合成为完整 articulation DOF 目标并在 Isaac 中执行。

职责边界:
    * 作为高层 demo 编排，可以串联对象配置、TCP 生成、IK、轨迹原语和日志。
    * 不直接创建 ``World`` 或导入机器人资产；这些由脚本入口和 env/assets 层完成。
    * 不在这里实现低层控制器；每个阶段最终委托 ``tasks.primitives`` 下发目标。

数组/坐标约定:
    内部数组大多是完整 articulation DOF 顺序；只有调用 cuMotion 时才切到 C-space 关节顺序，
    返回后再按关节名映射回完整目标。笛卡尔目标按当前示例的 world/base 对齐坐标表达，单位
    为 m；手型关节目标单位为 rad。这样可以同时控制机械臂、主动手指关节和 MJCF mimic
    follower，同时避免把 IK 结果误写到灵巧手 DOF。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tempfile

import numpy as np

from manipulation_project.backends.cumotion.context import CuMotionConfig, CuMotionContext
from manipulation_project.backends.cumotion.tcp_urdf_builder import write_tcp_urdf
from manipulation_project.planning.requests import IKRequest
from manipulation_project.objects.capsule_rope import CapsuleRopeConfig, endpoint_center
from manipulation_project.robots.joint_groups import target_vector_from_mapping
from manipulation_project.robots.mimic import expand_targets_with_mjcf_equalities
from manipulation_project.tasks.move_tcp_line import MoveTcpLineConfig, build_tcp_line_command_trajectory
from manipulation_project.tasks.primitives import (
    ExecutableTask,
    HoldTask,
    MoveFullJointTrajectoryTask,
    MoveJointTargetTask,
    TaskRuntime,
    hold_joint_target,
    move_full_joint_trajectory,
    move_joint_target,
    step_joint_target,
)
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
        approach_line_sample_hz: 从 approach 点下沉到抓取点的 TCP 直线 IK 采样频率。
        prep/move/approach/close/lift/wiggle/final/post...: 各阶段持续时间，单位 s。
        wiggle_axis: 抬升后摆动方向，世界坐标向量。
        cuMotion 后端参数从 robot config 的 ``cumotion`` 段读取。
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
    approach_line_sample_hz: float = 100.0
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
            approach_line_sample_hz=float(grasp.get("approach_line_sample_hz", cls.approach_line_sample_hz)),
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
            无返回值；发现非法端点、负时长或无效手型/TCP 参数时抛出 ``ValueError``。
        """

        if self.endpoint not in {"left", "right"}:
            raise ValueError("endpoint must be left or right")
        nonnegative = {
            "approach_distance": self.approach_distance,
            "lift_height": self.lift_height,
            "approach_line_sample_hz": self.approach_line_sample_hz,
            "prep_duration": self.prep_duration,
            "move_duration": self.move_duration,
            "approach_duration": self.approach_duration,
            "close_duration": self.close_duration,
            "lift_duration": self.lift_duration,
            "wiggle_amplitude": self.wiggle_amplitude,
            "wiggle_duration": self.wiggle_duration,
            "final_hold_duration": self.final_hold_duration,
            "post_joint_sweep_duration": self.post_joint_sweep_duration,
        }
        # 时长和距离允许为 0，用于跳过某些阶段或立即下发目标；负数没有物理意义，
        # 会导致 step 计数和 smoothstep 插值难以解释。
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.wiggle_cycles < 0:
            raise ValueError("wiggle_cycles cannot be negative")
        if self.approach_line_sample_hz <= 0:
            raise ValueError("approach_line_sample_hz must be positive")
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

    # endpoint_center 给出端块几何中心，target_world_offset 用于把 TCP 对准更适合夹捏的点，
    # lift_height 只在 z 方向叠加，保持抓取水平位置不变。
    return (
        np.asarray(endpoint_center(rope_config, config.endpoint), dtype=float)
        + np.asarray(config.target_world_offset, dtype=float)
        + np.asarray([0.0, 0.0, lift_height], dtype=float)
    )


class PinchGraspTask:
    """机械臂+灵巧手对 rope 端点 box 的脚本化夹捏任务。

    输入:
        初始化时传入抓取配置、rope 场景配置、MJCF 路径和 cuMotion 后端配置。
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
        cumotion_config: CuMotionConfig,
        tcp_frame_name: str | None = None,
    ) -> None:
        """保存任务配置和 IK/TCP 资源路径。

        参数:
            config: 夹捏抓取配置。
            rope_config: rope 对象配置，用于定位端点。
            mjcf_path: 组合 MJCF 文件路径，用于计算 pinch TCP 和 mimic 关系。
            cumotion_config: cuMotion 后端配置，通常来自 robot config 的 ``cumotion`` 段。
            tcp_frame_name: 写入临时 URDF 的 TCP frame 名称。
        返回:
            无返回值。
        """

        self.config = config
        self.rope_config = rope_config
        self.mjcf_path = Path(mjcf_path)
        self.cumotion_config = cumotion_config
        self.cumotion_config.validate()
        self.parent_frame = cumotion_config.flange_frame
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
            base_urdf_path = Path(self.cumotion_config.urdf_path)
            tcp_urdf = Path(temp_dir) / f"{base_urdf_path.stem}_{self.tcp_frame_name}.urdf"
            write_tcp_urdf(base_urdf_path, tcp_urdf, tcp)
            context = CuMotionContext(
                replace(
                    self.cumotion_config,
                    urdf_path=tcp_urdf,
                    default_tcp_frame=self.tcp_frame_name,
                )
            )
            ik_joint_names = context.joint_names()
            dof_names = list(robot.dof_names)
            dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
            # cuMotion 模型和 Isaac articulation 可能来自不同资产文件。这里按名称检查能尽早
            # 发现 URDF/MJCF 关节名不一致，而不是在写目标数组时静默错位。
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
                    position_tolerance=context.config.position_tolerance,
                    orientation_tolerance=context.config.orientation_tolerance,
                )
            )
            initial_all = np.asarray(robot.get_joint_positions(), dtype=float)
            pre_pinch_all = target_vector_from_mapping(dof_names, self.config.pre_targets, base=initial_all)
            approach_all = pre_pinch_all.copy()
            set_joint_targets_by_indices(approach_all, arm_indices, approach.joint_positions)
            approach_line_config = MoveTcpLineConfig(
                tcp_frame_name=self.tcp_frame_name,
                start_position=None,
                target_position=tuple(float(value) for value in pinch_world),
                orientation_mode="current",
                duration_s=self.config.approach_duration,
                sample_hz=self.config.approach_line_sample_hz,
                phase="approach_box",
            )
            # approach_all 是接近点的完整姿态；从这里开始构建一条短 TCP 直线下沉轨迹，
            # 比直接 IK 到抓取点再关节插值更接近“沿竖直方向靠近端块”的任务意图。
            grasp_line_trajectory, grasp_line_diagnostics = build_tcp_line_command_trajectory(
                dof_names=dof_names,
                command_indices=np.arange(len(dof_names), dtype=int),
                current_positions=approach_all,
                config=approach_line_config,
                context=context,
            )
            grasp_joint_positions = np.asarray(grasp_line_trajectory.positions[-1], dtype=float)[arm_indices]
            lift = solver.solve(
                IKRequest(
                    target_position=lifted_world,
                    target_orientation=ik_orientation,
                    warm_start=grasp_joint_positions,
                    position_tolerance=context.config.position_tolerance,
                    orientation_tolerance=context.config.orientation_tolerance,
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
                        warm_start=warm,
                        position_tolerance=context.config.position_tolerance,
                        orientation_tolerance=context.config.orientation_tolerance,
                    )
                )
                wiggles.append((target, result))
                warm = result.joint_positions

        # 把 cuMotion IK 解写回完整 articulation 目标。手部关节用稀疏映射覆盖，其它 DOF
        # 沿用上一阶段目标，保证未参与阶段切换的关节不被意外归零。
        grasp_open_all = np.asarray(grasp_line_trajectory.positions[-1], dtype=float).copy()
        grasp_closed_all = target_vector_from_mapping(dof_names, self.config.closed_targets, base=grasp_open_all)
        lifted_all = grasp_closed_all.copy()
        set_joint_targets_by_indices(lifted_all, arm_indices, lift.joint_positions)

        wiggle_all_targets = []
        for _world, result in wiggles:
            wiggle_all = grasp_closed_all.copy()
            set_joint_targets_by_indices(wiggle_all, arm_indices, result.joint_positions)
            wiggle_all_targets.append(wiggle_all)

        # 末尾扫动第 1 个机械臂关节是 scripted demo 的额外扰动，用于观察夹持是否稳固。
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
            "approach_line_trajectory": grasp_line_trajectory,
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
                "grasp_success": True,
                "grasp_error": grasp_line_diagnostics.max_position_error,
                "approach_line_start": grasp_line_diagnostics.start_position,
                "approach_line_target": grasp_line_diagnostics.target_position,
                "approach_line_max_error": grasp_line_diagnostics.max_position_error,
                "lift_success": lift.success,
                "lift_error": lift.position_error,
                "wiggles": [(world_target, result.success, result.position_error) for world_target, result in wiggles],
            },
        }

    def execution_tasks(self, plan: dict[str, object]) -> list[ExecutableTask]:
        """把抓取 plan 拆成可顺序执行的任务原语列表。"""

        # plan 阶段只生成目标数组；这里把它们转换成可执行原语，确保 run 的主循环只需要
        # 顺序调用 ``task.run``，便于之后插入/删除阶段。
        tasks: list[ExecutableTask] = [
            MoveJointTargetTask(
                start_all=plan["initial_all"],
                target_all=plan["pre_pinch_all"],
                duration=self.config.prep_duration,
                phase="pre_pinch",
            ),
            MoveJointTargetTask(
                start_all=plan["pre_pinch_all"],
                target_all=plan["approach_all"],
                duration=self.config.move_duration,
                phase="move_to_approach",
            ),
            MoveFullJointTrajectoryTask(
                trajectory=plan["approach_line_trajectory"],
                phase="approach_box",
            ),
            MoveJointTargetTask(
                start_all=plan["grasp_open_all"],
                target_all=plan["grasp_closed_all"],
                duration=self.config.close_duration,
                phase="close_fingers",
            ),
            MoveJointTargetTask(
                start_all=plan["grasp_closed_all"],
                target_all=plan["lifted_all"],
                duration=self.config.lift_duration,
                phase="lift",
            ),
        ]
        previous_target = plan["lifted_all"]
        for index, wiggle_all in enumerate(plan["wiggle_all_targets"], start=1):
            tasks.append(
                MoveJointTargetTask(
                    start_all=previous_target,
                    target_all=wiggle_all,
                    duration=self.config.wiggle_duration,
                    phase=f"wiggle_{index}",
                )
            )
            previous_target = wiggle_all
        if plan["wiggle_all_targets"]:
            tasks.append(
                MoveJointTargetTask(
                    start_all=previous_target,
                    target_all=plan["lifted_all"],
                    duration=self.config.wiggle_duration,
                    phase="wiggle_return_center",
                )
            )
        tasks.append(
            HoldTask(
                target_all=plan["lifted_all"],
                duration=self.config.final_hold_duration,
                phase="final",
            )
        )
        previous_target = plan["lifted_all"]
        for index, sweep_all in enumerate(plan["post_joint_sweep_targets"], start=1):
            tasks.append(
                MoveJointTargetTask(
                    start_all=previous_target,
                    target_all=sweep_all,
                    duration=self.config.post_joint_sweep_duration,
                    phase=f"post_joint_1_sweep_{index}",
                )
            )
            previous_target = sweep_all
        return tasks

    def run(
        self,
        *,
        robot,
        world,
        articulation_action_type,
        controller,
        simulation_app,
        render: bool,
        drive_logger=None,
    ) -> dict[str, object]:
        """规划并执行完整夹捏抓取脚本。

        参数:
            robot: Isaac articulation。
            world: Isaac world。
            articulation_action_type: Isaac action 类型构造器。
            controller: runtime 关节控制器，负责主动关节 action 和 mimic follower 下发。
            simulation_app: Isaac app，用于检测仿真窗口是否仍运行。
            render: 是否渲染每个仿真步。
            drive_logger: 可选关节跟踪日志器。
        返回:
            ``plan`` 字典，额外写入 ``steps`` 表示实际执行的 physics step 数。
        """

        # 先规划再构造 runtime，确保 IK/目标生成失败时不会推进 world，也不会写入半段日志。
        plan = self.plan(robot)
        runtime = TaskRuntime(
            robot=robot,
            world=world,
            articulation_action_type=articulation_action_type,
            controller=controller,
            simulation_app=simulation_app,
            render=render,
            drive_logger=drive_logger,
        )
        step = 0
        for task in self.execution_tasks(plan):
            step = task.run(runtime, step)
        plan["steps"] = step
        return plan
    # 文件结束：本类只定义抓取任务配置、规划和执行编排，不在导入时产生仿真副作用。
    pass
