"""PhysX implicit position drive 控制器封装。

控制器负责把“命令关节目标”扩展成完整 articulation target：
主动关节直接使用轨迹目标，mimic/follower 关节根据 MJCF equality 自动补齐。

关节数组顺序始终以 Isaac articulation 的 ``dof_names`` 为准。命令空间可以是完整 DOF
子集，follower 目标在运行时按 MJCF 多项式从主动关节状态推导；所有位置单位为 rad，
速度单位为 rad/s，max_force 的量纲由 PhysX 关节类型决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from manipulation_project.robots.joint_groups import resolve_joint_indices
from manipulation_project.robots.classification import component_for_name
from manipulation_project.robots.mimic import (
    MimicFollowerTargetMapper,
    mjcf_equality_follower_joint_names,
    parse_mjcf_joint_frictionloss,
)
from manipulation_project.utils.math_utils import expand_scalar_or_vector


@dataclass(frozen=True)
class ComponentDriveSettings:
    """单个部件的 runtime drive 增益、最大力和摩擦配置。

    输入字段:
        stiffness/damping: 主动关节 drive 增益；可以是标量元组或与同部件 DOF 等长的元组。
        follower_stiffness/follower_damping: mimic follower 关节 drive 增益。
        max_force: 主动关节最大力/力矩。
        follower_max_force: follower 最大力/力矩；为 ``None`` 时沿用 ``max_force``。
        joint_friction/follower_joint_friction: 默认关节摩擦；MJCF frictionloss 会覆盖同名关节。
    输出:
        作为 ``ImplicitDriveSettings`` 的 arm/hand/default 子配置。
    """

    stiffness: tuple[float, ...] = (1000.0,)
    damping: tuple[float, ...] = (50.0,)
    follower_stiffness: tuple[float, ...] = (50000.0,)
    follower_damping: tuple[float, ...] = (50.0,)
    max_force: float = 100.0
    follower_max_force: float | None = None
    joint_friction: float = 0.5
    follower_joint_friction: float | None = None


@dataclass(frozen=True)
class ImplicitDriveSettings:
    """按部件分组的 implicit position drive 配置。

    输入字段:
        default: 未识别部件时使用的回退参数。
        arm: 机械臂主动/从动关节参数。
        hand: 灵巧手主动/从动关节参数。
    输出:
        传给 ``ImplicitDriveController`` 后用于写入 articulation runtime 参数。
    """

    default: ComponentDriveSettings = field(default_factory=ComponentDriveSettings)
    arm: ComponentDriveSettings | None = None
    hand: ComponentDriveSettings | None = None

    def component(self, name: str) -> ComponentDriveSettings:
        """根据关节名选择部件配置。

        参数:
            name: articulation DOF 名称。
        返回:
            ``ComponentDriveSettings``；未知名称使用 ``default``。
        """

        group = component_for_name(name)
        if group == "arm" and self.arm is not None:
            return self.arm
        if group == "hand" and self.hand is not None:
            return self.hand
        return self.default


class ImplicitDriveController:
    """配置并驱动 Isaac articulation 的 position drive。

    输入:
        robot: Isaac articulation 对象，需提供 ``dof_names``、``num_dof`` 和 action API。
        joint_names: 控制命令空间中的主动关节名；可以通过上层配置选择 ``all``。
        settings: runtime drive 参数。
        mjcf_path: 可选 MJCF 路径，用于解析 mimic follower 和 frictionloss。
    输出:
        实例暴露 ``command_indices``、``follower_indices``、``driven_indices``，
        并可通过 ``build_full_targets`` / ``apply`` 生成和下发完整 DOF 目标。
    """

    def __init__(
        self,
        robot,
        *,
        joint_names: list[str],
        settings: ImplicitDriveSettings,
        mjcf_path: str | Path | None = None,
    ) -> None:
        """初始化控制器索引和 mimic follower 映射。

        参数:
            robot: Isaac articulation 对象。
            joint_names: 主动命令关节名列表。
            settings: drive 设置。
            mjcf_path: 可选 MJCF 文件路径。
        返回:
            无返回值；索引和 follower 控制关系保存在实例属性中。
        """

        self.robot = robot
        self.dof_names = list(robot.dof_names)
        # 命令关节只覆盖上层希望主动控制的子空间；这里先解析成 Isaac DOF index，
        # 之后所有目标构造都以该索引写入完整数组，避免依赖输入顺序碰巧等于 DOF 顺序。
        self.command_indices = resolve_joint_indices(self.dof_names, joint_names)
        # MJCF equality 中声明的 follower 不应作为独立命令输入，但 PhysX drive 仍需要
        # 给它们设置目标和增益，否则 mimic 关节可能滞后或偏离主关节多项式约束。
        self.follower_joint_names = mjcf_equality_follower_joint_names(mjcf_path)
        self.follower_indices = np.asarray(
            [index for index, name in enumerate(self.dof_names) if name in self.follower_joint_names],
            dtype=int,
        )
        self.driven_indices = np.unique(np.concatenate([self.command_indices, self.follower_indices])).astype(int)
        self.follower_mapper = MimicFollowerTargetMapper(self.dof_names, mjcf_path)
        self.settings = settings
        self.mjcf_path = mjcf_path

    def configure_runtime(self) -> None:
        """写入 runtime 增益、max effort、摩擦和重力设置。

        参数:
            无，使用初始化时保存的 robot/settings/mjcf_path。
        返回:
            无返回值；副作用是修改 articulation controller、max efforts、friction 和 gravity。
        """

        # 先构造完整 DOF 的 gain 数组，未驱动关节保持 0，防止意外把控制参数写到
        # 任务未管理的自由度上。
        stiffness = np.zeros(self.robot.num_dof, dtype=float)
        damping = np.zeros(self.robot.num_dof, dtype=float)
        follower_index_set = {int(index) for index in self.follower_indices}
        active_indices = np.asarray([index for index in self.driven_indices if int(index) not in follower_index_set], dtype=int)
        self._assign_component_gains(stiffness, damping, active_indices, follower=False)
        self._assign_component_gains(stiffness, damping, self.follower_indices, follower=True)

        # Isaac articulation controller 一次性接收完整 DOF gain 数组。切换 position
        # 模式后，后续 ``apply`` 只需下发 position/velocity target。
        controller = self.robot.get_articulation_controller()
        controller.switch_control_mode("position")
        controller.set_gains(kps=stiffness, kds=damping)

        # max effort 可能已经由资产或其它初始化流程设置过；能读到 runtime 值时先拷贝，
        # 只覆盖本控制器负责的关节，减少对其它 DOF 的副作用。
        max_efforts = np.zeros(self.robot.num_dof, dtype=float)
        view = getattr(self.robot, "_articulation_view", None)
        if view is not None and hasattr(view, "get_max_efforts"):
            runtime_max_efforts = view.get_max_efforts()
            if runtime_max_efforts is not None:
                max_efforts = np.asarray(runtime_max_efforts, dtype=float).reshape(-1)[: self.robot.num_dof].copy()

        for index in active_indices:
            settings = self.settings.component(self.dof_names[int(index)])
            if settings.max_force > 0:
                max_efforts[int(index)] = abs(float(settings.max_force))
        for index in self.follower_indices:
            settings = self.settings.component(self.dof_names[int(index)])
            follower_max_force = settings.follower_max_force
            if follower_max_force is None:
                follower_max_force = settings.max_force
            if follower_max_force > 0:
                max_efforts[int(index)] = abs(float(follower_max_force))
        controller.set_max_efforts(max_efforts)

        if view is not None and hasattr(view, "set_friction_coefficients"):
            # frictionloss 是资产级物理参数，应优先于通用配置；缺失时再按 arm/hand/default
            # 使用运行时默认摩擦，保证不同来源的资产仍有稳定阻尼。
            friction_by_name = parse_mjcf_joint_frictionloss(self.mjcf_path)
            friction = np.asarray([self._joint_friction(name, friction_by_name) for name in self.dof_names], dtype=float)
            view.set_friction_coefficients(friction.reshape(1, self.robot.num_dof))
        self.robot.disable_gravity()

    def _assign_component_gains(
        self,
        stiffness: np.ndarray,
        damping: np.ndarray,
        indices: np.ndarray,
        *,
        follower: bool,
    ) -> None:
        """按 arm/hand 分组写入 runtime stiffness/damping。"""

        for group in ("arm", "hand", "default"):
            group_indices = np.asarray(
                [index for index in indices if component_for_name(self.dof_names[int(index)]) == group],
                dtype=int,
            )
            if not group_indices.size:
                continue
            settings = self.settings.component(self.dof_names[int(group_indices[0])])
            stiffness_values = settings.follower_stiffness if follower else settings.stiffness
            damping_values = settings.follower_damping if follower else settings.damping
            label = f"{group} follower" if follower else f"{group} active"
            stiffness[group_indices] = expand_scalar_or_vector(stiffness_values, len(group_indices), f"{label} stiffness")
            damping[group_indices] = expand_scalar_or_vector(damping_values, len(group_indices), f"{label} damping")

    def _joint_friction(self, name: str, friction_by_name: dict[str, float]) -> float:
        """按关节名读取 runtime friction，资产 frictionloss 优先。"""

        if name in friction_by_name:
            return float(friction_by_name[name])
        settings = self.settings.component(name)
        if name in self.follower_joint_names and settings.follower_joint_friction is not None:
            return float(settings.follower_joint_friction)
        return float(settings.joint_friction)

    def build_full_targets(
        self,
        command_positions: np.ndarray,
        command_velocities: np.ndarray | None = None,
        *,
        base_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """构造包含 mimic follower 的完整 DOF 位置/速度目标。

        参数:
            command_positions: 命令空间主动关节目标，单位 rad。
            command_velocities: 可选命令空间速度目标，单位 rad/s；为空时填 0。
            base_positions: 可选完整 DOF 基准位置；为空时读取机器人当前关节位置。
        返回:
            ``(full_positions, full_velocities)``，shape 都为 ``(robot.num_dof,)``。
        """

        command_positions = np.asarray(command_positions, dtype=float).reshape(-1)
        if command_positions.size != self.command_indices.size:
            raise ValueError(f"command_positions expected {self.command_indices.size} values, got {command_positions.size}")
        if command_velocities is None:
            command_velocities = np.zeros_like(command_positions)
        else:
            command_velocities = np.asarray(command_velocities, dtype=float).reshape(-1)
            if command_velocities.size != self.command_indices.size:
                raise ValueError(
                    f"command_velocities expected {self.command_indices.size} values, got {command_velocities.size}"
                )

        # 完整目标以当前姿态或上一帧目标为基准，只替换命令关节。这样未控制 DOF 不会在
        # 第一帧被隐式置零，也能让 follower 基于最新主关节目标连续更新。
        if base_positions is None:
            full_positions = np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1).copy()
        else:
            full_positions = np.asarray(base_positions, dtype=float).reshape(-1).copy()
        full_velocities = np.zeros(self.robot.num_dof, dtype=float)
        full_positions[self.command_indices] = command_positions
        full_velocities[self.command_indices] = command_velocities

        # follower 目标来自 MJCF mimic 多项式，但速度需要结合当前实际状态估计，才能在
        # implicit drive 中减少主从关节之间的瞬态误差。
        self.follower_mapper.apply_from_actual(
            full_positions,
            full_velocities,
            np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1),
            np.asarray(self.robot.get_joint_velocities(), dtype=float).reshape(-1),
        )
        return full_positions, full_velocities

    def apply(self, articulation_action_type, joint_positions: np.ndarray, joint_velocities: np.ndarray | None = None) -> None:
        """向 articulation 下发完整 DOF 目标。

        参数:
            articulation_action_type: Isaac action 类型构造器。
            joint_positions: 完整 DOF 位置目标数组，单位 rad。
            joint_velocities: 可选完整 DOF 速度目标数组，单位 rad/s。
        返回:
            无返回值；副作用是调用 ``robot.apply_action``。
        """

        # 这里下发完整 DOF 数组，不传 joint_indices；Isaac 会按 articulation DOF 顺序解释。
        # 调用方必须保证数组已经由 ``build_full_targets`` 或等价流程构造完成。
        self.robot.apply_action(
            articulation_action_type(
                joint_positions=np.asarray(joint_positions, dtype=float),
                joint_velocities=None if joint_velocities is None else np.asarray(joint_velocities, dtype=float),
            )
        )
