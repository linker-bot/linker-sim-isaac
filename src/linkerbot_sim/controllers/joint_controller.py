"""Isaac articulation 关节控制器。

本模块把项目里的“命令关节目标”转换成 Isaac Sim 可以执行的 articulation action。主动关节
可以使用位置、速度或 effort 控制；位置和速度控制又可以选择 Isaac/PhysX 内置的 implicit
drive，或在 Python 侧显式计算 PD effort 后下发。mimic/follower 关节不参与主动命令空间，
无论主动关节使用哪种控制模式，都统一读取 master 实际状态并用 Isaac position drive 跟随。

控制约定:
    * position + implicit: 下发 ``joint_positions`` 和 ``joint_velocities``，由 PhysX drive 算力矩。
    * position + explicit: 读取位置/速度残差，计算 ``kp * q_err + kd * v_err``，再下发 effort。
    * velocity + implicit: 下发 ``joint_velocities``，由 PhysX velocity drive 算力矩。
    * velocity + explicit: 读取速度残差，计算 ``kd * v_err``，再下发 effort。
    * effort + direct: 直接下发调用方给出的 effort。

Isaac 的 ``ArticulationAction`` 支持 position、velocity 和 effort 字段；同一 DOF 上 effort
优先级最高。因此本控制器按 DOF 分组下发 action，避免主动关节的 effort 命令和 follower 的
position drive 目标互相覆盖。关节数组顺序始终以 Isaac articulation 的 ``dof_names`` 为准。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from linkerbot_sim.controllers.types import (
    ControlMethod,
    ControlMode,
    ControlTargets,
    JointControlSettings,
)
from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.robots.joint_groups import resolve_joint_indices
from linkerbot_sim.robots.mimic import (
    MimicFollowerTargetMapper,
    mjcf_equality_follower_joint_names,
    parse_mjcf_joint_frictionloss,
)
from linkerbot_sim.utils.math_utils import expand_scalar_or_vector


class JointController:
    """配置并驱动 Isaac articulation 的主动关节和 mimic follower。

    输入:
        robot: Isaac articulation 对象，需提供 ``dof_names``、``num_dof`` 和 action API。
        joint_names: 上层主动命令空间中的关节名；可以通过配置选择 ``all``。
        settings: runtime 关节控制参数。
        mjcf_path: 可选 MJCF 路径，用于解析 mimic follower 和 frictionloss。
    输出:
        实例暴露 ``command_indices``、``follower_indices``、``driven_indices``，
        并可通过 ``build_control_targets`` / ``apply_targets`` 生成和下发完整 DOF 目标。
    """

    def __init__(
        self,
        robot,
        *,
        joint_names: list[str],
        settings: JointControlSettings,
        mjcf_path: str | Path | None = None,
    ) -> None:
        """初始化控制器索引和 mimic follower 映射。

        参数:
            robot: Isaac articulation 对象。
            joint_names: 主动命令关节名列表；若包含 follower，初始化时会自动剔除。
            settings: 关节控制设置。
            mjcf_path: 可选 MJCF 文件路径。
        返回:
            无返回值；索引、控制模式和 follower 控制关系保存在实例属性中。
        """

        self.robot = robot
        self.dof_names = list(robot.dof_names)
        # 上层配置可以用 ``all`` 或显式关节名描述命令空间。这里先转成 Isaac DOF index，
        # 后续所有目标数组都按 articulation 的 dof_names 顺序写入，避免依赖外部列表顺序。
        requested_indices = resolve_joint_indices(self.dof_names, joint_names)
        # MJCF equality 中的 dependent joint 是 follower。它们需要被驱动，但不能作为独立
        # 命令输入，否则用户给 master 和 follower 同时下目标时会破坏 mimic 关系。
        self.follower_joint_names = mjcf_equality_follower_joint_names(mjcf_path)
        self.follower_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.dof_names)
                if name in self.follower_joint_names
            ],
            dtype=int,
        )
        follower_index_set = {int(index) for index in self.follower_indices}
        # command_indices 只保留主动关节。即使用户的 controlled_joints 包含 follower，
        # 初始化时也会剔除，确保 follower 只由 master 实际状态推导。
        self.command_indices = np.asarray(
            [
                int(index)
                for index in requested_indices
                if int(index) not in follower_index_set
            ],
            dtype=int,
        )
        # driven_indices 用于日志和诊断：它包含控制器实际会下发 action 的全部 DOF，
        # 即主动关节加 follower，但不包含 articulation 中其它自由度。
        self.driven_indices = np.unique(
            np.concatenate([self.command_indices, self.follower_indices])
        ).astype(int)
        self.follower_mapper = MimicFollowerTargetMapper(self.dof_names, mjcf_path)
        self.settings = settings
        self.mjcf_path = mjcf_path

        # 这些数组只描述主动关节的显式控制参数。implicit 模式也会缓存增益，便于同一套
        # 数据结构支持运行时模式切换和测试断言。
        self._active_stiffness = np.zeros(self.robot.num_dof, dtype=float)
        self._active_damping = np.zeros(self.robot.num_dof, dtype=float)
        self._active_effort_limits = np.zeros(self.robot.num_dof, dtype=float)
        self._active_specs: dict[int, tuple[ControlMode, ControlMethod]] = {}
        self.last_commanded_efforts = np.full(self.robot.num_dof, np.nan, dtype=float)

    def configure_runtime(self) -> None:
        """写入 runtime 控制模式、增益、max effort 和摩擦设置。

        参数:
            无，使用初始化时保存的 robot/settings/mjcf_path。
        返回:
            无返回值；副作用是修改 articulation controller、max efforts 和 friction。
        """

        stiffness = np.zeros(self.robot.num_dof, dtype=float)
        damping = np.zeros(self.robot.num_dof, dtype=float)
        # 每次 configure 都重新构造 runtime 缓存，避免修改 YAML/配置后复用旧控制器对象时
        # 残留上一轮模式、增益或 effort limit。
        self._active_stiffness = np.zeros(self.robot.num_dof, dtype=float)
        self._active_damping = np.zeros(self.robot.num_dof, dtype=float)
        self._active_effort_limits = np.zeros(self.robot.num_dof, dtype=float)
        self._active_specs = {}

        self._assign_active_runtime_parameters(stiffness, damping)
        self._assign_follower_runtime_parameters(stiffness, damping)

        # Isaac controller 接收完整 DOF gain/max_effort 数组。未由本控制器管理的 DOF 保持 0
        # 或已有 runtime 值，避免把控制参数写到当前动作不负责的自由度上。
        controller = self.robot.get_articulation_controller()
        controller.set_gains(kps=stiffness, kds=damping)
        controller.set_max_efforts(self._runtime_max_efforts())
        self._configure_effort_modes(controller)
        self._switch_control_modes(controller)
        self._configure_joint_friction()

    def _assign_active_runtime_parameters(
        self, stiffness: np.ndarray, damping: np.ndarray
    ) -> None:
        """按部件写入主动关节的控制增益和 effort 限幅。"""

        for group in ("arm", "hand", "default"):
            group_indices = np.asarray(
                [
                    index
                    for index in self.command_indices
                    if component_for_name(self.dof_names[int(index)]) == group
                ],
                dtype=int,
            )
            if not group_indices.size:
                continue
            settings = self.settings.component(self.dof_names[int(group_indices[0])])
            kp_values = expand_scalar_or_vector(
                settings.stiffness, len(group_indices), f"{group} active stiffness"
            )
            kd_values = expand_scalar_or_vector(
                settings.damping, len(group_indices), f"{group} active damping"
            )
            # 显式 position/velocity 控制在 Python 侧计算 effort；这些缓存就是每步计算残差时
            # 使用的 kp/kd 和限幅。implicit 模式缓存同样的值，但实际 effort 由 PhysX drive 算。
            self._active_stiffness[group_indices] = kp_values
            self._active_damping[group_indices] = kd_values
            self._active_effort_limits[group_indices] = abs(
                float(settings.active_effort_limit())
            )
            for index in group_indices:
                self._active_specs[int(index)] = (settings.mode, settings.method)

            # 只有 Isaac/PhysX implicit 模式需要把主动关节 gain 写到 runtime drive 上。
            # 显式模式和 direct effort 模式的主动关节 gain 保持 0，避免 runtime drive 与
            # Python 侧 effort action 同时发力。
            if settings.mode == "position" and settings.method == "implicit":
                stiffness[group_indices] = kp_values
                damping[group_indices] = kd_values
            elif settings.mode == "velocity" and settings.method == "implicit":
                damping[group_indices] = kd_values

    def _assign_follower_runtime_parameters(
        self, stiffness: np.ndarray, damping: np.ndarray
    ) -> None:
        """按部件写入 follower 的 Isaac position drive 增益。"""

        for group in ("arm", "hand", "default"):
            group_indices = np.asarray(
                [
                    index
                    for index in self.follower_indices
                    if component_for_name(self.dof_names[int(index)]) == group
                ],
                dtype=int,
            )
            if not group_indices.size:
                continue
            settings = self.settings.component(self.dof_names[int(group_indices[0])])
            # follower 的控制语义固定为 position drive，不随主动控制模式变化。因此即使
            # active_joints 处于 velocity/effort 模式，这里也写入 follower stiffness/damping。
            stiffness[group_indices] = expand_scalar_or_vector(
                settings.follower_stiffness,
                len(group_indices),
                f"{group} follower stiffness",
            )
            damping[group_indices] = expand_scalar_or_vector(
                settings.follower_damping,
                len(group_indices),
                f"{group} follower damping",
            )

    def _runtime_max_efforts(self) -> np.ndarray:
        """构造完整 DOF 的 runtime max effort 数组。"""

        max_efforts = np.zeros(self.robot.num_dof, dtype=float)
        view = getattr(self.robot, "_articulation_view", None)
        # 如果 Isaac runtime 已经有资产导入或其它初始化写入的 max effort，先拷贝一份；
        # 后面只覆盖本控制器负责的 DOF，减少对未控制关节的副作用。
        if view is not None and hasattr(view, "get_max_efforts"):
            runtime_max_efforts = view.get_max_efforts()
            if runtime_max_efforts is not None:
                max_efforts = (
                    np.asarray(runtime_max_efforts, dtype=float)
                    .reshape(-1)[: self.robot.num_dof]
                    .copy()
                )

        # 主动关节使用 active effort limit：implicit drive 作为 max force，显式/direct 模式
        # 作为 action effort 的对称限幅。
        for index in self.command_indices:
            limit = float(self._active_effort_limits[int(index)])
            max_efforts[int(index)] = limit if limit > 0 else 0.0
        # follower 不接受 active effort 命令；这里的 max effort 始终表示 position drive 的
        # 最大输出力/力矩。
        for index in self.follower_indices:
            settings = self.settings.component(self.dof_names[int(index)])
            follower_max_force = settings.follower_max_force
            if follower_max_force is None:
                follower_max_force = settings.max_force
            max_efforts[int(index)] = (
                abs(float(follower_max_force)) if follower_max_force > 0 else 0.0
            )
        return max_efforts

    def _configure_effort_modes(self, controller) -> None:
        """把需要 effort action 的主动关节设置成 force effort mode。"""

        # 只有 Python 侧显式控制和 direct effort 需要 effort action。implicit position/velocity
        # 由 PhysX drive 根据 target 和 runtime gain 自己求力矩。
        effort_indices = np.asarray(
            [
                index
                for index in self.command_indices
                if self._active_specs.get(int(index))
                in {
                    ("position", "explicit"),
                    ("velocity", "explicit"),
                    ("effort", "direct"),
                }
            ],
            dtype=int,
        )
        if effort_indices.size and hasattr(controller, "set_effort_modes"):
            controller.set_effort_modes("force", joint_indices=effort_indices)

    def _switch_control_modes(self, controller) -> None:
        """按 DOF 设置 Isaac runtime 控制模式。"""

        runtime_modes: list[tuple[int, str]] = []
        for index in self.command_indices:
            mode, method = self._active_specs[int(index)]
            # explicit position/velocity 最终都是下发 effort，所以 Isaac runtime mode 需要切到
            # effort；implicit 模式才保持 position 或 velocity。
            isaac_mode = mode if method == "implicit" else "effort"
            runtime_modes.append((int(index), isaac_mode))
        for index in self.follower_indices:
            # follower 永远是 position drive，和主动关节 runtime mode 分开设置。
            runtime_modes.append((int(index), "position"))

        if not runtime_modes:
            return
        if not hasattr(controller, "switch_dof_control_mode"):
            raise RuntimeError(
                "The articulation controller must provide switch_dof_control_mode"
            )
        for index, mode in runtime_modes:
            controller.switch_dof_control_mode(dof_index=int(index), mode=mode)

    def _configure_joint_friction(self) -> None:
        """按关节名写入 runtime friction，资产 frictionloss 优先。"""

        view = getattr(self.robot, "_articulation_view", None)
        if view is None or not hasattr(view, "set_friction_coefficients"):
            return
        # MJCF frictionloss 是资产作者给出的关节级物理参数，优先级高于 YAML 默认摩擦。
        # YAML 仍作为缺省值，保证 URDF-only 或缺少 frictionloss 的资产也有稳定阻尼。
        friction_by_name = parse_mjcf_joint_frictionloss(self.mjcf_path)
        friction = np.asarray(
            [self._joint_friction(name, friction_by_name) for name in self.dof_names],
            dtype=float,
        )
        view.set_friction_coefficients(friction.reshape(1, self.robot.num_dof))

    def _joint_friction(self, name: str, friction_by_name: dict[str, float]) -> float:
        """按关节名读取 runtime friction。"""

        if name in friction_by_name:
            return float(friction_by_name[name])
        settings = self.settings.component(name)
        if (
            name in self.follower_joint_names
            and settings.follower_joint_friction is not None
        ):
            return float(settings.follower_joint_friction)
        return float(settings.joint_friction)

    def build_control_targets(
        self,
        command_positions: np.ndarray | None = None,
        command_velocities: np.ndarray | None = None,
        command_efforts: np.ndarray | None = None,
        *,
        base_positions: np.ndarray | None = None,
    ) -> ControlTargets:
        """构造包含 mimic follower 的完整 DOF 控制目标。

        参数:
            command_positions: 命令空间主动关节位置目标，单位 rad；为空时沿用基准位置。
            command_velocities: 命令空间主动关节速度目标，单位 rad/s；为空时填 0。
            command_efforts: 命令空间主动关节 effort 目标；为空时填 0。
            base_positions: 可选完整 DOF 基准位置；为空时读取机器人当前关节位置。
        返回:
            ``ControlTargets``，三个数组 shape 都为 ``(robot.num_dof,)``。
        """

        if base_positions is None:
            # 没有外部基准时，用当前仿真状态初始化完整目标。这样未命令的 DOF 不会在
            # 第一帧因为默认 0 被突然拉回零位。
            full_positions = (
                np.asarray(self.robot.get_joint_positions(), dtype=float)
                .reshape(-1)
                .copy()
            )
        else:
            # 连续轨迹通常把上一帧目标作为 base_positions 传入，保证未主动命令的 DOF
            # 沿用上一帧目标，而不是每步读取实际状态造成命令抖动。
            full_positions = self._full_vector(base_positions, "base_positions").copy()
        full_velocities = np.zeros(self.robot.num_dof, dtype=float)
        full_efforts = np.zeros(self.robot.num_dof, dtype=float)

        # command_* 数组只覆盖主动命令空间；长度必须等于 command_indices，顺序也由
        # command_indices 对应的 joint_names 决定。
        if command_positions is not None:
            full_positions[self.command_indices] = self._command_vector(
                command_positions, "command_positions"
            )
        if command_velocities is not None:
            full_velocities[self.command_indices] = self._command_vector(
                command_velocities, "command_velocities"
            )
        if command_efforts is not None:
            full_efforts[self.command_indices] = self._command_vector(
                command_efforts, "command_efforts"
            )

        # follower 目标最后覆盖，确保即使调用方传入完整/稀疏目标时误写了 follower，
        # 运行时仍以 master 实际状态推导的 mimic 关系为准。
        self._apply_follower_targets(full_positions, full_velocities)
        return ControlTargets(full_positions, full_velocities, full_efforts)

    def targets_from_full_state(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray | None = None,
        joint_efforts: np.ndarray | None = None,
    ) -> ControlTargets:
        """从完整 DOF 目标构造控制目标，并刷新 follower position drive 目标。

        参数:
            joint_positions: 完整 DOF 位置目标，单位 rad。
            joint_velocities: 可选完整 DOF 速度目标，单位 rad/s。
            joint_efforts: 可选完整 DOF effort 目标。
        返回:
            ``ControlTargets``，可直接交给 ``apply_targets``。
        """

        positions = self._full_vector(joint_positions, "joint_positions").copy()
        # 完整 DOF 入口主要用于执行步骤：上游已经生成了 articulation 顺序的目标。
        # 速度/effort 缺省为 0，随后 follower 位置和速度会按 master 实际状态重算。
        velocities = (
            np.zeros(self.robot.num_dof, dtype=float)
            if joint_velocities is None
            else self._full_vector(joint_velocities, "joint_velocities").copy()
        )
        efforts = (
            np.zeros(self.robot.num_dof, dtype=float)
            if joint_efforts is None
            else self._full_vector(joint_efforts, "joint_efforts").copy()
        )
        self._apply_follower_targets(positions, velocities)
        return ControlTargets(positions, velocities, efforts)

    def _apply_follower_targets(
        self, target_positions: np.ndarray, target_velocities: np.ndarray
    ) -> None:
        """用 master 实际状态覆盖 follower 的 position drive 目标。"""

        # 注意这里特意读取 actual master，而不是使用 target master。主关节还没有跟上命令时，
        # follower 贴着实际姿态走，可以减少从动关节超前造成的接触抖动和 mimic 误差。
        self.follower_mapper.apply_from_actual(
            target_positions,
            target_velocities,
            np.asarray(self.robot.get_joint_positions(), dtype=float).reshape(-1),
            np.asarray(self.robot.get_joint_velocities(), dtype=float).reshape(-1),
        )

    def apply_targets(self, articulation_action_type, targets: ControlTargets) -> None:
        """按当前配置向 articulation 下发主动关节和 follower 目标。

        参数:
            articulation_action_type: Isaac ``ArticulationAction`` 类型构造器。
            targets: 完整 DOF 控制目标。
        返回:
            无返回值；副作用是调用 ``robot.apply_action``。
        """

        actual_positions = np.asarray(
            self.robot.get_joint_positions(), dtype=float
        ).reshape(-1)
        actual_velocities = np.asarray(
            self.robot.get_joint_velocities(), dtype=float
        ).reshape(-1)
        # 每次下发前重置 commanded effort 日志缓存。implicit drive 和 follower position drive
        # 没有 Python 侧 effort command，因此对应 DOF 会保持 nan。
        self.last_commanded_efforts = np.full(self.robot.num_dof, np.nan, dtype=float)
        # 主动关节可能按 arm/hand/default 配成不同控制模式。按 mode/method 分组后下发
        # 多个带 joint_indices 的 action，可以避免在同一个 DOF 上同时写 position 和 effort。
        for mode, method, indices in self._active_groups():
            if mode == "position" and method == "implicit":
                self._apply_position_action(articulation_action_type, targets, indices)
            elif mode == "velocity" and method == "implicit":
                self._apply_velocity_action(articulation_action_type, targets, indices)
            elif mode == "position" and method == "explicit":
                efforts = self._explicit_position_efforts(
                    targets, actual_positions, actual_velocities, indices
                )
                self.last_commanded_efforts[indices] = efforts
                self._apply_effort_action(articulation_action_type, efforts, indices)
            elif mode == "velocity" and method == "explicit":
                efforts = self._explicit_velocity_efforts(
                    targets, actual_velocities, indices
                )
                self.last_commanded_efforts[indices] = efforts
                self._apply_effort_action(articulation_action_type, efforts, indices)
            elif mode == "effort" and method == "direct":
                efforts = self._clip_efforts(
                    targets.efforts[indices], self._active_effort_limits[indices]
                )
                self.last_commanded_efforts[indices] = efforts
                self._apply_effort_action(articulation_action_type, efforts, indices)
            else:
                raise ValueError(f"Unsupported control mode/method: {mode}/{method}")

        # follower 最后单独下发 position action。即使主动关节使用 effort action，follower
        # 也不会被 effort 字段覆盖，始终交给 Isaac position drive 跟随。
        if self.follower_indices.size:
            self._apply_position_action(
                articulation_action_type, targets, self.follower_indices
            )

    def _active_groups(self) -> list[tuple[ControlMode, ControlMethod, np.ndarray]]:
        """把主动关节按控制模式和方法分组。"""

        groups: dict[tuple[ControlMode, ControlMethod], list[int]] = {}
        for index in self.command_indices:
            spec = self._active_specs.get(int(index))
            if spec is None:
                # 如果调用方忘记先 configure_runtime，仍按 settings 推导分组；不过真正的
                # runtime gain/mode 写入仍应由 configure_runtime 完成。
                settings = self.settings.component(self.dof_names[int(index)])
                spec = (settings.mode, settings.method)
            groups.setdefault(spec, []).append(int(index))
        return [
            (mode, method, np.asarray(indices, dtype=int))
            for (mode, method), indices in groups.items()
        ]

    def _apply_position_action(
        self, articulation_action_type, targets: ControlTargets, indices: np.ndarray
    ) -> None:
        """向指定 DOF 下发 Isaac position target。"""

        if not indices.size:
            return
        self.robot.apply_action(
            articulation_action_type(
                joint_positions=targets.positions[indices],
                joint_velocities=targets.velocities[indices],
                joint_indices=indices,
            )
        )

    def _apply_velocity_action(
        self, articulation_action_type, targets: ControlTargets, indices: np.ndarray
    ) -> None:
        """向指定 DOF 下发 Isaac velocity target。"""

        if not indices.size:
            return
        self.robot.apply_action(
            articulation_action_type(
                joint_velocities=targets.velocities[indices],
                joint_indices=indices,
            )
        )

    def _apply_effort_action(
        self, articulation_action_type, efforts: np.ndarray, indices: np.ndarray
    ) -> None:
        """向指定 DOF 下发 Isaac effort target。"""

        if not indices.size:
            return
        self.robot.apply_action(
            articulation_action_type(
                joint_efforts=np.asarray(efforts, dtype=float),
                joint_indices=indices,
            )
        )

    def _explicit_position_efforts(
        self,
        targets: ControlTargets,
        actual_positions: np.ndarray,
        actual_velocities: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """计算显式位置 PD effort。"""

        # 显式位置控制在 Python 侧实现和 Isaac implicit drive 等价的 PD 形式：
        # effort = kp * (q_target - q_actual) + kd * (v_target - v_actual)。
        position_error = targets.positions[indices] - actual_positions[indices]
        velocity_error = targets.velocities[indices] - actual_velocities[indices]
        efforts = (
            self._active_stiffness[indices] * position_error
            + self._active_damping[indices] * velocity_error
        )
        return self._clip_efforts(efforts, self._active_effort_limits[indices])

    def _explicit_velocity_efforts(
        self,
        targets: ControlTargets,
        actual_velocities: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """计算显式速度 D effort。"""

        # 显式速度控制不使用 stiffness，只把速度残差通过 damping 映射成 effort。
        velocity_error = targets.velocities[indices] - actual_velocities[indices]
        efforts = self._active_damping[indices] * velocity_error
        return self._clip_efforts(efforts, self._active_effort_limits[indices])

    def _clip_efforts(self, efforts: np.ndarray, limits: np.ndarray) -> np.ndarray:
        """按关节 effort limit 做对称限幅；limit<=0 时输出 0。"""

        efforts = np.asarray(efforts, dtype=float)
        limits = np.asarray(limits, dtype=float)
        # limit<=0 代表当前关节不允许输出 effort。这里直接置 0，而不是让 np.clip 使用
        # 反向上下界，避免配置错误时产生难读的数值行为。
        return np.where(limits > 0.0, np.clip(efforts, -limits, limits), 0.0)

    def _command_vector(self, values: np.ndarray, label: str) -> np.ndarray:
        """校验命令空间向量长度。"""

        # reshape(-1) 允许调用方传 list、列向量或一维数组；长度校验仍严格按 command_indices。
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != self.command_indices.size:
            raise ValueError(
                f"{label} expected {self.command_indices.size} values, got {vector.size}"
            )
        return vector

    def _full_vector(self, values: np.ndarray, label: str) -> np.ndarray:
        """校验完整 DOF 向量长度。"""

        # 完整 DOF 数组必须严格等于 articulation.num_dof。任何缺列或多列都会导致 action
        # 写到错误关节，因此这里尽早报错。
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != self.robot.num_dof:
            raise ValueError(
                f"{label} expected {self.robot.num_dof} values, got {vector.size}"
            )
        return vector
