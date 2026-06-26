"""关节轨迹跟踪日志。

该模块把“目标关节状态”和“实际关节状态”展开成稳定的 CSV 列名，
便于用 ``scripts/plot_joint_tracking_logs.py`` 或其它工具绘制误差曲线。
位置单位为 rad，速度单位为 rad/s。可选 effort 列记录 commanded/measured/applied effort，
量纲由 PhysX 关节类型决定。

logger 不参与控制决策，写文件失败以异常形式暴露给调用方；禁用日志时可以传 ``None``
路径，字段构造和数组校验仍保持一致，便于测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.logging.csv_writer import CsvWriter
from manipulation_project.logging.config import JointLoggingConfig
from manipulation_project.logging.effort_logger import (
    commanded_efforts_from_controller,
    read_joint_efforts,
)


class JointTrackingLogger:
    """记录每个关节的目标值、实际值和误差。

    输入:
        path: CSV 输出路径；为 ``None`` 时禁用写文件。
        joint_names: 记录的关节名顺序，必须和传入数组顺序一致。
        flush_interval_steps: 每隔多少个写入步 flush 一次。
    输出:
        CSV 列名中使用 ``qd/q`` 表示期望/实际位置，``vd/v`` 表示期望/实际速度，
        ``tau_cmd/tau_measured/tau_applied`` 表示三类 effort。
    """

    def __init__(
        self,
        path: str | Path | None,
        joint_names: list[str],
        *,
        flush_interval_steps: int = 1,
        config: JointLoggingConfig | None = None,
    ) -> None:
        """创建关节跟踪 logger。

        参数:
            path: CSV 输出路径或 ``None``。
            joint_names: 需要记录的关节名，定义数组和列的顺序。
            flush_interval_steps: 自动 flush 的仿真步间隔。
            config: 日志列和采样开关；为空时使用默认配置。
        返回:
            无返回值；内部会创建 ``CsvWriter``。
        """

        self.joint_names = joint_names
        self.config = config or JointLoggingConfig()
        fieldnames = ["step", "time_s", "phase", "drive_update"]
        for name in self.joint_names:
            if self.config.log_command_position:
                fieldnames.append(f"qd_{name}_rad")
            if self.config.log_actual_position:
                fieldnames.append(f"q_{name}_rad")
            if self.config.log_command_velocity:
                fieldnames.append(f"vd_{name}_rad_s")
            if self.config.log_actual_velocity:
                fieldnames.append(f"v_{name}_rad_s")
            if self.config.log_command_position and self.config.log_actual_position:
                fieldnames.append(f"pos_err_{name}_rad")
            if self.config.log_command_velocity and self.config.log_actual_velocity:
                fieldnames.append(f"vel_err_{name}_rad_s")
            if self.config.log_command_effort:
                fieldnames.append(f"tau_cmd_{name}")
            if self.config.log_action_effort:
                fieldnames.append(f"tau_action_{name}")
            if self.config.log_measured_effort:
                fieldnames.append(f"tau_measured_{name}")
            if self.config.log_applied_effort:
                fieldnames.append(f"tau_applied_{name}")
        self.writer = CsvWriter(
            path, fieldnames, flush_interval_rows=flush_interval_steps
        )

    def should_write(self, step: int) -> bool:
        """判断当前 step 是否需要写日志。

        只是转发 ``JointLoggingConfig`` 的采样策略，调用方应在返回 ``True`` 时再读取实际状态，
        以免降采样配置失效。
        """

        return self.config.should_write_step(step)

    def collect_efforts(
        self, robot, controller, joint_indices: np.ndarray
    ) -> dict[str, np.ndarray | None]:
        """按日志开关读取 commanded/action/measured/applied effort。

        measured/applied effort 读取可能触发 Isaac/PhysX buffer clone，是日志里相对昂贵的部分。
        因此调用方应通过本方法按需读取，而不是每步无条件读取。
        """

        result: dict[str, np.ndarray | None] = {
            "commanded_effort": None,
            "action_effort": None,
            "measured_effort": None,
            "applied_effort": None,
        }
        if self.config.log_command_effort or self.config.log_action_effort:
            action_effort = commanded_efforts_from_controller(controller, joint_indices)
            if self.config.log_command_effort:
                result["commanded_effort"] = action_effort
            if self.config.log_action_effort:
                result["action_effort"] = action_effort
        if self.config.log_measured_effort or self.config.log_applied_effort:
            sample = read_joint_efforts(
                robot,
                joint_indices,
                measured=self.config.log_measured_effort,
                applied=self.config.log_applied_effort,
            )
            if self.config.log_measured_effort:
                result["measured_effort"] = sample.measured
            if self.config.log_applied_effort:
                result["applied_effort"] = sample.applied
        return result

    def collect_step_values(
        self, robot, controller, targets, joint_indices: np.ndarray
    ) -> dict[str, np.ndarray | None]:
        """按日志开关收集一帧需要写入的关节值。

        actual position/velocity 和 measured/applied effort 都需要访问 Isaac runtime；只有对应列开启且
        当前 step 需要写日志时才调用本方法，可以降低高频仿真里的日志开销。
        """

        indices = np.asarray(joint_indices, dtype=int).reshape(-1)
        values: dict[str, np.ndarray | None] = {
            "desired_position": targets.positions[indices]
            if self.config.log_command_position
            else None,
            "actual_position": None,
            "desired_velocity": targets.velocities[indices]
            if self.config.log_command_velocity
            else None,
            "actual_velocity": None,
        }
        if self.config.log_actual_position:
            values["actual_position"] = np.asarray(
                robot.get_joint_positions(), dtype=float
            ).reshape(-1)[indices]
        if self.config.log_actual_velocity:
            values["actual_velocity"] = np.asarray(
                robot.get_joint_velocities(), dtype=float
            ).reshape(-1)[indices]
        values.update(self.collect_efforts(robot, controller, indices))
        return values

    def write(
        self,
        *,
        step: int,
        time_s: float,
        phase: str,
        drive_update: bool,
        desired_position: np.ndarray | None,
        actual_position: np.ndarray | None,
        desired_velocity: np.ndarray | None,
        actual_velocity: np.ndarray | None,
        commanded_effort: np.ndarray | None = None,
        action_effort: np.ndarray | None = None,
        measured_effort: np.ndarray | None = None,
        applied_effort: np.ndarray | None = None,
    ) -> None:
        """写入一个仿真步的跟踪数据。

        参数:
            step: 全局仿真/控制步号。
            time_s: 当前日志时间，单位 s。
            phase: 当前任务阶段名。
            drive_update: 当前帧是否刷新了驱动目标。
            desired_position: 目标关节位置数组，单位 rad；对应列关闭时可为 ``None``。
            actual_position: 实际关节位置数组，单位 rad；对应列关闭时可为 ``None``。
            desired_velocity: 目标关节速度数组，单位 rad/s；对应列关闭时可为 ``None``。
            actual_velocity: 实际关节速度数组，单位 rad/s；对应列关闭时可为 ``None``。
            commanded_effort: 控制器在 Python 侧下发的 effort；无该概念时可为 ``None``。
            action_effort: 控制器实际下发给 Isaac 的 effort action；无该概念时可为 ``None``。
            measured_effort: PhysX 求解器测得/计算的关节 effort；读取失败时可为 ``None``。
            applied_effort: Isaac runtime 当前 actuation effort；读取失败时可为 ``None``。
        返回:
            无返回值；副作用是写入 CSV 一行。
        """

        desired_position = self._optional_vector(desired_position, "desired_position")
        actual_position = self._optional_vector(actual_position, "actual_position")
        desired_velocity = self._optional_vector(desired_velocity, "desired_velocity")
        actual_velocity = self._optional_vector(actual_velocity, "actual_velocity")
        commanded_effort = self._optional_vector(commanded_effort, "commanded_effort")
        action_effort = self._optional_vector(action_effort, "action_effort")
        measured_effort = self._optional_vector(measured_effort, "measured_effort")
        applied_effort = self._optional_vector(applied_effort, "applied_effort")
        row: dict[str, float | int | str] = {
            "step": int(step),
            "time_s": f"{float(time_s):.9f}",
            "phase": phase,
            "drive_update": str(bool(drive_update)).lower(),
        }
        position_error = desired_position - actual_position
        velocity_error = desired_velocity - actual_velocity
        for index, name in enumerate(self.joint_names):
            if self.config.log_command_position:
                row[f"qd_{name}_rad"] = f"{desired_position[index]:.12g}"
            if self.config.log_actual_position:
                row[f"q_{name}_rad"] = f"{actual_position[index]:.12g}"
            if self.config.log_command_velocity:
                row[f"vd_{name}_rad_s"] = f"{desired_velocity[index]:.12g}"
            if self.config.log_actual_velocity:
                row[f"v_{name}_rad_s"] = f"{actual_velocity[index]:.12g}"
            if self.config.log_command_position and self.config.log_actual_position:
                row[f"pos_err_{name}_rad"] = f"{position_error[index]:.12g}"
            if self.config.log_command_velocity and self.config.log_actual_velocity:
                row[f"vel_err_{name}_rad_s"] = f"{velocity_error[index]:.12g}"
            if self.config.log_command_effort:
                row[f"tau_cmd_{name}"] = f"{commanded_effort[index]:.12g}"
            if self.config.log_action_effort:
                row[f"tau_action_{name}"] = f"{action_effort[index]:.12g}"
            if self.config.log_measured_effort:
                row[f"tau_measured_{name}"] = f"{measured_effort[index]:.12g}"
            if self.config.log_applied_effort:
                row[f"tau_applied_{name}"] = f"{applied_effort[index]:.12g}"
        self.writer.write(row)

    def _optional_vector(self, values: np.ndarray | None, label: str) -> np.ndarray:
        """读取可选日志数组；缺省时用 ``nan`` 占位。

        日志列关闭或采样读取失败时，调用方可以传 ``None``。这样写入函数仍能保持固定内部
        数组长度，同时避免在关闭某类列时强制读取对应 Isaac 状态。
        """

        if values is None:
            return np.full(len(self.joint_names), np.nan, dtype=float)
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != len(self.joint_names):
            raise ValueError(
                f"{label} expected {len(self.joint_names)} values, got {vector.size}"
            )
        return vector

    def close(self) -> None:
        """关闭内部 CSV writer；可重复调用。

        ``CsvWriter`` 会处理禁用日志和已关闭状态，因此任务执行器可以在 ``finally`` 中无条件
        调用本方法。
        """

        self.writer.close()

    def __enter__(self) -> "JointTrackingLogger":
        """进入上下文管理器。

        返回:
            ``self``。
        """

        return self

    def __exit__(self, *_exc_info) -> None:
        """退出上下文时关闭日志文件。

        参数:
            *_exc_info: Python 上下文管理器传入的异常信息。
        返回:
            无返回值。
        """

        self.close()
