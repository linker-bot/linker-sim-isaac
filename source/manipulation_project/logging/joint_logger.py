"""关节轨迹跟踪日志。

该模块把“目标关节状态”和“实际关节状态”展开成稳定的 CSV 列名，
便于用 ``scripts/plot_joint_tracking_logs.py`` 或其它工具绘制误差曲线。
位置单位为 rad，速度单位为 rad/s。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.logging.csv_writer import CsvWriter


class JointTrackingLogger:
    """记录每个关节的目标值、实际值和误差。

    输入:
        path: CSV 输出路径；为 ``None`` 时禁用写文件。
        joint_names: 记录的关节名顺序，必须和传入数组顺序一致。
        flush_interval_steps: 每隔多少个写入步 flush 一次。
    输出:
        CSV 列名中使用 ``qd/q`` 表示期望/实际位置，``vd/v`` 表示期望/实际速度。
    """

    def __init__(self, path: str | Path | None, joint_names: list[str], *, flush_interval_steps: int = 1) -> None:
        """创建关节跟踪 logger。

        参数:
            path: CSV 输出路径或 ``None``。
            joint_names: 需要记录的关节名，定义数组和列的顺序。
            flush_interval_steps: 自动 flush 的仿真步间隔。
        返回:
            无返回值；内部会创建 ``CsvWriter``。
        """

        self.joint_names = joint_names
        fieldnames = ["step", "time_s", "phase", "drive_update"]
        for name in self.joint_names:
            fieldnames.extend(
                [
                    f"qd_{name}_rad",
                    f"q_{name}_rad",
                    f"vd_{name}_rad_s",
                    f"v_{name}_rad_s",
                    f"pos_err_{name}_rad",
                    f"vel_err_{name}_rad_s",
                ]
            )
        self.writer = CsvWriter(path, fieldnames, flush_interval_rows=flush_interval_steps)

    def write(
        self,
        *,
        step: int,
        time_s: float,
        phase: str,
        drive_update: bool,
        desired_position: np.ndarray,
        actual_position: np.ndarray,
        desired_velocity: np.ndarray,
        actual_velocity: np.ndarray,
    ) -> None:
        """写入一个仿真步的跟踪数据。

        参数:
            step: 全局仿真/控制步号。
            time_s: 当前日志时间，单位 s。
            phase: 当前任务阶段名。
            drive_update: 当前帧是否刷新了驱动目标。
            desired_position: 目标关节位置数组，单位 rad。
            actual_position: 实际关节位置数组，单位 rad。
            desired_velocity: 目标关节速度数组，单位 rad/s。
            actual_velocity: 实际关节速度数组，单位 rad/s。
        返回:
            无返回值；副作用是写入 CSV 一行。
        """

        desired_position = np.asarray(desired_position, dtype=float).reshape(-1)
        actual_position = np.asarray(actual_position, dtype=float).reshape(-1)
        desired_velocity = np.asarray(desired_velocity, dtype=float).reshape(-1)
        actual_velocity = np.asarray(actual_velocity, dtype=float).reshape(-1)
        row: dict[str, float | int | str] = {
            "step": int(step),
            "time_s": f"{float(time_s):.9f}",
            "phase": phase,
            "drive_update": str(bool(drive_update)).lower(),
        }
        position_error = desired_position - actual_position
        velocity_error = desired_velocity - actual_velocity
        for index, name in enumerate(self.joint_names):
            row[f"qd_{name}_rad"] = f"{desired_position[index]:.12g}"
            row[f"q_{name}_rad"] = f"{actual_position[index]:.12g}"
            row[f"vd_{name}_rad_s"] = f"{desired_velocity[index]:.12g}"
            row[f"v_{name}_rad_s"] = f"{actual_velocity[index]:.12g}"
            row[f"pos_err_{name}_rad"] = f"{position_error[index]:.12g}"
            row[f"vel_err_{name}_rad_s"] = f"{velocity_error[index]:.12g}"
        self.writer.write(row)

    def close(self) -> None:
        """关闭内部 CSV writer。

        参数:
            无。
        返回:
            无返回值。
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
