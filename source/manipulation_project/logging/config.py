"""CSV 日志配置。

日志采样会发生在 physics step 内部，读取状态和写 CSV 都有成本。本模块把“写哪些列”和
“是否读取较重的 PhysX effort 数据”集中成一个 dataclass，供脚本、执行器和任务原语复用。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointLoggingConfig:
    """关节 CSV 日志开关。

    输入字段:
        enabled: 是否启用 CSV 写入；为 ``False`` 时 logger 使用 ``path=None``。
        joint_tracking_path: 默认关节日志输出路径，仓库相对或绝对路径。
        flush_interval_s: 自动 flush 的仿真时间间隔，单位 s。
        interval_steps: 采样降频；每隔多少个 physics step 写一行。
        log_actual_position/log_actual_velocity: 是否读取实际位置/速度。
        log_command_position/log_command_velocity: 是否记录控制目标位置/速度。
        log_command_effort: 是否记录语义上的 effort command。
        log_action_effort: 是否记录控制器实际下发给 Isaac 的 effort action。
        log_measured_effort/log_applied_effort: 是否读取 PhysX measured/applied effort。
    输出:
        传给 ``JointTrackingLogger`` 后决定 CSV 列和 runtime 读取行为。
    """

    enabled: bool = True
    joint_tracking_path: Path | None = Path("logs/joint_tracking/run_pinch_grasp.csv")
    flush_interval_s: float = 0.05
    interval_steps: int = 1
    log_actual_position: bool = True
    log_actual_velocity: bool = True
    log_command_position: bool = True
    log_command_velocity: bool = True
    log_command_effort: bool = True
    log_action_effort: bool = False
    log_measured_effort: bool = False
    log_applied_effort: bool = False

    def should_write_step(self, step: int) -> bool:
        """判断当前 physics step 是否需要写日志。

        ``interval_steps`` 会被钳制到至少 1；当日志整体关闭时始终返回 ``False``。
        """

        return self.enabled and int(step) % max(1, int(self.interval_steps)) == 0

    def flush_interval_steps(self, physics_dt: float) -> int:
        """把 flush 时间间隔换算成日志行/physics step 数。

        参数:
            physics_dt: 仿真物理步长，单位 s。
        返回:
            至少为 1 的整数；无效步长会退回每行 flush，优先保证日志落盘。
        """

        if physics_dt <= 0:
            return 1
        return max(1, int(round(float(self.flush_interval_s) / float(physics_dt))))


def _bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    """读取布尔配置，并拒绝字符串等隐式真值。"""

    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"logging.{key} must be a boolean")


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    """读取数值配置并转换为 ``float``。"""

    value = data.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"logging.{key} must be a number")


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    """读取正整数配置，用于采样间隔等不能为 0 的字段。"""

    value = data.get(key, default)
    if isinstance(value, int) and value > 0:
        return int(value)
    raise ValueError(f"logging.{key} must be a positive integer")


def _path(data: Mapping[str, Any], key: str, default: Path | None) -> Path | None:
    """读取可选路径配置。

    ``None`` 表示禁用对应文件输出；字符串会延迟到 writer 层再解析父目录和创建文件。
    """

    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    raise ValueError(f"logging.{key} must be a path string or null")


def joint_logging_config_from_mapping(
    data: Mapping[str, Any] | None,
) -> JointLoggingConfig:
    """从 YAML mapping 构造 ``JointLoggingConfig``。

    参数:
        data: 完整任务配置；函数只读取其中的 ``logging`` 子 mapping。
    返回:
        ``JointLoggingConfig``，所有缺失字段使用 dataclass 默认值。
    """

    logging = data.get("logging", {}) if data is not None else {}
    if not isinstance(logging, Mapping):
        raise ValueError("logging config must contain a mapping under key 'logging'")
    default = JointLoggingConfig()
    return JointLoggingConfig(
        enabled=_bool(logging, "enabled", default.enabled),
        joint_tracking_path=_path(
            logging, "joint_tracking_path", default.joint_tracking_path
        ),
        flush_interval_s=_float(logging, "flush_interval_s", default.flush_interval_s),
        interval_steps=_int(logging, "interval_steps", default.interval_steps),
        log_actual_position=_bool(
            logging, "actual_position", default.log_actual_position
        ),
        log_actual_velocity=_bool(
            logging, "actual_velocity", default.log_actual_velocity
        ),
        log_command_position=_bool(
            logging, "command_position", default.log_command_position
        ),
        log_command_velocity=_bool(
            logging, "command_velocity", default.log_command_velocity
        ),
        log_command_effort=_bool(logging, "command_effort", default.log_command_effort),
        log_action_effort=_bool(logging, "action_effort", default.log_action_effort),
        log_measured_effort=_bool(
            logging, "measured_effort", default.log_measured_effort
        ),
        log_applied_effort=_bool(logging, "applied_effort", default.log_applied_effort),
    )


def override_logging_config(
    config: JointLoggingConfig, **updates: Any
) -> JointLoggingConfig:
    """用命令行参数覆盖日志配置。

    只应用值不为 ``None`` 的更新，便于 CLI 把“参数未传”和“显式关闭/设 0”区分开。
    返回新的不可变配置对象，不修改传入实例。
    """

    return replace(
        config, **{key: value for key, value in updates.items() if value is not None}
    )
