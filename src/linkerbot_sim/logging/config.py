"""CSV 日志配置。

日志采样会发生在 physics step 内部，读取状态和写 CSV 都有成本。本模块把“写哪些列”和
“是否读取较重的 PhysX effort 数据”集中成一个 dataclass，供脚本、执行层和日志器复用。

本模块只负责 profile 名规范化、YAML 严格解析及不可变配置覆盖；除显式加载函数外不访问
文件系统，也不创建 logger。实际列定义、状态采样和 CSV 生命周期由日志运行层负责。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import CONFIGS_ROOT, repo_path


LOGGING_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_LOGGING_KEYS = frozenset(
    {
        "enabled",
        "joint_tracking_path",
        "flush_interval_s",
        "interval_steps",
        "log_actual_position",
        "log_actual_velocity",
        "log_command_position",
        "log_command_velocity",
        "log_command_effort",
        "log_action_effort",
        "log_measured_effort",
        "log_applied_effort",
    }
)


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
    不变量与生命周期:
        解析入口保证时间间隔为有限正数、步数为正整数且开关为严格 bool。对象冻结，通常在
        启动加载一次；CLI 覆盖通过 ``replace`` 生成新对象，不修改共享实例。
    """

    enabled: bool = True
    joint_tracking_path: Path | None = Path("logs/joint_tracking/pinch_grasp.csv")
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

        参数:
            step: 当前 physics step 计数；会转换为整数参与取模。
        返回:
            日志启用且 step 落在采样间隔上时为 ``True``。``interval_steps`` 会被钳制到
            至少 1；当日志整体关闭时始终返回 ``False``。
        """

        return self.enabled and int(step) % max(1, int(self.interval_steps)) == 0

    def flush_interval_steps(self, physics_dt: float) -> int:
        """把 flush 时间间隔换算成日志行/physics step 数。

        参数:
            physics_dt: 仿真物理步长，单位 s。
        返回:
            至少为 1 的整数；无效步长会退回每行 flush，优先保证日志落盘。
        副作用:
            无；只做单位换算，不刷新实际文件。
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
    """读取严格正有限数值并转换为 ``float``。"""

    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"logging.{key} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"logging.{key} must be finite and positive")
    return parsed


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    """读取正整数配置，用于采样间隔等不能为 0 的字段。"""

    value = data.get(key, default)
    if not isinstance(value, bool) and isinstance(value, int) and value > 0:
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
        if not str(value).strip():
            raise ValueError(f"logging.{key} must not be empty")
        return Path(value)
    raise ValueError(f"logging.{key} must be a path string or null")


def load_joint_logging_profile(
    name: str,
    *,
    logging_root: str | Path = CONFIGS_ROOT / "logging",
) -> JointLoggingConfig:
    """按名称加载一份日志 profile 并通过严格 schema 边界。

    参数:
        name: 不含目录和扩展名的 profile 名称。
        logging_root: profile 根目录；相对路径按仓库路径规则解析。
    返回:
        冻结的 :class:`JointLoggingConfig`。
    异常:
        ValueError: 名称或 YAML 内容不合法。
        FileNotFoundError: 对应 YAML 文件不存在。
    副作用:
        读取一个 YAML 文件，不创建 CSV。
    """

    profile_name = normalize_logging_profile_name(name)
    path = repo_path(logging_root) / f"{profile_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Logging profile {profile_name!r} was not found: {path}"
        )
    return joint_logging_config_from_mapping(load_yaml(path), source_path=path)


def normalize_logging_profile_name(value: object) -> str:
    """规范化日志 profile 文件 stem，并阻止越出配置根目录。

    参数:
        value: 待校验名称，必须为字符串。
    返回:
        去除首尾空白且匹配 ``[A-Za-z0-9][A-Za-z0-9_-]*`` 的名称。
    异常:
        ValueError: 类型或字符集合不合法。
    """

    if not isinstance(value, str):
        raise ValueError("logging profile name must be a string")
    name = value.strip()
    if LOGGING_PROFILE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"logging profile name must match [A-Za-z0-9][A-Za-z0-9_-]*, got {value!r}"
        )
    return name


def joint_logging_config_from_mapping(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> JointLoggingConfig:
    """从 YAML mapping 构造 ``JointLoggingConfig``。

    参数:
        data: 完整 logging profile；只接受当前显式 schema。
        source_path: 可选源文件路径，用于错误定位。
    返回:
        ``JointLoggingConfig``，所有缺失字段使用 dataclass 默认值。
    异常:
        ValueError: 顶层/字段结构、未知键、数值范围、路径或布尔类型不合法。
    副作用:
        无；不会修改输入 mapping 或创建输出文件。
    """

    source = str(source_path) if source_path is not None else "<mapping>"
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: logging profile must be a mapping")
    canonical = dict(data)
    _reject_unknown_keys(canonical, {"logging"}, label="logging profile")
    logging = canonical.get("logging", {})
    if not isinstance(logging, Mapping):
        raise ValueError("logging config must contain a mapping under key 'logging'")
    _reject_unknown_keys(logging, _LOGGING_KEYS, label="logging")
    default = JointLoggingConfig()
    return JointLoggingConfig(
        enabled=_bool(logging, "enabled", default.enabled),
        joint_tracking_path=_path(
            logging, "joint_tracking_path", default.joint_tracking_path
        ),
        flush_interval_s=_float(logging, "flush_interval_s", default.flush_interval_s),
        interval_steps=_int(logging, "interval_steps", default.interval_steps),
        log_actual_position=_bool(
            logging, "log_actual_position", default.log_actual_position
        ),
        log_actual_velocity=_bool(
            logging, "log_actual_velocity", default.log_actual_velocity
        ),
        log_command_position=_bool(
            logging, "log_command_position", default.log_command_position
        ),
        log_command_velocity=_bool(
            logging, "log_command_velocity", default.log_command_velocity
        ),
        log_command_effort=_bool(
            logging, "log_command_effort", default.log_command_effort
        ),
        log_action_effort=_bool(
            logging, "log_action_effort", default.log_action_effort
        ),
        log_measured_effort=_bool(
            logging, "log_measured_effort", default.log_measured_effort
        ),
        log_applied_effort=_bool(
            logging, "log_applied_effort", default.log_applied_effort
        ),
    )


def _reject_unknown_keys(
    data: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """拒绝未知 mapping 键，并在错误中报告完整点路径。"""

    unsupported = sorted(str(key) for key in data if key not in allowed)
    if unsupported:
        keys = ", ".join(unsupported)
        paths = ", ".join(f"{label}.{key}" for key in unsupported)
        raise ValueError(
            f"{label} contains unsupported keys: {keys} (full paths: {paths})"
        )


def override_logging_config(
    config: JointLoggingConfig, **updates: Any
) -> JointLoggingConfig:
    """用命令行参数覆盖日志配置。

    只应用值不为 ``None`` 的更新，便于 CLI 把“参数未传”和“显式关闭/设 0”区分开。
    返回新的不可变配置对象，不修改传入实例。

    参数:
        config: 已解析的基础日志配置。
        **updates: 字段名到覆盖值；``None`` 项会被忽略。
    返回:
        由 ``dataclasses.replace`` 创建的新配置。
    异常:
        TypeError: ``updates`` 包含不存在的 dataclass 字段。
    副作用:
        无；不会创建或操作日志文件。
    """

    return replace(
        config, **{key: value for key, value in updates.items() if value is not None}
    )
