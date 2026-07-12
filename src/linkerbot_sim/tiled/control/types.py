"""tiled 同步 command 的纯数据类型与 canonical 值域。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


SUPPORTED_COMMAND_KINDS = frozenset(
    {
        "hold",
        "joint_position_target",
        "joint_delta_pos",
        "ee_pose_target",
        "ee_delta_pos",
        "ee_delta_pose",
        "ee_linear_path",
    }
)
SUPPORTED_INTERPOLATIONS = frozenset({"linear", "smoothstep"})
SUPPORTED_POSE_REFERENCE_FRAMES = frozenset({"env", "base", "world"})
SUPPORTED_ORIENTATION_MODES = frozenset({"free", "current", "target"})


@dataclass(frozen=True)
class TiledCommandAction:
    """作用于 tiled env batch 的固定步长 command action。"""

    kind: str
    values: np.ndarray | None = None
    decimation: int | None = None
    interpolation: str = "smoothstep"
    tcp_frame_name: str | None = None
    pose_reference_frame: str = "env"
    duration_s: float | None = None
    sample_dt_s: float | None = None
    target_position: np.ndarray | None = None
    target_offset: np.ndarray | None = None
    orientation_mode: str = "free"
    target_orientation_wxyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        """在进入 runtime 热路径前校验 action 元数据。"""

        if self.kind not in SUPPORTED_COMMAND_KINDS:
            raise ValueError(f"Unsupported tiled command kind: {self.kind!r}")
        if self.decimation is not None and int(self.decimation) < 1:
            raise ValueError("TiledCommandAction.decimation must be positive")
        if self.duration_s is not None:
            duration_s = float(self.duration_s)
            if not np.isfinite(duration_s) or duration_s <= 0.0:
                raise ValueError("TiledCommandAction.duration_s must be positive")
            if self.kind != "ee_linear_path":
                raise ValueError("duration_s is only supported by ee_linear_path")
            if self.decimation is not None:
                raise ValueError("duration_s and decimation cannot both be set")
        if self.sample_dt_s is not None:
            sample_dt_s = float(self.sample_dt_s)
            if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
                raise ValueError("TiledCommandAction.sample_dt_s must be positive")
            if self.kind != "ee_linear_path":
                raise ValueError("sample_dt_s is only supported by ee_linear_path")
        if self.interpolation not in SUPPORTED_INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation!r}")
        if self.pose_reference_frame not in SUPPORTED_POSE_REFERENCE_FRAMES:
            raise ValueError(
                f"Unsupported pose_reference_frame: {self.pose_reference_frame!r}"
            )
        if self.orientation_mode not in SUPPORTED_ORIENTATION_MODES:
            raise ValueError("orientation_mode must be one of: free, current, target")
        for label, value in (
            ("values", self.values),
            ("target_position", self.target_position),
            ("target_offset", self.target_offset),
            ("target_orientation_wxyz", self.target_orientation_wxyz),
        ):
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"TiledCommandAction.{label} must contain finite values"
                )
        if self.kind == "hold" and self.values is not None:
            values = np.asarray(self.values)
            if values.size:
                raise ValueError("hold action values must be None or empty")
        if self.kind == "ee_linear_path":
            target_count = sum(
                value is not None
                for value in (self.values, self.target_offset, self.target_position)
            )
            if target_count != 1:
                raise ValueError(
                    "ee_linear_path requires exactly one of values, "
                    "target_offset, or target_position"
                )
            if (
                self.orientation_mode == "target"
                and self.target_orientation_wxyz is None
            ):
                raise ValueError(
                    "ee_linear_path orientation_mode='target' requires "
                    "target_orientation_quat_wxyz"
                )
            if (
                self.orientation_mode != "target"
                and self.target_orientation_wxyz is not None
            ):
                raise ValueError(
                    "target_orientation_quat_wxyz cannot be combined with "
                    f"orientation_mode={self.orientation_mode!r}"
                )
        elif any(
            value is not None
            for value in (
                self.target_position,
                self.target_offset,
                self.target_orientation_wxyz,
            )
        ):
            raise ValueError(
                "target_position, target_offset, and target orientation are only "
                "supported by ee_linear_path"
            )


@dataclass(frozen=True)
class TiledCommandTarget:
    """转换完成的 ``(num_envs, command_dim)`` 关节目标及 IK 诊断。"""

    joint_positions: np.ndarray
    info: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_positions, dtype=float)
        if q.ndim != 2:
            raise ValueError("TiledCommandTarget.joint_positions must be 2D")
        if not np.all(np.isfinite(q)):
            raise ValueError(
                "TiledCommandTarget.joint_positions must contain finite values"
            )
        object.__setattr__(self, "joint_positions", q.copy())


@dataclass(frozen=True)
class TiledCommandTrajectory:
    """固定 tick 的 ``(ticks, num_envs, command_dim)`` command 轨迹。"""

    joint_positions: np.ndarray
    info: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_positions, dtype=float)
        if q.ndim != 3 or q.shape[0] < 1:
            raise ValueError(
                "TiledCommandTrajectory.joint_positions must have shape (T,N,C)"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError(
                "TiledCommandTrajectory.joint_positions must contain finite values"
            )
        object.__setattr__(self, "joint_positions", q.copy())


__all__ = [
    "SUPPORTED_COMMAND_KINDS",
    "SUPPORTED_INTERPOLATIONS",
    "SUPPORTED_ORIENTATION_MODES",
    "SUPPORTED_POSE_REFERENCE_FRAMES",
    "TiledCommandAction",
    "TiledCommandTarget",
    "TiledCommandTrajectory",
]
