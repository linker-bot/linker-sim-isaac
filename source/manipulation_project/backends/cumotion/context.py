"""cuMotion 机器人模型和共享资源上下文。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manipulation_project.utils.paths import repo_path


@dataclass(frozen=True)
class CuMotionConfig:
    """cuMotion 后端配置。"""

    xrdf_path: str | Path
    urdf_path: str | Path
    flange_frame: str
    default_tcp_frame: str | None = None
    cspace_seeds: np.ndarray | None = None
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.75
    ccd_max_iterations: int = 180
    bfgs_max_iterations: int = 80
    orientation_weight: float = 0.25

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CuMotionConfig":
        """从 robot YAML 或 cumotion 子映射构造后端配置。

        配置文件中的路径按仓库根目录解析；求解器容差、迭代次数和可选 seed
        都属于 ``cumotion``，任务配置只负责描述任务意图。
        """

        settings = data.get("cumotion", data)
        if not isinstance(settings, Mapping):
            raise ValueError("cuMotion config must be a mapping")
        missing = [key for key in ("xrdf_path", "urdf_path", "flange_frame") if not settings.get(key)]
        if missing:
            raise ValueError(f"cuMotion config is missing required key(s): {missing}")

        config = cls(
            xrdf_path=repo_path(settings["xrdf_path"]),
            urdf_path=repo_path(settings["urdf_path"]),
            flange_frame=str(settings["flange_frame"]),
            default_tcp_frame=str(settings.get("default_tcp_frame") or settings["flange_frame"]),
            cspace_seeds=_optional_seeds(settings.get("cspace_seeds")),
            position_tolerance=float(settings.get("position_tolerance", cls.position_tolerance)),
            orientation_tolerance=float(settings.get("orientation_tolerance", cls.orientation_tolerance)),
            ccd_max_iterations=int(settings.get("ccd_max_iterations", cls.ccd_max_iterations)),
            bfgs_max_iterations=int(settings.get("bfgs_max_iterations", cls.bfgs_max_iterations)),
            orientation_weight=float(settings.get("orientation_weight", cls.orientation_weight)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """检查 cuMotion 后端配置字段。"""

        if not str(self.xrdf_path):
            raise ValueError("xrdf_path cannot be empty")
        if not str(self.urdf_path):
            raise ValueError("urdf_path cannot be empty")
        if not self.flange_frame:
            raise ValueError("flange_frame cannot be empty")
        if self.default_tcp_frame is not None and not self.default_tcp_frame:
            raise ValueError("default_tcp_frame cannot be empty")
        if self.position_tolerance < 0 or self.orientation_tolerance < 0:
            raise ValueError("cuMotion tolerances cannot be negative")
        if self.ccd_max_iterations <= 0 or self.bfgs_max_iterations <= 0:
            raise ValueError("cuMotion iteration counts must be positive")
        if self.orientation_weight < 0:
            raise ValueError("orientation_weight cannot be negative")
        _optional_seeds(self.cspace_seeds)


def _optional_seeds(value) -> np.ndarray | None:
    """解析可选 C-space seed 数组。"""

    if value is None:
        return None
    seeds = np.asarray(value, dtype=float)
    if seeds.ndim not in {1, 2} or seeds.size == 0:
        raise ValueError("cspace_seeds must be a non-empty 1D or 2D array")
    return seeds


class CuMotionContext:
    """缓存 cuMotion robot description 和 kinematics。"""

    def __init__(self, config: CuMotionConfig) -> None:
        try:
            import cumotion
        except ImportError as exc:
            raise ImportError(
                "cuMotion is not installed in this Python environment. Install the NVIDIA "
                "cuMotion package from https://github.com/nvidia-isaac/cumotion/releases."
            ) from exc

        self.cumotion = cumotion
        self.config = config
        self.robot_description = cumotion.load_robot_from_file(str(config.xrdf_path), str(config.urdf_path))
        self.kinematics = self.robot_description.kinematics()

    def joint_names(self) -> list[str]:
        """返回 cuMotion C-space 主动关节名。"""

        return [str(self.kinematics.cspace_coord_name(index)) for index in range(self.kinematics.num_cspace_coords())]

    def frame_names(self) -> list[str]:
        """返回 cuMotion 可查询 frame 名。"""

        return [str(name) for name in self.kinematics.frame_names()]

    def has_frame(self, frame_name: str) -> bool:
        """检查 frame 是否存在。"""

        return str(frame_name) in set(self.frame_names())

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。"""

        from manipulation_project.backends.cumotion.inverse_kinematics import CuMotionInverseKinematics

        return CuMotionInverseKinematics(self, tcp_frame_name=tcp_frame_name or self.config.default_tcp_frame)

    def make_forward_kinematics(self):
        """创建正运动学组件。"""

        from manipulation_project.backends.cumotion.forward_kinematics import CuMotionForwardKinematics

        return CuMotionForwardKinematics(self)
