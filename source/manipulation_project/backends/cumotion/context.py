"""cuMotion 机器人模型和共享资源上下文。

``CuMotionContext`` 是后端对象的共享缓存：它加载 XRDF/URDF、持有 kinematics 对象，并为
FK/IK 封装提供统一的关节名和 frame 名查询。cuMotion 导入被放到构造函数中，保证没有安装
该库时仍可运行配置解析和非后端相关测试。

职责边界:
    * 配置类只把 YAML/Mapping 中的路径、frame 名和求解器参数规范化。
    * Context 只负责加载机器人描述和创建 FK/IK 包装器，不直接执行任务流程。
    * 具体求解误差、碰撞世界和轨迹适配由同包其它模块负责。

坐标/顺序约定:
    cuMotion 的关节顺序来自 XRDF/URDF 解析后的 C-space，而不一定等于 Isaac articulation
    的完整 DOF 顺序。调用方必须通过 ``joint_names`` 做名称对齐后再把结果写回控制器。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manipulation_project.utils.paths import repo_path


@dataclass(frozen=True)
class CuMotionConfig:
    """cuMotion 后端配置。

    路径字段指向 cuMotion 使用的 XRDF/URDF；frame 字段必须与 URDF link 名一致。
    容差单位为米和弧度，seed 数组按 cuMotion C-space 关节顺序排列。
    """

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

        # robot YAML 可能把后端字段直接放在顶层，也可能放在 ``cumotion`` 子节点下。
        # 这里统一取出后再校验必填项，让脚本和测试可以复用同一个构造入口。
        settings = data.get("cumotion", data)
        if not isinstance(settings, Mapping):
            raise ValueError("cuMotion config must be a mapping")
        missing = [
            key
            for key in ("xrdf_path", "urdf_path", "flange_frame")
            if not settings.get(key)
        ]
        if missing:
            raise ValueError(f"cuMotion config is missing required key(s): {missing}")

        # 路径立即按仓库根目录解析，避免后续后端对象依赖调用脚本当前工作目录。
        # seed 和容差保持数值形式，不在这里检查与机器人自由度是否匹配；该信息只有
        # cuMotion 加载模型后才能可靠获得。
        config = cls(
            xrdf_path=repo_path(settings["xrdf_path"]),
            urdf_path=repo_path(settings["urdf_path"]),
            flange_frame=str(settings["flange_frame"]),
            default_tcp_frame=str(
                settings.get("default_tcp_frame") or settings["flange_frame"]
            ),
            cspace_seeds=_optional_seeds(settings.get("cspace_seeds")),
            position_tolerance=float(
                settings.get("position_tolerance", cls.position_tolerance)
            ),
            orientation_tolerance=float(
                settings.get("orientation_tolerance", cls.orientation_tolerance)
            ),
            ccd_max_iterations=int(
                settings.get("ccd_max_iterations", cls.ccd_max_iterations)
            ),
            bfgs_max_iterations=int(
                settings.get("bfgs_max_iterations", cls.bfgs_max_iterations)
            ),
            orientation_weight=float(
                settings.get("orientation_weight", cls.orientation_weight)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """检查 cuMotion 后端配置字段。

        该检查只覆盖无需加载第三方库即可判断的边界条件，例如空路径、负容差和非正迭代
        次数。frame 是否真实存在、seed 长度是否匹配 C-space 等模型相关问题交给
        ``CuMotionContext`` 或具体求解器处理。
        """

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
    """解析可选 C-space seed 数组。

    返回值可以是一条 seed（1D）或多条 seed（2D）。这里只做形状和非空校验，具体长度由
    cuMotion 在求解时结合机器人 C-space 检查。
    """

    if value is None:
        return None
    seeds = np.asarray(value, dtype=float)
    if seeds.ndim not in {1, 2} or seeds.size == 0:
        raise ValueError("cspace_seeds must be a non-empty 1D or 2D array")
    return seeds


class CuMotionContext:
    """缓存 cuMotion robot description 和 kinematics。

    实例化该类是进入 cuMotion 后端的边界：如果第三方库缺失，会在这里给出明确错误；
    如果加载成功，后续 FK/IK 对象共享同一份 robot description，避免重复解析资产文件。
    """

    def __init__(self, config: CuMotionConfig) -> None:
        """延迟导入 cuMotion 并加载机器人描述文件。"""

        # cuMotion 是可选且体量较大的依赖。把导入错误包装成项目语义更清晰的提示，
        # 可以让用户区分“后端未安装”和“配置文件写错”。
        try:
            import cumotion
        except ImportError as exc:
            raise ImportError(
                "cuMotion is not installed in this Python environment. Install the NVIDIA "
                "cuMotion package from https://github.com/nvidia-isaac/cumotion/releases."
            ) from exc

        self.cumotion = cumotion
        self.config = config
        # XRDF 提供 cuMotion 的语义配置，URDF 提供几何/运动链描述；二者必须同时加载，
        # 才能得到后续 FK/IK 共享的 kinematics 对象。
        self.robot_description = cumotion.load_robot_from_file(
            str(config.xrdf_path), str(config.urdf_path)
        )
        self.kinematics = self.robot_description.kinematics()

    def joint_names(self) -> list[str]:
        """返回 cuMotion C-space 主动关节名。"""

        return [
            str(self.kinematics.cspace_coord_name(index))
            for index in range(self.kinematics.num_cspace_coords())
        ]

    def frame_names(self) -> list[str]:
        """返回 cuMotion 可查询 frame 名。"""

        return [str(name) for name in self.kinematics.frame_names()]

    def has_frame(self, frame_name: str) -> bool:
        """检查 frame 是否存在。"""

        return str(frame_name) in set(self.frame_names())

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。"""

        from manipulation_project.backends.cumotion.inverse_kinematics import (
            CuMotionInverseKinematics,
        )

        return CuMotionInverseKinematics(
            self, tcp_frame_name=tcp_frame_name or self.config.default_tcp_frame
        )

    def make_forward_kinematics(self):
        """创建正运动学组件。"""

        from manipulation_project.backends.cumotion.forward_kinematics import (
            CuMotionForwardKinematics,
        )

        return CuMotionForwardKinematics(self)
