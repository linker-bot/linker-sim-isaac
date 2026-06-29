"""双 Isaac articulation 执行 runtime 数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSideRuntime:
    """单侧机器人执行时需要的对象引用。"""

    side: str
    articulation: object
    joint_controller: object
    drive_logger: object | None = None


@dataclass(frozen=True)
class DualRobotRuntime:
    """左右两台 Isaac articulation 的共享执行 runtime。"""

    left: RobotSideRuntime
    right: RobotSideRuntime
    simulation_world: object
    articulation_action_type: object
    simulation_app: object | None
    render_enabled: bool

    def side(self, side: str) -> RobotSideRuntime:
        """返回指定侧 runtime。"""

        normalized = str(side).lower()
        if normalized == "left":
            return self.left
        if normalized == "right":
            return self.right
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
