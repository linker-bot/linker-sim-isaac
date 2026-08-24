"""Mirror timeline、IK、线性运动与完整规划资源 owner。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from linkerbot_sim.mirror.lifecycle import close_result_stopped


MIRROR_V1_MOTION_OPERATIONS = frozenset(
    {
        "motion.plan_timeline",
        "motion.joint_goal",
        "motion.joint_delta",
        "motion.joint_trajectory",
        "motion.plan_cspace_goal",
        "motion.plan_cspace_delta",
        "motion.ik_pose",
        "motion.ik_offset",
        "motion.plan_linear_pose_path",
        "motion.hold",
    }
)
MOTION_OPERATIONS = frozenset(
    {
        "motion.plan_timeline",
        "motion.joint_goal",
        "motion.joint_delta",
        "motion.joint_trajectory",
        "motion.plan_cspace_goal",
        "motion.plan_cspace_delta",
        "motion.ik_pose",
        "motion.ik_offset",
        "motion.plan_linear_pose_path",
        "motion.hold",
        "motion.joint_effort",
    }
)
MIRROR_V3_MOTION_OPERATIONS = frozenset(
    set(MOTION_OPERATIONS) | {"motion.hybrid_force_position"}
)


@dataclass
class MirrorMotionOwner:
    """完整 motion/planning 只由 Mirror 持有，绝不进入 Kaleidoscope。"""

    backend: object
    _closed: bool = field(default=False, init=False, repr=False)

    def execute(
        self,
        operation: str,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: Callable[[], bool],
        protocol: str = "linkerbot.mirror.v1",
    ) -> object:
        if self._closed:
            raise RuntimeError("MirrorMotionOwner is closed")
        allowed = (
            MIRROR_V3_MOTION_OPERATIONS
            if protocol == "linkerbot.mirror.v3"
            else MOTION_OPERATIONS
        )
        if operation not in allowed:
            raise ValueError(f"unsupported Mirror motion operation: {operation!r}")
        callback = getattr(self.backend, "execute", None)
        if not callable(callback):
            raise RuntimeError("motion backend does not implement execute")
        return callback(
            operation,
            dict(arguments),
            request_id=request_id,
            should_cancel=should_cancel,
            protocol=protocol,
        )

    def tare_wrench(
        self,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel: Callable[[], bool],
    ) -> object:
        """Run the dedicated v3 tare transaction on the same owner thread."""

        if self._closed:
            raise RuntimeError("MirrorMotionOwner is closed")
        callback = getattr(self.backend, "tare_wrench", None)
        if not callable(callback):
            raise RuntimeError("motion backend does not implement tare_wrench")
        return callback(
            dict(arguments),
            request_id=request_id,
            should_cancel=should_cancel,
        )

    def close(self) -> object:
        if self._closed:
            return True
        callback = getattr(self.backend, "close", None)
        result = True if not callable(callback) else callback()
        if close_result_stopped(result):
            self._closed = True
        return result


__all__ = [
    "MIRROR_V1_MOTION_OPERATIONS",
    "MIRROR_V3_MOTION_OPERATIONS",
    "MOTION_OPERATIONS",
    "MirrorMotionOwner",
]
