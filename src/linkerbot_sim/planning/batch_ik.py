"""后端无关的数组 batch IK 契约。

这里先定义纯 Python 协议和结果结构，不直接依赖 cuRobo。真正的 cuRobo batch/array
IK 适配可以实现为 ``BatchIKBackend``。这样 control adapter 可以先用 fake solver
测试 shape、失败 fallback 和 info mask。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BatchIKResult:
    """每个 env 一个 IK 目标的求解结果。

    shape 约定:
        joint_positions: ``(N, C)``，C 是 cuRobo/command-space 关节维度。
        success: ``(N,)``，每个 env 是否成功。
        position_error: ``(N,)``，每个 env 的位置误差。
        orientation_error: 可选 ``(N,)``，没有姿态约束时可为 None。
        status: 长度 N 的后端状态字符串；缺省时根据 success 自动填充。
    """

    joint_positions: np.ndarray
    success: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray | None = None
    status: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """把输入规整为 numpy 数组并校验 batch 维度一致。"""

        q = np.asarray(self.joint_positions, dtype=float)
        success = np.asarray(self.success, dtype=bool).reshape(-1)
        position_error = np.asarray(self.position_error, dtype=float).reshape(-1)
        if q.ndim != 2:
            raise ValueError("BatchIKResult.joint_positions must be 2D")
        if success.shape != (q.shape[0],):
            raise ValueError("BatchIKResult.success must have shape (N,)")
        if position_error.shape != (q.shape[0],):
            raise ValueError("BatchIKResult.position_error must have shape (N,)")
        orientation_error = None
        if self.orientation_error is not None:
            orientation_error = np.asarray(self.orientation_error, dtype=float).reshape(
                -1
            )
            if orientation_error.shape != (q.shape[0],):
                raise ValueError("BatchIKResult.orientation_error must have shape (N,)")
        status = tuple(str(item) for item in self.status)
        if status and len(status) != q.shape[0]:
            raise ValueError("BatchIKResult.status must have length N")
        if not status:
            status = tuple("SUCCESS" if ok else "FAILED" for ok in success)

        object.__setattr__(self, "joint_positions", q)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "position_error", position_error)
        object.__setattr__(self, "orientation_error", orientation_error)
        object.__setattr__(self, "status", status)


class BatchIKBackend(Protocol):
    """任意数组 batch IK 后端需要实现的最小协议。"""

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchIKResult:
        """一次求解 N 个 env 的 IK 问题。"""


def apply_ik_failure_fallback(
    result: BatchIKResult, fallback_joint_positions: np.ndarray
) -> np.ndarray:
    """对 IK 失败的 env 保留 fallback 关节目标。

    这个函数是 tiled runtime 的重要安全阀：某些 env 的 IK 失败不应让整个 batch
    抛异常或停止推进。调用方仍可通过 ``result.success`` 把失败信息交给上层策略/日志。
    """

    fallback = np.asarray(fallback_joint_positions, dtype=float)
    if fallback.shape != result.joint_positions.shape:
        raise ValueError("fallback_joint_positions must match IK joint_positions shape")
    return np.where(result.success[:, None], result.joint_positions, fallback)
