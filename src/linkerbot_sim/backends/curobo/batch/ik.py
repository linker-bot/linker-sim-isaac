"""cuRobo 数组 batch IK 后端。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.backends.curobo.tensor_adapter import (
    seed_config_from_state_or_seed,
    tensor_like_to_numpy,
)
from linkerbot_sim.backends.curobo.joint_mapping import CuroboJointMapping
from linkerbot_sim.backends.curobo.tool_pose import (
    update_active_tool_pose_criteria,
)
from linkerbot_sim.planning.batch_ik import BatchIKResult


class CuroboBatchIKSolver:
    """把 cuRobo ``InverseKinematics.solve_pose`` 封装为 ``BatchIKBackend``。

    该类按 duck typing 调用 context 中的 ``ik_solver``，因此单元测试可以用 fake solver，
    真实环境则由后续 ``CuroboContext`` 注入 cuRobo ``InverseKinematics`` 实例。
    """

    def __init__(
        self,
        context,
        *,
        tcp_frame_name: str | None = None,
        command_joint_names: Sequence[str] | None = None,
    ) -> None:
        """创建 batch IK adapter。"""

        self.context = context
        self.ik_solver = getattr(context, "ik_solver", None)
        if self.ik_solver is None:
            raise RuntimeError("CuroboContext.ik_solver is required")
        self.tcp_frame_name = str(tcp_frame_name or context.default_tcp_frame)
        self.cspace_joint_names = tuple(str(name) for name in context.joint_names())
        self.expected_cspace_width = len(self.cspace_joint_names)
        self.mapping = (
            None
            if command_joint_names is None
            else CuroboJointMapping.from_joint_names(
                cspace_joint_names=self.cspace_joint_names,
                command_joint_names=command_joint_names,
            )
        )

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchIKResult:
        """一次求解 N 个 env 的 IK。"""

        frame_name = str(tcp_frame_name or self.tcp_frame_name)
        if frame_name != self.tcp_frame_name:
            self._validate_frame(frame_name)
        positions = _require_2d_width(target_positions, 3, "target_positions")
        orientations = _optional_2d_width(
            target_orientations_wxyz,
            positions.shape[0],
            4,
            "target_orientations_wxyz",
        )
        seed_array = _require_2d(seeds, "seeds")
        cspace_seeds, base_command = self._prepare_seeds(seed_array)
        if cspace_seeds.shape[0] != positions.shape[0]:
            raise ValueError("target_positions and seeds must have same N")

        result = self._call_solve_pose(
            frame_name=frame_name,
            positions=positions,
            orientations=orientations,
            cspace_seeds=cspace_seeds,
        )
        cspace_positions = _result_positions(result, fallback=cspace_seeds)
        success = _result_success(result, rows=positions.shape[0])
        cspace_positions = np.where(success[:, None], cspace_positions, cspace_seeds)
        position_error = _result_errors(
            result,
            ("position_error", "position_errors"),
            rows=positions.shape[0],
            fallback=float("inf"),
        )
        orientation_error = (
            None
            if orientations is None
            else _result_errors(
                result,
                ("rotation_error", "orientation_error", "orientation_errors"),
                rows=positions.shape[0],
                fallback=float("inf"),
            )
        )
        status = tuple("SUCCESS" if ok else "FAILED" for ok in success)
        if self.mapping is not None:
            joint_positions = self.mapping.cspace_to_command(
                cspace_positions,
                base_command_positions=base_command,
            )
        else:
            joint_positions = cspace_positions
        return BatchIKResult(
            joint_positions=joint_positions,
            success=success,
            position_error=position_error,
            orientation_error=orientation_error,
            status=status,
        )

    def compute_tcp_poses(
        self,
        command_positions: np.ndarray,
        *,
        tcp_frame_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量计算 command-space 对应 TCP 位姿。

        上层 world-frame wrapper 需要用同一个 backend 刷新 TCP world pose。IK 输入可以是
        command-space superset，因此这里复用同一套 joint mapping，
        只把 cuRobo C-space 列交给 context FK。
        """

        frame_name = str(tcp_frame_name or self.tcp_frame_name)
        if frame_name != self.tcp_frame_name:
            self._validate_frame(frame_name)
        positions = _require_2d(command_positions, "command_positions")
        if self.mapping is None:
            if positions.shape[1] != self.expected_cspace_width:
                raise ValueError("command_positions width must match cuRobo C-space")
            cspace_positions = positions
        else:
            cspace_positions = self.mapping.command_to_cspace(positions)
        compute = getattr(self.context, "compute_tcp_poses", None)
        if not callable(compute):
            raise RuntimeError("CuroboContext.compute_tcp_poses is required")
        return compute(cspace_positions, tcp_frame_name=frame_name)

    def _call_solve_pose(
        self,
        *,
        frame_name: str,
        positions: np.ndarray,
        orientations: np.ndarray | None,
        cspace_seeds: np.ndarray,
    ):
        """构造 goal pose 并调用 cuRobo/fake ``solve_pose``。"""

        if hasattr(self.context, "goal_tool_pose_from_arrays"):
            goal = self.context.goal_tool_pose_from_arrays(
                positions=positions,
                orientations_wxyz=orientations,
                tool_frames=(frame_name,),
            )
        else:
            goal = {
                "tool_frames": (frame_name,),
                "positions": positions,
                "orientations_wxyz": orientations,
            }
        update_active_tool_pose_criteria(
            self.context,
            self.ik_solver,
            active_tool_frame=frame_name,
            orientation_free=orientations is None,
            tool_frames=(frame_name,),
        )
        current_state = None
        if hasattr(self.context, "joint_state_from_positions"):
            current_state = self.context.joint_state_from_positions(cspace_seeds)
        seed_config = seed_config_from_state_or_seed(current_state, cspace_seeds)
        if current_state is None:
            return self.ik_solver.solve_pose(goal, seed_config=seed_config)
        return self.ik_solver.solve_pose(
            goal,
            current_state=current_state,
            seed_config=seed_config,
        )

    def _prepare_seeds(self, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把输入 seed 规范化为 cuRobo C-space seed。"""

        if self.mapping is None:
            if seeds.shape[1] != self.expected_cspace_width:
                raise ValueError("seeds width must match cuRobo C-space")
            return seeds, seeds
        if seeds.shape[1] != self.mapping.command_width:
            raise ValueError("seeds width must match command-space")
        return self.mapping.command_to_cspace(seeds), seeds

    def _validate_frame(self, frame_name: str) -> None:
        """检查 frame 是否属于 context 注册的 tool frames。"""

        frame_names = set(str(name) for name in self.context.frame_names())
        if frame_name not in frame_names:
            raise ValueError(f"Unknown cuRobo frame: {frame_name}")


def _result_positions(result, *, fallback: np.ndarray) -> np.ndarray:
    """从 cuRobo/fake result 中读取解。"""

    for name in ("solution", "joint_positions", "position"):
        value = getattr(result, name, None)
        if value is not None:
            array = tensor_like_to_numpy(value)
            if array.ndim == 3:
                array = array[:, 0, :]
            if array.shape == fallback.shape:
                return array
    return fallback.copy()


def _result_success(result, *, rows: int) -> np.ndarray:
    """从 cuRobo/fake result 中读取 success mask。"""

    value = getattr(result, "success", None)
    if value is None:
        return np.zeros(rows, dtype=bool)
    success = np.asarray(tensor_like_to_numpy(value), dtype=bool)
    if success.ndim > 1:
        success = success.any(axis=tuple(range(1, success.ndim)))
    success = success.reshape(-1)
    if success.size != rows:
        raise ValueError("cuRobo IK success mask has wrong length")
    return success


def _result_errors(
    result,
    names: tuple[str, ...],
    *,
    rows: int,
    fallback: float,
) -> np.ndarray:
    """读取 result 中的误差向量。"""

    for name in names:
        value = getattr(result, name, None)
        if value is None:
            continue
        error = np.asarray(tensor_like_to_numpy(value), dtype=float)
        if error.ndim > 1:
            error = error.reshape(error.shape[0], -1).min(axis=1)
        error = error.reshape(-1)
        if error.size == rows:
            return error
    return np.full(rows, fallback, dtype=float)


def _require_2d(values: np.ndarray, label: str) -> np.ndarray:
    """把输入规范化为二维 float 数组。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    return array


def _require_2d_width(values: np.ndarray, width: int, label: str) -> np.ndarray:
    """校验二维数组宽度。"""

    array = _require_2d(values, label)
    if array.shape[1] != int(width):
        raise ValueError(f"{label} must have shape (N, {width})")
    return array


def _optional_2d_width(
    values: np.ndarray | None,
    rows: int,
    width: int,
    label: str,
) -> np.ndarray | None:
    """读取可选二维数组，并支持单行广播。"""

    if values is None:
        return None
    array = _require_2d_width(values, width, label)
    if array.shape[0] == 1 and rows != 1:
        return np.repeat(array, rows, axis=0)
    if array.shape[0] != rows:
        raise ValueError(f"{label} first dimension must be 1 or N")
    return array
