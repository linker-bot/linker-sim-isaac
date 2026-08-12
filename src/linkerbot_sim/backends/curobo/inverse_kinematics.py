"""cuRobo 逆运动学封装。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.curobo.collision_capability import (
    collision_capability_message,
    context_supports_collision_queries,
)
from linkerbot_sim.backends.curobo.tensor_adapter import (
    seed_config_from_state_or_seed,
)
from linkerbot_sim.backends.curobo.tool_pose import (
    goal_tool_pose_from_single_tcp_target,
    update_active_tool_pose_criteria,
)
from linkerbot_sim.planning.requests import IKRequest
from linkerbot_sim.planning.results import IKResult
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class CuroboInverseKinematics:
    """使用 cuRobo ``InverseKinematics.solve_pose`` 求解单个 TCP 目标。"""

    def __init__(self, context, *, tcp_frame_name: str | None = None) -> None:
        """保存 context 和默认 TCP frame。"""

        self.context = context
        self.ik_solver = context.ik_solver
        self.tcp_frame_name = str(tcp_frame_name or context.default_tcp_frame)
        self.expected_cspace_width = len(context.joint_names())

    def joint_names(self) -> list[str]:
        """返回 IK 输入 seed 和输出解使用的 C-space 关节名顺序。"""

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回当前 context 注册的 tool frames。"""

        return self.context.frame_names()

    def solve(self, request: IKRequest) -> IKResult:
        """求解单个 IK 请求。"""

        request.validate_structure()
        if request.avoid_collisions and not context_supports_collision_queries(
            self.context,
            consumer="ik",
        ):
            return IKResult(
                joint_positions=np.asarray([], dtype=float),
                success=False,
                position_error=float("inf"),
                orientation_error=(
                    None if request.target_orientation is None else float("inf")
                ),
                message=(
                    "cuRobo collision-aware IK cannot satisfy "
                    "avoid_collisions=True: "
                    + collision_capability_message(self.context, consumer="ik")
                ),
                status="COLLISION_UNSUPPORTED",
                num_solutions=0,
            )
        frame_name = str(request.tcp_frame_name or self.tcp_frame_name)
        self._validate_request_model_match(request, frame_name)
        seed = _seed_matrix(
            request.warm_start_ik_cspace_seed,
            expected_width=self.expected_cspace_width,
        )
        goal = goal_tool_pose_from_single_tcp_target(
            self.context,
            tcp_frame_name=frame_name,
            target_position=request.target_position,
            target_orientation=request.target_orientation,
            seed=seed,
        )
        update_active_tool_pose_criteria(
            self.context,
            self.ik_solver,
            active_tool_frame=frame_name,
            orientation_free=request.target_orientation is None,
        )
        current_state = (
            None if seed is None else self.context.joint_state_from_positions(seed)
        )
        result = self.ik_solver.solve_pose(
            goal,
            current_state=current_state,
            seed_config=seed_config_from_state_or_seed(current_state, seed),
        )
        success = _result_success(result)
        solution = _result_solution(
            result,
            fallback=np.asarray(
                [] if seed is None else seed.reshape(-1),
                dtype=float,
            ),
        )
        return IKResult(
            joint_positions=solution,
            success=success,
            position_error=_result_error(
                result,
                names=("position_error", "position_errors"),
                fallback=0.0 if success else float("inf"),
            ),
            orientation_error=(
                None
                if request.target_orientation is None
                else _result_error(
                    result,
                    names=("rotation_error", "orientation_error", "orientation_errors"),
                    fallback=0.0 if success else float("inf"),
                )
            ),
            status="SUCCESS" if success else str(getattr(result, "status", "FAILED")),
            num_solutions=1,
        )

    def _validate_request_model_match(
        self, request: IKRequest, frame_name: str
    ) -> None:
        """检查与当前 cuRobo context 相关的请求字段。"""

        if (
            request.position_tolerance is not None
            or request.orientation_tolerance is not None
        ):
            raise ValueError(
                "cuRobo IK does not support per-request tolerance overrides; "
                "configure curobo.kinematics.ik position_tolerance and "
                "orientation_tolerance instead"
            )
        if frame_name not in set(self.context.frame_names()):
            raise ValueError(f"cuRobo frame {frame_name!r} not found")
        if request.warm_start_ik_cspace_seed is not None:
            size = (
                np.asarray(request.warm_start_ik_cspace_seed, dtype=float)
                .reshape(-1)
                .size
            )
            if size != self.expected_cspace_width:
                raise ValueError(
                    "warm_start_ik_cspace_seed expected "
                    f"{self.expected_cspace_width} values, got {size}"
                )


def _seed_matrix(seed, *, expected_width: int) -> np.ndarray | None:
    """把 warm-start seed 规范化为 ``(1, C)`` 矩阵。"""

    if seed is None:
        return None
    array = np.asarray(seed, dtype=float).reshape(1, -1)
    if array.shape[1] != int(expected_width):
        raise ValueError(
            f"warm_start_ik_cspace_seed expected {expected_width} values, got {array.shape[1]}"
        )
    return array


def _result_success(result) -> bool:
    """读取 cuRobo IK result 的 success。"""

    value = getattr(result, "success", False)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _result_solution(result, *, fallback: np.ndarray) -> np.ndarray:
    """读取 cuRobo IK result 的第一条解。"""

    for name in ("solution", "joint_positions", "position"):
        value = getattr(result, name, None)
        if value is None:
            continue
        array = tensor_like_to_numpy(value, dtype=float)
        if array.ndim == 3:
            array = array[:, 0, :]
        if array.ndim == 2:
            return np.asarray(array[0], dtype=float).reshape(-1)
        if array.ndim == 1:
            return np.asarray(array, dtype=float).reshape(-1)
    return fallback.copy()


def _result_error(result, *, names: tuple[str, ...], fallback: float) -> float:
    """读取 cuRobo IK result 中的误差标量。"""

    for name in names:
        value = getattr(result, name, None)
        if value is None:
            continue
        error = tensor_like_to_numpy(value, dtype=float).reshape(-1)
        if error.size:
            return float(np.min(error))
    return float(fallback)
