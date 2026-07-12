"""cuRobo BatchMotionPlanner 的 command/C-space 映射、padding 与逐行结果。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.backends.curobo.batch.result_adapter import (
    batch_result_row_positions,
    batch_success_matrix,
)
from linkerbot_sim.backends.curobo.batch.types import (
    CuroboBatchJointProblem,
    CuroboBatchJointResult,
)
from linkerbot_sim.backends.curobo.collision_capability import (
    collision_capability_message,
    context_supports_collision_queries,
)
from linkerbot_sim.backends.curobo.joint_mapping import CuroboJointMapping
from linkerbot_sim.trajectories.retiming import trajectory_sample_times


class CuroboBatchJointPlanner:
    """独占一个 context/BatchMotionPlanner 的纯 row-order joint batch planner。"""

    def __init__(
        self,
        context: object,
        *,
        batch_planner: object | None = None,
        planner_joint_names: tuple[str, ...] | None = None,
    ) -> None:
        self.context = context
        self.batch_planner = (
            getattr(context, "batch_motion_planner", None)
            if batch_planner is None
            else batch_planner
        )
        if self.batch_planner is None:
            raise RuntimeError("cuRobo BatchMotionPlanner is required")
        self.planner_joint_names = (
            tuple(str(name) for name in context.joint_names())
            if planner_joint_names is None
            else tuple(str(name) for name in planner_joint_names)
        )

    def plan(self, problem: CuroboBatchJointProblem) -> CuroboBatchJointResult:
        """执行一次固定容量 plan_cspace，并移除所有 padding rows。"""

        rows = problem.current_positions.shape[0]
        if problem.avoid_collisions and not context_supports_collision_queries(
            self.context, consumer="batch_planner"
        ):
            return CuroboBatchJointResult.failed(
                problem,
                status="COLLISION_UNSUPPORTED",
                message=(
                    "cuRobo collision-aware batch planning cannot satisfy "
                    "avoid_collisions=True: "
                    + collision_capability_message(
                        self.context,
                        consumer="batch_planner",
                    )
                ),
            )
        batch_size = int(getattr(self.batch_planner, "batch_size", rows) or rows)
        if rows > batch_size:
            return CuroboBatchJointResult.failed(
                problem,
                status="BATCH_TOO_SMALL",
                message=(
                    f"cuRobo BatchMotionPlanner batch_size={batch_size} cannot plan "
                    f"{rows} rows"
                ),
            )
        mapping = _command_to_curobo_mapping(
            planner_joint_names=self.planner_joint_names,
            command_joint_names=problem.command_joint_names,
        )
        current_cspace = mapping.command_to_cspace(problem.current_positions)
        goal_cspace = mapping.command_to_cspace(problem.goal_positions)
        padded_current = _pad_batch_rows(current_cspace, batch_size)
        padded_goal = _pad_batch_rows(goal_cspace, batch_size)
        current_state = self.context.joint_state_from_positions(padded_current)
        goal_state = self.context.joint_state_from_positions(padded_goal)
        result = self.batch_planner.plan_cspace(goal_state, current_state)
        if result is None:
            return CuroboBatchJointResult.failed(
                problem,
                status="NO_RESULT",
                message="cuRobo BatchMotionPlanner returned None",
            )
        success_matrix = batch_success_matrix(result, rows=batch_size)
        row_success = success_matrix[:rows].any(axis=1)
        if not np.all(row_success):
            failed = np.flatnonzero(~row_success)
            return CuroboBatchJointResult.failed(
                problem,
                status="FAILED",
                message=f"cuRobo BatchMotionPlanner failed rows {failed.tolist()}",
                success=row_success,
            )
        seed_indices = np.argmax(success_matrix, axis=1)
        times = trajectory_sample_times(
            duration_s=problem.duration_s,
            sample_dt_s=problem.sample_dt_s,
            include_start=True,
        )
        base_command = _base_command_positions(
            current=problem.current_positions,
            goal=problem.goal_positions,
            times=times,
            duration_s=problem.duration_s,
        )
        cspace_rows = [
            batch_result_row_positions(
                result,
                row=row,
                seed_index=int(seed_indices[row]),
                joint_names=self.planner_joint_names,
                sample_dt_s=problem.sample_dt_s,
                query_times=times,
                duration_s=problem.duration_s,
                start_position=current_cspace[row],
            )
            for row in range(rows)
        ]
        positions = mapping.cspace_to_command(
            np.stack(cspace_rows, axis=0).reshape(-1, len(self.planner_joint_names)),
            base_command_positions=base_command.reshape(
                -1, len(problem.command_joint_names)
            ),
        ).reshape(rows, times.size, len(problem.command_joint_names))
        return CuroboBatchJointResult(
            success=row_success,
            times=times,
            positions=positions,
            status=tuple("SUCCESS" for _ in range(rows)),
            message="cuRobo BatchMotionPlanner trajectory generated",
        )


def _command_to_curobo_mapping(
    *,
    planner_joint_names: tuple[str, ...],
    command_joint_names: tuple[str, ...],
) -> CuroboJointMapping:
    """构造 command-space 到 cuRobo C-space 的 name-based 映射并补充错误语义。"""

    try:
        return CuroboJointMapping.from_joint_names(
            cspace_joint_names=planner_joint_names,
            command_joint_names=command_joint_names,
        )
    except ValueError as exc:
        raise ValueError(
            "command joint_names do not contain all cuRobo planner joints"
        ) from exc


def _pad_batch_rows(values: np.ndarray, batch_size: int) -> np.ndarray:
    """用最后一个真实 row 填满固定 capture batch。"""

    array = np.asarray(values, dtype=float)
    if array.shape[0] == int(batch_size):
        return array
    if array.shape[0] > int(batch_size):
        raise ValueError("cannot pad array larger than batch_size")
    padding = np.repeat(array[-1:, :], int(batch_size) - array.shape[0], axis=0)
    return np.vstack([array, padding])


def _base_command_positions(
    *,
    current: np.ndarray,
    goal: np.ndarray,
    times: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """为非 cuRobo command 列生成 current-to-goal linear baseline。"""

    alpha = (np.asarray(times, dtype=float) / float(duration_s)).reshape(1, -1, 1)
    return (
        np.asarray(current, dtype=float)[:, None, :]
        + (np.asarray(goal, dtype=float) - np.asarray(current, dtype=float))[:, None, :]
        * alpha
    )


__all__ = ["CuroboBatchJointPlanner"]
