"""tiled runtime 的 cuMotion batched IK 适配层。

本模块把 cuMotion 的 ``CollisionFreeIkSolver.solve_array`` 封装成 tiled
step-control 可消费的 ``BatchedIKSolver``。它和现有
``CuMotionInverseKinematics`` 的关键区别是：

* 每个 env 的 seed 都由调用方显式传入，shape 为 ``(N, C)``。
* 不把成功解写入共享 ``IkConfig.cspace_seeds``，避免 env 间 warm-start 泄漏。
* 可选支持 cuMotion C-space 与 controller command-space 的关节名映射。

该 solver 要求运行环境提供 cuMotion batch IK API；tiled 性能路径不再回退到 per-env
``solve_ik`` loop。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.backends.cumotion.context import (
    resolve_tcp_frame_name,
    validate_cumotion_frame,
)
from linkerbot_sim.backends.cumotion.pose_adapter import rotation_from_quat_wxyz
from linkerbot_sim.tiled.batched_ik import BatchedIKResult


@dataclass(frozen=True)
class CuMotionJointMapping:
    """cuMotion C-space 与 controller command-space 的关节名映射。

    ``cspace_joint_names`` 来自 ``CuMotionContext.joint_names()``；``command_joint_names``
    来自 tiled controller 对外暴露的 command-space。若二者相同，可以不创建 mapping。
    如果 command-space 是 superset，本 mapping 会从 command seed 中抽取 C-space seed，
    并把 IK 解写回 command-space 对应列，未参与 IK 的列保留 base command seed。
    """

    cspace_joint_names: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    command_indices_for_cspace: tuple[int, ...]

    @classmethod
    def from_joint_names(
        cls,
        *,
        cspace_joint_names: Sequence[str],
        command_joint_names: Sequence[str],
    ) -> "CuMotionJointMapping":
        """按关节名创建 C-space 到 command-space 的列映射。"""

        cspace_names = tuple(str(name) for name in cspace_joint_names)
        command_names = tuple(str(name) for name in command_joint_names)
        index_by_name = {name: index for index, name in enumerate(command_names)}
        missing = [name for name in cspace_names if name not in index_by_name]
        if missing:
            raise ValueError(
                "command_joint_names missing cuMotion C-space joints: "
                + ", ".join(missing)
            )
        return cls(
            cspace_joint_names=cspace_names,
            command_joint_names=command_names,
            command_indices_for_cspace=tuple(
                index_by_name[name] for name in cspace_names
            ),
        )

    @property
    def cspace_width(self) -> int:
        """返回 cuMotion C-space 维度。"""

        return len(self.cspace_joint_names)

    @property
    def command_width(self) -> int:
        """返回 controller command-space 维度。"""

        return len(self.command_joint_names)

    def command_to_cspace(self, command_positions: np.ndarray) -> np.ndarray:
        """从 command-space 数组抽取 cuMotion C-space 列。"""

        command = _require_2d(command_positions, "command_positions")
        if command.shape[1] != self.command_width:
            raise ValueError(
                "command_positions width must match command_joint_names length"
            )
        return command[:, self.command_indices_for_cspace]

    def cspace_to_command(
        self,
        cspace_positions: np.ndarray,
        *,
        base_command_positions: np.ndarray,
    ) -> np.ndarray:
        """把 C-space IK 解写回 command-space，未参与 IK 的列保留 base command。"""

        cspace = _require_2d(cspace_positions, "cspace_positions")
        base_command = _require_2d(base_command_positions, "base_command_positions")
        if cspace.shape[1] != self.cspace_width:
            raise ValueError("cspace_positions width must match cuMotion C-space")
        if base_command.shape[1] != self.command_width:
            raise ValueError("base_command_positions width must match command-space")
        if cspace.shape[0] != base_command.shape[0]:
            raise ValueError("cspace_positions and base command must have same N")
        command = base_command.copy()
        command[:, self.command_indices_for_cspace] = cspace
        return command


class BatchedCuMotionIKSolver:
    """cuMotion tiled batched IK solver。

    参数:
        context: 已加载机器人模型的 ``CuMotionContext`` 或测试 fake。
        tcp_frame_name: 默认 TCP frame；单次 ``solve`` 仍可传入覆盖值。
        command_joint_names: 可选 command-space 关节顺序。提供后，solver 输入/输出都是
            command-space；内部自动抽取和回填 cuMotion C-space。
        position_tolerance/orientation_tolerance: 可选覆盖；不传则使用 context 配置。
    """

    def __init__(
        self,
        context,
        *,
        tcp_frame_name: str | None = None,
        command_joint_names: Sequence[str] | None = None,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> None:
        """创建 tiled IK solver，并建立 command-space 到 cuMotion C-space 的映射。

        当 ``command_joint_names`` 存在时，外部仍按控制器 command joints 输入/输出；
        内部只把 cuMotion 认识的 C-space 关节切出来求解，再回填到 command-space。
        """

        self.context = context
        self.cumotion = context.cumotion
        self.kinematics = context.kinematics
        self.tcp_frame_name = resolve_tcp_frame_name(
            context,
            tcp_frame_name=tcp_frame_name,
            label="tcp_frame_name",
        )
        self.cspace_joint_names = tuple(str(name) for name in context.joint_names())
        self.expected_cspace_width = int(getattr(context, "expected_cspace_width", 0))
        if self.expected_cspace_width <= 0:
            self.expected_cspace_width = len(self.cspace_joint_names)
        if len(self.cspace_joint_names) != self.expected_cspace_width:
            raise ValueError("cuMotion joint_names width does not match context width")
        self.mapping = (
            None
            if command_joint_names is None
            else CuMotionJointMapping.from_joint_names(
                cspace_joint_names=self.cspace_joint_names,
                command_joint_names=command_joint_names,
            )
        )
        ik_config = context.config.kinematics.ik
        self.position_tolerance = (
            float(ik_config.position_tolerance)
            if position_tolerance is None
            else float(position_tolerance)
        )
        self.orientation_tolerance = (
            float(ik_config.orientation_tolerance)
            if orientation_tolerance is None
            else float(orientation_tolerance)
        )
        self.collision_free_params = dict(
            getattr(ik_config, "collision_free_params", {}) or {}
        )
        self._batch_solvers: dict[str, object] = {}
        self.last_backend = "uninitialized"

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchedIKResult:
        """求解 N 个 env 的 IK。

        ``seeds`` 是每个 env 的显式 warm-start。若构造时提供了 ``command_joint_names``，
        ``seeds`` 使用 command-space 宽度；否则使用 cuMotion C-space 宽度。
        """

        frame_name = str(tcp_frame_name or self.tcp_frame_name)
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

        return self._solve_batch_api(
            frame_name=frame_name,
            positions=positions,
            orientations=orientations,
            cspace_seeds=cspace_seeds,
            base_command=base_command,
        )

    def compute_tcp_poses(
        self,
        command_positions: np.ndarray,
        *,
        tcp_frame_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量计算 command-space 关节位置对应的 TCP 位姿。

        返回值仍在 cuMotion 机器人 base frame 下，shape 分别为 ``(N, 3)`` 和
        ``(N, 4)``，四元数顺序为 wxyz。这里用同一个 command/C-space 关节名映射，保证
        tiled Isaac runtime 的 delta IK warm-start 与 batch IK 求解使用同一组关节列。
        """

        frame_name = str(tcp_frame_name or self.tcp_frame_name)
        self._validate_frame(frame_name)
        command = _require_2d(command_positions, "command_positions")
        cspace_positions, _ = self._prepare_seeds(command)
        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        for row in cspace_positions:
            positions.append(
                np.asarray(
                    self.kinematics.position(row, frame_name), dtype=float
                ).reshape(3)
            )
            orientations.append(
                _rotation_to_quat_wxyz(self.kinematics.orientation(row, frame_name))
            )
        return np.vstack(positions), np.vstack(orientations)

    def _solve_batch_api(
        self,
        *,
        frame_name: str,
        positions: np.ndarray,
        orientations: np.ndarray | None,
        cspace_seeds: np.ndarray,
        base_command: np.ndarray,
    ) -> BatchedIKResult:
        """调用 cuMotion ``solve_array`` 一次性求解整个 tiled batch。"""

        solver = self._batch_solver(frame_name)
        target_array = self._task_space_target_array(
            positions=positions,
            orientations=orientations,
        )
        try:
            results = solver.solve_array(
                target_array,
                [np.asarray(seed, dtype=float).reshape(-1) for seed in cspace_seeds],
            )
        except Exception as exc:
            raise RuntimeError(f"cuMotion batch IK solve_array failed: {exc}") from exc

        success_rows: list[bool] = []
        q_rows: list[np.ndarray] = []
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        status_rows: list[str] = []
        has_orientation_error = orientations is not None
        for env_id in range(positions.shape[0]):
            problem = results.problem(env_id)
            status_name = _status_name(problem.status())
            positions_for_problem = list(problem.cspace_positions())
            success = status_name == "SUCCESS" and bool(positions_for_problem)
            q = _select_closest_cspace_position(
                positions_for_problem,
                seed=cspace_seeds[env_id],
                width=self.expected_cspace_width,
            )
            if q is None:
                q = cspace_seeds[env_id]
                success = False
            success_rows.append(success)
            q_rows.append(q)
            status_rows.append(status_name if status_name else "FAILED")
            position_errors.append(
                self._position_error(q, positions[env_id], frame_name)
                if success
                else float("inf")
            )
            if has_orientation_error:
                orientation_errors.append(
                    self._orientation_error(q, orientations[env_id], frame_name)
                    if success
                    else float("inf")
                )

        self.last_backend = "collision_free_solve_array"
        return self._result_from_cspace_rows(
            cspace_positions=np.vstack(q_rows),
            base_command=base_command,
            success=np.asarray(success_rows, dtype=bool),
            position_error=np.asarray(position_errors, dtype=float),
            orientation_error=(
                np.asarray(orientation_errors, dtype=float)
                if has_orientation_error
                else None
            ),
            status=tuple(status_rows),
        )

    def _result_from_cspace_rows(
        self,
        *,
        cspace_positions: np.ndarray,
        base_command: np.ndarray,
        success: np.ndarray,
        position_error: np.ndarray,
        orientation_error: np.ndarray | None,
        status: tuple[str, ...],
    ) -> BatchedIKResult:
        """把 cuMotion C-space 结果映射回 tiled command-space。"""

        if self.mapping is not None:
            joint_positions = self.mapping.cspace_to_command(
                cspace_positions,
                base_command_positions=base_command,
            )
        else:
            joint_positions = cspace_positions
        return BatchedIKResult(
            joint_positions=joint_positions,
            success=success,
            position_error=position_error,
            orientation_error=orientation_error,
            status=status,
        )

    def _prepare_seeds(self, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把调用方 seed 规范化为 cuMotion C-space seed。"""

        if self.mapping is None:
            if seeds.shape[1] != self.expected_cspace_width:
                raise ValueError(
                    "seeds width must match cuMotion expected_cspace_width"
                )
            return seeds, seeds
        if seeds.shape[1] != self.mapping.command_width:
            raise ValueError("seeds width must match command_joint_names length")
        return self.mapping.command_to_cspace(seeds), seeds

    def _batch_solver(self, frame_name: str):
        """按 TCP frame 缓存 cuMotion ``CollisionFreeIkSolver``。"""

        if frame_name in self._batch_solvers:
            return self._batch_solvers[frame_name]
        required = (
            "CollisionFreeIkSolver",
            "create_default_collision_free_ik_solver_config",
            "create_collision_free_ik_solver",
        )
        if not all(hasattr(self.cumotion, name) for name in required):
            raise RuntimeError("cuMotion collision-free batch IK API is missing")
        if not hasattr(self.context, "robot_description"):
            raise RuntimeError("CuMotionContext.robot_description is required")
        try:
            collision_world = (
                self.context.empty_collision_world()
                if hasattr(self.context, "empty_collision_world")
                else self.context.collision_world()
            )
            config = self.cumotion.create_default_collision_free_ik_solver_config(
                self.context.robot_description,
                frame_name,
                collision_world.world_view,
            )
            _apply_collision_free_params(
                config,
                self.collision_free_params,
                self.cumotion.CollisionFreeIkSolverConfig.ParamValue,
            )
            solver = self.cumotion.create_collision_free_ik_solver(config)
        except Exception as exc:
            raise RuntimeError(
                f"failed to create cuMotion batch IK solver: {exc}"
            ) from exc
        if not hasattr(solver, "solve_array"):
            raise RuntimeError("CollisionFreeIkSolver.solve_array is required")
        self._batch_solvers[frame_name] = solver
        return solver

    def _task_space_target_array(
        self,
        *,
        positions: np.ndarray,
        orientations: np.ndarray | None,
    ):
        """把 tiled target arrays 转成 cuMotion TaskSpaceTargetArray。"""

        try:
            solver_type = self.cumotion.CollisionFreeIkSolver
            translation = solver_type.TranslationConstraintArray.target(
                [[np.asarray(row, dtype=float).reshape(3)] for row in positions],
                self.position_tolerance,
            )
            if orientations is None:
                orientation = solver_type.OrientationConstraintArray.none()
            else:
                orientation = solver_type.OrientationConstraintArray.target(
                    [
                        [rotation_from_quat_wxyz(self.cumotion, quat)]
                        for quat in orientations
                    ],
                    self.orientation_tolerance,
                )
            return solver_type.TaskSpaceTargetArray(translation, orientation)
        except Exception as exc:
            raise RuntimeError(
                f"failed to build cuMotion TaskSpaceTargetArray: {exc}"
            ) from exc

    def _position_error(
        self,
        cspace_position: np.ndarray,
        target_position: np.ndarray,
        frame_name: str,
    ) -> float:
        """用 FK 复算 batch IK 的位置误差。"""

        try:
            actual = np.asarray(
                self.kinematics.position(cspace_position, frame_name),
                dtype=float,
            ).reshape(3)
            target = np.asarray(target_position, dtype=float).reshape(3)
            return float(np.linalg.norm(actual - target))
        except Exception:
            return float("inf")

    def _orientation_error(
        self,
        cspace_position: np.ndarray,
        target_orientation: np.ndarray,
        frame_name: str,
    ) -> float:
        """用 FK 复算 batch IK 的姿态误差。"""

        try:
            actual = self.kinematics.orientation(cspace_position, frame_name)
            target = rotation_from_quat_wxyz(self.cumotion, target_orientation)
            return float(self.cumotion.Rotation3.distance(target, actual))
        except Exception:
            return float("inf")

    def _validate_frame(self, frame_name: str) -> None:
        """检查 TCP frame 是否存在于 cuMotion robot description。"""

        validate_cumotion_frame(self.context, frame_name, label="tcp_frame_name")


def _require_2d(values: np.ndarray, label: str) -> np.ndarray:
    """把输入规范化成二维 float 数组；一维输入视为单行 batch。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    return array


def _require_2d_width(values: np.ndarray, width: int, label: str) -> np.ndarray:
    """校验二维数组列数等于指定 width。"""

    array = _require_2d(values, label)
    if array.shape[1] != int(width):
        raise ValueError(f"{label} must have shape (N, {width})")
    return array


def _optional_2d_width(
    values: np.ndarray | None,
    num_rows: int,
    width: int,
    label: str,
) -> np.ndarray | None:
    """读取可选 batched 数组，并支持单行广播到 ``num_rows``。"""

    if values is None:
        return None
    array = _require_2d_width(values, width, label)
    if array.shape[0] == 1 and num_rows != 1:
        return np.repeat(array, num_rows, axis=0)
    if array.shape[0] != num_rows:
        raise ValueError(f"{label} first dimension must be 1 or N")
    return array


def _select_closest_cspace_position(
    positions: Sequence[np.ndarray],
    *,
    seed: np.ndarray,
    width: int,
) -> np.ndarray | None:
    """从 batch IK 候选解中选择最接近该 env seed 的结果。"""

    seed_array = np.asarray(seed, dtype=float).reshape(width)
    candidates = []
    for value in positions:
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.size == width:
            candidates.append(q.reshape(width))
    if not candidates:
        return None
    return min(candidates, key=lambda q: float(np.linalg.norm(q - seed_array)))


def _status_name(status) -> str:
    """把 pybind enum 或测试 fake status 规整成字符串。"""

    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    text = str(status)
    return text.rsplit(".", maxsplit=1)[-1]


def _rotation_to_quat_wxyz(rotation) -> np.ndarray:
    """把 cuMotion 或测试 fake rotation 对象转换为项目 wxyz 四元数。"""

    if hasattr(rotation, "quaternion_wxyz"):
        return np.asarray(rotation.quaternion_wxyz, dtype=float).reshape(4)
    return np.asarray(
        [rotation.w(), rotation.x(), rotation.y(), rotation.z()],
        dtype=float,
    )


def _apply_collision_free_params(
    config,
    params: Mapping[str, object],
    param_value_type,
) -> None:
    """把配置中的 collision-free IK 后端参数写入 cuMotion config。"""

    for name, value in params.items():
        ok = config.set_param(str(name), param_value_type(value))
        if not ok:
            raise ValueError(f"Invalid collision-free IK parameter: {name}")
