"""cuRobo 的纯 CUDA batch IK/waypoint adapter。

本模块不导入项目 planning/collision 包。context 只需提供已经构造的 kinematics-only IK solver、
DeviceCfg 和 cuRobo public types；所有 joint mapping index 在构造时缓存到 GPU。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from linkerbot_sim.backends.curobo.tensor_adapter import (
    seed_config_from_state_or_seed,
)
from linkerbot_sim.backends.curobo.tool_pose import (
    update_active_tool_pose_criteria,
)
from linkerbot_sim.backends.curobo.kinematics.types import (
    BatchIKTensorResult,
    BatchIKWaypointTensorResult,
    assert_finite_async,
    require_cuda_tensor,
)


class CuroboDeviceBatchIKSolver:
    """实现 Kaleidoscope ``DeviceBatchIKSolver`` 的 cuRobo adapter。"""

    def __init__(
        self,
        context: object,
        *,
        tcp_frame_name: str | None = None,
        command_joint_names: Sequence[str] | None = None,
    ) -> None:
        # adapter 是 context 的唯一上层 owner。scene assembly 成功后只把 adapter 交给
        # action term；Session/ReplicatedScene 不再保存或重复关闭 context。
        self.context = context
        self.ik_solver = getattr(context, "ik_solver", None)
        if self.ik_solver is None:
            raise RuntimeError("kinematics context must expose ik_solver")
        self.tcp_frame_name = str(
            tcp_frame_name or getattr(context, "default_tcp_frame", "")
        )
        frame_names = tuple(str(name) for name in context.frame_names())
        if not self.tcp_frame_name or self.tcp_frame_name not in frame_names:
            raise ValueError("tcp_frame_name must be registered by the context")
        self.cspace_joint_names = tuple(str(name) for name in context.joint_names())
        command_names = (
            self.cspace_joint_names
            if command_joint_names is None
            else tuple(str(name) for name in command_joint_names)
        )
        if len(set(command_names)) != len(command_names):
            raise ValueError("command_joint_names cannot contain duplicates")
        missing = set(self.cspace_joint_names) - set(command_names)
        if missing:
            raise ValueError(
                f"command joints are missing cuRobo C-space names: {sorted(missing)}"
            )
        device_cfg = getattr(context, "device_cfg", None)
        self.device = torch.device(getattr(device_cfg, "device", "cpu"))
        if self.device.type != "cuda":
            raise ValueError("Kaleidoscope cuRobo IK requires a CUDA DeviceCfg")
        self.dtype = getattr(device_cfg, "dtype", torch.float32)
        if self.dtype != torch.float32:
            raise ValueError("Kaleidoscope cuRobo IK currently requires float32")
        self.command_dim = len(command_names)
        self._cspace_dim = len(self.cspace_joint_names)
        command_index = {name: index for index, name in enumerate(command_names)}
        self._cspace_from_command = torch.tensor(
            [command_index[name] for name in self.cspace_joint_names],
            device=self.device,
            dtype=torch.int64,
        )
        self._closed = False

    def solve(
        self,
        *,
        target_positions: torch.Tensor,
        target_orientations_wxyz: torch.Tensor | None,
        seeds: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> BatchIKTensorResult:
        self._require_open()
        positions, orientations, command_seed, cspace_seed, active = self._inputs(
            target_positions,
            target_orientations_wxyz,
            seeds,
            active_mask,
        )
        result = self._solve_pose(
            positions=positions,
            orientations=orientations,
            cspace_seed=cspace_seed,
        )
        success = (
            _result_success(result, rows=positions.shape[0], reference=cspace_seed)
            & active
        )
        cspace_solution = _result_positions(result, fallback=cspace_seed)
        cspace_solution = torch.where(success[:, None], cspace_solution, cspace_seed)
        command_solution = command_seed.clone()
        command_solution.index_copy_(1, self._cspace_from_command, cspace_solution)
        position_error = _result_errors(
            result,
            ("position_error", "position_errors"),
            rows=positions.shape[0],
            reference=cspace_seed,
        )
        orientation_error = (
            None
            if orientations is None
            else _result_errors(
                result,
                ("rotation_error", "orientation_error", "orientation_errors"),
                rows=positions.shape[0],
                reference=cspace_seed,
            )
        )
        return BatchIKTensorResult(
            joint_positions=command_solution,
            success=success,
            position_error=position_error,
            orientation_error=orientation_error,
        )

    def solve_waypoints(
        self,
        *,
        target_positions: torch.Tensor,
        target_orientations_wxyz: torch.Tensor | None,
        seeds: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> BatchIKWaypointTensorResult:
        self._require_open()
        positions = require_cuda_tensor(
            target_positions, name="cuRobo waypoint positions", ndim=3
        )
        if positions.device != self.device or positions.shape[2:] != (3,):
            raise ValueError(
                "waypoint positions must have shape (T,N,3) on solver device"
            )
        orientations = target_orientations_wxyz
        if orientations is not None:
            orientations = require_cuda_tensor(
                orientations, name="cuRobo waypoint orientations", ndim=3
            )
            if orientations.shape != (*positions.shape[:2], 4):
                raise ValueError("waypoint orientations must have shape (T,N,4)")
        # 使用第一个 waypoint 只为复用 seed/mask/device 校验；实际 position steps 保持原 tensor。
        _, _, command, cspace, active = self._inputs(
            positions[0],
            None if orientations is None else orientations[0],
            seeds,
            active_mask,
        )
        steps, rows = positions.shape[:2]
        trajectory = torch.empty(
            (steps, rows, self.command_dim), device=self.device, dtype=self.dtype
        )
        path_success = active.clone()
        solving = active.clone()
        first_failure = torch.full((rows,), -1, device=self.device, dtype=torch.int64)
        max_position_error = torch.zeros(rows, device=self.device, dtype=self.dtype)
        max_orientation_error = (
            None
            if orientations is None
            else torch.zeros(rows, device=self.device, dtype=self.dtype)
        )
        for step_index in range(steps):
            result = self._solve_pose(
                positions=positions[step_index],
                orientations=(
                    None if orientations is None else orientations[step_index]
                ),
                cspace_seed=cspace,
            )
            success = _result_success(result, rows=rows, reference=cspace)
            solution = _result_positions(result, fallback=cspace)
            position_error = _result_errors(
                result,
                ("position_error", "position_errors"),
                rows=rows,
                reference=cspace,
            )
            attempted = solving
            max_position_error = torch.where(
                attempted,
                torch.maximum(max_position_error, position_error),
                max_position_error,
            )
            if max_orientation_error is not None:
                orientation_error = _result_errors(
                    result,
                    ("rotation_error", "orientation_error", "orientation_errors"),
                    rows=rows,
                    reference=cspace,
                )
                max_orientation_error = torch.where(
                    attempted,
                    torch.maximum(max_orientation_error, orientation_error),
                    max_orientation_error,
                )
            accepted = attempted & success
            failed = attempted & ~success
            first_failure = torch.where(
                failed,
                torch.full_like(first_failure, step_index),
                first_failure,
            )
            path_success &= ~failed
            cspace = torch.where(accepted[:, None], solution, cspace).contiguous()
            command = command.clone()
            command.index_copy_(1, self._cspace_from_command, cspace)
            solving = accepted
            trajectory[step_index].copy_(command)
        return BatchIKWaypointTensorResult(
            joint_positions=trajectory,
            success=path_success,
            first_failure_step=first_failure,
            position_error=max_position_error,
            orientation_error=max_orientation_error,
        )

    def _inputs(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor | None,
        seeds: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        positions = require_cuda_tensor(
            positions, name="cuRobo target positions", ndim=2, dtype=self.dtype
        )
        command = require_cuda_tensor(
            seeds, name="cuRobo command seeds", ndim=2, dtype=self.dtype
        )
        if positions.device != self.device or command.device != self.device:
            raise ValueError("cuRobo inputs must live on solver.device")
        if positions.shape[1:] != (3,) or command.shape != (
            positions.shape[0],
            self.command_dim,
        ):
            raise ValueError("cuRobo target/seed shapes are inconsistent")
        if orientations is not None:
            orientations = require_cuda_tensor(
                orientations,
                name="cuRobo target orientations",
                ndim=2,
                dtype=self.dtype,
            )
            if orientations.device != self.device or orientations.shape != (
                positions.shape[0],
                4,
            ):
                raise ValueError("cuRobo orientations must have shape (N,4)")
        if active_mask is None:
            active = torch.ones(
                positions.shape[0], device=self.device, dtype=torch.bool
            )
        else:
            active = require_cuda_tensor(
                active_mask, name="cuRobo active mask", ndim=1, dtype=torch.bool
            )
            if active.device != self.device or active.shape != (positions.shape[0],):
                raise ValueError("cuRobo active mask must have shape (N,)")
        assert_finite_async(positions, name="cuRobo target positions")
        assert_finite_async(command, name="cuRobo command seeds")
        cspace = command.index_select(1, self._cspace_from_command).contiguous()
        return positions, orientations, command, cspace, active

    def _solve_pose(
        self,
        *,
        positions: torch.Tensor,
        orientations: torch.Tensor | None,
        cspace_seed: torch.Tensor,
    ) -> object:
        if orientations is None:
            quaternion = torch.zeros(
                (positions.shape[0], 4), device=self.device, dtype=self.dtype
            )
            quaternion[:, 0] = 1.0
        else:
            norm = torch.linalg.vector_norm(orientations, dim=1, keepdim=True)
            torch._assert_async(torch.all(norm > 1.0e-8), "IK quaternion norm is zero")
            quaternion = orientations / norm
        update_active_tool_pose_criteria(
            self.context,
            self.ik_solver,
            active_tool_frame=self.tcp_frame_name,
            orientation_free=orientations is None,
            tool_frames=(self.tcp_frame_name,),
        )
        goal_type = getattr(getattr(self.context, "types", None), "GoalToolPose", None)
        if goal_type is None:
            raise RuntimeError("kinematics context must expose types.GoalToolPose")
        goal = goal_type(
            tool_frames=[self.tcp_frame_name],
            position=positions[:, None, None, None, :],
            quaternion=quaternion[:, None, None, None, :],
        )
        state = self.context.joint_state_from_positions(cspace_seed)
        return self.ik_solver.solve_pose(
            goal,
            current_state=state,
            seed_config=seed_config_from_state_or_seed(state, cspace_seed),
        )

    def close(self) -> None:
        """幂等关闭唯一 context；失败时不伪装成功，允许 action term 重试。"""

        if self._closed:
            return
        close = getattr(self.context, "close", None)
        if not callable(close):
            raise TypeError("kinematics context must implement close()")
        close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("cuRobo device batch IK solver is closed")


def _result_positions(result: object, *, fallback: torch.Tensor) -> torch.Tensor:
    for name in ("solution", "joint_positions", "position"):
        value = getattr(result, name, None)
        if value is None:
            continue
        tensor = require_cuda_tensor(
            value,
            name=f"cuRobo result {name}",
            dtype=fallback.dtype,
        )
        if tensor.device != fallback.device:
            raise ValueError(
                f"cuRobo result {name} must live on {fallback.device}, "
                f"got {tensor.device}"
            )
        if tensor.ndim == 3:
            tensor = tensor[:, 0, :]
        if tensor.shape != fallback.shape:
            raise ValueError(
                f"cuRobo result {name} must have shape {tuple(fallback.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.detach().contiguous()
    return fallback.clone()


def _result_success(
    result: object, *, rows: int, reference: torch.Tensor
) -> torch.Tensor:
    value = getattr(result, "success", None)
    if value is None:
        return torch.zeros(rows, device=reference.device, dtype=torch.bool)
    success = require_cuda_tensor(
        value,
        name="cuRobo result success",
        dtype=torch.bool,
    ).detach()
    if success.device != reference.device:
        raise ValueError(
            "cuRobo result success must live on "
            f"{reference.device}, got {success.device}"
        )
    if success.ndim > 1:
        success = success.reshape(success.shape[0], -1).any(dim=1)
    success = success.reshape(-1)
    if success.shape != (rows,):
        raise ValueError("cuRobo success mask has the wrong shape")
    return success.contiguous()


def _result_errors(
    result: object,
    names: tuple[str, ...],
    *,
    rows: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    for name in names:
        value = getattr(result, name, None)
        if value is None:
            continue
        error = require_cuda_tensor(
            value,
            name=f"cuRobo result {name}",
            dtype=reference.dtype,
        ).detach()
        if error.device != reference.device:
            raise ValueError(
                f"cuRobo result {name} must live on {reference.device}, "
                f"got {error.device}"
            )
        if error.ndim > 1:
            error = error.reshape(error.shape[0], -1).amin(dim=1)
        error = error.reshape(-1)
        if error.shape == (rows,):
            return error.contiguous()
    return torch.full(
        (rows,),
        torch.finfo(reference.dtype).max,
        device=reference.device,
        dtype=reference.dtype,
    )


__all__ = ["CuroboDeviceBatchIKSolver"]
