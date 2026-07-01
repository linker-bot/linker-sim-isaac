"""cuMotion motion 客户参数描述。

本模块只保存轻量 Python 参数对象，不创建 Isaac runtime，也不加载 cuMotion context。
客户脚本可以直接导入这些 spec 来描述 TCP 和临时实验动作。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
    SpecifiedPathFamily,
)
from linkerbot_sim.planning.requests import (
    CompositePath,
    CSpaceWaypointPath,
    IKRequest,
    MotionRequest,
    SpecifiedPathRequest,
    TaskSpacePath,
)
from linkerbot_sim.backends.cumotion.tcp_frame import TcpTransform


MoveExecutionMode: TypeAlias = Literal["single", "selected_side", "dual_cspace"]
OverlayTiming: TypeAlias = Literal["sync", "before", "after"]
IkOrientationMode: TypeAlias = Literal["current", "target", "none"]


@dataclass(frozen=True)
class CartesianTcpFrameSpec:
    """客户脚本传入的末端相对 TCP 变换。"""

    frame_name: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if not str(self.frame_name):
            raise ValueError("TCP frame_name cannot be empty")
        np.asarray(self.xyz, dtype=float).reshape(3)
        np.asarray(self.rpy, dtype=float).reshape(3)


@dataclass(frozen=True)
class DualArmTcpSpec:
    """双臂运行时使用的左右 TCP spec。"""

    left: CartesianTcpFrameSpec
    right: CartesianTcpFrameSpec

    def validate(self) -> None:
        self.left.validate()
        self.right.validate()
        if self.left.frame_name == self.right.frame_name:
            raise ValueError("left/right TCP frame_name must be unique")


@dataclass(frozen=True)
class HandMoveSpec:
    """直接在 controller command-space 中执行单手动作。"""

    side: str
    joint_positions: Mapping[str, float] | Sequence[float]
    duration_s: float | None = None
    phase: str | None = None

    def validate(self, *, require_side: bool = False) -> None:
        _normalize_side_required(self.side)
        _validate_hand_joint_positions(self.joint_positions)
        _validate_duration(self.duration_s)


@dataclass(frozen=True)
class DualHandMoveSpec:
    """同步执行左右手 command-space 动作。"""

    left: HandMoveSpec | None = None
    right: HandMoveSpec | None = None
    duration_s: float | None = None
    phase: str | None = None

    def validate(self, *, require_side: bool = False) -> None:
        if self.left is None and self.right is None:
            raise ValueError("DualHandMoveSpec requires at least one hand target")
        if self.left is not None:
            self.left.validate()
            if _normalize_side_required(self.left.side) != "left":
                raise ValueError("DualHandMoveSpec.left must use side='left'")
        if self.right is not None:
            self.right.validate()
            if _normalize_side_required(self.right.side) != "right":
                raise ValueError("DualHandMoveSpec.right must use side='right'")
        _validate_duration(self.duration_s)
        if self.duration_s is None and all(
            hand is None or hand.duration_s is None
            for hand in (self.left, self.right)
        ):
            raise ValueError("DualHandMoveSpec requires duration_s")


@dataclass(frozen=True)
class RawJointSequenceSideSpec:
    """单侧 command-space raw target 序列。"""

    joint_positions: Mapping[str, Sequence[float]] | Sequence[Sequence[float]]

    def validate(self) -> None:
        _validate_raw_joint_sequence_side(self.joint_positions)


@dataclass(frozen=True)
class RawJointSequenceMoveSpec:
    """直接按 physics step 刷新 controller command-space target。"""

    left: RawJointSequenceSideSpec | None = None
    right: RawJointSequenceSideSpec | None = None
    step_interval: int = 1
    phase: str | None = None

    def validate(self, *, require_side: bool = False) -> None:
        if self.left is None and self.right is None:
            raise ValueError("RawJointSequenceMoveSpec requires at least one side")
        if isinstance(self.step_interval, bool):
            raise ValueError("step_interval must be a positive integer")
        if int(self.step_interval) != self.step_interval or int(self.step_interval) <= 0:
            raise ValueError("step_interval must be a positive integer")
        if self.left is not None:
            self.left.validate()
        if self.right is not None:
            self.right.validate()


@dataclass(frozen=True)
class CommandOverlaySpec:
    """arm/cuMotion motion 前后或同步叠加的 command-space 手部动作。"""

    timing: OverlayTiming = "sync"
    left_hand: HandMoveSpec | None = None
    right_hand: HandMoveSpec | None = None

    def validate(self) -> None:
        if self.timing not in {"sync", "before", "after"}:
            raise ValueError("overlay timing must be one of: sync, before, after")
        if self.left_hand is None and self.right_hand is None:
            raise ValueError("CommandOverlaySpec requires at least one hand target")
        if self.left_hand is not None:
            self.left_hand.validate()
            if _normalize_side_required(self.left_hand.side) != "left":
                raise ValueError("left_hand overlay must use side='left'")
        if self.right_hand is not None:
            self.right_hand.validate()
            if _normalize_side_required(self.right_hand.side) != "right":
                raise ValueError("right_hand overlay must use side='right'")


@dataclass(frozen=True)
class CumotionMoveSpec:
    """高级入口：直接传项目侧 planning request。"""

    request: IKRequest | MotionRequest | SpecifiedPathRequest
    side: str | None = None
    tcp_frame_name: str | None = None
    duration_s: float | None = None
    execution: MoveExecutionMode = "single"
    phase: str | None = None
    overlays: tuple[CommandOverlaySpec, ...] = ()

    def validate(self, *, require_side: bool = False) -> None:
        if require_side and self.execution != "dual_cspace":
            _normalize_side_required(self.side)
        if self.execution not in {"single", "selected_side", "dual_cspace"}:
            raise ValueError(
                "execution must be one of: single, selected_side, dual_cspace"
            )
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        _validate_duration(self.duration_s)
        _validate_overlays(self.overlays)
        self.request.validate_structure()


@dataclass(frozen=True)
class IkOffsetMoveSpec:
    """相对当前 TCP pose 的 IK 位移动作。"""

    tcp_frame_name: str
    tcp_offset: tuple[float, float, float]
    duration_s: float
    side: str | None = None
    phase: str | None = None
    orientation_mode: IkOrientationMode = "current"
    target_orientation: tuple[float, float, float, float] | None = None
    overlays: tuple[CommandOverlaySpec, ...] = ()

    def validate(self, *, require_side: bool = False) -> None:
        if require_side:
            _normalize_side_required(self.side)
        if not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        np.asarray(self.tcp_offset, dtype=float).reshape(3)
        if self.orientation_mode not in {"current", "target", "none"}:
            raise ValueError(
                "orientation_mode must be one of: current, target, none"
            )
        if self.orientation_mode == "target" and self.target_orientation is None:
            raise ValueError("target_orientation is required when orientation_mode='target'")
        if self.target_orientation is not None:
            np.asarray(self.target_orientation, dtype=float).reshape(4)
        _validate_duration(self.duration_s)
        _validate_overlays(self.overlays)


@dataclass(frozen=True)
class CSpaceGoalPlanMoveSpec:
    """规划到选定侧 arm C-space 的绝对关节角目标。"""

    joint_positions: tuple[float, ...]
    duration_s: float
    side: str | None = None
    tcp_frame_name: str | None = None
    phase: str | None = None
    overlays: tuple[CommandOverlaySpec, ...] = ()

    def validate(self, *, require_side: bool = False) -> None:
        if require_side:
            _normalize_side_required(self.side)
        joints = np.asarray(self.joint_positions, dtype=float).reshape(-1)
        if joints.size == 0:
            raise ValueError("joint_positions cannot be empty")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        _validate_duration(self.duration_s)
        _validate_overlays(self.overlays)


@dataclass(frozen=True)
class CSpaceDeltaPlanMoveSpec:
    """在当前 C-space 上叠加关节增量，并调用目标式 planner。"""

    joint_deltas: tuple[float, ...]
    duration_s: float
    side: str | None = None
    tcp_frame_name: str | None = None
    phase: str | None = None
    overlays: tuple[CommandOverlaySpec, ...] = ()

    def validate(self, *, require_side: bool = False) -> None:
        if require_side:
            _normalize_side_required(self.side)
        deltas = np.asarray(self.joint_deltas, dtype=float).reshape(-1)
        if deltas.size == 0:
            raise ValueError("joint_deltas cannot be empty")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        _validate_duration(self.duration_s)
        _validate_overlays(self.overlays)


@dataclass(frozen=True)
class SpecifiedPathMoveSpec:
    """执行调用方指定的 C-space、task-space 或 composite 路径。"""

    tcp_frame_name: str
    path: CSpaceWaypointPath | TaskSpacePath | CompositePath
    duration_s: float
    side: str | None = None
    phase: str | None = None
    overlays: tuple[CommandOverlaySpec, ...] = ()

    def validate(self, *, require_side: bool = False) -> None:
        if require_side:
            _normalize_side_required(self.side)
        if not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        _validate_duration(self.duration_s)
        _validate_overlays(self.overlays)


MoveSpec: TypeAlias = (
    CumotionMoveSpec
    | IkOffsetMoveSpec
    | CSpaceGoalPlanMoveSpec
    | CSpaceDeltaPlanMoveSpec
    | SpecifiedPathMoveSpec
    | HandMoveSpec
    | DualHandMoveSpec
    | RawJointSequenceMoveSpec
)


def tcp_transform_from_spec(spec: CartesianTcpFrameSpec) -> TcpTransform:
    """把客户 TCP spec 转为末端相对 ``TcpTransform``。"""

    spec.validate()
    return TcpTransform.from_xyz_rpy(
        frame_name=str(spec.frame_name),
        xyz=spec.xyz,
        rpy=spec.rpy,
    )


def tcp_transforms_from_dual_spec(
    spec: DualArmTcpSpec,
) -> tuple[TcpTransform, TcpTransform]:
    """把双臂 TCP spec 转为末端相对 ``TcpTransform``。"""

    spec.validate()
    return (
        tcp_transform_from_spec(spec.left),
        tcp_transform_from_spec(spec.right),
    )


def side_tcp_frame_name(tcp: DualArmTcpSpec, side: str) -> str:
    """按 side 返回双臂 TCP frame 名称。"""

    normalized = _normalize_side_required(side)
    return tcp.left.frame_name if normalized == "left" else tcp.right.frame_name


def motion_type_name(move: MoveSpec | CumotionMoveSpec) -> str:
    """返回用于日志和默认 phase 的运动类型名称。"""

    if isinstance(move, IkOffsetMoveSpec):
        return "ik"
    if isinstance(move, CSpaceGoalPlanMoveSpec):
        return "motion"
    if isinstance(move, CSpaceDeltaPlanMoveSpec):
        return "motion"
    if isinstance(move, SpecifiedPathMoveSpec):
        return "specified_path"
    if isinstance(move, HandMoveSpec):
        return "hand"
    if isinstance(move, DualHandMoveSpec):
        return "dual_hand"
    if isinstance(move, CumotionMoveSpec):
        if isinstance(move.request, IKRequest):
            return "ik"
        if isinstance(move.request, MotionRequest):
            return "motion"
        if isinstance(move.request, SpecifiedPathRequest):
            return "specified_path"
    raise TypeError(f"unsupported move spec type: {type(move).__name__}")


def default_move_phase(
    move: MoveSpec | CumotionMoveSpec,
    *,
    side: str | None = None,
    dual_cspace: bool = False,
) -> str:
    """为缺省 phase 生成稳定名称。"""

    move_type = motion_type_name(move)
    if dual_cspace:
        return f"dual_cspace_{move_type}"
    if side is None:
        return f"cumotion_{move_type}"
    return f"dual_{_normalize_side_required(side)}_{move_type}"


def specified_path_planner_config(
    base_config: MotionPlannerBackendConfig,
    *,
    path: CSpaceWaypointPath | TaskSpacePath | CompositePath,
) -> MotionPlannerBackendConfig:
    """基于 profile 配置构造本次指定路径 planner 配置。"""

    return MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        graph_search=base_config.graph_search,
        trajectory_generation=base_config.trajectory_generation,
        trajectory_optimization=base_config.trajectory_optimization,
        specified_path=SpecifiedPathConfig(
            family=specified_path_family_for_path(path),
            validate_collision_after_generation=(
                base_config.specified_path.validate_collision_after_generation
            ),
            cspace_waypoints=base_config.specified_path.cspace_waypoints,
            task_space_segments=base_config.specified_path.task_space_segments,
            composite=base_config.specified_path.composite,
        ),
    )


def specified_path_family_for_path(
    path: CSpaceWaypointPath | TaskSpacePath | CompositePath,
) -> SpecifiedPathFamily:
    """按 path 类型选择 specified_path family。"""

    if isinstance(path, CSpaceWaypointPath):
        return "cspace_waypoints"
    if isinstance(path, TaskSpacePath):
        return "task_space_segments"
    if isinstance(path, CompositePath):
        return "composite"
    raise TypeError(f"unsupported specified path type: {type(path).__name__}")


def normalize_move_sequence(moves: Sequence[MoveSpec]) -> tuple[MoveSpec, ...]:
    """把客户传入的 move 序列冻结成 tuple 并拒绝空动作。"""

    normalized = tuple(moves)
    if not normalized:
        raise ValueError("moves cannot be empty")
    return normalized


def _validate_duration(duration_s: float | None) -> None:
    if duration_s is not None and float(duration_s) < 0.0:
        raise ValueError("duration_s cannot be negative")


def _validate_overlays(overlays: Sequence[CommandOverlaySpec]) -> None:
    for overlay in tuple(overlays):
        if not isinstance(overlay, CommandOverlaySpec):
            raise TypeError("overlays must contain CommandOverlaySpec values")
        overlay.validate()


def _validate_hand_joint_positions(
    joint_positions: Mapping[str, float] | Sequence[float],
) -> None:
    if isinstance(joint_positions, Mapping):
        if not joint_positions:
            raise ValueError("hand joint_positions cannot be empty")
        for name, value in joint_positions.items():
            if not str(name):
                raise ValueError("hand joint name cannot be empty")
            float(value)
        return
    values = np.asarray(joint_positions, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("hand joint_positions cannot be empty")


def _validate_raw_joint_sequence_side(
    joint_positions: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> None:
    if isinstance(joint_positions, Mapping):
        if not joint_positions:
            raise ValueError("raw joint sequence mapping cannot be empty")
        sample_count: int | None = None
        for name, values in joint_positions.items():
            if not str(name):
                raise ValueError("raw joint sequence joint name cannot be empty")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("raw joint sequence mapping values must be lists")
            array = np.asarray(values, dtype=float).reshape(-1)
            if array.size == 0:
                raise ValueError("raw joint sequence samples cannot be empty")
            if sample_count is None:
                sample_count = int(array.size)
            elif array.size != sample_count:
                raise ValueError("raw joint sequence mapping sample counts must match")
        return
    if isinstance(joint_positions, (str, bytes)):
        raise ValueError("raw joint sequence joint_positions must be a mapping or matrix")
    matrix = np.asarray(joint_positions, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("raw joint sequence matrix must have shape (N, dof)")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("raw joint sequence matrix cannot be empty")


def _normalize_side_required(side: str | None) -> str:
    if side is None:
        raise ValueError("side is required")
    normalized = str(side).lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized
