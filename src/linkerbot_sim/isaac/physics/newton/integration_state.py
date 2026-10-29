"""Newton ``SolverMuJoCo`` 的设备中立 persistent integration state。

Newton 的 ``State``/``Control`` 已经拥有 ``qpos``、``qvel`` 和 ``ctrl`` 的权威值；
``SolverMuJoCo`` 仍会在 MuJoCo ``Data`` 中跨 step 保留时间、actuator activation 与
warm-start acceleration。若 snapshot/reset/clone 只恢复 Newton 状态，这三类值会让后继
轨迹悄悄分叉。本模块只保存 ``TIME|ACT|WARMSTART``，避免建立第二份 generalized state。

CPU/CUDA 共用 MuJoCo Warp 的 float32 状态和生命周期；CUDA 使用 owner stream，
CPU 同步执行。主状态仍由 Newton 拥有，不保留原生 MuJoCo C 状态分支。
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class SolverIntegrationStateStore(Protocol):
    """``SolverMuJoCo`` persistent state 的统一所有权接口。"""

    @property
    def execution(self) -> Literal["cpu", "cuda"]:
        """返回 store 使用的物理执行设备类别。"""

    @property
    def signature(self) -> int:
        """返回固定的 MuJoCo ``TIME|ACT|WARMSTART`` state signature。"""

    @property
    def width(self) -> int:
        """返回每个 world 的 persistent state 列数。"""

    @property
    def activation_width(self) -> int:
        """返回 persistent layout 中 actuator activation 的列数。"""

    def initialize(
        self,
        solver: object,
        *,
        world_count: int,
        device: object,
        stream: object | None = None,
    ) -> None:
        """绑定 solver、分配 canonical/baseline buffer，并提交初始 baseline。"""

    def capture(self) -> None:
        """把 engine 当前 persistent state 捕获到 canonical buffer。"""

    def restore(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        """把全部或选中 world 的值写回 engine，并回读 canonical buffer。"""

    def reset(self, active_world_mask: object | None = None) -> None:
        """把全部或选中 world 恢复到 committed baseline。"""

    def commit(self) -> None:
        """捕获 engine 当前值并替换 committed baseline。"""

    def borrow(self) -> object:
        """借用 store-owned canonical buffer；调用方不得替换其 storage。"""

    def validate(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        """校验 state 与可选 world mask 的形状、dtype 和设备。"""


class _StoreMetadata:
    """初始化状态与只读 layout metadata。"""

    _execution: Literal["cpu", "cuda"]

    def __init__(self) -> None:
        self._initialized = False
        self._signature = 0
        self._width = 0
        self._activation_width = 0

    @property
    def execution(self) -> Literal["cpu", "cuda"]:
        return self._execution

    @property
    def signature(self) -> int:
        self._require_initialized()
        return self._signature

    @property
    def width(self) -> int:
        return self._width

    @property
    def activation_width(self) -> int:
        return self._activation_width

    def _require_uninitialized(self) -> None:
        if self._initialized:
            raise RuntimeError("solver integration state store is already initialized")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("solver integration state store is not initialized")


class WarpSolverIntegrationStateStore(_StoreMetadata):
    """CPU/CUDA 共用的 MuJoCo Warp persistent state store。"""

    def __init__(self, *, execution: Literal["cpu", "cuda"]) -> None:
        super().__init__()
        if execution not in {"cpu", "cuda"}:
            raise ValueError("solver integration execution must be cpu or cuda")
        self._execution = execution
        self._solver: object | None = None
        self._device = ""
        self._stream: object | None = None
        self._state: object | None = None
        self._baseline: object | None = None
        self._all_world_mask: object | None = None
        self._world_count = 0

    def initialize(
        self,
        solver: object,
        *,
        world_count: int,
        device: object,
        stream: object | None = None,
    ) -> None:
        """分配固定 Warp buffer；构造后热路径不会再创建 selector 或 host staging。"""

        self._require_uninitialized()
        if type(world_count) is not int or world_count < 1:
            raise ValueError("solver integration world_count must be positive")
        device_name = str(device)
        if self.execution == "cuda":
            if stream is None or not _is_canonical_cuda_device(device_name):
                raise ValueError(
                    "CUDA solver integration requires a cuda:N device and owner stream"
                )
        elif device_name != "cpu" or stream is not None:
            raise ValueError(
                "CPU solver integration requires device=cpu and stream=None"
            )
        if bool(getattr(solver, "use_mujoco_cpu", False)):
            raise ValueError("integration state requires a MuJoCo Warp solver")
        _validate_update_interval(solver)

        import mujoco
        import mujoco_warp
        import warp as wp

        mjw_data = getattr(solver, "mjw_data", None)
        actual_world_count = int(getattr(mjw_data, "nworld", -1))
        if actual_world_count != world_count:
            raise RuntimeError(
                "SolverMuJoCo integration world count mismatch: "
                f"actual={actual_world_count}, expected={world_count}"
            )
        signature = int(
            mujoco_warp.State.TIME | mujoco_warp.State.ACT | mujoco_warp.State.WARMSTART
        )
        mj_model = getattr(solver, "mj_model", None)
        width = int(mujoco.mj_stateSize(mj_model, signature))
        if width < 1:
            raise RuntimeError("SolverMuJoCo returned an empty persistent state layout")

        self._solver = solver
        self._device = device_name
        self._stream = stream
        self._world_count = world_count
        self._signature = signature
        self._width = width
        self._activation_width = int(getattr(mj_model, "na", 0))
        # 这些分配都发生在 owner stream 建立后、CUDA graph capture 之前。常驻全选 mask
        # 避免 mujoco_warp.get_state/set_state 在每个 physics tick 临时分配 wp.ones。
        with self._scope():
            self._state = wp.zeros(
                (world_count, width),
                dtype=wp.float32,
                device=device,
            )
            self._baseline = wp.zeros(
                (world_count, width),
                dtype=wp.float32,
                device=device,
            )
            self._all_world_mask = wp.ones(
                world_count,
                dtype=wp.bool,
                device=device,
            )
            self._capture_on_owner_stream(mujoco_warp)
            wp.copy(self._baseline, self._state, stream=stream)
        self._initialized = True

    def capture(self) -> None:
        self._require_initialized()
        import mujoco_warp

        with self._scope():
            self._capture_on_owner_stream(mujoco_warp)

    def restore(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        self.validate(values, active_world_mask=active_world_mask)
        import mujoco_warp

        assert self._solver is not None
        assert self._state is not None
        assert self._all_world_mask is not None
        active = (
            self._all_world_mask if active_world_mask is None else active_world_mask
        )
        with self._scope():
            mujoco_warp.set_state(
                self._solver.mjw_model,
                self._solver.mjw_data,
                values,
                self._signature,
                active=active,
            )
            # 只有 engine 接受写入后才更新 canonical buffer；selected get 会保留未选行。
            mujoco_warp.get_state(
                self._solver.mjw_model,
                self._solver.mjw_data,
                self._state,
                self._signature,
                active=active,
            )

    def reset(self, active_world_mask: object | None = None) -> None:
        self._require_initialized()
        assert self._baseline is not None
        self.restore(self._baseline, active_world_mask=active_world_mask)

    def commit(self) -> None:
        self._require_initialized()
        import mujoco_warp
        import warp as wp

        assert self._baseline is not None
        assert self._state is not None
        with self._scope():
            self._capture_on_owner_stream(mujoco_warp)
            wp.copy(self._baseline, self._state, stream=self._stream)

    def borrow(self) -> object:
        self._require_initialized()
        assert self._state is not None
        return self._state

    def validate(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        self._require_initialized()
        import warp as wp

        expected = (self._world_count, self._width)
        if tuple(getattr(values, "shape", ())) != expected:
            raise ValueError(
                "solver integration state must have shape "
                f"{expected}, got {getattr(values, 'shape', None)!r}"
            )
        if getattr(values, "dtype", None) != wp.float32:
            raise TypeError("solver integration state must use Warp float32")
        if str(getattr(values, "device", "")) != self._device:
            raise ValueError(
                "solver integration state must live on the store device: "
                f"expected={self._device}, "
                f"actual={getattr(values, 'device', None)!r}"
            )
        if active_world_mask is None:
            return
        if tuple(getattr(active_world_mask, "shape", ())) != (self._world_count,):
            raise ValueError(
                f"solver active world mask must have shape ({self._world_count},)"
            )
        if getattr(active_world_mask, "dtype", None) != wp.bool:
            raise TypeError("solver active world mask must use Warp bool")
        if str(getattr(active_world_mask, "device", "")) != self._device:
            raise ValueError("solver active world mask must live on the store device")

    @contextmanager
    def _scope(self):
        import warp as wp

        stream_scope = (
            wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False)
            if self._stream is not None
            else nullcontext()
        )
        with wp.ScopedDevice(self._device), stream_scope:
            yield

    def _capture_on_owner_stream(self, mujoco_warp: object) -> None:
        assert self._solver is not None
        assert self._state is not None
        assert self._all_world_mask is not None
        mujoco_warp.get_state(
            self._solver.mjw_model,
            self._solver.mjw_data,
            self._state,
            self._signature,
            active=self._all_world_mask,
        )


def create_solver_integration_state_store(
    execution: Literal["cpu", "cuda"],
) -> SolverIntegrationStateStore:
    return WarpSolverIntegrationStateStore(execution=execution)


def _validate_update_interval(solver: object) -> None:
    if int(getattr(solver, "update_data_interval", -1)) != 1:
        raise RuntimeError(
            "Newton requires SolverMuJoCo update_data_interval=1 so q/qd remain "
            "authoritative across reset and clone"
        )


def _is_canonical_cuda_device(device: str) -> bool:
    prefix, separator, index = device.partition(":")
    return (
        prefix == "cuda"
        and separator == ":"
        and index.isdecimal()
        and str(int(index)) == index
    )


__all__ = [
    "WarpSolverIntegrationStateStore",
    "SolverIntegrationStateStore",
    "create_solver_integration_state_store",
]
