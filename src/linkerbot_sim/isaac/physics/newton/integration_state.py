"""Newton ``SolverMuJoCo`` 的设备中立 persistent integration state。

Newton 的 ``State``/``Control`` 已经拥有 ``qpos``、``qvel`` 和 ``ctrl`` 的权威值；
``SolverMuJoCo`` 仍会在 MuJoCo ``Data`` 中跨 step 保留时间、actuator activation 与
warm-start acceleration。若 snapshot/reset/clone 只恢复 Newton 状态，这三类值会让后继
轨迹悄悄分叉。本模块只保存 ``TIME|ACT|WARMSTART``，避免建立第二份 generalized state。

CUDA 与 CPU 后端暴露完全相同的生命周期。CUDA 实现始终使用 Warp device buffer 和
owner stream，不经过 NumPy；CPU 实现只服务 Mirror 的单 world，并直接调用 MuJoCo C API。
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np


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
    """两种 store 共享的初始化状态与只读 layout metadata。"""

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


class CudaSolverIntegrationStateStore(_StoreMetadata):
    """MuJoCo-Warp persistent state store，所有数据始终驻留 CUDA。"""

    _execution: Literal["cuda"] = "cuda"

    def __init__(self) -> None:
        super().__init__()
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
            raise ValueError("CUDA solver integration world_count must be positive")
        if stream is None:
            raise ValueError("CUDA solver integration state requires an owner stream")
        device_name = str(device)
        if not _is_canonical_cuda_device(device_name):
            raise ValueError(
                "CUDA solver integration state requires a canonical cuda:N device"
            )
        if bool(getattr(solver, "use_mujoco_cpu", False)):
            raise ValueError("CUDA integration state cannot bind a MuJoCo CPU solver")
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
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
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
        import warp as wp

        assert self._stream is not None
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
            self._capture_on_owner_stream(mujoco_warp)

    def restore(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        self.validate(values, active_world_mask=active_world_mask)
        import mujoco_warp
        import warp as wp

        assert self._solver is not None
        assert self._state is not None
        assert self._all_world_mask is not None
        assert self._stream is not None
        active = (
            self._all_world_mask if active_world_mask is None else active_world_mask
        )
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
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
        assert self._stream is not None
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
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
            raise TypeError("CUDA solver integration state must use Warp float32")
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
            raise TypeError("CUDA solver active world mask must use Warp bool")
        if str(getattr(active_world_mask, "device", "")) != self._device:
            raise ValueError(
                "CUDA solver active world mask must live on the store device"
            )

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


class CpuSolverIntegrationStateStore(_StoreMetadata):
    """MuJoCo C CPU persistent state store；产品合同严格限定单 world。"""

    _execution: Literal["cpu"] = "cpu"

    def __init__(self) -> None:
        super().__init__()
        self._mujoco: object | None = None
        self._model: object | None = None
        self._data: object | None = None
        self._state: np.ndarray | None = None
        self._baseline: np.ndarray | None = None

    def initialize(
        self,
        solver: object,
        *,
        world_count: int,
        device: object,
        stream: object | None = None,
    ) -> None:
        """绑定单 world ``mjData``；CPU 路径不存在 Warp stream 或 CUDA graph。"""

        self._require_uninitialized()
        if type(world_count) is not int or world_count != 1:
            raise ValueError(
                "MuJoCo CPU integration state supports exactly one world; "
                f"got {world_count}"
            )
        if str(device) != "cpu":
            raise ValueError("MuJoCo CPU integration state requires device='cpu'")
        if stream is not None:
            raise ValueError("MuJoCo CPU integration state requires stream=None")
        if not bool(getattr(solver, "use_mujoco_cpu", False)):
            raise ValueError("CPU integration state requires use_mujoco_cpu=True")
        solver_model = getattr(solver, "model", None)
        solver_world_count = getattr(solver_model, "world_count", 1)
        if int(solver_world_count) != 1:
            raise ValueError(
                "MuJoCo CPU solver model must contain exactly one Newton world"
            )
        _validate_update_interval(solver)

        import mujoco

        signature = int(
            mujoco.mjtState.mjSTATE_TIME
            | mujoco.mjtState.mjSTATE_ACT
            | mujoco.mjtState.mjSTATE_WARMSTART
        )
        model = getattr(solver, "mj_model", None)
        data = getattr(solver, "mj_data", None)
        if model is None or data is None:
            raise RuntimeError("MuJoCo CPU solver must expose mj_model and mj_data")
        width = int(mujoco.mj_stateSize(model, signature))
        if width < 1:
            raise RuntimeError("SolverMuJoCo returned an empty persistent state layout")

        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._signature = signature
        self._width = width
        self._activation_width = int(getattr(model, "na", 0))
        # MuJoCo 的公开 Python ABI 使用 mjtNum(double)。显式 float64 也让 snapshot
        # serialization 不受 Newton model 的 Warp float32 storage 影响。
        self._state = np.empty((1, width), dtype=np.float64)
        self._baseline = np.empty((1, width), dtype=np.float64)
        mujoco.mj_getState(model, data, self._state[0], signature)
        np.copyto(self._baseline, self._state)
        self._initialized = True

    def capture(self) -> None:
        self._require_initialized()
        assert self._mujoco is not None
        assert self._model is not None
        assert self._data is not None
        assert self._state is not None
        self._mujoco.mj_getState(
            self._model,
            self._data,
            self._state[0],
            self._signature,
        )

    def restore(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        self.validate(values, active_world_mask=active_world_mask)
        if not _cpu_world_is_selected(active_world_mask):
            return
        assert isinstance(values, np.ndarray)
        assert self._mujoco is not None
        assert self._model is not None
        assert self._data is not None
        # signature 不含 QPOS/QVEL/CTRL；mj_setState 因而不会越过 Newton 的状态所有权。
        self._mujoco.mj_setState(
            self._model,
            self._data,
            values[0],
            self._signature,
        )
        self.capture()

    def reset(self, active_world_mask: object | None = None) -> None:
        self._require_initialized()
        assert self._baseline is not None
        self.restore(self._baseline, active_world_mask=active_world_mask)

    def commit(self) -> None:
        self._require_initialized()
        assert self._baseline is not None
        assert self._state is not None
        self.capture()
        np.copyto(self._baseline, self._state)

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
        expected = (1, self._width)
        if not isinstance(values, np.ndarray) or values.shape != expected:
            raise ValueError(
                "CPU solver integration state must be a NumPy array with shape "
                f"{expected}, got {getattr(values, 'shape', None)!r}"
            )
        if values.dtype != np.dtype(np.float64):
            raise TypeError("CPU solver integration state must use NumPy float64")
        if not values.flags.c_contiguous:
            raise ValueError("CPU solver integration state must be C-contiguous")
        _cpu_world_is_selected(active_world_mask)


def create_solver_integration_state_store(
    execution: Literal["cpu", "cuda"],
) -> SolverIntegrationStateStore:
    """按已解析的 physics execution 创建 store，不从产品 mode 推断设备。"""

    if execution == "cpu":
        return CpuSolverIntegrationStateStore()
    if execution == "cuda":
        return CudaSolverIntegrationStateStore()
    raise ValueError(
        f"solver integration state execution must be 'cpu' or 'cuda'; got {execution!r}"
    )


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


def _cpu_world_is_selected(active_world_mask: object | None) -> bool:
    """把单 world CPU selector 归一化为 bool，并明确拒绝 CUDA/多 world mask。"""

    if active_world_mask is None:
        return True
    if type(active_world_mask) is bool or isinstance(active_world_mask, np.bool_):
        return bool(active_world_mask)
    if isinstance(active_world_mask, np.ndarray):
        if active_world_mask.shape != (1,):
            raise ValueError("CPU solver active world mask must have shape (1,)")
        if active_world_mask.dtype != np.dtype(np.bool_):
            raise TypeError("CPU solver active world mask must use NumPy bool")
        return bool(active_world_mask[0])

    # Warp CPU mask 只在调用方确实传入 Warp array 时才触发可选依赖 import；普通 Mirror
    # CPU 使用 bool/NumPy 时不会初始化 Warp runtime。禁止 CUDA mask 隐式搬回 host。
    try:
        import warp as wp
    except ModuleNotFoundError as exc:
        raise TypeError(
            "CPU solver active world mask must be bool, NumPy bool[1], or Warp CPU bool[1]"
        ) from exc
    if not isinstance(active_world_mask, wp.array):
        raise TypeError(
            "CPU solver active world mask must be bool, NumPy bool[1], or Warp CPU bool[1]"
        )
    if tuple(active_world_mask.shape) != (1,):
        raise ValueError("CPU solver active world mask must have shape (1,)")
    if active_world_mask.dtype != wp.bool:
        raise TypeError("CPU Warp active world mask must use Warp bool")
    if str(active_world_mask.device) != "cpu":
        raise ValueError("CPU solver active world mask cannot reside on CUDA")
    return bool(np.asarray(active_world_mask.numpy(), dtype=np.bool_)[0])


__all__ = [
    "CpuSolverIntegrationStateStore",
    "CudaSolverIntegrationStateStore",
    "SolverIntegrationStateStore",
    "create_solver_integration_state_store",
]
