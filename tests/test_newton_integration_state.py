from __future__ import annotations

from enum import IntFlag
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.newton.integration_state import (
    CpuSolverIntegrationStateStore,
    CudaSolverIntegrationStateStore,
    SolverIntegrationStateStore,
    create_solver_integration_state_store,
)


class _StateBits(IntFlag):
    TIME = 1
    ACT = 8
    WARMSTART = 32


class _FakeWarpArray:
    def __init__(
        self,
        values: object,
        *,
        dtype: object,
        device: object,
    ) -> None:
        self.values = np.asarray(values).copy()
        self.dtype = dtype
        self.device = str(device)
        self.shape = self.values.shape

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise AssertionError("CUDA integration state must not stage through NumPy")
        return self.values.copy()


class _FakeScopedStream:
    def __init__(self, stream: object, **_kwargs: object) -> None:
        self.stream = stream

    def __enter__(self) -> _FakeScopedStream:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _install_fake_cuda_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, object, object]:
    float32 = object()
    bool_dtype = object()
    warp = ModuleType("warp")
    warp.float32 = float32
    warp.bool = bool_dtype
    warp.array = _FakeWarpArray
    warp.ScopedStream = _FakeScopedStream
    warp.zeros = lambda shape, *, dtype, device: _FakeWarpArray(
        np.zeros(shape), dtype=dtype, device=device
    )
    warp.ones = lambda shape, *, dtype, device: _FakeWarpArray(
        np.ones(shape, dtype=np.bool_), dtype=dtype, device=device
    )

    def copy(
        destination: _FakeWarpArray, source: _FakeWarpArray, **_kwargs: object
    ) -> None:
        np.copyto(destination.values, source.values)

    warp.copy = copy

    mujoco = ModuleType("mujoco")
    mujoco.mj_stateSize = lambda model, signature: (
        1 + model.na + model.nv if signature == 41 else 0
    )

    mujoco_warp = ModuleType("mujoco_warp")
    mujoco_warp.State = SimpleNamespace(
        TIME=_StateBits.TIME,
        ACT=_StateBits.ACT,
        WARMSTART=_StateBits.WARMSTART,
    )

    def get_state(
        _model: object,
        data: object,
        destination: _FakeWarpArray,
        signature: int,
        *,
        active: _FakeWarpArray,
    ) -> None:
        assert signature == 41
        selected = active.values.astype(np.bool_)
        destination.values[selected] = data.persistent[selected]

    def set_state(
        _model: object,
        data: object,
        source: _FakeWarpArray,
        signature: int,
        *,
        active: _FakeWarpArray,
    ) -> None:
        assert signature == 41
        selected = active.values.astype(np.bool_)
        data.persistent[selected] = source.values[selected]

    mujoco_warp.get_state = get_state
    mujoco_warp.set_state = set_state
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "mujoco_warp", mujoco_warp)
    return warp, float32, bool_dtype


def _install_fake_cpu_mujoco(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    mujoco = ModuleType("mujoco")
    mujoco.mjtState = SimpleNamespace(
        mjSTATE_TIME=_StateBits.TIME,
        mjSTATE_ACT=_StateBits.ACT,
        mjSTATE_WARMSTART=_StateBits.WARMSTART,
    )
    mujoco.mj_stateSize = lambda model, signature: (
        1 + model.na + model.nv if signature == 41 else 0
    )

    def get_state(
        model: object, data: object, values: np.ndarray, signature: int
    ) -> None:
        assert signature == 41
        values[:] = np.concatenate(
            (
                np.asarray([data.time], dtype=np.float64),
                np.asarray(data.act, dtype=np.float64),
                np.asarray(data.qacc_warmstart, dtype=np.float64),
            )
        )
        assert values.size == 1 + model.na + model.nv

    def set_state(
        model: object, data: object, values: np.ndarray, signature: int
    ) -> None:
        assert signature == 41
        data.time = float(values[0])
        data.act[:] = values[1 : 1 + model.na]
        data.qacc_warmstart[:] = values[1 + model.na :]

    mujoco.mj_getState = get_state
    mujoco.mj_setState = set_state
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    return mujoco


def _fake_cpu_solver() -> SimpleNamespace:
    model = SimpleNamespace(na=2, nv=3)
    data = SimpleNamespace(
        time=0.0,
        act=np.zeros(model.na, dtype=np.float64),
        qacc_warmstart=np.zeros(model.nv, dtype=np.float64),
        qpos=np.zeros(4, dtype=np.float64),
        qvel=np.zeros(model.nv, dtype=np.float64),
        ctrl=np.zeros(2, dtype=np.float64),
    )
    return SimpleNamespace(
        use_mujoco_cpu=True,
        update_data_interval=1,
        model=SimpleNamespace(world_count=1),
        mj_model=model,
        mj_data=data,
    )


def test_factory_exposes_one_device_neutral_protocol() -> None:
    cpu = create_solver_integration_state_store("cpu")
    cuda = create_solver_integration_state_store("cuda")

    assert isinstance(cpu, CpuSolverIntegrationStateStore)
    assert isinstance(cuda, CudaSolverIntegrationStateStore)
    assert isinstance(cpu, SolverIntegrationStateStore)
    assert isinstance(cuda, SolverIntegrationStateStore)
    assert cpu.execution == "cpu"
    assert cuda.execution == "cuda"
    with pytest.raises(ValueError, match="'cpu' or 'cuda'"):
        create_solver_integration_state_store("metal")  # type: ignore[arg-type]


def test_cuda_store_keeps_state_device_resident_and_restores_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warp, float32, bool_dtype = _install_fake_cuda_dependencies(monkeypatch)
    width = 6
    data = SimpleNamespace(
        nworld=2,
        persistent=np.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
            dtype=np.float32,
        ),
    )
    solver = SimpleNamespace(
        use_mujoco_cpu=False,
        update_data_interval=1,
        mj_model=SimpleNamespace(na=2, nv=3),
        mjw_model=object(),
        mjw_data=data,
    )
    store = CudaSolverIntegrationStateStore()
    owner_stream = object()
    store.initialize(
        solver,
        world_count=2,
        device="cuda:0",
        stream=owner_stream,
    )

    assert store.signature == 41
    assert store.width == width
    assert store.activation_width == 2
    saved = _FakeWarpArray(data.persistent, dtype=float32, device="cuda:0")
    data.persistent.fill(0.0)
    selected = _FakeWarpArray([False, True], dtype=bool_dtype, device="cuda:0")
    store.restore(saved, active_world_mask=selected)

    np.testing.assert_array_equal(data.persistent[0], np.zeros(width))
    np.testing.assert_array_equal(data.persistent[1], saved.values[1])
    canonical = store.borrow()
    assert isinstance(canonical, _FakeWarpArray)
    # 未选 engine row 保持零；selected get 同时不覆盖 canonical 中原有的第一行。
    np.testing.assert_array_equal(canonical.values[0], saved.values[0])
    np.testing.assert_array_equal(canonical.values[1], saved.values[1])

    committed = np.full((2, width), 17.0, dtype=np.float32)
    data.persistent[:] = committed
    store.commit()
    data.persistent.fill(-3.0)
    store.reset()
    np.testing.assert_array_equal(data.persistent, committed)
    assert warp is sys.modules["warp"]


def test_cuda_store_rejects_host_or_wrong_device_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _warp, float32, bool_dtype = _install_fake_cuda_dependencies(monkeypatch)
    solver = SimpleNamespace(
        use_mujoco_cpu=False,
        update_data_interval=1,
        mj_model=SimpleNamespace(na=0, nv=1),
        mjw_model=object(),
        mjw_data=SimpleNamespace(
            nworld=1,
            persistent=np.zeros((1, 2), dtype=np.float32),
        ),
    )
    store = CudaSolverIntegrationStateStore()
    store.initialize(solver, world_count=1, device="cuda:0", stream=object())

    with pytest.raises(TypeError, match="Warp float32"):
        store.validate(np.zeros((1, 2), dtype=np.float32))
    wrong_device = _FakeWarpArray(np.zeros((1, 2)), dtype=float32, device="cuda:1")
    with pytest.raises(ValueError, match="store device"):
        store.validate(wrong_device)
    wrong_mask = _FakeWarpArray([True], dtype=bool_dtype, device="cuda:1")
    with pytest.raises(ValueError, match="mask must live"):
        store.validate(store.borrow(), active_world_mask=wrong_mask)


def test_cpu_store_restores_only_persistent_fields_and_commits_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cpu_mujoco(monkeypatch)
    solver = _fake_cpu_solver()
    store = CpuSolverIntegrationStateStore()
    store.initialize(solver, world_count=1, device="cpu", stream=None)

    solver.mj_data.time = 1.25
    solver.mj_data.act[:] = [2.0, 3.0]
    solver.mj_data.qacc_warmstart[:] = [4.0, 5.0, 6.0]
    store.capture()
    saved = np.asarray(store.borrow()).copy()

    solver.mj_data.time = -1.0
    solver.mj_data.act.fill(-2.0)
    solver.mj_data.qacc_warmstart.fill(-3.0)
    # 这些字段由 Newton State/Control 拥有，persistent restore 绝不能触碰。
    solver.mj_data.qpos[:] = [10.0, 11.0, 12.0, 13.0]
    solver.mj_data.qvel[:] = [20.0, 21.0, 22.0]
    solver.mj_data.ctrl[:] = [30.0, 31.0]
    store.restore(saved, active_world_mask=np.asarray([True]))

    assert solver.mj_data.time == pytest.approx(1.25)
    np.testing.assert_array_equal(solver.mj_data.act, [2.0, 3.0])
    np.testing.assert_array_equal(solver.mj_data.qacc_warmstart, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(solver.mj_data.qpos, [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(solver.mj_data.qvel, [20.0, 21.0, 22.0])
    np.testing.assert_array_equal(solver.mj_data.ctrl, [30.0, 31.0])

    store.commit()
    solver.mj_data.time = 99.0
    solver.mj_data.act.fill(99.0)
    solver.mj_data.qacc_warmstart.fill(99.0)
    store.reset()
    np.testing.assert_array_equal(np.asarray(store.borrow()), saved)


def test_cpu_false_selector_is_a_valid_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cpu_mujoco(monkeypatch)
    solver = _fake_cpu_solver()
    store = CpuSolverIntegrationStateStore()
    store.initialize(solver, world_count=1, device="cpu")
    values = np.asarray(store.borrow()).copy()
    solver.mj_data.time = 7.0

    store.restore(values, active_world_mask=False)
    assert solver.mj_data.time == pytest.approx(7.0)
    store.restore(values, active_world_mask=np.asarray([False]))
    assert solver.mj_data.time == pytest.approx(7.0)


def test_cpu_accepts_only_single_world_host_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cpu_mujoco(monkeypatch)
    solver = _fake_cpu_solver()
    store = CpuSolverIntegrationStateStore()
    store.initialize(solver, world_count=1, device="cpu")
    state = store.borrow()

    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        store.validate(state, active_world_mask=np.asarray([True, False]))
    with pytest.raises(TypeError, match="NumPy bool"):
        store.validate(state, active_world_mask=np.asarray([1], dtype=np.int64))
    with pytest.raises(TypeError, match="float64"):
        store.validate(np.zeros((1, store.width), dtype=np.float32))

    # Warp CPU selector 可以在 CPU 边界读取；CUDA selector 必须被明确拒绝，不能隐式下载。
    bool_dtype = object()
    warp = ModuleType("warp")
    warp.array = _FakeWarpArray
    warp.bool = bool_dtype
    monkeypatch.setitem(sys.modules, "warp", warp)
    store.validate(
        state,
        active_world_mask=_FakeWarpArray([True], dtype=bool_dtype, device="cpu"),
    )
    with pytest.raises(ValueError, match="cannot reside on CUDA"):
        store.validate(
            state,
            active_world_mask=_FakeWarpArray([True], dtype=bool_dtype, device="cuda:0"),
        )


@pytest.mark.parametrize(
    ("world_count", "device", "stream", "message"),
    (
        (2, "cpu", None, "exactly one world"),
        (1, "cuda:0", None, "device='cpu'"),
        (1, "cpu", object(), "stream=None"),
    ),
)
def test_cpu_store_fails_closed_before_allocating_invalid_runtime(
    world_count: int,
    device: str,
    stream: object | None,
    message: str,
) -> None:
    store = CpuSolverIntegrationStateStore()

    with pytest.raises(ValueError, match=message):
        store.initialize(
            _fake_cpu_solver(),
            world_count=world_count,
            device=device,
            stream=stream,
        )


def test_real_mujoco_cpu_state_round_trip_excludes_newton_owned_fields() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint name='joint' type='hinge'/>"
        "<geom type='sphere' size='0.1' mass='1'/></body></worldbody>"
        "<actuator><motor joint='joint'/></actuator></mujoco>"
    )
    data = mujoco.MjData(model)
    solver = SimpleNamespace(
        use_mujoco_cpu=True,
        update_data_interval=1,
        model=SimpleNamespace(world_count=1),
        mj_model=model,
        mj_data=data,
    )
    store = CpuSolverIntegrationStateStore()
    store.initialize(solver, world_count=1, device="cpu")

    data.time = 2.5
    data.qacc_warmstart[:] = 4.0
    store.capture()
    saved = np.asarray(store.borrow()).copy()
    data.time = 0.0
    data.qacc_warmstart[:] = 0.0
    data.qpos[:] = 7.0
    data.qvel[:] = 8.0
    data.ctrl[:] = 9.0
    store.restore(saved)

    assert data.time == pytest.approx(2.5)
    np.testing.assert_array_equal(data.qacc_warmstart, [4.0])
    np.testing.assert_array_equal(data.qpos, [7.0])
    np.testing.assert_array_equal(data.qvel, [8.0])
    np.testing.assert_array_equal(data.ctrl, [9.0])
