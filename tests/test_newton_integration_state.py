from __future__ import annotations

from enum import IntFlag
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.newton.integration_state import (
    WarpSolverIntegrationStateStore,
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
    warp.ScopedDevice = _FakeScopedStream
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


def test_factory_exposes_one_device_neutral_protocol() -> None:
    cpu = create_solver_integration_state_store("cpu")
    cuda = create_solver_integration_state_store("cuda")

    assert isinstance(cpu, WarpSolverIntegrationStateStore)
    assert isinstance(cuda, WarpSolverIntegrationStateStore)
    assert isinstance(cpu, SolverIntegrationStateStore)
    assert isinstance(cuda, SolverIntegrationStateStore)
    assert cpu.execution == "cpu"
    assert cuda.execution == "cuda"
    with pytest.raises(ValueError, match="cpu or cuda"):
        create_solver_integration_state_store("metal")  # type: ignore[arg-type]


@pytest.mark.parametrize("execution", ["cpu", "cuda"])
def test_warp_store_keeps_state_device_resident_and_restores_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
    execution: str,
) -> None:
    warp, float32, bool_dtype = _install_fake_cuda_dependencies(monkeypatch)
    device = "cpu" if execution == "cpu" else "cuda:0"
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
    store = WarpSolverIntegrationStateStore(execution=execution)
    owner_stream = object() if execution == "cuda" else None
    store.initialize(
        solver,
        world_count=2,
        device=device,
        stream=owner_stream,
    )

    assert store.signature == 41
    assert store.width == width
    assert store.activation_width == 2
    saved = _FakeWarpArray(data.persistent, dtype=float32, device=device)
    data.persistent.fill(0.0)
    selected = _FakeWarpArray([False, True], dtype=bool_dtype, device=device)
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
    store = WarpSolverIntegrationStateStore(execution="cuda")
    store.initialize(solver, world_count=1, device="cuda:0", stream=object())

    with pytest.raises(TypeError, match="Warp float32"):
        store.validate(np.zeros((1, 2), dtype=np.float32))
    wrong_device = _FakeWarpArray(np.zeros((1, 2)), dtype=float32, device="cuda:1")
    with pytest.raises(ValueError, match="store device"):
        store.validate(wrong_device)
    wrong_mask = _FakeWarpArray([True], dtype=bool_dtype, device="cuda:1")
    with pytest.raises(ValueError, match="mask must live"):
        store.validate(store.borrow(), active_world_mask=wrong_mask)


@pytest.mark.parametrize("execution", ["cpu", "cuda"])
def test_real_warp_state_round_trip_excludes_newton_owned_fields(execution):
    wp = pytest.importorskip("warp")
    mujoco = pytest.importorskip("mujoco")
    mjw = pytest.importorskip("mujoco_warp")
    if execution == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    device = "cpu" if execution == "cpu" else "cuda:0"
    stream = wp.Stream(device) if execution == "cuda" else None
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint name='j'/><geom type='sphere' size='.1' mass='1'/>"
        "</body></worldbody><actuator><motor joint='j'/></actuator></mujoco>"
    )
    with wp.ScopedDevice(device):
        wm = mjw.put_model(model)
        wd = mjw.make_data(model, nworld=1, nconmax=4, njmax=8)
    solver = SimpleNamespace(
        use_mujoco_cpu=False,
        update_data_interval=1,
        mj_model=model,
        mjw_model=wm,
        mjw_data=wd,
    )
    store = create_solver_integration_state_store(execution)
    store.initialize(solver, world_count=1, device=device, stream=stream)
    with store._scope():
        wd.time.fill_(2.5)
        wd.qacc_warmstart.fill_(4.0)
        store.capture()
        saved = wp.clone(store.borrow())
        wd.time.zero_()
        wd.qacc_warmstart.zero_()
        wd.qpos.fill_(7.0)
        wd.qvel.fill_(8.0)
        wd.ctrl.fill_(9.0)
        store.restore(saved)
    if stream is not None:
        wp.synchronize_stream(stream)
    assert wd.time.numpy()[0] == pytest.approx(2.5)
    np.testing.assert_array_equal(wd.qacc_warmstart.numpy(), [[4.0]])
    np.testing.assert_array_equal(wd.qpos.numpy(), [[7.0]])
    np.testing.assert_array_equal(wd.qvel.numpy(), [[8.0]])
    np.testing.assert_array_equal(wd.ctrl.numpy(), [[9.0]])
