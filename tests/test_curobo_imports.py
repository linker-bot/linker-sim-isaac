from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import linkerbot_sim.backends.curobo.context as context_module
import linkerbot_sim.backends.curobo.runtime_imports as runtime_imports
from linkerbot_sim.backends.curobo.config import CuroboConfig, CuroboRobotConfig
from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.backends.curobo.runtime_imports import ensure_torch_device_usable
from linkerbot_sim.backends.curobo.warp_compat import (
    ensure_warp_func_module_keyword_compatible,
    ensure_warp_torch_namespace_compatible,
)


def test_ensure_torch_device_usable_skips_cpu_device() -> None:
    ensure_torch_device_usable(SimpleNamespace(), "cpu")


def test_ensure_torch_device_usable_rejects_missing_cuda() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    try:
        ensure_torch_device_usable(torch, "cuda:0")
    except RuntimeError as exc:
        assert "torch.cuda.is_available()" in str(exc)
    else:
        raise AssertionError("missing CUDA was accepted")


def test_ensure_torch_device_usable_reports_kernel_smoke_failure() -> None:
    class _FakeTensor:
        def __add__(self, _value):
            raise RuntimeError("no kernel image is available for execution")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Fake GPU",
            get_device_capability=lambda _index: (12, 0),
        ),
        ones=lambda _shape, *, device: _FakeTensor(),
    )

    try:
        ensure_torch_device_usable(torch, "cuda:0")
    except RuntimeError as exc:
        message = str(exc)
        assert "CUDA smoke test failed" in message
        assert "Fake GPU" in message
        assert "(12, 0)" in message
    else:
        raise AssertionError("broken CUDA kernel path was accepted")


def test_ensure_torch_device_usable_accepts_cuda_smoke_success() -> None:
    class _FakeTensor:
        def __add__(self, _value):
            return self

        def detach(self):
            return self

        def cpu(self):
            return np.asarray([2.0])

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True, synchronize=lambda _device: None
        ),
        ones=lambda _shape, *, device: _FakeTensor(),
    )

    ensure_torch_device_usable(torch, "cuda:0")


def test_require_curobo_kernel_backend_accepts_cuda_core(monkeypatch) -> None:
    backends = ModuleType("curobo._src.curobolib.backends")
    backends.get_backend_name = lambda: "cuda_core"
    monkeypatch.setattr(
        runtime_imports.importlib,
        "import_module",
        lambda _name: backends,
    )

    assert runtime_imports.require_curobo_kernel_backend() == "cuda_core"


def test_require_curobo_kernel_backend_rejects_pybind(monkeypatch) -> None:
    backends = ModuleType("curobo._src.curobolib.backends")
    backends.get_backend_name = lambda: "pybind"
    monkeypatch.setattr(
        runtime_imports.importlib,
        "import_module",
        lambda _name: backends,
    )

    with pytest.raises(RuntimeError, match="backend is 'pybind'; expected 'cuda_core'"):
        runtime_imports.require_curobo_kernel_backend()


def test_require_curobo_kernel_backend_reports_backend_resolution_failure(
    monkeypatch,
) -> None:
    def fail_import(_name: str):
        raise ImportError("cuda.core is unavailable")

    monkeypatch.setattr(runtime_imports.importlib, "import_module", fail_import)

    with pytest.raises(
        RuntimeError,
        match="failed to resolve cuRobo kernel backend",
    ) as exc:
        runtime_imports.require_curobo_kernel_backend()

    assert isinstance(exc.value.__cause__, ImportError)


def test_curobo_context_requires_cuda_core_before_cuda_resource_creation(
    monkeypatch,
) -> None:
    calls: list[object] = []
    config = _fake_context_config(calls)
    monkeypatch.setattr(
        context_module,
        "import_curobo_module",
        lambda: SimpleNamespace(__version__="0.8.0"),
    )

    def reject_backend(*, expected: str) -> str:
        calls.append(("backend", expected))
        raise RuntimeError("cuRobo kernel backend is 'pybind'; expected 'cuda_core'")

    monkeypatch.setattr(context_module, "require_curobo_kernel_backend", reject_backend)
    monkeypatch.setattr(
        context_module,
        "materialize_curobo_config",
        lambda *_args, **_kwargs: calls.append("materialize"),
    )
    monkeypatch.setattr(
        context_module,
        "import_torch_module",
        lambda: calls.append("torch") or SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="backend is 'pybind'"):
        CuroboContext(config)

    assert calls == [("version", "0.8.0"), ("backend", "cuda_core")]


def test_curobo_context_records_cuda_core_and_allows_repeated_close(
    monkeypatch,
) -> None:
    calls: list[object] = []
    config = _fake_context_config(calls)
    torch = SimpleNamespace()
    monkeypatch.setattr(
        context_module,
        "import_curobo_module",
        lambda: SimpleNamespace(__version__="0.8.0"),
    )
    monkeypatch.setattr(
        context_module,
        "require_curobo_kernel_backend",
        lambda *, expected: calls.append(("backend", expected)) or "cuda_core",
    )
    monkeypatch.setattr(
        context_module,
        "materialize_curobo_config",
        lambda value, *, cache_root=None: value,
    )
    monkeypatch.setattr(context_module, "import_torch_module", lambda: torch)
    monkeypatch.setattr(
        context_module,
        "ensure_torch_device_usable",
        lambda module, device: calls.append(("device", module, device)),
    )
    monkeypatch.setattr(
        context_module,
        "import_curobo_public",
        lambda name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(CuroboContext, "_make_device_cfg", lambda self: object())
    monkeypatch.setattr(CuroboContext, "_make_kinematics", lambda self: object())

    context = CuroboContext(config)
    context.close()
    context.close()

    assert context.kernel_backend == "cuda_core"
    assert calls[:2] == [("version", "0.8.0"), ("backend", "cuda_core")]
    assert calls[2] == ("device", torch, "cuda:0")


def test_ensure_warp_func_module_keyword_compatible_patches_signature_without_keyword(
    monkeypatch,
) -> None:
    warp = _warp_module_without_module_keyword()
    monkeypatch.setitem(sys.modules, "warp", warp)

    ensure_warp_func_module_keyword_compatible()

    assert getattr(warp.func, "_linkerbot_accepts_module_keyword")

    def sample_func() -> None:
        pass

    registered = warp.func(sample_func, module="curobo.fake.kernel")
    assert registered.func is sample_func
    assert "sample_func" in warp._modules["curobo.fake.kernel"].functions

    unique_registered = warp.func(module="unique")(sample_func)
    assert unique_registered.func is sample_func
    assert unique_registered.module.name == "sample_func"


def test_ensure_warp_func_module_keyword_compatible_keeps_signature_with_keyword(
    monkeypatch,
) -> None:
    warp = ModuleType("warp")

    def decorator_factory_func(f=None, *, name=None, module=None):
        return (f, name, module)

    warp.func = decorator_factory_func
    monkeypatch.setitem(sys.modules, "warp", warp)

    ensure_warp_func_module_keyword_compatible()

    assert warp.func is decorator_factory_func


def test_ensure_warp_torch_namespace_compatible_maps_top_level_converter(
    monkeypatch,
) -> None:
    warp = ModuleType("warp")
    warp.device_from_torch = lambda device: f"warp:{device}"
    monkeypatch.setitem(sys.modules, "warp", warp)

    ensure_warp_torch_namespace_compatible()
    ensure_warp_torch_namespace_compatible()

    assert warp.torch.device_from_torch("cuda:0") == "warp:cuda:0"


def test_ensure_warp_torch_namespace_compatible_keeps_existing_namespace(
    monkeypatch,
) -> None:
    warp = ModuleType("warp")
    existing = SimpleNamespace(device_from_torch=lambda device: device)
    warp.torch = existing
    monkeypatch.setitem(sys.modules, "warp", warp)

    ensure_warp_torch_namespace_compatible()

    assert warp.torch is existing


def _warp_module_without_module_keyword() -> ModuleType:
    warp = ModuleType("warp")
    modules: dict[str, _FakeWarpModule] = {}

    def func_without_module_keyword(f=None, *, name=None):
        return (f, name)

    def get_module(name: str) -> "_FakeWarpModule":
        return modules.setdefault(name, _FakeWarpModule(name))

    warp.func = func_without_module_keyword
    warp.context = SimpleNamespace(
        Function=_FakeWarpFunction,
        Module=_FakeWarpModule,
        get_module=get_module,
    )
    warp.codegen = SimpleNamespace(
        make_full_qualified_name=lambda func: func.__name__,
    )
    warp._modules = modules
    return warp


def _fake_context_config(calls: list[object]) -> CuroboConfig:
    """构造真实 typed config，只替换本测试需要观察的 bundle version hook。"""

    return CuroboConfig(
        robot=CuroboRobotConfig.from_mapping(
            {
                "robot_config_path": "configs/robots/ar5v2_l.yaml",
                "default_tcp_frame": "tool",
            }
        ),
        task_bundle=SimpleNamespace(
            validate_curobo_version=lambda version: calls.append(("version", version))
        ),
    )


class _FakeWarpModule:
    def __init__(self, name: str | None, loader=None) -> None:
        self.name = name if name is not None else "None"
        self.loader = loader
        self.functions: dict[str, _FakeWarpFunction] = {}

    def register_function(
        self,
        func: "_FakeWarpFunction",
        scope_locals,
        skip_adding_overload: bool = False,
    ) -> None:
        del skip_adding_overload
        existing = scope_locals.get(func.func.__name__)
        self.functions[func.key] = (
            existing if isinstance(existing, _FakeWarpFunction) else func
        )


class _FakeWarpFunction:
    def __init__(
        self,
        *,
        func,
        key: str,
        namespace: str,
        module: _FakeWarpModule,
        value_func,
        scope_locals,
        doc: str,
    ) -> None:
        del namespace, value_func, doc
        self.func = func
        self.key = key
        self.module = module
        module.register_function(self, scope_locals)
