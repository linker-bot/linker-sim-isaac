from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

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
