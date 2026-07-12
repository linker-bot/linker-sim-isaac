"""cuRobo 与 torch 的延迟导入、设备可用性检查。"""

from __future__ import annotations

import importlib

from linkerbot_sim.backends.curobo.warp_compat import (
    ensure_warp_func_module_keyword_compatible,
    ensure_warp_torch_namespace_compatible,
)


def import_curobo_module():
    """导入 cuRobo 顶层包，并把缺依赖转换为可执行诊断。"""

    ensure_warp_func_module_keyword_compatible()
    ensure_warp_torch_namespace_compatible()
    try:
        return importlib.import_module("curobo")
    except ModuleNotFoundError as exc:
        missing = exc.name or "curobo"
        if missing == "curobo":
            raise RuntimeError(
                "cuRobo is not installed. Sync the project simulation dependencies "
                "into .venv with 'uv sync --all-extras'."
            ) from exc
        raise RuntimeError(
            "cuRobo is installed but a runtime dependency is missing: "
            f"{missing!r}. Sync .venv from the project lockfile with "
            "'uv sync --all-extras'."
        ) from exc


def import_curobo_public(module_name: str):
    """延迟导入一个 cuRobo public module。"""

    ensure_warp_func_module_keyword_compatible()
    ensure_warp_torch_namespace_compatible()
    try:
        return importlib.import_module(f"curobo.{module_name}")
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        raise RuntimeError(
            f"failed to import cuRobo public module curobo.{module_name}: "
            f"missing dependency {missing!r}"
        ) from exc


def import_torch_module():
    """延迟导入 cuRobo tensor 边界所需的 torch。"""

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required by the cuRobo backend but is not installed"
        ) from exc


def ensure_torch_device_usable(torch_module: object, device: str) -> None:
    """在构建 context 前用一个 tensor kernel 验证目标 CUDA device。"""

    device_name = str(device).strip()
    if not device_name.startswith("cuda"):
        return
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise RuntimeError(
            f"cuRobo config requests {device_name!r}, but torch.cuda.is_available() "
            "is False. Install a CUDA-enabled PyTorch build before using cuRobo."
        )
    try:
        probe = torch_module.ones((1,), device=device_name)
        (probe + 1.0).detach().cpu()
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device_name)
    except Exception as exc:
        summary = _torch_cuda_device_summary(torch_module, device_name)
        raise RuntimeError(
            "cuRobo CUDA smoke test failed before context initialization on "
            f"{device_name!r}{summary}. This usually means the installed PyTorch/CUDA "
            "wheel does not contain kernels for the current GPU architecture; install "
            "a PyTorch build that supports this device before running cuRobo."
        ) from exc


def _torch_cuda_device_summary(torch_module: object, device: str) -> str:
    """尽力读取 CUDA device name/capability，供 smoke-test 错误消息使用。"""

    cuda = getattr(torch_module, "cuda", None)
    try:
        index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    except ValueError:
        index = 0
    parts: list[str] = []
    for label, getter in (
        ("name", getattr(cuda, "get_device_name", None)),
        ("capability", getattr(cuda, "get_device_capability", None)),
    ):
        if not callable(getter):
            continue
        try:
            parts.append(f"{label}={getter(index)}")
        except Exception:
            continue
    return "" if not parts else " (" + ", ".join(parts) + ")"


__all__ = [
    "ensure_torch_device_usable",
    "import_curobo_module",
    "import_curobo_public",
    "import_torch_module",
]
