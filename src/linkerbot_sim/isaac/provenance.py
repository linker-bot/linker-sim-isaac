"""Isaac 启动后的依赖来源采集与目标版本校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any

from linkerbot_sim.isaac.extensions import enumerate_enabled_kit_extensions
from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    active_physics_execution,
    normalize_physics_backend,
    normalize_physics_execution,
)


@dataclass(frozen=True)
class ModuleProvenance:
    """一个运行时模块的发行版版本与实际加载路径。"""

    distribution: str
    version: str
    module: str
    path: str | None
    module_version: str | None = None


@dataclass(frozen=True)
class KitExtensionProvenance:
    """Kit 扩展管理器实际启用的扩展版本与解析路径。"""

    name: str
    version: str
    path: str | None


@dataclass(frozen=True)
class PhysicsEngineProvenance:
    """Unified physics registry 中一个后端的注册与 active 状态。"""

    name: str
    active: bool


@dataclass(frozen=True)
class RuntimeProvenance:
    """影响 Isaac、Warp 与 cuRobo ABI 的运行时信息。"""

    python: str
    executable: str
    platform: str
    physics_backend: str
    physics_execution: str
    physics_engines: tuple[PhysicsEngineProvenance, ...]
    isaacsim: ModuleProvenance
    torch: ModuleProvenance
    warp: ModuleProvenance
    newton: ModuleProvenance | None
    mujoco_warp: ModuleProvenance | None
    pxr: ModuleProvenance
    torch_cuda: str | None
    cuda_available: bool
    cuda_device: int
    cuda_device_name: str | None
    cuda_device_capability: tuple[int, int] | None
    nvidia_driver: str | None
    usd_core_installed: bool
    kit_extensions: tuple[KitExtensionProvenance, ...]
    curobo: ModuleProvenance | None = None
    curobo_backend: str | None = None
    curobo_commit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回可序列化且字段稳定的诊断映射。"""

        return asdict(self)


def collect_runtime_provenance(
    *,
    cuda_device: int,
    include_curobo: bool = False,
    physics_execution: str | None = None,
) -> RuntimeProvenance:
    """在 SimulationApp 启动后采集实际依赖闭包。

    PhysX closure 不 import ``newton``/``mujoco_warp``，避免可选 wheel 缺失导致
    PhysX 启动失败；只有 Newton profile 才采集两者的真实来源和版本。
    """

    if type(cuda_device) is not int or cuda_device < 0:
        raise ValueError("cuda_device must be a non-negative integer")

    torch = importlib.import_module("torch")
    warp = importlib.import_module("warp")
    pxr_usd = importlib.import_module("pxr.Usd")
    backend = active_physics_backend()
    execution = (
        active_physics_execution()
        if physics_execution is None
        else normalize_physics_execution(physics_execution)
    )
    newton_info = None
    mujoco_warp_info = None
    if backend == "newton":
        newton = importlib.import_module("newton")
        mujoco_warp = importlib.import_module("mujoco_warp")
        newton_info = _module_provenance("newton", "newton", newton)
        mujoco_warp_info = _module_provenance("mujoco-warp", "mujoco_warp", mujoco_warp)
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(
        callable(getattr(cuda, "is_available", None)) and cuda.is_available()
    )
    device_name = None
    device_capability = None
    if cuda_available:
        device_name = str(cuda.get_device_name(cuda_device))
        device_capability = tuple(
            int(value) for value in cuda.get_device_capability(cuda_device)
        )

    curobo_info = None
    curobo_backend = None
    if include_curobo:
        from linkerbot_sim.backends.curobo.runtime_imports import (
            import_curobo_module,
            require_curobo_kernel_backend,
        )

        curobo = import_curobo_module()
        curobo_info = _module_provenance("nvidia-curobo", "curobo", curobo)
        curobo_backend = require_curobo_kernel_backend("cuda_core")
        curobo_commit = _distribution_vcs_commit("nvidia-curobo")
    else:
        curobo_commit = None

    return RuntimeProvenance(
        python=platform.python_version(),
        executable=str(Path(sys.executable).resolve()),
        platform=platform.platform(),
        physics_backend=backend,
        physics_execution=execution,
        # Newton Kit 明确排除了 SimulationManager；尝试 import 它本身会污染待证明闭包。
        physics_engines=(() if backend == "newton" else _physics_engine_provenance()),
        isaacsim=_module_provenance("isaacsim", "isaacsim"),
        torch=_module_provenance("torch", "torch", torch),
        warp=_module_provenance("warp-lang", "warp", warp),
        newton=newton_info,
        mujoco_warp=mujoco_warp_info,
        pxr=_module_provenance("isaacsim-kernel", "pxr.Usd", pxr_usd),
        torch_cuda=getattr(getattr(torch, "version", None), "cuda", None),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        cuda_device_name=device_name,
        cuda_device_capability=device_capability,
        nvidia_driver=_nvidia_driver_version(),
        usd_core_installed=_distribution_installed("usd-core"),
        kit_extensions=_kit_extension_provenance(),
        curobo=curobo_info,
        curobo_backend=curobo_backend,
        curobo_commit=curobo_commit,
    )


def validate_target_runtime(
    provenance: RuntimeProvenance,
    *,
    require_curobo: bool = False,
    expected_physics_backend: object = "physx",
    physics_execution: object = "cpu",
    experience_family: str = "mirror",
    rendering_required: bool = False,
) -> None:
    """严格校验七个正式 Kit 的 Isaac 6.0.1 运行时闭包。"""

    expected_backend = normalize_physics_backend(expected_physics_backend)
    execution = normalize_physics_execution(physics_execution)
    family = str(experience_family).strip().lower()
    if type(rendering_required) is not bool:
        raise TypeError("rendering_required must be boolean")
    if family not in {"mirror", "kaleidoscope"}:
        raise ValueError(f"unsupported experience family {experience_family!r}")
    newton_runtime = expected_backend == "newton"
    expected = {
        "python": (provenance.python, "3.12."),
        "isaacsim": (provenance.isaacsim.version, "6.0.1.0"),
        "torch": (provenance.torch.version, "2.11.0"),
        "warp": (provenance.warp.version, "1.13.0"),
        "warp.__version__": (provenance.warp.module_version, "1.13.0"),
        "torch_cuda": (provenance.torch_cuda, "12.8"),
    }
    if expected_backend == "newton":
        if provenance.newton is None:
            expected["newton"] = (None, "1.2.1")
            expected["newton.__version__"] = (None, "1.2.1")
        else:
            expected["newton"] = (provenance.newton.version, "1.2.1")
            expected["newton.__version__"] = (
                provenance.newton.module_version,
                "1.2.1",
            )
        if provenance.mujoco_warp is None:
            expected["mujoco-warp"] = (None, "3.8.0.3")
            expected["mujoco_warp.__version__"] = (None, "3.8.0.3")
        else:
            expected["mujoco-warp"] = (
                provenance.mujoco_warp.version,
                "3.8.0.3",
            )
            expected["mujoco_warp.__version__"] = (
                provenance.mujoco_warp.module_version,
                "3.8.0.3",
            )
    if require_curobo:
        if provenance.curobo is None:
            raise RuntimeError("cuRobo provenance is required but was not collected")
        expected["curobo"] = (provenance.curobo.version, "0.8.0")
    mismatches = [
        f"{name}={actual!r} expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if not str(actual).startswith(wanted)
    ]
    if require_curobo and provenance.curobo_backend != "cuda_core":
        mismatches.append(
            f"curobo_backend={provenance.curobo_backend!r} expected 'cuda_core'"
        )
    if require_curobo and provenance.curobo_commit != (
        "4ea77366ca48ee453e7df139e39fa6532af49f3b"
    ):
        mismatches.append(
            f"curobo_commit={provenance.curobo_commit!r} expected "
            "'4ea77366ca48ee453e7df139e39fa6532af49f3b'"
        )
    if provenance.physics_backend != expected_backend:
        mismatches.append(
            f"physics_backend={provenance.physics_backend!r} "
            f"expected {expected_backend!r}"
        )
    if provenance.physics_execution != execution:
        mismatches.append(
            f"physics_execution={provenance.physics_execution!r} expected {execution!r}"
        )
    if not provenance.cuda_available:
        mismatches.append("torch.cuda.is_available() is false")
    if provenance.nvidia_driver is None:
        mismatches.append("NVIDIA driver version is unavailable")
    # The dev extra installs usd-core for offline tests.  It is safe in a Kit
    # process only when the actually imported pxr module is still Isaac's
    # bundled copy; reject the installed distribution if it shadows that path.
    if provenance.usd_core_installed and not _path_is_within(
        provenance.pxr.path,
        _module_package_root(provenance.isaacsim.path),
    ):
        mismatches.append(
            "usd-core must not be installed in the simulation environment "
            f"(loaded pxr={provenance.pxr.path!r})"
        )

    extensions = {item.name: item for item in provenance.kit_extensions}
    required_extensions = {
        "isaacsim.simulation_app",
        "isaacsim.asset.importer.urdf",
        "isaacsim.asset.importer.mjcf",
        "omni.warp.core",
    }
    if newton_runtime:
        required_extensions.update(
            {
                "omni.kit.loop-isaac",
                "omni.kit.usd.layers",
            }
        )
        if rendering_required:
            required_extensions.update(
                {
                    "omni.hydra.rtx",
                    "omni.kit.viewport.utility",
                    "omni.kit.viewport.window",
                }
            )
            if family == "mirror":
                required_extensions.update(
                    {
                        "omni.kit.manipulator.camera",
                        "omni.syntheticdata",
                        "omni.usd.schema.omni_lens_distortion",
                    }
                )
    elif family == "kaleidoscope":
        required_extensions.add("isaacsim.core.api")
        if rendering_required:
            required_extensions.update(
                {
                    "omni.hydra.rtx",
                    "omni.kit.viewport.utility",
                    "omni.kit.viewport.window",
                }
            )
    else:
        required_extensions.update(
            {
                "isaacsim.core.api",
                "isaacsim.sensors.experimental.rtx",
            }
        )
    if expected_backend == "physx":
        required_extensions.add("omni.physics.physx")
    missing_extensions = sorted(required_extensions - extensions.keys())
    if missing_extensions:
        mismatches.append("missing Kit extensions: " + ", ".join(missing_extensions))

    if family == "kaleidoscope":
        forbidden_exact = {
            "isaacsim.pip.newton",
            "isaacsim.sensors.experimental.rtx",
        }
        forbidden_prefixes = [
            "isaacsim.physics.newton",
            "isaacsim.sensors.camera",
            "omni.replicator",
            "omni.syntheticdata",
        ]
        if not rendering_required:
            # ``isaacsim.core.api`` 在 Isaac Sim 6.0.1 中硬依赖
            # ``omni.hydra.usdrt_delegate``；它只是 USD Runtime delegate。无渲染闭包
            # 仍只拒绝真正的 RTX/viewport 实现，显式 viewport 闭包则允许这些命名空间。
            forbidden_prefixes.extend(
                (
                    "omni.hydra.rtx",
                    "omni.hydra.pxr",
                    "omni.kit.viewport",
                    "omni.kit.renderer",
                    "omni.renderer",
                    "omni.rtx",
                )
            )
        forbidden_kaleidoscope = sorted(
            name
            for name in extensions
            if name in forbidden_exact
            or any(name.startswith(prefix) for prefix in forbidden_prefixes)
        )
        if forbidden_kaleidoscope:
            mismatches.append(
                "Kaleidoscope has forbidden enabled extensions: "
                + ", ".join(forbidden_kaleidoscope)
            )

    registry_engines = sorted(item.name for item in provenance.physics_engines)
    active_engines = sorted(
        item.name for item in provenance.physics_engines if item.active
    )
    # 项目 Newton experience 的要求不是“没有 active engine”这么弱，而是 registry 必须为空。
    # 某个 physics extension 即使当前 inactive，也可能已经安装 stage-update callback、持有
    # SimulationManager 状态，或在下一次 timeline play 时自动接管。允许这种 entry 会把
    # 项目 Newton owner 与 Isaac owner 的互斥会变成启动时序竞态，因此按全部 entry fail closed。
    if newton_runtime and registry_engines:
        mismatches.append(
            f"Newton physics registry engines={registry_engines!r} expected []"
        )
    elif not newton_runtime and active_engines != [expected_backend]:
        mismatches.append(
            f"active physics registry engines={active_engines!r} "
            f"expected [{expected_backend!r}]"
        )
    expected_extension_versions = {"omni.warp.core": "1.13.0"}
    if newton_runtime:
        expected_extension_versions["omni.kit.usd.layers"] = "2.6.1"
        forbidden_exact = {
            "isaacsim.core.api",
            "isaacsim.core.cloner",
            "isaacsim.core.experimental.prims",
            "isaacsim.core.simulation_manager",
            "isaacsim.core.utils",
        }
        forbidden_prefixes = (
            "isaacsim.physics.",
            "isaacsim.sensors.physx",
            "omni.physics.",
            "omni.physx",
        )
        forbidden = sorted(
            name
            for name in extensions
            if name in forbidden_exact
            or any(name.startswith(prefix) for prefix in forbidden_prefixes)
        )
        if forbidden:
            # registry 审计与 extension closure 审计缺一不可：有些 physics-owner extension
            # 尚未向 SimulationManager 注册 engine，却已经加载了原生插件或 stage callback。
            mismatches.append(
                "Newton has forbidden enabled extensions: " + ", ".join(forbidden)
            )
        if not rendering_required:
            renderer_prefixes = (
                "omni.hydra.rtx",
                "omni.kit.viewport",
                "omni.syntheticdata",
            )
            unexpected_renderers = sorted(
                name
                for name in extensions
                if any(name.startswith(prefix) for prefix in renderer_prefixes)
            )
            if unexpected_renderers:
                mismatches.append(
                    "Newton physics-only Kit has renderer extensions: "
                    + ", ".join(unexpected_renderers)
                )
    for name, expected_version in expected_extension_versions.items():
        extension = extensions.get(name)
        if extension is not None and not extension.version.startswith(expected_version):
            mismatches.append(
                f"{name}={extension.version!r} expected {expected_version!r}"
            )

    warp_extension = extensions.get("omni.warp.core")
    if warp_extension is not None and not _path_is_within(
        provenance.warp.path, warp_extension.path
    ):
        mismatches.append(
            "warp module path is outside enabled omni.warp.core extension: "
            f"module={provenance.warp.path!r}, extension={warp_extension.path!r}"
        )

    if expected_backend == "newton":
        for name, module in (
            ("newton", provenance.newton),
            ("mujoco_warp", provenance.mujoco_warp),
        ):
            if module is None or module.path is None:
                mismatches.append(f"{name} module path is unavailable")

    if mismatches:
        raise RuntimeError(
            "runtime dependency validation failed: " + "; ".join(mismatches)
        )


def format_runtime_provenance(provenance: RuntimeProvenance) -> str:
    """生成单行 JSON，便于启动日志和故障报告采集。"""

    return json.dumps(provenance.as_dict(), ensure_ascii=True, sort_keys=True)


def _module_provenance(
    distribution: str,
    module_name: str,
    module: object | None = None,
) -> ModuleProvenance:
    """读取发行版版本和模块实际文件；缺少 metadata 时保留 unknown。"""

    imported = importlib.import_module(module_name) if module is None else module
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    module_file = getattr(imported, "__file__", None)
    module_version = getattr(imported, "__version__", None)
    return ModuleProvenance(
        distribution=distribution,
        version=str(version),
        module=module_name,
        path=None if module_file is None else str(Path(module_file).resolve()),
        module_version=None if module_version is None else str(module_version),
    )


def _module_package_root(module_path: str | None) -> str | None:
    """由包 ``__init__`` 路径返回包目录；未知路径保持未知。"""

    if module_path is None:
        return None
    return str(Path(module_path).resolve().parent)


def _path_is_within(path: str | None, directory: str | None) -> bool:
    """仅当两个来源均可解析且模块路径属于给定目录时返回真。"""

    if path is None or directory is None:
        return False
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _distribution_installed(distribution: str) -> bool:
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _physics_engine_provenance() -> tuple[PhysicsEngineProvenance, ...]:
    """读取 unified physics registry；缺失接口交给严格校验统一报告。"""

    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except (ImportError, ModuleNotFoundError):
        return ()
    getter = getattr(SimulationManager, "get_available_physics_engines", None)
    if not callable(getter):
        return ()
    engines = (
        PhysicsEngineProvenance(name=str(name).strip().lower(), active=bool(active))
        for name, active in getter()
    )
    return tuple(sorted(engines, key=lambda item: item.name))


def _distribution_vcs_commit(distribution: str) -> str | None:
    """从 PEP 610 direct_url 元数据读取不可变 VCS commit。"""

    try:
        metadata = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    content = metadata.read_text("direct_url.json")
    if content is None:
        return None
    try:
        direct_url = json.loads(content)
    except json.JSONDecodeError:
        return None
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return None if commit is None else str(commit)


def _nvidia_driver_version(
    version_path: Path = Path("/proc/driver/nvidia/version"),
) -> str | None:
    """读取 Linux NVIDIA kernel module 版本，不依赖 nvidia-smi 子进程。"""

    try:
        first_line = version_path.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, OSError, UnicodeError):
        return None
    match = re.search(r"\b[0-9]{3,}\.[0-9]+(?:\.[0-9]+)?\b", first_line)
    return None if match is None else match.group(0)


def _kit_extension_provenance() -> tuple[KitExtensionProvenance, ...]:
    """采集完整 enabled closure；无法枚举时让启动校验 fail closed。"""

    return tuple(
        KitExtensionProvenance(
            name=item.name,
            version=item.version,
            path=item.path,
        )
        for item in enumerate_enabled_kit_extensions()
    )


__all__ = [
    "KitExtensionProvenance",
    "ModuleProvenance",
    "PhysicsEngineProvenance",
    "RuntimeProvenance",
    "collect_runtime_provenance",
    "format_runtime_provenance",
    "validate_target_runtime",
]
