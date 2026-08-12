"""由 :class:`IsaacSessionSpec` 驱动的 Kit ``SimulationApp`` 启动边界。

本模块只负责进程级 Kit 资源：选择七个正式 experience 之一、构造 SimulationApp 参数、
验证唯一物理后端并采集运行时来源。场景、任务、相机 cadence 和产品 transport 都不属于
这里。所有 Isaac/Omni 导入保持在显式启动之后，使纯配置和接口测试不会隐式初始化 Kit。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import traceback

from linkerbot_sim.isaac.physics.backend import (
    clear_runtime_physics_backend,
    normalize_physics_backend,
    set_runtime_physics_backend,
)
from linkerbot_sim.isaac.physics.exclusivity import (
    validate_newton_exclusivity,
)
from linkerbot_sim.isaac.spec import (
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)


_EULA_ENV_VAR = "OMNI_KIT_ACCEPT_EULA"
_ACCEPTED_EULA_VALUES = frozenset({"y", "yes", "1"})
_APPS_ROOT = Path(__file__).resolve().parents[3] / "apps"


def _experience_path(spec: IsaacSessionSpec) -> Path:
    """返回严格规格对应的唯一正式 Kit 文件。

    selector 不接受 backend 字符串或默认 family，因此调用方无法组合出产品范围之外的
    legacy Newton、通用 PhysX 或错误渲染闭包。Mirror PhysX 共用一个含 RTX 资源的 Kit；
    Mirror/Kaleidoscope 的 Newton 与 Kaleidoscope PhysX 均根据 ``render.enabled``
    在物理闭包和显式 viewport 闭包之间精确选择。
    """

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    physics = spec.physics
    if spec.experience_family == "kaleidoscope":
        if isinstance(physics, IsaacNewtonCudaSpec):
            filename = (
                "linkerbot_sim.kaleidoscope.newton_viewport.python.kit"
                if spec.render.enabled
                else "linkerbot_sim.kaleidoscope.newton.python.kit"
            )
        else:
            filename = (
                "linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit"
                if spec.render.enabled
                else "linkerbot_sim.kaleidoscope.physx_cuda.python.kit"
            )
    elif isinstance(physics, IsaacPhysxCpuSpec):
        filename = "linkerbot_sim.mirror.physx.python.kit"
    elif isinstance(physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec)):
        filename = (
            "linkerbot_sim.mirror.newton_render.python.kit"
            if spec.render.enabled
            else "linkerbot_sim.mirror.newton.python.kit"
        )
    else:  # pragma: no cover - IsaacSessionSpec 已封闭该不可能分支。
        raise TypeError(
            "Mirror requires physx_cpu, newton_cpu, or newton_cuda; "
            f"got {type(physics).__name__}"
        )
    return _APPS_ROOT / filename


def _require_eula_acceptance() -> None:
    """要求部署环境显式接受 Kit EULA，项目不会代替用户写入该状态。"""

    value = os.environ.get(_EULA_ENV_VAR)
    if value is not None and value.lower() in _ACCEPTED_EULA_VALUES:
        return
    raise RuntimeError(
        "Isaac Sim EULA has not been accepted. Set "
        f"{_EULA_ENV_VAR}=Y (or YES/1) in the deployment environment before "
        "launching; linkerbot_sim will not set it automatically."
    )


def _kit_config(spec: IsaacSessionSpec) -> dict[str, object]:
    """把纯 session 规格投影为 SimulationApp 唯一支持的启动参数。

    ``active_gpu`` 与 ``physics_gpu`` 都从 session 根部唯一的 ``compute`` 选择派生。
    PhysX CPU 不使用 ``physics_gpu`` 做物理计算，但 RTX、cuRobo 和其余 CUDA 消费者仍
    使用同一张卡，因此不能把 CPU 物理错误映射为隐式 ``cuda:0``。
    """

    app = spec.app
    render = spec.render
    gui = app.gui
    cuda_device = spec.compute.cuda_device

    hide_ui = app.hide_ui
    if hide_ui is None:
        hide_ui = False if gui else None
    disable_viewport_updates = app.disable_viewport_updates
    if disable_viewport_updates is None:
        disable_viewport_updates = not gui
    fast_shutdown = app.fast_shutdown
    if fast_shutdown is None:
        fast_shutdown = not gui

    extra_args = [
        f"--/rtx/materialDb/syncLoads={str(app.material_sync_loads).lower()}",
        f"--/rtx/hydra/materialSyncLoads={str(app.hydra_material_sync_loads).lower()}",
    ]
    if not gui:
        # viewport.window 即使在 --no-window 下也会默认创建一个无主交互 viewport；它会与
        # Newton SyntheticData camera 各自持有一份 Hydra product。headless session 不需要
        # 交互视角，只保留相机显式拥有并负责销毁的 window；GUI session 继续使用扩展默认
        # viewport。USD 0.25.11 首次 scene population 的 _MarkInstancerDirty 内部告警与此
        # 资源去重无关，不能在这里通过多建或提前更新 viewport 规避。
        extra_args.insert(
            0,
            "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
        )
    if not gui and app.hide_ui is None:
        extra_args.insert(0, "--/app/window/hideUi=1")

    return {
        "headless": not gui,
        "hide_ui": hide_ui,
        "disable_viewport_updates": disable_viewport_updates,
        "fast_shutdown": fast_shutdown,
        "multi_gpu": False,
        "max_gpu_count": max(1, cuda_device + 1),
        "active_gpu": cuda_device,
        "physics_gpu": cuda_device,
        "width": render.width,
        "height": render.height,
        "window_width": render.window_width,
        "window_height": render.window_height,
        "renderer": render.renderer,
        "anti_aliasing": render.anti_aliasing,
        "samples_per_pixel_per_frame": render.samples_per_pixel_per_frame,
        "denoiser": render.denoiser,
        "extra_args": extra_args,
    }


def _configure_and_validate_physics_backend(spec: IsaacSessionSpec) -> None:
    """在任何 model/``World`` 创建前验证并登记唯一物理 owner。"""

    physics = spec.physics
    if isinstance(physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec)):
        # Newton Kit 不加载 SimulationManager。先对实际 enabled extension closure 做排他
        # 审计，再登记仅用于 API dispatch 的 backend；真正物理 owner 由 session factory
        # 随后创建，启动失败时 launch 负责撤销此登记。
        validate_newton_exclusivity(phase="startup")
        set_runtime_physics_backend("newton", execution=spec.physics_execution)
        return

    from isaacsim.core.simulation_manager import SimulationManager

    actual = normalize_physics_backend(SimulationManager.get_active_physics_engine())
    if actual != "physx":
        raise RuntimeError(
            f"Isaac physics backend mismatch: configured='physx', active={actual!r}"
        )
    assert isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec))
    SimulationManager.set_device(spec.physics_device)
    actual_device = str(SimulationManager.get_physics_sim_device())
    if actual_device != spec.physics_device:
        raise RuntimeError(
            "PhysX physics device mismatch before World creation: "
            f"configured={spec.physics_device!r}, active={actual_device!r}"
        )


def _close_failed_launch(app: object, *, exit_code: int = 1) -> None:
    """释放尚未交给 :class:`IsaacSession` 的原生 App。"""

    close = getattr(app, "close", None)
    if not callable(close):
        return
    try:
        try:
            supports_exit_code = "exit_code" in inspect.signature(close).parameters
        except (TypeError, ValueError):
            supports_exit_code = False
        if supports_exit_code:
            close(exit_code=int(exit_code))
        else:
            close()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def _physics_only_simulation_app_type(simulation_app_type: type) -> type:
    """返回不会探测未加载 viewport extension 的 SimulationApp 变体。"""

    class PhysicsOnlySimulationApp(simulation_app_type):
        def _wait_for_viewport(self) -> None:
            return None

    PhysicsOnlySimulationApp.__name__ = "PhysicsOnlySimulationApp"
    return PhysicsOnlySimulationApp


def launch_simulation_app(spec: IsaacSessionSpec):
    """按严格 session 规格启动唯一 Kit experience 并验证运行时闭包。"""

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    _require_eula_acceptance()

    # 必须延迟导入：这些包在 import 时会初始化 Kit 插件状态，任何纯 Python 模块都不应
    # 通过类型注解或 facade 间接触发它们。
    import isaacsim  # noqa: F401
    from isaacsim.simulation_app import SimulationApp

    experience_path = _experience_path(spec)
    if not experience_path.is_file():
        raise RuntimeError(f"Isaac Sim experience not found: {experience_path}")
    newton_runtime = isinstance(
        spec.physics,
        (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec),
    )
    app_config = _kit_config(spec)
    if newton_runtime and spec.render.enabled:
        # Newton render/viewport experience 需要 viewport update。无渲染 Mirror 与
        # Kaleidoscope Newton Kit 都不含 viewport，必须保留 strict app spec 中的禁用值。
        app_config["disable_viewport_updates"] = False
    simulation_app_type = SimulationApp
    if (newton_runtime and not spec.render.enabled) or (
        spec.experience_family == "kaleidoscope" and not spec.render.enabled
    ):
        simulation_app_type = _physics_only_simulation_app_type(SimulationApp)
    app = simulation_app_type(app_config, experience=str(experience_path))
    backend_registered = False
    try:
        _configure_and_validate_physics_backend(spec)
        backend_registered = newton_runtime
        from linkerbot_sim.isaac.provenance import (
            collect_runtime_provenance,
            format_runtime_provenance,
            validate_target_runtime,
        )

        physics_execution = spec.physics_execution
        provenance = collect_runtime_provenance(
            cuda_device=spec.compute.cuda_device,
            include_curobo=False,
            physics_execution=physics_execution,
        )
        validate_target_runtime(
            provenance,
            expected_physics_backend=("newton" if newton_runtime else "physx"),
            physics_execution=physics_execution,
            experience_family=spec.experience_family,
            rendering_required=spec.render.enabled,
        )
        if os.environ.get("LINKERBOT_RUNTIME_PROVENANCE", "1") != "0":
            print(
                "LINKERBOT_RUNTIME_PROVENANCE " + format_runtime_provenance(provenance),
                flush=True,
            )
    except BaseException as launch_error:
        # `_configure_and_validate_physics_backend` 可能在登记前拒绝另一种 execution；只释放
        # 当前 App 真正取得的登记，不能误清仍存活 session 的 Newton backend。
        if backend_registered:
            clear_runtime_physics_backend(backend="newton")
        traceback.print_exception(launch_error)
        sys.stderr.flush()
        print(
            f"SIMULATION_APP_LAUNCH_FAILED {type(launch_error).__name__}: "
            f"{launch_error}",
            flush=True,
        )
        try:
            _close_failed_launch(app, exit_code=1)
        except BaseException as cleanup_error:
            launch_error.add_note(
                "SimulationApp cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    return app


__all__ = ["launch_simulation_app"]
