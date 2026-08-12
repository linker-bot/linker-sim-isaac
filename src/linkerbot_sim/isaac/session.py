"""SimulationApp、USD stage 与唯一物理 runtime 的 session 所有权边界。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import sys
import traceback

from linkerbot_sim.isaac.app import launch_simulation_app
from linkerbot_sim.isaac.lifecycle import (
    close_simulation_app,
    register_simulation_app_physics_runtime,
)
from linkerbot_sim.isaac.physics.core_api import (
    ExperimentalArticulationAction,
    create_single_articulation_core_view,
    use_experimental_core,
)
from linkerbot_sim.isaac.physics.factory import create_physics_runtime
from linkerbot_sim.isaac.physics.physx_pipeline import (
    configure_physx_fabric_outputs,
)
from linkerbot_sim.isaac.physics.backend import clear_runtime_physics_backend
from linkerbot_sim.isaac.physics.runtime import PhysicsRuntime
from linkerbot_sim.isaac.spec import (
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacSessionSpec,
)
from linkerbot_sim.isaac.world import build_physx_world


@dataclass
class IsaacSession:
    """持有一个 App、一个 stage 和一个精确物理 owner。

    公共结构故意不提供 ``world`` 或 ``physics_manager``。PhysX runtime 可以在自己的具体
    类型上拥有 Isaac ``World``，Newton 则直接拥有 Model/State/Control/Solver。
    产品层只通过 ``physics_runtime`` 窄合同步进与关闭，避免两个对象都声称拥有物理时间。
    """

    app: object
    stage: object
    physics_runtime: PhysicsRuntime
    articulation_action_type: object
    single_articulation_type: object
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self, *, exit_code: int = 0) -> None:
        """按 runtime → importer → App 顺序幂等关闭该 session 的精确资源。"""

        if self._closed:
            return
        close_simulation_app(
            self.app,
            exit_code=exit_code,
            physics_runtime=self.physics_runtime,
        )
        self._closed = True


def create_isaac_session_from_spec(
    *,
    spec: IsaacSessionSpec,
    app_launcher: Callable[[IsaacSessionSpec], object] = launch_simulation_app,
    physics_runtime_factory: Callable[..., PhysicsRuntime] = create_physics_runtime,
    world_builder: Callable[..., object] = build_physx_world,
    fabric_output_configurer: Callable[..., object] = configure_physx_fabric_outputs,
    app_closer: Callable[..., None] = close_simulation_app,
) -> IsaacSession:
    """从唯一纯规格执行 App → stage → physics owner 创建事务。

    该签名不接受产品配置、旧 runtime settings 或 backend 字符串。注入点只为纯测试和
    真实 Isaac smoke 服务，所有注入实现也必须消费同一个 ``spec``，因此不能重新形成一条
    参数含义不同的兼容启动链。Newton 在这里仅创建 runtime；产品导入完整资产后再
    调用 ``initialize_worlds`` finalize 与 execution 对应的 Newton model。
    """

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    app = app_launcher(spec)
    physics_runtime: PhysicsRuntime | None = None
    try:
        stage = _active_or_new_stage()
        articulation_action_type, single_articulation_type = _runtime_core_types(
            spec=spec
        )
        physics_runtime = physics_runtime_factory(
            app=app,
            stage=stage,
            spec=spec,
            world_builder=world_builder,
            fabric_output_configurer=fabric_output_configurer,
        )
        register_simulation_app_physics_runtime(app, physics_runtime)
        return IsaacSession(
            app=app,
            stage=stage,
            physics_runtime=physics_runtime,
            articulation_action_type=articulation_action_type,
            single_articulation_type=single_articulation_type,
        )
    except BaseException as exc:
        traceback.print_exception(exc)
        sys.stderr.flush()
        print(
            f"ISAAC_SESSION_CREATE_FAILED {type(exc).__name__}: {exc}",
            flush=True,
        )
        close_kwargs: dict[str, object] = {"exit_code": 1}
        if physics_runtime is not None:
            close_kwargs["physics_runtime"] = physics_runtime
        try:
            app_closer(app, **close_kwargs)
        except BaseException as cleanup_error:
            # 构造错误决定 session 为什么不可用；cleanup 错误只作为附加诊断，不能覆盖
            # importer/World 的真正失败原因，也不能误导上层重试策略。
            exc.add_note(
                "IsaacSession cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if physics_runtime is None and isinstance(
            spec.physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec)
        ):
            # App 启动已经取得一次 backend registration，但 runtime 尚未返回，因而没有
            # App/runtime binding 能在 lifecycle 中代为释放。计数式释放只归还本 App 的
            # 登记；若已有另一个同 execution session，它的登记仍然保留。
            try:
                clear_runtime_physics_backend(backend="newton")
            except BaseException as cleanup_error:
                exc.add_note(
                    "Newton backend registration cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise


def _active_or_new_stage() -> object:
    """取得活动 USD stage；context 尚未创建时只创建一次。"""

    import omni.usd

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        create_stage = getattr(context, "new_stage", None)
        if not callable(create_stage):
            raise RuntimeError("no active USD stage and context cannot create one")
        create_stage()
        stage = context.get_stage()
    if stage is None:
        raise RuntimeError("USD stage creation did not produce an active stage")
    return stage


def _runtime_core_types(*, spec: IsaacSessionSpec) -> tuple[object, object]:
    """在 App 启动后选择 articulation handle，避免纯模块 import Isaac Core。"""

    physics_backend = (
        "newton"
        if isinstance(spec.physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec))
        else "physx"
    )
    if use_experimental_core(physics_backend=physics_backend):
        return (
            ExperimentalArticulationAction,
            partial(
                create_single_articulation_core_view,
                physics_backend=physics_backend,
            ),
        )
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    return ArticulationAction, SingleArticulation


__all__ = ["IsaacSession", "create_isaac_session_from_spec"]
