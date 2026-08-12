"""Isaac ``World`` 所有权明确的 PhysX runtime。"""

from __future__ import annotations

from linkerbot_sim.isaac.physics.runtime import PhysicsCapabilities


class PhysxRuntime:
    """持有唯一 Isaac ``World`` 并提供统一物理生命周期。

    ``SimulationApp`` 仍负责最终 native plugin teardown；本对象负责在该时刻之前关闭业务
    入口，确保 session 之外没有第二个对象继续推进或读取 World。
    """

    backend = "physx"
    capabilities = PhysicsCapabilities(
        supports_multiple_worlds=True,
        rendering=True,
        dynamic_chain=True,
        selected_reset=True,
        cuda_graph=False,
    )

    def __init__(self, world: object, *, kind: str) -> None:
        if kind not in {"physx_cpu", "physx_cuda"}:
            raise ValueError("PhysxRuntime.kind must be 'physx_cpu' or 'physx_cuda'")
        self.world = world
        self.kind = kind
        self.execution = "cpu" if kind == "physx_cpu" else "cuda"
        self.scene = getattr(world, "scene", None)
        self.closed = False

    def reset(self) -> None:
        self._require_open()
        self.world.reset()

    def forward(self) -> None:
        self._require_open()
        callback = getattr(self.world, "forward", None)
        if callable(callback):
            callback()

    def step(self, *, render: bool = False) -> None:
        self._require_open()
        self.world.step(render=bool(render))

    def render(self) -> None:
        self._require_open()
        callback = getattr(self.world, "render", None)
        if callable(callback):
            callback()
            return
        step = getattr(self.world, "step", None)
        if not callable(step):
            raise RuntimeError("PhysX World does not expose a render-only API")
        try:
            step(render=True, step_sim=False)
        except TypeError as exc:
            raise RuntimeError(
                "PhysX World cannot render without advancing physics"
            ) from exc

    def pre_render(self) -> None:
        # PhysX/Fabric 在 World.render/step 内完成 physics-to-USD 同步；保留显式钩子只为
        # 统一生命周期，不额外执行 stage update。
        self._require_open()

    def close(self) -> None:
        """幂等关闭 Python 入口；native World 最终由 SimulationApp 销毁。"""

        self.closed = True

    def get_physics_dt(self) -> float:
        self._require_open()
        return float(self.world.get_physics_dt())

    def get_rendering_dt(self) -> float:
        self._require_open()
        getter = getattr(self.world, "get_rendering_dt", None)
        return self.get_physics_dt() if not callable(getter) else float(getter())

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("PhysX runtime is closed")


__all__ = ["PhysxRuntime"]
