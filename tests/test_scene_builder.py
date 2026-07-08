from __future__ import annotations

import sys
from types import ModuleType

from linkerbot_sim.envs.scene_builder import build_world


def test_build_world_passes_ground_height_to_default_ground(monkeypatch) -> None:
    """World 构建不启动 Isaac，也应能验证默认地面高度的参数传递。"""

    class FakePhysicsContext:
        def __init__(self) -> None:
            self.gravity: float | None = None

        def set_gravity(self, value: float) -> None:
            self.gravity = value

    class FakeScene:
        def __init__(self) -> None:
            self.default_ground_plane_kwargs: dict[str, float] | None = None

        def add_default_ground_plane(self, **kwargs: float) -> None:
            self.default_ground_plane_kwargs = kwargs

    class FakeWorld:
        def __init__(
            self,
            *,
            stage_units_in_meters: float,
            physics_dt: float | None,
            rendering_dt: float | None,
        ) -> None:
            self.stage_units_in_meters = stage_units_in_meters
            self.physics_dt = physics_dt
            self.rendering_dt = rendering_dt
            self.scene = FakeScene()
            self.physics_context = FakePhysicsContext()

        def get_physics_context(self) -> FakePhysicsContext:
            return self.physics_context

    world_module = ModuleType("isaacsim.core.api.world")
    setattr(world_module, "World", FakeWorld)

    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", ModuleType("isaacsim.core.api"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.api.world", world_module)

    world = build_world(
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        add_ground=True,
        ground_height=0.12,
    )

    assert world.stage_units_in_meters == 1.0
    assert world.physics_dt == 1.0 / 240.0
    assert world.rendering_dt == 1.0 / 60.0
    assert world.physics_context.gravity == -9.81
    assert world.scene.default_ground_plane_kwargs == {"z_position": 0.12}
