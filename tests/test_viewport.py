from __future__ import annotations

import sys
from types import ModuleType

from linkerbot_sim.envs.visual_settings import ViewportViewSettings
from linkerbot_sim.visualization.viewport import set_default_viewport_view


def test_set_default_viewport_view_passes_configured_camera_prim_path(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeVec3d(tuple):
        def __new__(cls, x: float, y: float, z: float):
            return super().__new__(cls, (x, y, z))

    viewports_module = ModuleType("isaacsim.core.utils.viewports")
    viewports_module.set_camera_view = lambda **kwargs: calls.append(kwargs)
    pxr_module = ModuleType("pxr")
    gf_module = ModuleType("pxr.Gf")
    gf_module.Vec3d = FakeVec3d
    pxr_module.Gf = gf_module

    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.utils", ModuleType("isaacsim.core.utils")
    )
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.viewports", viewports_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    set_default_viewport_view(
        ViewportViewSettings(
            eye=(2.0, -1.0, 1.2),
            target=(0.1, 0.0, 0.4),
            prim_path="/World/ConfiguredViewportCamera",
        )
    )

    assert calls == [
        {
            "eye": (2.0, -1.0, 1.2),
            "target": (0.1, 0.0, 0.4),
            "camera_prim_path": "/World/ConfiguredViewportCamera",
        }
    ]
