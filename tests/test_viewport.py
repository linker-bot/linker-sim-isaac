from __future__ import annotations

import sys
from types import ModuleType

import pytest

from linkerbot_sim.configuration.scenes import ViewportSettings
from linkerbot_sim.visualization.viewport import (
    set_default_viewport_view,
    set_viewport_camera_navigation_enabled,
)


def test_set_default_viewport_view_passes_configured_camera_prim_path(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeVec3d(tuple):
        def __new__(cls, x: float, y: float, z: float):
            return super().__new__(cls, (x, y, z))

    viewport = object()
    utility_module = ModuleType("omni.kit.viewport.utility")
    utility_module.get_active_viewport = lambda: viewport
    camera_state_module = ModuleType("omni.kit.viewport.utility.camera_state")

    class FakeCameraState:
        def __init__(self, path: str, actual_viewport: object) -> None:
            calls.append({"camera_prim_path": path, "viewport": actual_viewport})

        def set_position_world(self, value: object, immediate: bool) -> None:
            calls[-1]["eye"] = value
            calls[-1]["eye_immediate"] = immediate

        def set_target_world(self, value: object, immediate: bool) -> None:
            calls[-1]["target"] = value
            calls[-1]["target_immediate"] = immediate

    camera_state_module.ViewportCameraState = FakeCameraState
    pxr_module = ModuleType("pxr")
    gf_module = ModuleType("pxr.Gf")
    gf_module.Vec3d = FakeVec3d
    pxr_module.Gf = gf_module

    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility_module)
    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility.camera_state",
        camera_state_module,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    set_default_viewport_view(
        ViewportSettings(
            eye=(2.0, -1.0, 1.2),
            target=(0.1, 0.0, 0.4),
            prim_path="/World/ConfiguredViewportCamera",
        )
    )

    assert calls == [
        {
            "viewport": viewport,
            "eye": (2.0, -1.0, 1.2),
            "eye_immediate": True,
            "target": (0.1, 0.0, 0.4),
            "target_immediate": True,
            "camera_prim_path": "/World/ConfiguredViewportCamera",
        }
    ]


def test_camera_navigation_is_toggled_on_only_the_requested_window() -> None:
    class FakeLayer:
        visible = True

    class FakeWindow:
        def __init__(self) -> None:
            self.layer = FakeLayer()
            self.lookups: list[tuple[str, str]] = []

        def _find_viewport_layer(self, name: str, category: str) -> object:
            self.lookups.append((name, category))
            return self.layer

    main = FakeWindow()
    sensor = FakeWindow()

    set_viewport_camera_navigation_enabled(sensor, enabled=False)

    assert main.layer.visible is True
    assert sensor.layer.visible is False
    assert sensor.lookups == [("Camera", "manipulator")]


def test_camera_navigation_rejects_a_window_without_camera_layer() -> None:
    class FakeWindow:
        def _find_viewport_layer(self, _name: str, _category: str) -> None:
            return None

    with pytest.raises(RuntimeError, match="manipulator layer is unavailable"):
        set_viewport_camera_navigation_enabled(FakeWindow(), enabled=False)
