from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "linkerbot_sim.app.interactive.scene",
        "linkerbot_sim.app.interactive.tiled",
        "linkerbot_sim.app.runtime.scene_runtime",
        "linkerbot_sim.app.runtime.reset",
        "linkerbot_sim.snapshots.scene_adapter",
        "linkerbot_sim.snapshots.tiled_adapter",
        "linkerbot_sim.snapshots.debug_tiled_adapter",
    ),
)
def test_unsupported_runtime_module_names_are_not_importable(
    module_name: str,
) -> None:
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module(module_name)

    assert exc_info.value.name == module_name


@pytest.mark.parametrize(
    ("module_name", "unsupported_symbols"),
    (
        (
            "linkerbot_sim.app.runtime.single_scene_runtime",
            ("SceneRuntime", "create_scene_runtime"),
        ),
        (
            "linkerbot_sim.app.interactive.single_scene.cli",
            ("create_scene_runtime", "run_interactive_scene_motion"),
        ),
        (
            "linkerbot_sim.app.interactive.tiled_scene.runtime.core",
            ("IsaacTiledInteractiveRuntime",),
        ),
        (
            "linkerbot_sim.app.interactive.tiled_scene.cli",
            ("create_tiled_runtime",),
        ),
        (
            "linkerbot_sim.snapshots",
            (
                "get_scene_snapshot",
                "set_scene_snapshot",
                "get_tiled_snapshot",
                "set_tiled_snapshot",
            ),
        ),
    ),
)
def test_current_runtime_modules_do_not_export_unsupported_symbols(
    module_name: str,
    unsupported_symbols: tuple[str, ...],
) -> None:
    module = importlib.import_module(module_name)

    assert [name for name in unsupported_symbols if hasattr(module, name)] == []
