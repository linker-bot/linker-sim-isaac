"""普通 SingleSceneRuntime 的 canonical 交互入口。"""

from linkerbot_sim.app.interactive.single_scene.cli import main

__all__ = ["main", "run_single_scene_interactive_motion"]


def __getattr__(name: str):
    if name != "run_single_scene_interactive_motion":
        raise AttributeError(name)
    from linkerbot_sim.app.interactive.single_scene.runtime import (
        run_single_scene_interactive_motion,
    )

    return run_single_scene_interactive_motion
