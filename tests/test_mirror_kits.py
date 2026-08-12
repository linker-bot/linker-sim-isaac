from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _kit(name: str) -> dict[str, object]:
    path = REPO_ROOT / "apps" / name
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_mirror_has_three_backend_render_specific_experiences() -> None:
    physx = _kit("linkerbot_sim.mirror.physx.python.kit")
    newton = _kit("linkerbot_sim.mirror.newton.python.kit")
    newton_render = _kit("linkerbot_sim.mirror.newton_render.python.kit")

    assert set(physx["dependencies"]) == {"isaacsim.exp.base.python"}
    base = {
        "isaacsim.simulation_app",
        "isaacsim.asset.importer.mjcf",
        "isaacsim.asset.importer.urdf",
        "omni.kit.loop-isaac",
        "omni.kit.usd.layers",
        "omni.warp.core",
    }
    assert set(newton["dependencies"]) == base
    assert set(newton_render["dependencies"]) == base | {
        "omni.hydra.rtx",
        "omni.kit.manipulator.camera",
        "omni.kit.viewport.window",
        "omni.syntheticdata",
        "omni.usd.schema.omni_lens_distortion",
    }


def test_newton_mirror_render_disables_dlss_frame_generation() -> None:
    settings = _kit("linkerbot_sim.mirror.newton_render.python.kit")["settings"]

    assert settings["rtx-transient"]["dlssg"]["enabled"] is False
    assert (
        settings["exts"]["omni.kit.viewport.window"]["startup"]["cameraManipulator"][
            "enabled"
        ]
        is True
    )


def test_newton_mirror_experiences_do_not_enable_isaac_physics_owners() -> None:
    for name in (
        "linkerbot_sim.mirror.newton.python.kit",
        "linkerbot_sim.mirror.newton_render.python.kit",
    ):
        dependencies = set(_kit(name)["dependencies"])
        assert all(
            not dependency.startswith("linkerbot_sim.") for dependency in dependencies
        )
        assert all("physx" not in dependency.casefold() for dependency in dependencies)
