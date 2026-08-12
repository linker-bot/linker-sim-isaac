from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PHYSX_KIT = "linkerbot_sim.kaleidoscope.physx_cuda.python.kit"
NEWTON_KIT = "linkerbot_sim.kaleidoscope.newton.python.kit"


def _kit(name: str) -> dict[str, object]:
    path = ROOT / "apps" / name
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_kaleidoscope_kit_is_minimal_physx_cuda_fabric_closure() -> None:
    document = _kit(PHYSX_KIT)
    dependencies = set(document["dependencies"])
    required = {
        "isaacsim.simulation_app",
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.prims",
        "isaacsim.core.simulation_manager",
        "isaacsim.asset.importer.mjcf",
        "isaacsim.asset.importer.urdf",
        "omni.physics.physx",
        "omni.physics.stageupdate",
        "omni.physx.fabric",
        "omni.physx.tensors",
    }
    assert required <= dependencies
    assert "isaacsim.exp.base" not in dependencies
    assert "isaacsim.exp.base.python" not in dependencies
    forbidden = (
        "camera",
        "viewport",
        "rtx",
        "replicator",
        "synthetic",
        "telemetry",
        "newton",
    )
    assert not any(
        fragment in dependency for dependency in dependencies for fragment in forbidden
    )


def test_kaleidoscope_kit_disables_readback_and_all_render_outputs() -> None:
    settings = _kit(PHYSX_KIT)["settings"]
    assert settings["physics"]["suppressReadback"] is True
    assert settings["physics"]["fabricUseGPUInterop"] is True
    for name in (
        "fabricUpdateTransformations",
        "fabricUpdateVelocities",
        "fabricUpdateForceSensors",
        "fabricUpdateJointStates",
        "fabricUpdatePoints",
    ):
        assert settings["physics"][name] is False
    assert settings["app"]["useFabricSceneDelegate"] is False


def test_kaleidoscope_newton_kit_is_independent_project_owned_closure() -> None:
    document = _kit(NEWTON_KIT)
    dependencies = set(document["dependencies"])
    assert dependencies == {
        "isaacsim.simulation_app",
        "isaacsim.asset.importer.mjcf",
        "isaacsim.asset.importer.urdf",
        "omni.kit.loop-isaac",
        "omni.kit.usd.layers",
        "omni.warp.core",
    }
    assert not any(name.startswith("linkerbot_sim.") for name in dependencies)

    # Python wheels 由项目 runtime 直接导入；Isaac Newton 与 PhysX 均不能成为第二个
    # physics owner，也不能通过另一个产品 Kit 间接进入闭包。
    excluded = set(document["settings"]["app"]["extensions"]["excluded"])
    assert {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.simulation_manager",
        "isaacsim.pip.newton",
        "isaacsim.physics.newton",
        "isaacsim.physics.newton.tensors",
        "omni.physics.physx",
        "omni.physics.stageupdate",
        "omni.physx.fabric",
        "omni.physx.tensors",
    } <= excluded
    assert document["settings"]["app"]["useFabricSceneDelegate"] is False


def test_all_kaleidoscope_kits_exclude_render_and_interactive_consumers() -> None:
    forbidden_fragments = (
        "camera",
        "viewport",
        "rtx",
        "replicator",
        "synthetic",
        "telemetry",
    )
    for name in (PHYSX_KIT, NEWTON_KIT):
        document = _kit(name)
        dependencies = set(document["dependencies"])
        assert not any(
            fragment in dependency.casefold()
            for dependency in dependencies
            for fragment in forbidden_fragments
        )
