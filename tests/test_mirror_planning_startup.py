from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import linkerbot_sim.mirror.scene_assembly as mirror_assembly
from linkerbot_sim.configuration import load_mirror_config


@pytest.mark.parametrize(
    ("policy", "expected_events"),
    (
        ("lazy", ["resources", "state_stream", "motion"]),
        (
            "prewarm",
            ["resources", "snapshot", "prewarm:independent", "state_stream", "motion"],
        ),
    ),
)
def test_scene_planning_startup_runs_before_interactive_resources(
    policy: str,
    expected_events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_mirror_config("physx_cpu")
    config = replace(config, scene=replace(config.scene, planning_startup=policy))
    events: list[str] = []
    snapshot = object()

    class CollisionRegistry:
        def snapshot(self):
            events.append("snapshot")
            return snapshot

    class PlanningRegistry:
        def prewarm_interactive_planners(self, value, *, coordination: str):
            assert value is snapshot
            events.append(f"prewarm:{coordination}")
            return (0, 1)

        def close(self) -> None:
            pass

    resources = SimpleNamespace(
        scene=config.scene,
        session=object(),
        planning_registry=PlanningRegistry(),
        collision_registry=CollisionRegistry(),
        sensor_cameras=(),
        camera_output=None,
        loggers=(),
        hybrid_control_logger=None,
        hybrid_diagnostics_provider=None,
        robots_by_id={},
        object_state_views={},
    )

    class MotionBackend:
        def hybrid_diagnostics(self) -> dict[str, object]:
            return {"active": False}

    def create_resources(**_kwargs: object):
        events.append("resources")
        return resources

    monkeypatch.setattr(mirror_assembly, "prepare_mcap_output", lambda *_a, **_k: None)
    monkeypatch.setattr(
        mirror_assembly,
        "create_mirror_scene_resources",
        create_resources,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "start_interactive_state_stream",
        lambda *_a, **_k: events.append("state_stream") or None,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "MirrorTimelineBackend",
        lambda *_a, **_k: events.append("motion") or MotionBackend(),
    )

    mirror_assembly.build_mirror_assembly(config)

    assert events == expected_events


def test_disabled_telemetry_does_not_prepare_retained_mcap_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_mirror_config("physx_cpu")
    config = replace(
        config,
        outputs=replace(
            config.outputs,
            telemetry=replace(config.outputs.telemetry, enabled=False),
        ),
    )

    class AssemblyCaptured(RuntimeError):
        pass

    monkeypatch.setattr(
        mirror_assembly,
        "prepare_mcap_output",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled telemetry must not prepare its retained MCAP path"
        ),
    )
    monkeypatch.setattr(
        mirror_assembly,
        "create_mirror_scene_resources",
        lambda **_kwargs: (_ for _ in ()).throw(AssemblyCaptured("captured")),
    )

    with pytest.raises(AssemblyCaptured, match="captured"):
        mirror_assembly.build_mirror_assembly(config)
