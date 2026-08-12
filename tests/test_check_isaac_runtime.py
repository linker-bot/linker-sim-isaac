from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import check_isaac_runtime as check


def test_runtime_markers_are_emitted_before_fast_app_close(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LINKERBOT_ISAAC_RUNTIME_WORKER", "1")
    provenance = SimpleNamespace()
    events: list[str] = []
    session_specs: list[object] = []

    def close() -> None:
        events.extend(capsys.readouterr().out.splitlines())
        events.append("SESSION_CLOSED")

    session = SimpleNamespace(close=close)

    monkeypatch.setattr(
        check,
        "create_isaac_session_from_spec",
        lambda *, spec: session_specs.append(spec) or session,
    )
    monkeypatch.setattr(
        check,
        "_physics_owner_probe",
        lambda _session, spec: {
            "backend": "newton",
            "kind": spec.physics.kind,
        },
    )
    monkeypatch.setattr(
        check,
        "collect_runtime_provenance",
        lambda **_kwargs: provenance,
    )
    monkeypatch.setattr(
        check,
        "validate_target_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        check,
        "format_runtime_provenance",
        lambda value: json.dumps({"provenance": value is provenance}),
    )

    check.main(["--profile", "mirror-newton-cuda"])
    events.extend(capsys.readouterr().out.splitlines())

    assert events[0].startswith("LINKERBOT_PHYSICS_SCENE_RUNTIME_VALID ")
    assert events[1].startswith("LINKERBOT_DEPENDENCY_RUNTIME_VALID ")
    assert events[2] == "SESSION_CLOSED"
    assert session_specs[0].physics.kind == "newton_cuda"


def test_kaleidoscope_newton_profile_selects_multi_world_cuda_spec() -> None:
    spec = check._session_spec("kaleidoscope-newton-cuda", cuda_device=3)

    assert spec.experience_family == "kaleidoscope"
    assert spec.physics.kind == "newton_cuda"
    assert spec.compute.cuda_device == 3
    assert spec.compute_device == "cuda:3"
    assert spec.physics_device == "cuda:3"
    assert spec.physics.world_count == 2
    assert spec.render.enabled is False


@pytest.mark.parametrize(
    ("profile", "rendering"),
    (
        ("mirror-newton-cpu", False),
        ("mirror-newton-cpu-render", True),
    ),
)
def test_mirror_newton_cpu_profiles_select_single_world_cpu_spec(
    profile: str,
    rendering: bool,
) -> None:
    spec = check._session_spec(profile, cuda_device=3)

    assert spec.experience_family == "mirror"
    assert spec.physics.kind == "newton_cpu"
    assert spec.physics.world_count == 1
    assert spec.compute_device == "cuda:3"
    assert spec.physics_device == "cpu"
    assert spec.physics_execution == "cpu"
    assert spec.render.enabled is rendering


@pytest.mark.parametrize(
    ("profile", "physics_kind"),
    (
        ("kaleidoscope-physx-cuda-viewport", "physx_cuda"),
        ("kaleidoscope-newton-cuda-viewport", "newton_cuda"),
    ),
)
def test_kaleidoscope_viewport_profiles_select_one_visible_world(
    profile: str,
    physics_kind: str,
) -> None:
    spec = check._session_spec(profile, cuda_device=2)

    assert spec.experience_family == "kaleidoscope"
    assert spec.physics.kind == physics_kind
    assert spec.compute_device == "cuda:2"
    assert spec.physics_device == "cuda:2"
    assert spec.render.enabled is True
    assert spec.render.visible_world_indices == (0,)


def test_help_exits_before_starting_runtime_worker(monkeypatch, capsys) -> None:
    supervised: list[object] = []
    monkeypatch.delenv("LINKERBOT_ISAAC_RUNTIME_WORKER", raising=False)
    monkeypatch.setattr(
        check,
        "run_supervised_worker",
        lambda **kwargs: supervised.append(kwargs),
    )

    with pytest.raises(SystemExit) as raised:
        check.main(["--help"])

    assert raised.value.code == 0
    assert "mirror-newton-cuda" in capsys.readouterr().out
    assert supervised == []
