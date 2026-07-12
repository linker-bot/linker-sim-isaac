from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

from linkerbot_sim.configs import cli


class _ResolvedConfig:
    fingerprint = "a" * 64
    sources = {
        "runtime.mode": "runtime:unit",
        "runtime.paths.cache_root": "runtime:unit",
    }

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime": {
                "mode": "single_scene",
                "paths": {"cache_root": "/sensitive/private/cache"},
            },
        }


def test_resolve_runtime_profile_loads_selected_env_and_uses_no_cli_overrides(
    monkeypatch,
) -> None:
    profile = SimpleNamespace(profiles=SimpleNamespace(env="scene1"))
    resolved = _ResolvedConfig()
    calls: list[tuple[object, dict[str, object], object, object]] = []

    monkeypatch.setattr(cli, "load_runtime_profile", lambda name: profile)
    monkeypatch.setattr(
        cli,
        "load_profile_yaml",
        lambda group, name: {"loaded": f"{group}/{name}"},
    )

    def fake_resolve(
        profile_arg,
        *,
        cli_overrides,
        env_config,
        expected_mode,
    ):
        calls.append(
            (
                profile_arg,
                cli_overrides,
                env_config,
                expected_mode,
            )
        )
        return resolved

    monkeypatch.setattr(cli, "resolve_runtime_config", fake_resolve)
    graph_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "validate_profile_graph",
        lambda **kwargs: graph_calls.append(kwargs),
    )

    assert cli.resolve_runtime_profile("experiment") == (profile, resolved)
    assert calls == [
        (
            profile,
            {},
            {"loaded": "env/scene1"},
            None,
        )
    ]
    assert graph_calls == [
        {
            "runtime_profile": "experiment",
            "profile": profile,
            "resolved": resolved,
            "env_config": {"loaded": "env/scene1"},
        }
    ]


def test_validate_config_default_output_is_safe_for_startup_logs(monkeypatch) -> None:
    resolved = _ResolvedConfig()
    monkeypatch.setattr(
        cli,
        "resolve_runtime_profile",
        lambda _name: (object(), resolved),
    )
    stdout = StringIO()

    assert cli.main(["--runtime-profile", "unit"], stdout=stdout) == 0

    payload = json.loads(stdout.getvalue())
    assert payload == {
        "event": "config_validated",
        "fingerprint": "a" * 64,
        "runtime_profile": "unit",
    }
    assert "/sensitive/private/cache" not in stdout.getvalue()
    assert "cache_root" not in stdout.getvalue()


def test_dump_effective_config_includes_values_and_per_field_sources(
    monkeypatch,
) -> None:
    resolved = _ResolvedConfig()
    monkeypatch.setattr(
        cli,
        "resolve_runtime_profile",
        lambda _name: (object(), resolved),
    )
    stdout = StringIO()

    assert (
        cli.main(
            ["--runtime-profile", "unit", "--dump-effective-config"],
            stdout=stdout,
        )
        == 0
    )

    payload = json.loads(stdout.getvalue())
    assert payload["runtime_profile"] == "unit"
    assert payload["fingerprint"] == "a" * 64
    assert payload["effective"] == resolved.as_dict()
    assert payload["sources"] == resolved.sources


def test_validate_config_returns_nonzero_for_invalid_profile(monkeypatch) -> None:
    def fail(_name: str):
        raise ValueError("runtime.planner.resources.max_workers must be positive")

    monkeypatch.setattr(cli, "resolve_runtime_profile", fail)
    stdout = StringIO()
    stderr = StringIO()

    assert cli.main(["--runtime-profile", "bad"], stdout=stdout, stderr=stderr) == 1
    assert stdout.getvalue() == ""
    assert "CONFIG_INVALID ValueError" in stderr.getvalue()
    assert "runtime.planner.resources.max_workers" in stderr.getvalue()
