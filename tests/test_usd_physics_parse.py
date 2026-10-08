from types import SimpleNamespace
import sys

import pytest

from linkerbot_sim.isaac import usd_physics_parse as parser


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("env_locked", [False, True])
def test_pinned_parser_restores_effective_concurrency(monkeypatch, fail, env_locked):
    state = {"limit": 8}
    work = SimpleNamespace(
        GetConcurrencyLimit=lambda: state["limit"],
        SetConcurrencyLimit=lambda value: state.update(
            limit=8 if env_locked else value
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(Usd=SimpleNamespace(GetVersion=lambda: (0, 25, 11)), Work=work),
    )
    calls = []

    def set_native(value):
        calls.append(value)
        state["limit"] = value

    monkeypatch.setattr(parser, "_set_pinned_work_limit", set_native)
    try:
        with parser.serialized_usd_physics_parse():
            assert state["limit"] == 1
            with parser.serialized_usd_physics_parse():
                assert state["limit"] == 1
            assert state["limit"] == 1
            if fail:
                raise ValueError("import failed")
    except ValueError:
        assert fail
    assert state["limit"] == 8
    assert calls == ([1, 8] if env_locked else [])


def test_parser_does_not_run_when_effective_limit_cannot_be_set(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(GetVersion=lambda: (0, 25, 11)),
            Work=SimpleNamespace(
                GetConcurrencyLimit=lambda: 8, SetConcurrencyLimit=lambda value: None
            ),
        ),
    )
    monkeypatch.setattr(parser, "_set_pinned_work_limit", lambda value: None)
    with pytest.raises(RuntimeError, match="cannot serialize"):
        with parser.serialized_usd_physics_parse():
            pytest.fail("must not run the unsafe parser")


def test_other_usd_versions_do_not_change_concurrency(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(GetVersion=lambda: (0, 26, 5)),
            Work=SimpleNamespace(
                GetConcurrencyLimit=lambda: pytest.fail("unexpected Work override")
            ),
        ),
    )
    with parser.serialized_usd_physics_parse():
        pass
