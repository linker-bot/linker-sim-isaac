from __future__ import annotations

import pytest

from linkerbot_sim.isaac.lifecycle import (
    bound_simulation_app_physics_runtime,
    close_simulation_app,
    register_simulation_app_physics_runtime,
)
from linkerbot_sim.assets import robot_import


class _FakeApp:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls

    def close(self) -> None:
        self.calls.append("app")


class _FakeRuntime:
    backend = "physx"
    execution = "cpu"

    def __init__(self, calls: list[object], *, error: BaseException | None = None):
        self.calls = calls
        self.error = error

    def close(self) -> None:
        self.calls.append("physics")
        if self.error is not None:
            raise self.error


def test_close_simulation_app_releases_exact_runtime_before_native_shutdown(
    monkeypatch,
) -> None:
    calls: list[object] = []
    app = _FakeApp(calls)
    runtime = _FakeRuntime(calls)
    register_simulation_app_physics_runtime(app, runtime)
    monkeypatch.setattr(
        robot_import,
        "release_imported_asset_files",
        lambda: calls.append("imports"),
    )

    close_simulation_app(app)

    assert calls == ["physics", "imports", "app"]
    assert bound_simulation_app_physics_runtime(app) is None


def test_close_simulation_app_without_binding_never_releases_global_owner(
    monkeypatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        robot_import,
        "release_imported_asset_files",
        lambda: calls.append("imports"),
    )

    close_simulation_app(_FakeApp(calls))

    assert calls == ["imports", "app"]


def test_close_simulation_app_forwards_nonzero_exit_code(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        robot_import,
        "release_imported_asset_files",
        lambda: calls.append("imports"),
    )

    class _ExitAwareApp:
        def close(self, *, exit_code: int = 0) -> None:
            calls.append(("app", exit_code))

    close_simulation_app(_ExitAwareApp(), exit_code=1)

    assert calls == ["imports", ("app", 1)]


def test_close_simulation_app_always_attempts_native_after_runtime_error(
    monkeypatch,
) -> None:
    calls: list[object] = []
    app = _FakeApp(calls)
    runtime = _FakeRuntime(calls, error=RuntimeError("physics close failed"))
    register_simulation_app_physics_runtime(app, runtime)
    monkeypatch.setattr(
        robot_import,
        "release_imported_asset_files",
        lambda: calls.append("imports"),
    )

    with pytest.raises(RuntimeError, match="physics close failed"):
        close_simulation_app(app)

    assert calls == ["physics", "imports", "app"]
    # close 失败时保留 identity，不能把半关闭 owner 伪装成已释放。
    assert bound_simulation_app_physics_runtime(app) is runtime

    # retry 只重做失败的 physics 阶段，不会再次关闭 importer 或 native App。
    runtime.error = None
    close_simulation_app(app)
    assert calls == ["physics", "imports", "app", "physics"]
    assert bound_simulation_app_physics_runtime(app) is None


def test_close_rejects_runtime_from_another_app_before_teardown() -> None:
    first_calls: list[object] = []
    second_calls: list[object] = []
    app = _FakeApp(first_calls)
    bound = _FakeRuntime(first_calls)
    wrong = _FakeRuntime(second_calls)
    register_simulation_app_physics_runtime(app, bound)

    with pytest.raises(RuntimeError, match="different physics runtime"):
        close_simulation_app(app, physics_runtime=wrong)

    assert first_calls == []
    assert second_calls == []
    assert bound_simulation_app_physics_runtime(app) is bound
