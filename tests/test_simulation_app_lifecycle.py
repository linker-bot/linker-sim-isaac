from __future__ import annotations

from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app
from linkerbot_sim.assets import robot_import


class _FakeApp:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close(self) -> None:
        self.calls.append("app")


def test_close_simulation_app_releases_import_files_before_native_shutdown(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        robot_import,
        "release_imported_asset_files",
        lambda: calls.append("imports"),
    )

    close_simulation_app(_FakeApp(calls))

    assert calls == ["imports", "app"]
