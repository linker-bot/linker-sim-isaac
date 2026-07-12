from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType

import pytest

from linkerbot_sim.configs.runtime import SimulationAppSettings
from tools.object_assets.flexible.rope import build_asset as rope_entry
from tools.object_assets.rigid.tblock import build_asset as tblock_entry


class _FakeSimulationApp:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.parametrize(
    ("entry", "config", "writer_name"),
    [
        (
            rope_entry,
            {
                "object": {
                    "asset_path": "assets/test_rope.usda",
                    "root_path": "/TestRope",
                },
                "rope": {},
            },
            "write_capsule_rope_asset",
        ),
        (
            tblock_entry,
            {
                "object": {
                    "asset_path": "assets/test_tblock.usda",
                    "root_path": "/TestTBlock",
                },
                "tblock": {},
            },
            "write_tblock_asset",
        ),
    ],
)
def test_asset_entrypoint_launches_typed_headless_app_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    entry: ModuleType,
    config: dict[str, object],
    writer_name: str,
) -> None:
    app = _FakeSimulationApp()
    launched: list[SimulationAppSettings] = []

    monkeypatch.setattr(
        entry,
        "parse_args",
        lambda: argparse.Namespace(config=Path("asset.yaml"), output=None),
    )
    monkeypatch.setattr(entry, "load_yaml", lambda _path: config)

    def launch(settings: SimulationAppSettings) -> _FakeSimulationApp:
        launched.append(settings)
        return app

    monkeypatch.setattr(entry, "launch_simulation_app", launch)
    monkeypatch.setattr(entry, writer_name, lambda _config, _output: Path("asset.usda"))

    entry.main()

    assert len(launched) == 1
    assert launched[0].gui is False
    assert app.close_count == 1
