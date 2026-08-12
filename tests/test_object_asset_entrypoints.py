from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType

import pytest

from linkerbot_sim.isaac.spec import IsaacSessionSpec
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
def test_asset_entrypoint_creates_spec_session_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    entry: ModuleType,
    config: dict[str, object],
    writer_name: str,
) -> None:
    app = _FakeSimulationApp()
    launched: list[IsaacSessionSpec] = []

    monkeypatch.setattr(
        entry,
        "parse_args",
        lambda: argparse.Namespace(
            config=Path("asset.yaml"), output=None, cuda_device=3
        ),
    )
    monkeypatch.setattr(entry, "load_yaml", lambda _path: config)

    def launch(*, spec: IsaacSessionSpec) -> _FakeSimulationApp:
        launched.append(spec)
        return app

    monkeypatch.setattr(entry, "create_isaac_session_from_spec", launch)
    monkeypatch.setattr(entry, writer_name, lambda _config, _output: Path("asset.usda"))

    entry.main()

    assert len(launched) == 1
    assert launched[0].experience_family == "mirror"
    assert launched[0].compute.cuda_device == 3
    assert launched[0].compute_device == "cuda:3"
    assert launched[0].physics_device == "cpu"
    assert launched[0].physics.kind == "physx_cpu"
    assert launched[0].app.gui is False
    assert app.close_count == 1
