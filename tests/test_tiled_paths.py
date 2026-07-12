from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.paths import (
    env_local_suffix,
    env_origins,
    env_root_path,
    env_root_paths,
    make_env_local_prim_path,
    prim_paths_from_suffix,
)


def test_env_root_paths_use_configured_namespace() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 3,
                "base_env_path": "/World/Tiled",
                "env_prefix": "scene",
            }
        }
    )

    assert env_root_path(config, 2) == "/World/Tiled/scene_2"
    assert env_root_paths(config) == (
        "/World/Tiled/scene_0",
        "/World/Tiled/scene_1",
        "/World/Tiled/scene_2",
    )


def test_make_env_local_prim_path_rewrites_world_paths() -> None:
    assert (
        make_env_local_prim_path("/World/envs/env_0", "/World/Robot")
        == "/World/envs/env_0/Robot"
    )
    assert (
        make_env_local_prim_path("/World/envs/env_0", "/World/Foo/Bar")
        == "/World/envs/env_0/Foo/Bar"
    )


def test_make_env_local_prim_path_does_not_double_namespace() -> None:
    assert (
        make_env_local_prim_path("/World/envs/env_0", "/World/envs/env_0/Robot")
        == "/World/envs/env_0/Robot"
    )


@pytest.mark.parametrize("path", ["World/Robot", "/World", "/World//Robot"])
def test_make_env_local_prim_path_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError):
        make_env_local_prim_path("/World/envs/env_0", path)


def test_env_origins_use_grid_spacing() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 5,
                "spacing": 2.5,
                "num_per_row": 3,
            }
        }
    )

    np.testing.assert_allclose(
        env_origins(config),
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [2.5, 2.5, 0.0],
        ],
    )


def test_env_local_suffix_round_trip() -> None:
    suffix = env_local_suffix("/World/envs/env_0", "/World/envs/env_0/Robot/root")

    assert suffix == "Robot/root"
    assert prim_paths_from_suffix(
        ("/World/envs/env_0", "/World/envs/env_1"), suffix
    ) == (
        "/World/envs/env_0/Robot/root",
        "/World/envs/env_1/Robot/root",
    )
