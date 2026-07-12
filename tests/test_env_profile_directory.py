from __future__ import annotations

from pathlib import Path

from linkerbot_sim.configs.profiles import (
    load_env_profile_directory,
    load_profile_yaml,
    profile_path,
)
from linkerbot_sim.tiled.config import TiledEnvConfig


def test_load_profile_yaml_accepts_directory_env_profile() -> None:
    config = load_profile_yaml("env", "scene3_tiled")
    tiled = TiledEnvConfig.from_env_config(config)

    assert (
        profile_path("env", "scene3_tiled")
        == Path("configs/envs/scene3_tiled/base.yaml").resolve()
    )
    assert config["env"]["name"] == "scene3_tiled"
    assert tiled.enabled is True
    assert tiled.num_envs == int(config["tiled"]["num_envs"])
    assert tiled.clone.replicate_physics is bool(
        config["tiled"]["clone"]["replicate_physics"]
    )
    assert [item.env_id for item in tiled.per_env] == [0, 1, 2, 3]
    assert tiled.per_env[1].object_root_poses["Tblock"].xyz == (0.12, 0.04, -0.4)
    assert tiled.per_env[0].camera_poses["world_rgbd"].xyz == (0.08, 0.0, 0.08)
    assert "world_rgbd" not in tiled.per_env[1].camera_poses


def test_load_env_profile_directory_merges_per_env_yaml(tmp_path: Path) -> None:
    profile_dir = tmp_path / "demo_tiled"
    env_dir = profile_dir / "envs"
    env_dir.mkdir(parents=True)
    (profile_dir / "base.yaml").write_text(
        """
env:
  name: demo_tiled
robots:
  - label: single
    robot_profile: ar5v2_l6v1_l
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
objects:
  - name: block
    object_profile: TblockV1_default
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
sensors:
  cameras:
    world_rgbd:
      enabled: false
      env_ids: [3]
      prim_path: /World/WorldRGBD
tiled:
  enabled: true
  per_env_config_dir: envs
""",
        encoding="utf-8",
    )
    (env_dir / "env_003.yaml").write_text(
        """
env_id: 3
objects:
  block:
    root_pose:
      xyz: [0.3, 0.0, -0.4]
      rpy: [0.0, 1.0, 0.0]
cameras:
  world_rgbd:
    pose:
      xyz: [0.1, 0.2, 0.3]
      rpy: [0.0, 1.0, 0.0]
metadata:
  replay_id: case_003
""",
        encoding="utf-8",
    )

    config = load_env_profile_directory(profile_dir)
    tiled = TiledEnvConfig.from_env_config(config)

    assert tiled.num_envs == 4
    assert [item.env_id for item in tiled.per_env] == [3]
    assert tiled.per_env[0].object_root_poses["block"].xyz == (0.3, 0.0, -0.4)
    assert tiled.per_env[0].camera_poses["world_rgbd"].xyz == (0.1, 0.2, 0.3)
    assert tiled.per_env[0].metadata == {"replay_id": "case_003"}
