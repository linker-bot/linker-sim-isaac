"""Sensor camera expansion helpers for tiled envs."""

from __future__ import annotations

from dataclasses import replace

from linkerbot_sim.assets.robot_loader import RootPoseConfig
from linkerbot_sim.sensors.camera_config import (
    SceneSensorSettings,
    SensorCameraOutputSettings,
    SensorCameraSettings,
)
from linkerbot_sim.tiled.config import TiledEnvConfig, TiledPerEnvConfig
from linkerbot_sim.tiled.paths import env_root_paths, make_env_local_prim_path


def tiled_sensor_camera_settings(
    sensors: SceneSensorSettings,
    *,
    tiled_config: TiledEnvConfig,
) -> SceneSensorSettings:
    """Expand common sensor camera settings into one camera per tiled env.

    Base ``sensors.cameras`` keeps shared parameters such as resolution, modalities,
    clipping range, and output settings. ``tiled.per_env[].cameras`` may override only
    the camera pose for a given env.
    """

    if not sensors.cameras:
        return SceneSensorSettings()
    _validate_camera_pose_overrides(sensors, tiled_config.per_env)
    per_env_by_id = {item.env_id: item for item in tiled_config.per_env}
    expanded: list[SensorCameraSettings] = []
    for env_id, env_root in enumerate(env_root_paths(tiled_config)):
        per_env = per_env_by_id.get(env_id)
        env_label = _env_label(env_id)
        for camera in sensors.cameras:
            pose = _camera_pose(camera, per_env=per_env)
            expanded.append(
                replace(
                    camera,
                    name=f"{env_label}_{camera.name}",
                    prim_path=_env_local_camera_prim_path(camera, env_root=env_root),
                    parent_prim_path=_env_local_camera_parent_path(
                        camera, env_root=env_root
                    ),
                    pose_xyz=pose.xyz,
                    pose_rpy=pose.rpy,
                    output=_env_local_camera_output(
                        camera.output,
                        env_label=env_label,
                    ),
                )
            )
    return SceneSensorSettings(cameras=tuple(expanded))


def _camera_pose(
    camera: SensorCameraSettings,
    *,
    per_env: TiledPerEnvConfig | None,
) -> RootPoseConfig:
    """Return the per-env camera pose override or the base pose."""

    if per_env is not None and camera.name in per_env.camera_poses:
        return per_env.camera_poses[camera.name]
    return RootPoseConfig(xyz=camera.pose_xyz, rpy=camera.pose_rpy)


def _env_local_camera_prim_path(
    camera: SensorCameraSettings,
    *,
    env_root: str,
) -> str:
    """Rewrite the base camera prim path into one env namespace."""

    return make_env_local_prim_path(env_root, camera.prim_path)


def _env_local_camera_parent_path(
    camera: SensorCameraSettings,
    *,
    env_root: str,
) -> str:
    """Rewrite the camera parent into one env namespace."""

    parent = camera.parent_prim_path
    if parent is None or parent.rstrip("/") == "/World":
        return env_root
    return make_env_local_prim_path(env_root, parent)


def _env_local_camera_output(
    output: SensorCameraOutputSettings,
    *,
    env_label: str,
) -> SensorCameraOutputSettings:
    """Make per-env camera output paths/topics unique."""

    return replace(
        output,
        save_dir=_append_path_segment(output.save_dir, env_label),
        foxglove_topic_prefix=_append_topic_segment(
            output.foxglove_topic_prefix, env_label
        ),
    )


def _validate_camera_pose_overrides(
    sensors: SceneSensorSettings,
    per_env_configs: tuple[TiledPerEnvConfig, ...],
) -> None:
    """Reject per-env camera overrides that do not match base camera names."""

    camera_names = {camera.name for camera in sensors.cameras}
    for per_env in per_env_configs:
        unknown = sorted(set(per_env.camera_poses) - camera_names)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                "tiled.per_env camera pose override references unknown camera "
                f"for env_id {per_env.env_id}: {names}"
            )


def _append_path_segment(value: str | None, segment: str) -> str | None:
    """Append a POSIX-style segment to a configured output path."""

    if value is None:
        return None
    return value.rstrip("/") + "/" + segment


def _append_topic_segment(value: str | None, segment: str) -> str | None:
    """Append a topic segment while preserving an unset custom prefix."""

    if value is None:
        return None
    return value.rstrip("/") + "/" + segment


def _env_label(env_id: int) -> str:
    """Return the stable human-facing tiled env label."""

    return f"env_{int(env_id):03d}"
