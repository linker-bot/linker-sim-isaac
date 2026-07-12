"""tiled env 的相机配置展开工具。

env YAML 中的 ``sensors.cameras`` 保存共享相机模板，``env_ids`` 选择实际拥有资源的
子环境。tiled runtime 构建真实场景前，把模板展开成 ``env_000/camera`` 等独立相机，
并让保存路径和 Foxglove topic 带上 env 标签，避免多个 env 的输出互相覆盖。
"""

from __future__ import annotations

from dataclasses import replace

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.sensors.camera import (
    SensorCameraOutputSettings,
    SensorCameraSettings,
)
from linkerbot_sim.tiled.config import TiledEnvConfig, TiledPerEnvConfig
from linkerbot_sim.tiled.scene.paths import env_root_paths, make_env_local_prim_path


def tiled_sensor_camera_settings(
    sensors: SceneSensorSettings,
    *,
    tiled_config: TiledEnvConfig,
) -> SceneSensorSettings:
    """按 camera env_ids 把共享配置展开到 selected tiled env。

    基础 ``sensors.cameras`` 保存分辨率、modalities、clip range、输出设置等共享参数；
    ``tiled.per_env[].cameras`` 只允许覆盖某个 env 内的相机位姿，不能偷偷新增相机。
    """

    if not sensors.cameras:
        return SceneSensorSettings()
    env_ids_by_camera = {
        camera.name: frozenset(
            _resolved_camera_env_ids(
                camera,
                tiled_config=tiled_config,
            )
        )
        for camera in sensors.cameras
    }
    _validate_camera_pose_overrides(
        sensors,
        tiled_config.per_env,
        env_ids_by_camera=env_ids_by_camera,
    )
    per_env_by_id = {item.env_id: item for item in tiled_config.per_env}
    expanded: list[SensorCameraSettings] = []
    for env_id, env_root in enumerate(env_root_paths(tiled_config)):
        per_env = per_env_by_id.get(env_id)
        env_label = _env_label(env_id)
        for camera in sensors.cameras:
            if env_id not in env_ids_by_camera[camera.name]:
                continue
            pose = _camera_pose(camera, per_env=per_env)
            # replace 保留共享相机的其它字段，只改 env-local namespace、位姿和输出目标。
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
                    # Tiled Scene 展开已消费该 selector；必须清空，避免后续 Single Scene
                    # 相机创建器把仍带 env scope 的配置当成普通相机静默处理。
                    env_ids=None,
                    output=_env_local_camera_output(
                        camera.output,
                        env_label=env_label,
                    ),
                )
            )
    return SceneSensorSettings(cameras=tuple(expanded))


def _resolved_camera_env_ids(
    camera: SensorCameraSettings,
    *,
    tiled_config: TiledEnvConfig,
) -> tuple[int, ...]:
    """校验 tiled camera 显式且有界的 env 资源范围。"""

    if camera.env_ids is None:
        label = f"sensors.cameras.{camera.name}.env_ids"
        raise ValueError(f"{label} is required for a tiled profile")
    invalid = tuple(
        env_id
        for env_id in camera.env_ids
        if env_id < 0 or env_id >= tiled_config.num_envs
    )
    if invalid:
        raise ValueError(
            f"sensors.cameras.{camera.name}.env_ids contains out-of-range env id "
            f"for tiled.num_envs={tiled_config.num_envs}: {invalid[0]}"
        )
    return camera.env_ids


def _camera_pose(
    camera: SensorCameraSettings,
    *,
    per_env: TiledPerEnvConfig | None,
) -> RootPoseConfig:
    """返回 per-env 位姿覆盖；没有覆盖时使用基础相机位姿。"""

    if per_env is not None and camera.name in per_env.camera_poses:
        return per_env.camera_poses[camera.name]
    return RootPoseConfig(xyz=camera.pose_xyz, rpy=camera.pose_rpy)


def _env_local_camera_prim_path(
    camera: SensorCameraSettings,
    *,
    env_root: str,
) -> str:
    """把基础相机 prim path 改写到某个 env namespace 下。"""

    return make_env_local_prim_path(env_root, camera.prim_path)


def _env_local_camera_parent_path(
    camera: SensorCameraSettings,
    *,
    env_root: str,
) -> str:
    """把相机 parent prim 改写到某个 env namespace 下。"""

    parent = camera.parent_prim_path
    if parent is None or parent.rstrip("/") == "/World":
        # 未显式指定 parent 时，相机挂在 env root 下，避免跨 env 共享同一个 /World parent。
        return env_root
    return make_env_local_prim_path(env_root, parent)


def _env_local_camera_output(
    output: SensorCameraOutputSettings,
    *,
    env_label: str,
) -> SensorCameraOutputSettings:
    """为每个 env 生成互不冲突的输出路径和 topic 前缀。"""

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
    *,
    env_ids_by_camera: dict[str, frozenset[int]],
) -> None:
    """拒绝未知相机，以及超出相机资源环境范围的逐环境 pose 覆盖。

    该检查在创建 sensor 前完成，防止配置只在部分环境中静默失效。
    """

    camera_names = {camera.name for camera in sensors.cameras}
    for per_env in per_env_configs:
        unknown = sorted(set(per_env.camera_poses) - camera_names)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                "tiled.per_env camera pose override references unknown camera "
                f"for env_id {per_env.env_id}: {names}"
            )
        for camera_name in per_env.camera_poses:
            if per_env.env_id not in env_ids_by_camera[camera_name]:
                raise ValueError(
                    "tiled.per_env camera pose override for "
                    f"env_id {per_env.env_id} at cameras.{camera_name}.pose is "
                    f"outside sensors.cameras.{camera_name}.env_ids"
                )


def _append_path_segment(value: str | None, segment: str) -> str | None:
    """向配置的输出路径追加 POSIX 风格 path segment。"""

    if value is None:
        return None
    return value.rstrip("/") + "/" + segment


def _append_topic_segment(value: str | None, segment: str) -> str | None:
    """向 topic 前缀追加 segment，同时保留未配置前缀的 ``None`` 语义。"""

    if value is None:
        return None
    return value.rstrip("/") + "/" + segment


def _env_label(env_id: int) -> str:
    """返回稳定的人类可读 tiled env 标签。"""

    return f"env_{int(env_id):03d}"
