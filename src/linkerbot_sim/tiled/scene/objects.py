"""Object import/path helpers for tiled Isaac scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from linkerbot_sim.app.runtime.objects import (
    RuntimeObjectConfig,
    runtime_objects_from_env_config,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.paths import (
    env_local_suffix,
    make_env_local_prim_path,
    prim_paths_from_suffix,
)
from linkerbot_sim.tiled.scene.utils import _print_status


def env_local_runtime_object_configs(
    env_config: Mapping[str, object],
    *,
    env_root: str,
) -> tuple[RuntimeObjectConfig, ...]:
    """把 env objects[] 的 profile prim path 改写到 env root 下。"""

    result: list[RuntimeObjectConfig] = []
    for config in runtime_objects_from_env_config(env_config):
        root_path = config.profile.root_path
        result.append(
            replace(
                config,
                profile=replace(
                    config.profile,
                    prim_path=make_env_local_prim_path(
                        env_root, config.profile.prim_path
                    ),
                    root_path=(
                        None
                        if root_path is None
                        else make_env_local_prim_path(env_root, root_path)
                    ),
                ),
            )
        )
    return tuple(result)


def _tiled_object_prim_paths(
    *,
    object_configs: Sequence[RuntimeObjectConfig],
    env_zero: str,
    env_roots: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """返回每个 object name 在所有 env 下的 prim path。

    目录型 tiled profile 要求所有 env 拥有同一组对象；因此这里从 base/env_0 的对象
    配置推导路径，并把同一个 suffix 映射到每个 env root。
    """

    result: dict[str, tuple[str, ...]] = {}
    for config in object_configs:
        if config.name in result:
            raise ValueError(f"Duplicate tiled object name: {config.name}")
        suffix = env_local_suffix(env_zero, config.profile.prim_path)
        result[config.name] = prim_paths_from_suffix(env_roots, suffix)
    return result


def _apply_per_env_object_pose_overrides(
    *,
    stage: object,
    config: TiledEnvConfig,
    object_prim_paths: Mapping[str, Sequence[str]],
    status_prefix: str | None,
) -> int:
    """把 ``tiled.per_env`` 中的 object root_pose 写入 cloned prim。

    位姿是 env-local 的：对象 prim 位于 ``/World/envs/env_i`` 下面，写入的 translate/rotate
    会相对于各自 env root 生效，不需要手动加 ``env_origin``。
    """

    applied = 0
    for per_env in config.per_env:
        for object_name, root_pose in per_env.object_root_poses.items():
            if object_name not in object_prim_paths:
                raise ValueError(
                    f"tiled.per_env[{per_env.env_id}] references unknown object "
                    f"{object_name!r}; all envs must share the base objects collection"
                )
            prim_path = str(object_prim_paths[object_name][per_env.env_id])
            apply_root_pose_to_prim(stage, prim_path, root_pose)
            applied += 1
            _print_status(
                status_prefix,
                "OBJECT_POSE "
                f"env_id={per_env.env_id} name={object_name} "
                f"prim_path={prim_path} xyz={list(root_pose.xyz)} "
                f"rpy={list(root_pose.rpy)}",
            )
    return applied
