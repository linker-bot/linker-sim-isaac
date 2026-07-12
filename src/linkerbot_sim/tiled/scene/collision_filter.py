"""Isaac tiled scenes 的跨 env collision filtering 规划与 authoring。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from linkerbot_sim.tiled.config import TiledEnvConfig


def _filter_env_collisions(
    *,
    stage: object,
    config: TiledEnvConfig,
    env_roots: Sequence[str],
) -> bool:
    """按配置过滤不同 env root 之间的碰撞。

    默认 ``collision_groups``：每个 env 一个 ``UsdPhysics.CollisionGroup``，并在 physics
    scene 上打开 ``physxScene:invertCollisionGroupFilter``，把 ``filteredGroups`` 语义反转成
    白名单——每个 env 只与自身和 global(地面)碰撞。authoring 成本 O(E)，是 256+ env 的推荐
    路径，也是 Isaac ``GridCloner.filter_collisions`` 的等价做法。

    ``filtered_pairs`` 是逐 prim 两两写入 ``UsdPhysics.FilteredPairsAPI`` 的显式备选策略：
    它不依赖 collision group，但复杂度 O(E²)，仅适合少量 env 或需要规避 collision group 的场景。
    """

    if not config.clone.filter_collisions:
        return False

    if config.clone.collision_filter_strategy == "filtered_pairs":
        participant_paths = [
            _collision_filter_participant_paths(stage=stage, env_root=env_root)
            for env_root in env_roots
        ]
        return _apply_env_pair_filters(stage, participant_paths) > 0

    plan = _plan_env_collision_groups(
        env_roots=tuple(str(env_root) for env_root in env_roots),
        global_paths=_resolve_global_collision_paths(
            stage=stage,
            config=config,
            env_roots=env_roots,
        ),
        collision_root_path=str(config.clone.collision_root_path),
    )
    # 先在 scene 上打开 invert，再 author 各 collision group。invert 属于 PhysX schema，
    # 只有 Isaac app 内可用；group prim 本身只用 UsdPhysics，可离线校验结构。
    physics_scene_path = _resolve_physics_scene_path(
        stage,
        config.clone.physics_scene_path,
    )
    _author_scene_invert_collision_filter(stage, physics_scene_path)
    return _author_collision_group_prims(stage, plan) > 0


@dataclass(frozen=True)
class _CollisionGroupPlan:
    """env 间碰撞过滤的纯数据规划（不触碰 pxr）。

    invert 语义下 ``filtered_groups`` 是白名单(enable-list)：每个 env 组只列自身和 global，
    global 组列出全部 env 组。因此总目标数是 O(E)，而不是 pair 过滤的 O(E²)。
    """

    collision_root_path: str
    invert_filter: bool
    env_group_paths: tuple[str, ...]
    env_includes: tuple[str, ...]
    global_group_path: str | None
    global_includes: tuple[str, ...]

    def env_group_filtered_groups(self, index: int) -> tuple[str, ...]:
        """第 ``index`` 个 env 组的白名单：自身 + global(若存在)。"""

        targets = [self.env_group_paths[index]]
        if self.global_group_path is not None:
            targets.append(self.global_group_path)
        return tuple(targets)

    def global_group_filtered_groups(self) -> tuple[str, ...]:
        """global 组白名单：全部 env 组，使地面与每个 env 保持碰撞。"""

        return self.env_group_paths

    def total_filtered_group_targets(self) -> int:
        """需要写入的 filteredGroups 目标总数（用于复杂度断言，应为 O(E)）。"""

        env_targets = sum(
            len(self.env_group_filtered_groups(index))
            for index in range(len(self.env_group_paths))
        )
        global_targets = (
            len(self.global_group_filtered_groups())
            if self.global_group_path is not None
            else 0
        )
        return env_targets + global_targets


def _plan_env_collision_groups(
    *,
    env_roots: Sequence[str],
    global_paths: Sequence[str],
    collision_root_path: str,
) -> _CollisionGroupPlan:
    """把 env roots + global paths 规划成 O(E) 的碰撞组结构（纯数据，可离线单测）。"""

    roots = tuple(str(env_root) for env_root in env_roots)
    seen: set[str] = set()
    global_includes: list[str] = []
    for path in global_paths:
        text = str(path)
        if text and text not in seen:
            seen.add(text)
            global_includes.append(text)
    has_global = bool(global_includes)
    return _CollisionGroupPlan(
        collision_root_path=str(collision_root_path),
        invert_filter=True,
        env_group_paths=tuple(
            f"{collision_root_path}/env_{index}" for index in range(len(roots))
        ),
        env_includes=roots,
        global_group_path=f"{collision_root_path}/global" if has_global else None,
        global_includes=tuple(global_includes),
    )


def _author_collision_group_prims(stage: object, plan: _CollisionGroupPlan) -> int:
    """按 plan 创建 ``UsdPhysics.CollisionGroup``（含 collection 与 filteredGroups）。

    只用 ``UsdPhysics`` / ``Usd.CollectionAPI``，不依赖 PhysX schema，因此可以在没有 Isaac
    app 的环境里用内存 stage 直接校验 group 结构。返回写入的 filteredGroups 目标数。
    """

    from pxr import Sdf, Usd, UsdPhysics

    authored = 0
    if plan.global_group_path is not None:
        global_group = UsdPhysics.CollisionGroup.Define(stage, plan.global_group_path)
        includes = Usd.CollectionAPI.Apply(
            global_group.GetPrim(), "colliders"
        ).CreateIncludesRel()
        for include_path in plan.global_includes:
            includes.AddTarget(Sdf.Path(include_path))
        filtered = global_group.CreateFilteredGroupsRel()
        for target in plan.global_group_filtered_groups():
            filtered.AddTarget(Sdf.Path(target))
            authored += 1
    for index, group_path in enumerate(plan.env_group_paths):
        group = UsdPhysics.CollisionGroup.Define(stage, group_path)
        Usd.CollectionAPI.Apply(
            group.GetPrim(), "colliders"
        ).CreateIncludesRel().AddTarget(Sdf.Path(plan.env_includes[index]))
        filtered = group.CreateFilteredGroupsRel()
        for target in plan.env_group_filtered_groups(index):
            filtered.AddTarget(Sdf.Path(target))
            authored += 1
    return authored


def _author_scene_invert_collision_filter(
    stage: object, physics_scene_path: str
) -> None:
    """在 physics scene 上打开 ``invertCollisionGroupFilter``，使 filteredGroups 成为白名单。

    该属性属于 PhysX USD schema（``PhysxSchema.PhysxSceneAPI``），只有在 Isaac/omni.physx
    扩展加载后才可导入；tiled scene 构建始终运行在已启动的 Isaac app 内，因此在此延迟导入。
    """

    from pxr import PhysxSchema

    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if not scene_prim.IsValid():
        raise RuntimeError(
            "cannot enable inverted collision filter: no physics scene at "
            f"{physics_scene_path}"
        )
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateInvertCollisionGroupFilterAttr().Set(True)


def _collision_filter_participant_paths(
    *, stage: object, env_root: str
) -> tuple[str, ...]:
    """收集一个 env 下可用于 ``FilteredPairsAPI`` 的 physics prim path。"""

    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(env_root)
    if not root.IsValid():
        return ()

    result: list[str] = []

    def visit(prim: object) -> None:
        """深度优先收集能参与 collision filtering 的 prim path。"""

        if _is_collision_filter_participant(prim, UsdPhysics):
            result.append(str(prim.GetPath()))
            return
        for child in prim.GetChildren():
            visit(child)

    visit(root)
    return tuple(result)


def _is_collision_filter_participant(prim: object, usd_physics: object) -> bool:
    """判断 prim 是否能承载 ``FilteredPairsAPI``。"""

    return (
        prim.HasAPI(usd_physics.ArticulationRootAPI)
        or prim.HasAPI(usd_physics.RigidBodyAPI)
        or prim.HasAPI(usd_physics.CollisionAPI)
    )


def _apply_env_pair_filters(
    stage: object, participant_paths: Sequence[Sequence[str]]
) -> int:
    """给不同 env 的 participant author pairwise collision filters。"""

    from pxr import Sdf, UsdPhysics

    authored = 0
    for source_env_index, source_paths in enumerate(participant_paths):
        for target_paths in participant_paths[source_env_index + 1 :]:
            for source_path in source_paths:
                source = stage.GetPrimAtPath(source_path)
                if not source.IsValid():
                    continue
                filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(source)
                rel = filtered_pairs.CreateFilteredPairsRel()
                for target_path in target_paths:
                    target = stage.GetPrimAtPath(target_path)
                    if not target.IsValid():
                        continue
                    rel.AddTarget(Sdf.Path(target_path))
                    authored += 1
    return authored


def _resolve_physics_scene_path(stage: object, configured_path: str | None) -> str:
    """解析显式 PhysicsScene，或要求 stage 中恰好存在一个 scene。"""

    from pxr import UsdPhysics

    if configured_path is not None:
        prim = stage.GetPrimAtPath(configured_path)
        if not prim.IsValid():
            raise RuntimeError(
                f"tiled.clone.physics_scene_path does not exist: {configured_path}"
            )
        if not prim.IsA(UsdPhysics.Scene):
            raise RuntimeError(
                "tiled.clone.physics_scene_path is not a UsdPhysics.Scene: "
                f"{configured_path}"
            )
        return str(prim.GetPath())

    candidates = tuple(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No UsdPhysics.Scene found; create World before tiled scene or set "
            "tiled.clone.physics_scene_path"
        )
    raise RuntimeError(
        "Multiple UsdPhysics.Scene prims found; set "
        "tiled.clone.physics_scene_path explicitly: " + ", ".join(candidates)
    )


def _resolve_global_collision_paths(
    *,
    stage: object,
    config: TiledEnvConfig,
    env_roots: Sequence[str],
) -> tuple[str, ...]:
    """解析 auto/显式/追加 global collider paths，并做 stage 级校验。"""

    configured = config.clone.global_collision_paths
    if configured == "auto":
        base_paths: Sequence[str] = _global_collision_paths(stage, env_roots)
    else:
        base_paths = configured

    result: list[str] = []
    seen: set[str] = set()
    for path in (*base_paths, *config.clone.extra_global_collision_paths):
        text = str(path)
        if text in seen:
            continue
        _validate_global_collision_path(
            stage=stage,
            path=text,
            env_roots=env_roots,
        )
        seen.add(text)
        result.append(text)
    return tuple(result)


def _validate_global_collision_path(
    *,
    stage: object,
    path: str,
    env_roots: Sequence[str],
) -> None:
    """确认 global collection target 存在、与 env roots 互斥且包含 collider。"""

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"global collision path does not exist: {path}")
    for env_root in env_roots:
        root = str(env_root).rstrip("/")
        if path == root or path.startswith(f"{root}/") or root.startswith(f"{path}/"):
            raise RuntimeError(
                f"global collision path {path!r} overlaps tiled env root {root!r}"
            )
    if not _prim_subtree_has_collider(prim):
        raise RuntimeError(
            f"global collision path contains no UsdPhysics collider: {path}"
        )


def _prim_subtree_has_collider(root_prim: object) -> bool:
    """返回 collection target 子树内是否至少有一个 CollisionAPI prim。"""

    from pxr import Usd, UsdPhysics

    return any(
        prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(root_prim)
    )


def _global_collision_paths(stage: object, env_roots: Sequence[str]) -> list[str]:
    """收集需要和所有 env 保持碰撞的 stage-level prim。"""

    candidates = (
        "/World/defaultGroundPlane",
        "/World/GroundPlane",
        "/World/groundPlane",
        "/World/ground",
    )
    env_root_set = set(env_roots)
    return [
        path
        for path in candidates
        if path not in env_root_set and stage.GetPrimAtPath(path).IsValid()
    ]
