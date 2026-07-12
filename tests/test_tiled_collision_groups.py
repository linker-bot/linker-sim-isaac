"""纯数据校验 tiled env 碰撞组规划（不依赖 pxr / Isaac）。

这里只测试 ``_plan_env_collision_groups`` 的拓扑与复杂度：真正的 USD authoring 由
``_author_collision_group_prims``（仅 UsdPhysics）和 scene invert（PhysxSchema）负责，
在 Isaac 环境的 pxr 测试里覆盖。
"""

from __future__ import annotations

from linkerbot_sim.tiled.config import _normalize_collision_filter_strategy
from linkerbot_sim.tiled.scene.collision_filter import _plan_env_collision_groups


def test_plan_env_collision_groups_uses_invert_whitelist() -> None:
    plan = _plan_env_collision_groups(
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
        global_paths=("/World/defaultGroundPlane",),
        collision_root_path="/World/collisions",
    )

    assert plan.invert_filter is True
    assert plan.env_group_paths == (
        "/World/collisions/env_0",
        "/World/collisions/env_1",
    )
    # 每个 env 组只 include 自己的子树，彼此互斥（避免 collider 落入多个组）。
    assert plan.env_includes == ("/World/envs/env_0", "/World/envs/env_1")
    # invert 白名单：env 只与自身 + global 碰撞，不与其它 env 碰撞。
    assert plan.env_group_filtered_groups(0) == (
        "/World/collisions/env_0",
        "/World/collisions/global",
    )
    assert plan.env_group_filtered_groups(1) == (
        "/World/collisions/env_1",
        "/World/collisions/global",
    )
    # global 组与每个 env 都碰撞（地面不被过滤）。
    assert plan.global_group_path == "/World/collisions/global"
    assert plan.global_includes == ("/World/defaultGroundPlane",)
    assert plan.global_group_filtered_groups() == (
        "/World/collisions/env_0",
        "/World/collisions/env_1",
    )


def test_plan_env_collision_groups_without_global() -> None:
    plan = _plan_env_collision_groups(
        env_roots=("/World/envs/env_0",),
        global_paths=(),
        collision_root_path="/World/collisions",
    )

    assert plan.global_group_path is None
    assert plan.global_includes == ()
    # 没有 global 时 env 只与自身碰撞。
    assert plan.env_group_filtered_groups(0) == ("/World/collisions/env_0",)


def test_plan_env_collision_groups_dedupes_global_paths() -> None:
    plan = _plan_env_collision_groups(
        env_roots=("/World/envs/env_0",),
        global_paths=("/World/ground", "/World/ground", "/World/ground2"),
        collision_root_path="/c",
    )

    assert plan.global_includes == ("/World/ground", "/World/ground2")


def test_plan_env_collision_groups_target_count_is_linear() -> None:
    # 目标数 ~= E*(self+global) + E(global 白名单) = 3E，随 env 线性增长，而非 O(E^2)。
    for num_envs in (1, 8, 64, 256):
        plan = _plan_env_collision_groups(
            env_roots=tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
            global_paths=("/World/ground",),
            collision_root_path="/World/collisions",
        )
        assert plan.total_filtered_group_targets() == 3 * num_envs
        assert len(plan.env_group_paths) == num_envs


def test_normalize_collision_filter_strategy_accepts_only_canonical_values() -> None:
    assert _normalize_collision_filter_strategy(None) == "collision_groups"
    assert (
        _normalize_collision_filter_strategy("collision_groups") == "collision_groups"
    )
    assert _normalize_collision_filter_strategy("filtered_pairs") == "filtered_pairs"
    for value in ("groups", "collision-groups", "pairs", "nonsense"):
        try:
            _normalize_collision_filter_strategy(value)
        except ValueError as exc:
            assert "collision_groups or filtered_pairs" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"non-canonical strategy {value!r} was accepted")
