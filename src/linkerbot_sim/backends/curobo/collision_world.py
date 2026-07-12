"""项目 ``CollisionObject`` 到 cuRobo ``Scene`` 的适配。

cuRobo 的 collision world 更新入口是 ``solver.update_world(SceneCfg)``，公开 API 不暴露
World/WorldView/ObstacleHandle 增量对象。因此本模块采用“按当前规划快照重建 SceneCfg，
再整体推送给 IK / MotionPlanner / BatchMotionPlanner”的策略。

这比逐 obstacle handle 复杂度低，也更贴近 cuRobo 的公开 API。后续如果需要 tiled
``multi_env=True`` 的每 env 不同障碍物，可以在这个模块上扩展成 scene list / env-indexed
SceneCfg，而不影响上层 context API。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


@dataclass
class CuroboCollisionWorld:
    """持有当前 cuRobo SceneCfg 快照并同步到 context solver。"""

    context: object
    collision_objects: tuple[CollisionObject, ...] = ()

    def __post_init__(self) -> None:
        """创建初始 scene 并推送给所有可更新 solver。"""

        self.scene_cfg = make_curobo_scene_cfg(self.context, self.collision_objects)
        self.update_solvers()

    @property
    def num_enabled_obstacles(self) -> int:
        """返回 materialize 后真实进入 checker 的 obstacle 数量。"""

        return sum(materialized_obstacle_counts(self.scene_cfg).values())

    @property
    def num_canonical_obstacles(self) -> int:
        """返回启用的后端无关输入 obstacle 数量。"""

        return sum(1 for obj in self.collision_objects if obj.enabled)

    @property
    def materialized_counts(self) -> dict[str, int]:
        """按 cuRobo v0.8.0 cache 类型返回 materialized 数量。"""

        return materialized_obstacle_counts(self.scene_cfg)

    def sync(self, collision_objects: Sequence[CollisionObject]) -> None:
        """用最新项目碰撞对象重建 cuRobo SceneCfg 并推送。"""

        self.collision_objects = tuple(collision_objects)
        self.scene_cfg = make_curobo_scene_cfg(self.context, self.collision_objects)
        self.update_solvers()

    def update_solvers(self) -> None:
        """把当前 SceneCfg 同步给 IK、单轨迹 planner 和 batch planner。

        基础 URDF fallback 通常没有 collision spheres，也可能没有创建
        ``scene_collision_checker``。这种情况下调用 cuRobo ``update_world`` 会在后端内部访问
        空 checker，因此这里会先判断 solver 是否真的具备 scene collision checker。
        """

        solver_provider = getattr(self.context, "existing_solvers", None)
        solvers = (
            solver_provider()
            if callable(solver_provider)
            else tuple(
                getattr(self.context, name, None)
                for name in ("ik_solver", "motion_planner", "batch_motion_planner")
            )
        )
        for solver in solvers:
            if solver is None or not _solver_supports_scene_update(solver):
                continue
            solver.update_world(self.scene_cfg)


def make_curobo_scene_cfg(context, collision_objects: Sequence[CollisionObject]):
    """把项目碰撞对象列表转换成 cuRobo ``Scene`` / ``SceneCfg``。

    仅转换 enabled 对象；disabled 对象在规划快照中可以保留，但不会进入后端碰撞世界。
    输入支持 cuboid、sphere、capsule。cuRobo v0.8.0 的 GPU SceneData 只装载
    cuboid/mesh/voxel，因此 sphere/capsule 必须在这里转换为保守 cuboid，不能原样放进
    ``SceneCfg``。
    """

    scene = context.scene_module
    cuboids = []
    for obj in collision_objects:
        if not obj.enabled:
            continue
        converted = _make_curobo_obstacle(scene, obj)
        shape = obj.shape.lower()
        if shape in {"cuboid", "sphere", "capsule"}:
            cuboids.append(converted)
        else:
            raise ValueError(
                f"Unsupported cuRobo collision object shape: {obj.shape!r}"
            )
    scene_cfg = scene.Scene(cuboid=cuboids)
    validator = getattr(context, "validate_collision_cache_capacity", None)
    if callable(validator):
        validator(materialized_obstacle_counts(scene_cfg))
    return scene_cfg


def _make_curobo_obstacle(scene, obj: CollisionObject):
    """把单个 ``CollisionObject`` 转为 cuRobo obstacle dataclass。"""

    shape = obj.shape.lower()
    pose = _pose_list_wxyz_from_matrix(obj.pose_matrix())
    size = tuple(float(value) for value in obj.padded_size())
    if shape == "cuboid":
        _require_size(obj, size, expected=3)
        _require_positive_dimensions(obj, size)
        return scene.Cuboid(name=obj.name, pose=pose, dims=list(size))
    if shape == "sphere":
        _require_size(obj, size, expected=1)
        radius = float(size[0])
        _require_positive_dimensions(obj, (radius,))
        diameter = 2.0 * radius
        return scene.Cuboid(
            name=obj.name,
            pose=pose,
            dims=[diameter, diameter, diameter],
        )
    if shape == "capsule":
        _require_size(obj, size, expected=2)
        radius, length = float(size[0]), float(size[1])
        if radius <= 0.0 or length < 0.0:
            raise ValueError(
                f"capsule collision object {obj.name!r} requires radius > 0 "
                "and length >= 0"
            )
        diameter = 2.0 * radius
        return scene.Cuboid(
            name=obj.name,
            pose=pose,
            dims=[diameter, diameter, length + diameter],
        )
    raise ValueError(f"Unsupported cuRobo collision object shape: {obj.shape!r}")


def _pose_list_wxyz_from_matrix(pose_matrix: np.ndarray) -> list[float]:
    """把 4x4 位姿矩阵转成 cuRobo ``[x,y,z,qw,qx,qy,qz]`` pose list。"""

    matrix = np.asarray(pose_matrix, dtype=float).reshape(4, 4)
    quat_wxyz = matrix_to_quat_wxyz(matrix[:3, :3])
    return [
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    ]


def _require_size(
    obj: CollisionObject, size: tuple[float, ...], *, expected: int
) -> None:
    """校验指定 shape 的尺寸数量。"""

    if len(size) != expected:
        raise ValueError(
            f"{obj.shape} collision object {obj.name!r} expected {expected} size "
            f"values, got {len(size)}"
        )


def _require_positive_dimensions(obj: CollisionObject, size: tuple[float, ...]) -> None:
    """拒绝零或负尺寸，避免创建无效 GPU obstacle。"""

    if any(value <= 0.0 for value in size):
        raise ValueError(
            f"{obj.shape} collision object {obj.name!r} requires positive dimensions"
        )


def materialized_obstacle_counts(scene_cfg: object) -> dict[str, int]:
    """统计 cuRobo v0.8.0 SceneData 原生支持的 obstacle 数量。"""

    return {
        shape: len(getattr(scene_cfg, shape, ()) or ()) for shape in ("cuboid", "mesh")
    }


def _solver_supports_scene_update(solver: object) -> bool:
    """判断 cuRobo solver 是否可以安全调用 ``update_world``。"""

    update_world = getattr(solver, "update_world", None)
    if not callable(update_world):
        return False
    # 真实 cuRobo planner/solver 在没有 scene_collision_cfg 时会留下 None checker；这种配置下
    # 没有动态 obstacle 缓存，不能调用 update_world。
    if hasattr(solver, "scene_collision_checker"):
        return getattr(solver, "scene_collision_checker") is not None
    return True


__all__ = [
    "CuroboCollisionWorld",
    "make_curobo_scene_cfg",
    "materialized_obstacle_counts",
]
