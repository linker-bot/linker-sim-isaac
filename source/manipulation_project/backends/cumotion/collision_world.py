"""项目碰撞对象到 cuMotion World 的适配。

上层 planning 使用后端无关的 ``CollisionObject`` 描述障碍物；本模块把这些对象转换为
cuMotion ``World`` 中的 obstacle，并维护按名称索引的 handle。位置姿态以世界/机器人 base
坐标下的 4x4 齐次矩阵表达，尺寸单位为 m，padding 在 ``CollisionObject`` 中统一处理。

职责边界:
    * 只做形状名称、尺寸和位姿的后端适配。
    * 支持对已知对象做增量同步，但不从 Isaac stage 自动提取障碍物。
    * 不决定 IK 或 planner 是否避障；调用方通过请求里的碰撞模式选择是否使用 world。
    * inspector wrapper 只做距离/碰撞查询适配，不把诊断结果自动写入规划结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from manipulation_project.backends.cumotion.pose_adapter import pose_from_matrix
from manipulation_project.planning.collision_objects import CollisionObject


class CuMotionCollisionWorld:
    """持有 cuMotion ``World`` 和对应 obstacle handles。

    ``handles`` 和 ``obstacles`` 都用对象名称索引，便于同一个 world 在多次规划之间同步。
    构造后会立即创建并刷新 ``world_view``；后续对 ``World`` 的修改必须再调用 ``update`` 或
    ``sync``，solver/planner 才能看到新 pose、启停状态或新增/删除的 obstacle。
    """

    def __init__(
        self, context, collision_objects: Sequence[CollisionObject] = ()
    ) -> None:
        """根据项目碰撞对象初始化 cuMotion world view。"""

        self.context = context
        self.cumotion = context.cumotion
        # cuMotion 的 pybind 类型 ``World`` 不是公开构造入口；官方 Python API 通过
        # ``create_world()`` 工厂函数返回可用实例。这里显式走工厂，避免在真实 Isaac/cuMotion
        # 环境中触发 ``World: No constructor defined``。
        #
        # 每个 collision-free IK / MotionPlanner 请求通常构造一个 world 快照；handles 和几何
        # 签名保留下来，让调用方在复用同一 world 时可以通过 sync 增量更新。
        self.world = self.cumotion.create_world()

        # handle 是 cuMotion World.add_obstacle(...) 返回的 ObstacleHandle
        # 用于后续对 world 里的这个障碍物做操作：移动、启用、禁用、删除、查询
        self.handles = {}
        # obstacles 是创建出来的 cuMotion Obstacle 几何对象
        # 保存原始 obstacle 对象本身，主要用于本地记录/调试/可能的后续扩展
        self.obstacles = {}
        # 给 CollisionObject 生成一个几何签名，描述几何体形状及大小。
        self._geometry_signatures = {}

        # 添加 collision_objects
        for obj in collision_objects:
            self.add(obj)
        self.world_view = self.world.add_world_view()
        self.world_view.update()

    def add(self, obj: CollisionObject):
        """添加一个项目碰撞对象，并返回 cuMotion obstacle handle。

        disabled 对象不会加入 cuMotion world，返回 ``None``。如果同名对象已经存在，
        调用方应先 ``remove`` 或使用 ``sync``，避免本地 handle 字典覆盖旧 handle 后
        失去删除旧 obstacle 的机会。
        """

        # disabled 对象保留在请求中但不加入后端 world，可用于配置快速开关障碍物。
        if not obj.enabled:
            return None
        obstacle = self._make_obstacle(obj)
        handle = self.world.add_obstacle(
            obstacle, pose_from_matrix(self.cumotion, obj.pose_matrix())
        )
        self.handles[obj.name] = handle
        self.obstacles[obj.name] = obstacle
        self._geometry_signatures[obj.name] = _geometry_signature(obj)

        return handle

    def set_pose(self, name: str, pose_matrix) -> None:
        """更新已添加障碍物的 pose。

        只更新 cuMotion ``World`` 内部状态；如果已有 solver/planner 持有 ``world_view``，
        还需要调用 ``update`` 才会把修改同步到 view。
        """

        handle = self._handle(name)
        self.world.set_pose(handle, pose_from_matrix(self.cumotion, pose_matrix))

    def enable(self, name: str) -> None:
        """启用已添加障碍物。

        该操作保留原 obstacle handle 和几何属性，只改变参与距离/碰撞查询的开关。
        """

        self.world.enable_obstacle(self._handle(name))

    def disable(self, name: str) -> None:
        """禁用已添加障碍物。

        禁用后对象仍留在 ``handles`` 中，之后可通过 ``enable`` 或 ``sync`` 重新启用。
        """

        self.world.disable_obstacle(self._handle(name))

    def remove(self, name: str) -> None:
        """从 world 中删除已添加障碍物。

        cuMotion 的 obstacle handle 删除后不可再用；本方法会同步清理本地 handle、obstacle
        和几何签名缓存。
        """

        handle = self._handle(name)
        self.world.remove_obstacle(handle)
        self.handles.pop(name, None)
        self.obstacles.pop(name, None)
        self._geometry_signatures.pop(name, None)

    def sync(self, collision_objects: Sequence[CollisionObject]) -> None:
        """按对象名称同步一组障碍物。

        已存在对象会更新 pose 和 enabled 状态；新增对象会添加；缺失对象会删除。
        几何尺寸变化需要重新创建 obstacle，因此这里也采用 remove + add。
        """

        incoming = {obj.name: obj for obj in collision_objects}
        for name in tuple(self.handles):
            if name not in incoming:
                self.remove(name)
        for name, obj in incoming.items():
            if not obj.enabled:
                if name in self.handles:
                    self.disable(name)
                continue
            if name not in self.handles:
                self.add(obj)
                continue
            if self._geometry_signatures.get(name) != _geometry_signature(obj):
                self.remove(name)
                self.add(obj)
                continue
            self.set_pose(name, obj.pose_matrix())
            self.enable(name)
        self.update()

    def update(self) -> None:
        """刷新 world view，使最近的 world 修改对 solver/planner 可见。

        cuMotion 的 ``World`` 和 ``WorldView`` 是分离对象；添加、删除、移动或启停 obstacle 后
        必须调用本方法，后续 inspector、IK 或 planner 才会看到最新碰撞世界。
        """

        self.world_view.update()

    def make_world_inspector(self):
        """创建只查询 world obstacle 的 inspector。

        inspector 使用当前 ``world_view``；若之后继续修改 ``World``，应先调用 ``update`` 再查询。
        """

        return CuMotionWorldInspector(self.context, self.world_view)

    def make_robot_world_inspector(self):
        """创建机器人与 world 碰撞诊断 inspector。

        返回对象使用同一个 ``world_view`` 检查机器人碰撞球与 obstacle 的关系，也可清空或替换
        world view 来只检查自碰。
        """

        return CuMotionRobotWorldInspector(self.context, self.world_view)

    def _handle(self, name: str):
        """按名称读取 obstacle handle。"""

        try:
            return self.handles[name]
        except KeyError as exc:
            raise KeyError(f"cuMotion obstacle {name!r} not found") from exc

    def _make_obstacle(self, obj: CollisionObject):
        """把项目形状名称和尺寸映射为 cuMotion obstacle 属性。"""

        shape = obj.shape.lower()
        if shape not in {"cuboid", "sphere", "capsule"}:
            raise ValueError(f"Unsupported collision object shape: {obj.shape!r}")
        obstacle_type = getattr(self.cumotion.Obstacle.Type, shape.upper())
        obstacle = self.cumotion.create_obstacle(obstacle_type)
        size = np.asarray(obj.padded_size(), dtype=float)
        if shape == "cuboid":
            obstacle.set_attribute(
                self.cumotion.Obstacle.Attribute.SIDE_LENGTHS, size.reshape(3)
            )
        elif shape == "sphere":
            obstacle.set_attribute(
                self.cumotion.Obstacle.Attribute.RADIUS, float(size[0])
            )
        elif shape == "capsule":
            obstacle.set_attribute(
                self.cumotion.Obstacle.Attribute.RADIUS, float(size[0])
            )
            obstacle.set_attribute(
                self.cumotion.Obstacle.Attribute.HEIGHT, float(size[1])
            )
        return obstacle


@dataclass
class CuMotionWorldInspector:
    """轻量封装 cuMotion ``WorldInspector``。

    ``WorldInspector`` 是 cuMotion 提供的环境障碍物查询接口：它只看 ``WorldView`` 中的
    obstacle，不涉及机器人模型。这里的 wrapper 用于回答“某个点/球和当前环境障碍物的
    距离或碰撞关系是什么”，常用于单元测试、调试碰撞世界是否构造正确，以及打印诊断信息。

    该 wrapper 只负责把输入点规范化成 3D numpy 数组，并把 pybind 返回值转成 Python
    ``bool``/``float``/``list``。它不缓存查询结果；每次调用都读取当前 ``world_view``。
    """

    context: object
    world_view: object

    def __post_init__(self) -> None:
        self.cumotion = self.context.cumotion
        self.inspector = self.cumotion.create_world_inspector(self.world_view)

    def num_enabled_obstacles(self) -> int:
        """返回当前 world view 中启用的 obstacle 数量。

        该值反映最近一次 ``world_view.update()`` 后的状态；如果调用方刚修改过 ``World``，
        应先通过 ``CuMotionCollisionWorld.update`` 刷新。
        """

        return int(self.inspector.num_enabled_obstacles())

    def is_enabled(self, obstacle_handle) -> bool:
        """查询指定 obstacle handle 在当前 view 中是否启用。

        ``obstacle_handle`` 应来自同一个 ``World``；跨 world/view 混用 handle 的行为由
        cuMotion 后端决定，本 wrapper 不做额外转换。
        """

        return bool(self.inspector.is_enabled(obstacle_handle))

    def pose(self, obstacle_handle):
        """返回指定 obstacle 的 cuMotion ``Pose3``。

        返回对象保持后端原生类型，便于测试直接检查 translation/rotation；需要 numpy 矩阵时
        应在调用处显式转换。
        """

        return self.inspector.pose(obstacle_handle)

    def in_collision(self, center, radius: float, obstacle_handle=None) -> bool:
        """检查一个球是否与任意 obstacle 或指定 obstacle 碰撞。

        ``center`` 使用 world/base 坐标，``radius`` 单位为 m。传入 ``obstacle_handle`` 时只
        检查该 obstacle；否则检查当前 view 中所有 enabled obstacles。
        """

        center = np.asarray(center, dtype=float).reshape(3)
        if obstacle_handle is None:
            return bool(self.inspector.in_collision(center, float(radius)))
        return bool(self.inspector.in_collision(obstacle_handle, center, float(radius)))

    def min_distance(self, point, *, gradient=None) -> float:
        """返回点到所有 enabled obstacles 的最小 signed distance。

        ``gradient`` 可传入可写 numpy 数组，cuMotion 会在支持时写入距离梯度。
        """

        point = np.asarray(point, dtype=float).reshape(3)
        return float(self.inspector.min_distance(point, gradient))

    def distance_to(self, obstacle_handle, point, *, gradient=None) -> float:
        """返回点到指定 obstacle 的 signed distance。

        ``point`` 会按 world/base 坐标 reshape 为 ``(3,)``。距离为负表示点落在 obstacle 内部，
        正值表示外部间隙。
        """

        point = np.asarray(point, dtype=float).reshape(3)
        return float(self.inspector.distance_to(obstacle_handle, point, gradient))

    def distances_to(
        self, point, *, compute_distance_gradients: bool = True
    ) -> tuple[list[float], list[np.ndarray] | None] | None:
        """返回点到所有 enabled obstacles 的距离列表和可选梯度列表。

        cuMotion 在没有 enabled obstacle 时可能返回 ``None``。当
        ``compute_distance_gradients=False`` 或后端不返回梯度时，第二个返回值为 ``None``。
        """

        point = np.asarray(point, dtype=float).reshape(3)
        result = self.inspector.distances_to(
            point, bool(compute_distance_gradients)
        )
        if result is None:
            return None
        distances, gradients = result
        parsed_gradients = (
            None
            if gradients is None
            else [np.asarray(gradient, dtype=float) for gradient in gradients]
        )
        return (
            [float(distance) for distance in distances],
            parsed_gradients,
        )


@dataclass
class CuMotionRobotWorldInspector:
    """轻量封装 cuMotion ``RobotWorldInspector``。

    ``RobotWorldInspector`` 是 cuMotion 提供的机器人碰撞诊断接口：它根据机器人描述和可选
    ``WorldView``，检查机器人自身 collision spheres 的自碰，以及机器人 world-collision
    spheres 与环境 obstacle 的碰撞/距离关系。这里的 wrapper 用于回答“当前关节构型是否自碰、
    是否碰到环境、最近环境障碍物距离是多少”，不参与轨迹规划决策本身。

    输入的 ``cspace_position`` 始终按 ``CuMotionContext.joint_names()`` 的 C-space 顺序排列。
    该对象面向诊断和测试：它不会改变 planner/IK 行为，也不会自动写入 ``MotionResult``。
    """

    context: object
    world_view: object | None = None

    def __post_init__(self) -> None:
        self.cumotion = self.context.cumotion
        self.inspector = self.cumotion.create_robot_world_inspector(
            self.context.robot_description, self.world_view
        )

    def set_world_view(self, world_view) -> None:
        """切换用于机器人-环境碰撞查询的 world view。

        适合在同一个 robot inspector 上复用不同静态场景；传入 view 前应确保它已经
        ``update``，否则查询可能仍是旧 obstacle 状态。
        """

        self.inspector.set_world_view(world_view)

    def clear_world_view(self) -> None:
        """清空 world view；后续查询只涉及机器人自碰。

        清空后 ``in_collision_with_obstacle`` 和距离到 obstacle 的查询只返回空环境语义；
        ``in_self_collision`` 仍可继续使用。
        """

        self.inspector.clear_world_view()

    def in_self_collision(self, cspace_position) -> bool:
        """判断机器人在给定 C-space 构型下是否自碰。

        ``cspace_position`` 必须按 ``context.joint_names()`` 顺序排列；函数只检查机器人内部
        collision spheres，不使用 world obstacles。
        """

        return bool(
            self.inspector.in_self_collision(
                np.asarray(cspace_position, dtype=float).reshape(-1)
            )
        )

    def frames_in_self_collision(self, cspace_position) -> list[tuple[str, str]]:
        """返回发生自碰的 frame 名称对列表。

        返回值是 ``(frame_a, frame_b)`` 字符串元组列表，适合写入诊断日志或单元测试断言。
        空列表表示当前构型未检测到自碰 frame pair。
        """

        return [
            (str(first), str(second))
            for first, second in self.inspector.frames_in_self_collision(
                np.asarray(cspace_position, dtype=float).reshape(-1)
            )
        ]

    def in_collision_with_obstacle(self, cspace_position) -> bool:
        """判断机器人在给定 C-space 构型下是否碰到当前 world view 的 obstacle。

        只检查机器人 world-collision spheres 与已启用 obstacle 的关系；自碰仍需单独调用
        ``in_self_collision``。
        """

        return bool(
            self.inspector.in_collision_with_obstacle(
                np.asarray(cspace_position, dtype=float).reshape(-1)
            )
        )

    def min_distance_to_obstacle(self, cspace_position) -> float:
        """返回机器人碰撞球到 world obstacles 的最小 signed distance。

        距离为负表示至少一个 world-collision sphere 已侵入 obstacle；正值表示最小间隙，
        单位为 m。
        """

        return float(
            self.inspector.min_distance_to_obstacle(
                np.asarray(cspace_position, dtype=float).reshape(-1)
            )
        )

    def distance_to_obstacle(
        self, obstacle_handle, world_collision_sphere_index: int, cspace_position
    ) -> float:
        """返回某个机器人 world-collision sphere 到指定 obstacle 的距离。

        ``world_collision_sphere_index`` 使用 cuMotion robot description 中 world collision sphere
        的索引，可结合 ``world_collision_sphere_frame_name`` 定位对应 link/frame。
        """

        return float(
            self.inspector.distance_to_obstacle(
                obstacle_handle,
                int(world_collision_sphere_index),
                np.asarray(cspace_position, dtype=float).reshape(-1),
            )
        )

    def num_self_collision_spheres(self) -> int:
        """返回机器人自碰检测使用的 collision sphere 数量。

        该数量来自机器人描述文件，通常用于遍历 sphere frame 名或调试碰撞模型覆盖范围。
        """

        return int(self.inspector.num_self_collision_spheres())

    def num_world_collision_spheres(self) -> int:
        """返回机器人与 world 碰撞检测使用的 collision sphere 数量。

        world collision spheres 可以与自碰 spheres 数量不同，具体取决于 cuMotion 配置里的
        collision sphere 集合。
        """

        return int(self.inspector.num_world_collision_spheres())

    def self_collision_sphere_frame_name(self, index: int) -> str:
        """返回第 ``index`` 个自碰 sphere 所属 frame 名称。

        索引范围应小于 ``num_self_collision_spheres``；越界错误由 cuMotion 后端抛出。
        """

        return str(self.inspector.self_collision_sphere_frame_name(int(index)))

    def world_collision_sphere_frame_name(self, index: int) -> str:
        """返回第 ``index`` 个 world-collision sphere 所属 frame 名称。"""

        return str(self.inspector.world_collision_sphere_frame_name(int(index)))

    def self_collision_sphere_positions(self, cspace_position) -> list[np.ndarray]:
        """返回给定构型下所有自碰 sphere 的 3D 位置。

        每个返回数组 shape 为 ``(3,)``，单位 m，顺序与 ``self_collision_sphere_frame_name``
        的索引一致。
        """

        return [
            np.asarray(position, dtype=float).reshape(3)
            for position in self.inspector.self_collision_sphere_positions(
                np.asarray(cspace_position, dtype=float).reshape(-1)
            )
        ]

    def self_collision_sphere_radii(self) -> list[float]:
        """返回所有自碰 sphere 的半径，单位 m。

        半径顺序与 self-collision sphere 索引一致，可与 ``self_collision_sphere_positions``
        组合用于调试可视化。
        """

        return [float(radius) for radius in self.inspector.self_collision_sphere_radii()]


def _geometry_signature(obj: CollisionObject) -> tuple[str, tuple[float, ...]]:
    """返回用于判断 obstacle 是否需要重建的几何签名。"""

    shape = obj.shape.lower()
    size = tuple(float(value) for value in obj.padded_size())
    return shape, size


def make_collision_world(
    context, collision_objects: Sequence[CollisionObject] = ()
) -> CuMotionCollisionWorld:
    """构建 cuMotion collision world。

    参数:
        context: 已加载 robot description 和 cuMotion 模块的 ``CuMotionContext``。
        collision_objects: 可选规划层障碍物列表；会立即写入后端 ``World`` 并刷新 view。
    返回:
        ``CuMotionCollisionWorld``，供 IK、MotionPlanner 或诊断 inspector 复用。
    """

    return CuMotionCollisionWorld(context, collision_objects)
