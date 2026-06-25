"""项目碰撞对象到 cuMotion World 的适配。

上层 planning 使用后端无关的 ``CollisionObject`` 描述障碍物；本模块在进入 cuMotion IK 前
把它们转换为 cuMotion ``World`` 中的 obstacle。位置姿态以世界/机器人 base 坐标下的 4x4
齐次矩阵表达，尺寸单位为 m，padding 在 ``CollisionObject`` 中统一处理。

职责边界:
    * 只做形状名称、尺寸和位姿的后端适配。
    * 不从 Isaac stage 自动提取障碍物，也不维护动态碰撞对象。
    * 不决定 IK 是否避障；调用方通过 ``IKRequest.avoid_collisions`` 选择是否使用。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.backends.cumotion.pose_adapter import pose_from_matrix
from manipulation_project.planning.collision_objects import CollisionObject


class CuMotionCollisionWorld:
    """持有 cuMotion ``World`` 和对应 obstacle handles。

    ``handles`` 用对象名称索引，便于未来做增量更新或调试输出。当前实现构造后立即创建
    ``world_view`` 并刷新，满足 collision-free IK solver 对静态世界快照的要求。
    """

    def __init__(
        self, context, collision_objects: Sequence[CollisionObject] = ()
    ) -> None:
        """根据项目碰撞对象初始化 cuMotion world view。"""

        self.context = context
        self.cumotion = context.cumotion
        # 每个 collision-free IK 请求构造一个静态 world 快照；handles 保留下来便于未来做
        # 增量更新或输出调试信息。
        self.world = self.cumotion.World()
        self.handles = {}
        for obj in collision_objects:
            self.add(obj)
        self.world_view = self.world.add_world_view()
        self.world_view.update()

    def add(self, obj: CollisionObject) -> None:
        """添加一个项目碰撞对象。"""

        # disabled 对象保留在请求中但不加入后端 world，可用于配置快速开关障碍物。
        if not obj.enabled:
            return
        obstacle = self._make_obstacle(obj)
        handle = self.world.add_obstacle(
            obstacle, pose_from_matrix(self.cumotion, obj.pose_matrix())
        )
        self.handles[obj.name] = handle

    def update(self) -> None:
        """刷新 world view。"""

        self.world_view.update()

    def _make_obstacle(self, obj: CollisionObject):
        """把项目形状名称和尺寸映射为 cuMotion obstacle 属性。"""

        # 项目内部称 box，cuMotion API 称 cuboid；其它形状直接按枚举名映射。
        shape = obj.shape.lower()
        if shape == "box":
            shape = "cuboid"
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


def make_collision_world(
    context, collision_objects: Sequence[CollisionObject] = ()
) -> CuMotionCollisionWorld:
    """构建 cuMotion collision world。"""

    return CuMotionCollisionWorld(context, collision_objects)
