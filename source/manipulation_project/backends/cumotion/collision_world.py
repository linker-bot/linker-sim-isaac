"""项目碰撞对象到 cuMotion World 的适配。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manipulation_project.backends.cumotion.pose_adapter import pose_from_matrix
from manipulation_project.planning.collision_objects import CollisionObject


class CuMotionCollisionWorld:
    """持有 cuMotion ``World`` 和对应 obstacle handles。"""

    def __init__(self, context, collision_objects: Sequence[CollisionObject] = ()) -> None:
        self.context = context
        self.cumotion = context.cumotion
        self.world = self.cumotion.World()
        self.handles = {}
        for obj in collision_objects:
            self.add(obj)
        self.world_view = self.world.add_world_view()
        self.world_view.update()

    def add(self, obj: CollisionObject) -> None:
        """添加一个项目碰撞对象。"""

        if not obj.enabled:
            return
        obstacle = self._make_obstacle(obj)
        handle = self.world.add_obstacle(obstacle, pose_from_matrix(self.cumotion, obj.pose_matrix()))
        self.handles[obj.name] = handle

    def update(self) -> None:
        """刷新 world view。"""

        self.world_view.update()

    def _make_obstacle(self, obj: CollisionObject):
        shape = obj.shape.lower()
        if shape == "box":
            shape = "cuboid"
        obstacle_type = getattr(self.cumotion.Obstacle.Type, shape.upper())
        obstacle = self.cumotion.create_obstacle(obstacle_type)
        size = np.asarray(obj.padded_size(), dtype=float)
        if shape == "cuboid":
            obstacle.set_attribute(self.cumotion.Obstacle.Attribute.SIDE_LENGTHS, size.reshape(3))
        elif shape == "sphere":
            obstacle.set_attribute(self.cumotion.Obstacle.Attribute.RADIUS, float(size[0]))
        elif shape == "capsule":
            obstacle.set_attribute(self.cumotion.Obstacle.Attribute.RADIUS, float(size[0]))
            obstacle.set_attribute(self.cumotion.Obstacle.Attribute.HEIGHT, float(size[1]))
        return obstacle


def make_collision_world(context, collision_objects: Sequence[CollisionObject] = ()) -> CuMotionCollisionWorld:
    """构建 cuMotion collision world。"""

    return CuMotionCollisionWorld(context, collision_objects)
