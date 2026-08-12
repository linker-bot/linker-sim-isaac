"""Isaac 双物理后端的同构环境装配基础设施。

本包只处理 source env、后端专用布局、USD/PhysX clone、world 隔离以及 raw view 的装配；
它不知道强化学习任务、奖励或 Gymnasium。这样 ``Kaleidoscope`` 是产品语义，
``replicated_scene`` 只是可被产品组合的 Isaac 基础设施，不携带任何产品模式语义。
"""

from .physx_builder import build_replicated_physx_scene
from .types import ImportedReplicatedRobot, ReplicatedNewtonScene, ReplicatedPhysxScene
from .views import finalize_replicated_robot_views

__all__ = [
    "ImportedReplicatedRobot",
    "ReplicatedNewtonScene",
    "ReplicatedPhysxScene",
    "build_replicated_physx_scene",
    "finalize_replicated_robot_views",
]
