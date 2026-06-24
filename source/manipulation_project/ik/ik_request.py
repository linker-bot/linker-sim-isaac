"""兼容入口：IK 请求已迁移到 ``planning.requests``。"""

from manipulation_project.planning.requests import IKRequest

__all__ = ["IKRequest"]
