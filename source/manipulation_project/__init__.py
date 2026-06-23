"""本仓库的 Isaac Sim / Isaac Lab 机械臂操作工具包。

该 package 把脚本中容易混在一起的功能拆成可复用模块：
资产导入、控制器、IK、TCP、轨迹、任务、日志和场景构建。
顶层只暴露仓库根路径，具体功能请从各子包导入。
"""

from manipulation_project.utils.paths import REPO_ROOT

__all__ = ["REPO_ROOT"]
