"""本仓库的 Isaac Sim / Isaac Lab 机械臂操作工具包。

该 package 把脚本中容易混在一起的功能拆成可复用模块：资产导入、控制器、IK、TCP、轨迹、
任务、日志和场景构建。顶层只暴露仓库根路径，具体功能请从各子包导入。

职责边界:
	* 顶层包不启动 Isaac Sim，不导入 ``omni``、``isaacsim`` 或 cuMotion。
	* 顶层包不注册全局状态，也不读取 YAML；脚本入口和子模块负责各自资源生命周期。
	* 只提供轻量路径常量，方便外部脚本确认当前导入的是仓库内源码。

设计约定:
	* 坐标与单位遵循 Isaac/URDF 常用约定：长度 m、角度 rad、角速度 rad/s。
	* 对外四元数通常使用 ``wxyz``，只有调用 SciPy/cuMotion/Isaac API 时才在局部转换。
	* 关节数组顺序必须由显式关节名或 Isaac ``dof_names`` 决定，不能依赖不同资产文件的隐式顺序。
"""

from manipulation_project.utils.paths import REPO_ROOT

__all__ = ["REPO_ROOT"]
