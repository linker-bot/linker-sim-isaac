"""力矩/级联控制日志的预留入口。

当前项目主要记录关节位置和速度跟踪，所以先把 ``EffortLogger`` 指向 ``JointTrackingLogger``。
后续如果加入力矩控制或级联控制，可以在保持导入路径不变的情况下替换为更完整的 effort
日志实现。

职责边界:
	* 只提供兼容别名，不打开日志文件。
	* 不改变 ``JointTrackingLogger`` 的列名约定，避免现有绘图脚本失效。
	* 不在这里补写 effort 列；真正力矩日志需要单独的数据结构和 CSV schema。
"""

from __future__ import annotations

from manipulation_project.logging.joint_logger import JointTrackingLogger


EffortLogger = JointTrackingLogger
