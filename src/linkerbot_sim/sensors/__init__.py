"""仿真传感器配置和运行时辅助。

本包面向会产生数据的 sensor，例如 RGB-D 摄像机。GUI viewport 观察视角仍属于
``linkerbot_sim.configuration.visualization`` 和 ``linkerbot_sim.visualization``。
"""

from linkerbot_sim.sensors.config import SceneSensorSettings

__all__ = ["SceneSensorSettings"]
