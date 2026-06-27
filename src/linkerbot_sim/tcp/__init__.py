"""TCP（Tool Center Point）坐标系定义。

本包把不同末端参考点抽象成 ``TcpFrame``：既可以是机械臂法兰，也可以是手指闭合后
的夹捏中心。IK 和轨迹层只需要关心 TCP frame 名称、父 link、相对位姿。

约定：TCP 的 ``xyz`` 以父 link 坐标系表示，单位为米；``rpy`` 使用弧度并遵循固定轴
XYZ 顺序。需要传给 cuMotion 的自定义 TCP 会通过临时 URDF 固定关节显式加入运动树。
"""
