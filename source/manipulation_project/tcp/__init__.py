"""TCP（Tool Center Point）坐标系定义。

本包把不同末端参考点抽象成 ``TcpFrame``：既可以是机械臂法兰，也可以是手指闭合后
的夹捏中心。IK 和轨迹层只需要关心 TCP frame 名称、父 link、相对位姿。
"""
