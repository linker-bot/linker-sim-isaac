"""自定义固定偏移 TCP。

用于快速定义“某个 link 上偏移一段距离”的末端点，例如临时工具、夹爪中心或标定点。
"""

from __future__ import annotations

from manipulation_project.tcp.tcp_frame import TcpFrame


def make_custom_tcp(parent_frame: str, frame_name: str, xyz, rpy=(0.0, 0.0, 0.0)) -> TcpFrame:
    """创建固定到父 frame 的自定义 TCP。

    参数:
        parent_frame: 父 link/frame 名称。
        frame_name: 新 TCP frame 名称。
        xyz: TCP 相对父 frame 的平移，长度 3，单位 m。
        rpy: TCP 相对父 frame 的固定轴 XYZ 顺序（外旋 XYZ 顺序）RPY，长度 3，单位 rad。
    返回:
        ``TcpFrame`` 实例。
    """

    return TcpFrame.from_xyz_rpy(frame_name=frame_name, parent_frame=parent_frame, xyz=xyz, rpy=rpy)
