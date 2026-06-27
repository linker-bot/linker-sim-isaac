"""机械臂法兰 TCP 定义。

当动作只想把机械臂末端法兰当作 TCP 时，可直接使用本模块；没有额外平移或旋转。
法兰 frame 名必须与后端 robot description 中的 link/frame 名一致，否则 FK/IK 查询会失败。
"""

from __future__ import annotations

from manipulation_project.tcp.tcp_frame import TcpFrame


def make_flange_tcp(frame_name: str) -> TcpFrame:
    """创建位于机械臂法兰 link 自身的 TCP。

    参数:
        frame_name: 法兰 frame/link 名称，同时作为 TCP frame 和父 frame。
    返回:
        ``TcpFrame``，相对位姿为零。
    """

    return TcpFrame.from_xyz_rpy(frame_name=frame_name, parent_frame=frame_name)
