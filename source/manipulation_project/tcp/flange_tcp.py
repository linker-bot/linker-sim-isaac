"""AR5 法兰 TCP 默认定义。

当任务只想把机械臂末端法兰当作 TCP 时，可直接使用本模块；没有额外平移或旋转。
"""

from __future__ import annotations

from manipulation_project.tcp.tcp_frame import TcpFrame


DEFAULT_AR5_FLANGE_FRAME = "AR5V2_L_arm_flan_link"


def make_flange_tcp(frame_name: str = DEFAULT_AR5_FLANGE_FRAME) -> TcpFrame:
    """创建位于 AR5 法兰 link 自身的 TCP。

    参数:
        frame_name: 法兰 frame/link 名称，同时作为 TCP frame 和父 frame。
    返回:
        ``TcpFrame``，相对位姿为零。
    """

    return TcpFrame.from_xyz_rpy(frame_name=frame_name, parent_frame=frame_name)
