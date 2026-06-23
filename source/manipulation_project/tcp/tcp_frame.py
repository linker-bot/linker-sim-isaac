"""通用 TCP frame 数据类型。

TCP 在这里被表示为一个刚性固定到父 link 的局部坐标系。``xyz`` 和 ``rpy`` 都是
父 link 坐标系下的相对位姿。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TcpFrame:
    """固定在某个父 link 上的 TCP 坐标系。

    输入字段:
        frame_name: TCP frame 名称，写入 URDF 后供 IK 后端引用。
        parent_frame: 父 link/frame 名称。
        xyz: TCP 相对父 frame 的平移，shape ``(3,)``，单位 m。
        rpy: TCP 相对父 frame 的固定轴 XYZ 顺序（外旋 XYZ 顺序）RPY，shape ``(3,)``，单位 rad。
    输出:
        dataclass 实例本身作为 TCP 描述，通常传给 ``write_tcp_urdf``。
    """

    frame_name: str
    parent_frame: str
    xyz: np.ndarray
    rpy: np.ndarray

    @classmethod
    def from_xyz_rpy(
        cls,
        frame_name: str,
        parent_frame: str,
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
    ) -> "TcpFrame":
        """从可迭代输入构造 TCP frame。

        参数:
            frame_name: 新 TCP frame 名称。
            parent_frame: 父 frame 名称。
            xyz: 长度 3 的平移输入，单位 m。
            rpy: 长度 3 的固定轴 XYZ 顺序（外旋 XYZ 顺序）RPY 输入，单位 rad。
        返回:
            ``TcpFrame``，其中 ``xyz``/``rpy`` 已转换为 float ndarray。
        """

        return cls(
            frame_name=frame_name,
            parent_frame=parent_frame,
            xyz=np.asarray(xyz, dtype=float).reshape(3),
            rpy=np.asarray(rpy, dtype=float).reshape(3),
        )
