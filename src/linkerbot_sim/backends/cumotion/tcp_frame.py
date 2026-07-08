"""cuMotion TCP frame data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TcpFrame:
    """TCP transform bound to a concrete URDF parent link."""

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
        """从普通 Python 序列构造已绑定 URDF parent link 的 TCP frame。"""

        return cls(
            frame_name=str(frame_name),
            parent_frame=str(parent_frame),
            xyz=np.asarray(xyz, dtype=float).reshape(3),
            rpy=np.asarray(rpy, dtype=float).reshape(3),
        )
