"""cuMotion TCP transform data structures.

The public motion/script boundary describes a TCP as a Cartesian transform
relative to the robot endpoint. The backend binds that transform to a concrete
URDF parent link only when it writes the temporary cuMotion robot description.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TcpTransform:
    """TCP transform relative to an endpoint/flange frame."""

    frame_name: str
    xyz: np.ndarray
    rpy: np.ndarray

    @classmethod
    def from_xyz_rpy(
        cls,
        frame_name: str,
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
    ) -> "TcpTransform":
        """从普通 Python 序列构造末端相对 TCP transform。"""

        return cls(
            frame_name=str(frame_name),
            xyz=np.asarray(xyz, dtype=float).reshape(3),
            rpy=np.asarray(rpy, dtype=float).reshape(3),
        )


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


def bind_tcp_transform(transform: TcpTransform, *, parent_frame: str) -> TcpFrame:
    """Bind an endpoint-relative TCP transform to a concrete parent link."""

    return TcpFrame.from_xyz_rpy(
        frame_name=transform.frame_name,
        parent_frame=parent_frame,
        xyz=transform.xyz,
        rpy=transform.rpy,
    )
