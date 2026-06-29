"""生成带固定 TCP link 的临时 URDF。

cuMotion 只能对 URDF 中已有 link 求解。对于 pinch center 或自定义 TCP，
这里会复制基础机械臂 URDF，并在指定 parent frame 下追加一个 fixed joint/link。

生成文件通常写入临时目录或构建目录，并作为 ``CuMotionConfig.urdf_path`` 传给后端。
``TcpFrame.xyz`` 使用米，``TcpFrame.rpy`` 使用弧度，均相对于 parent link 坐标系。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

from linkerbot_sim.tcp.tcp_frame import TcpFrame


def write_tcp_urdf(
    urdf_path: str | Path, output_urdf_path: str | Path, tcp: TcpFrame
) -> Path:
    """写出追加了固定 TCP link 和 joint 的 URDF 副本。

    参数:
        urdf_path: 基础 URDF 路径。
        output_urdf_path: 输出 URDF 路径。
        tcp: TCP frame 描述，包含父 link、frame 名和相对位姿。
    返回:
        输出 URDF 的 ``Path``。
    """

    source_path = Path(urdf_path)
    output_path = Path(output_urdf_path)
    tree = ET.parse(source_path)
    root = tree.getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    joint_names = {joint.get("name") for joint in root.findall("joint")}
    if tcp.parent_frame not in link_names:
        raise ValueError(
            f"Parent frame {tcp.parent_frame!r} not found in {source_path}"
        )
    if tcp.frame_name in link_names:
        raise ValueError(
            f"TCP frame {tcp.frame_name!r} already exists in {source_path}"
        )

    joint_name = f"{tcp.frame_name}_joint"
    if joint_name in joint_names:
        raise ValueError(f"TCP joint {joint_name!r} already exists in {source_path}")

    ET.SubElement(root, "link", {"name": tcp.frame_name})
    tcp_joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(tcp_joint, "parent", {"link": tcp.parent_frame})
    ET.SubElement(tcp_joint, "child", {"link": tcp.frame_name})
    xyz = " ".join(f"{float(value):.9g}" for value in tcp.xyz)
    rpy = " ".join(f"{float(value):.9g}" for value in tcp.rpy)
    ET.SubElement(tcp_joint, "origin", {"xyz": xyz, "rpy": rpy})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def write_tcp_urdf_with_frames(
    urdf_path: str | Path,
    output_urdf_path: str | Path,
    tcp_frames: Sequence[TcpFrame],
) -> Path:
    """写出追加多个 fixed TCP link/joint 的 URDF 副本。"""

    source_path = Path(urdf_path)
    output_path = Path(output_urdf_path)
    tree = ET.parse(source_path)
    root = tree.getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    joint_names = {joint.get("name") for joint in root.findall("joint")}
    for tcp in tcp_frames:
        if tcp.parent_frame not in link_names:
            raise ValueError(
                f"Parent frame {tcp.parent_frame!r} not found in {source_path}"
            )
        if tcp.frame_name in link_names:
            raise ValueError(
                f"TCP frame {tcp.frame_name!r} already exists in {source_path}"
            )
        joint_name = f"{tcp.frame_name}_joint"
        if joint_name in joint_names:
            raise ValueError(f"TCP joint {joint_name!r} already exists in {source_path}")
        ET.SubElement(root, "link", {"name": tcp.frame_name})
        tcp_joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "fixed"})
        ET.SubElement(tcp_joint, "parent", {"link": tcp.parent_frame})
        ET.SubElement(tcp_joint, "child", {"link": tcp.frame_name})
        xyz = " ".join(f"{float(value):.9g}" for value in tcp.xyz)
        rpy = " ".join(f"{float(value):.9g}" for value in tcp.rpy)
        ET.SubElement(tcp_joint, "origin", {"xyz": xyz, "rpy": rpy})
        link_names.add(tcp.frame_name)
        joint_names.add(joint_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
