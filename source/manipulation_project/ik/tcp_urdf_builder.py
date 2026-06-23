"""生成带固定 TCP link 的临时 URDF。

IK 后端通常只能对 URDF 中已有 link 求解。对于 pinch center 或自定义 TCP，
这里会复制基础机械臂 URDF，并在指定 parent frame 下追加一个 fixed joint/link。
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from manipulation_project.tcp.tcp_frame import TcpFrame


def write_tcp_urdf(base_urdf_path: str | Path, output_urdf_path: str | Path, tcp: TcpFrame) -> Path:
    """写出追加了固定 TCP link 和 joint 的 URDF 副本。

    参数:
        base_urdf_path: 基础 URDF 路径。
        output_urdf_path: 输出 URDF 路径。
        tcp: TCP frame 描述，包含父 link、frame 名和相对位姿。
    返回:
        输出 URDF 的 ``Path``。
    """

    base_path = Path(base_urdf_path)
    output_path = Path(output_urdf_path)
    tree = ET.parse(base_path)
    root = tree.getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    joint_names = {joint.get("name") for joint in root.findall("joint")}
    if tcp.parent_frame not in link_names:
        raise ValueError(f"Parent frame {tcp.parent_frame!r} not found in {base_path}")
    if tcp.frame_name in link_names:
        raise ValueError(f"TCP frame {tcp.frame_name!r} already exists in {base_path}")

    joint_name = f"{tcp.frame_name}_joint"
    if joint_name in joint_names:
        raise ValueError(f"TCP joint {joint_name!r} already exists in {base_path}")

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
