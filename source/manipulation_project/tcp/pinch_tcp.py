"""从 L6 手 MJCF 运动树计算夹捏中心 TCP。

夹捏 TCP 的位置不是一个固定常量，而是由 thumb/index 指尖在“闭合手型”下的位置决定。
本模块读取 MJCF body/joint 层级，沿手掌基座到指尖的 body chain 做正运动学，
取拇指和食指 tip 的中点作为 TCP。
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from manipulation_project.robots.mimic import expand_targets_with_mjcf_equalities
from manipulation_project.tcp.tcp_frame import TcpFrame
from manipulation_project.utils.math_utils import axis_angle_to_matrix, make_transform, quat_wxyz_to_matrix


DEFAULT_HAND_BASE_BODY = "L6V1_L_hand_base_link"
DEFAULT_THUMB_TIP_BODY = "L6V1_L_hand_thumb_tip"
DEFAULT_INDEX_TIP_BODY = "L6V1_L_hand_index_tip"
DEFAULT_PINCH_TCP_FRAME = "ar5_l6_pinch_tcp"


def parse_vec3(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    """解析 MuJoCo/MJCF 的 vec3 字符串属性。

    参数:
        text: 形如 ``"x y z"`` 的字符串；为空时使用 ``default``。
        default: 缺省 vec3。
    返回:
        shape ``(3,)`` 的 float ndarray。
    """

    if not text:
        return np.asarray(default, dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return np.asarray(values, dtype=float)


def parse_quat_wxyz(text: str | None) -> np.ndarray:
    """解析 MuJoCo/MJCF 的四元数字符串属性。

    参数:
        text: 形如 ``"w x y z"`` 的字符串；为空时返回单位四元数。
    返回:
        shape ``(4,)`` 的 wxyz 顺序四元数。
    """

    if not text:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != 4:
        raise ValueError(f"Expected 4 quaternion values, got {text!r}")
    return np.asarray(values, dtype=float)


def mjcf_parent_map(root: ET.Element) -> dict[int, ET.Element]:
    """为 MJCF XML 树建立 child id 到 parent element 的映射。

    参数:
        root: MJCF XML 根节点。
    返回:
        ``id(child) -> parent`` 字典，用于从 tip body 向上回溯到 base body。
    """

    parent_by_child_id: dict[int, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parent_by_child_id[id(child)] = parent
    return parent_by_child_id


def find_mjcf_body(root: ET.Element, body_name: str) -> ET.Element:
    """按名称查找 MJCF body。

    参数:
        root: MJCF XML 根节点。
        body_name: 目标 body 的 ``name`` 属性。
    返回:
        匹配的 ``Element``。
    """

    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise ValueError(f"MJCF body not found: {body_name}")


def body_chain_between(root: ET.Element, base_body_name: str, tip_body_name: str) -> list[ET.Element]:
    """返回从基座 body 到指尖 body 的 body 链。

    参数:
        root: MJCF XML 根节点。
        base_body_name: 起点 body 名称，通常是手掌基座。
        tip_body_name: 终点 body 名称，通常是某根手指 tip。
    返回:
        按 ``base -> ... -> tip`` 排列的 body 元素列表，包含两端。
    """

    base = find_mjcf_body(root, base_body_name)
    tip = find_mjcf_body(root, tip_body_name)
    parent_by_child_id = mjcf_parent_map(root)
    chain = []
    node = tip
    while node.tag == "body":
        chain.append(node)
        if node is base:
            return list(reversed(chain))
        parent = parent_by_child_id.get(id(node))
        if parent is None:
            break
        node = parent
    raise ValueError(f"{tip_body_name} is not under {base_body_name} in MJCF")


def body_chain_local_transform(body_chain: list[ET.Element], joint_positions: dict[str, float]) -> np.ndarray:
    """沿一条 MJCF body 链计算局部齐次变换。

    参数:
        body_chain: ``body_chain_between`` 返回的 body 列表。
        joint_positions: ``关节名 -> 关节角(rad)`` 映射，缺失关节按 0 处理。
    返回:
        shape ``(4, 4)`` 的齐次变换矩阵，表示 tip 相对 base 的位姿。
    """

    transform = np.eye(4, dtype=float)
    for body in body_chain[1:]:
        # MJCF body 的 pos/quat 是该 body 相对父 body 的固定偏移；
        # joint 的 pos/axis/qpos 再在该 body 内贡献一个转动自由度。
        body_pos = parse_vec3(body.get("pos"))
        body_rot = quat_wxyz_to_matrix(parse_quat_wxyz(body.get("quat")))
        transform = transform @ make_transform(body_pos, body_rot)
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            joint_pos = parse_vec3(joint.get("pos"))
            joint_axis = parse_vec3(joint.get("axis"), default=(1.0, 0.0, 0.0))
            joint_value = float(joint_positions.get(joint_name, 0.0))
            transform = transform @ make_transform(joint_pos, axis_angle_to_matrix(joint_axis, joint_value))
    return transform


def fingertip_pinch_local_offset(
    mjcf_path: str | Path,
    hand_targets: dict[str, float],
    *,
    hand_base_body: str = DEFAULT_HAND_BASE_BODY,
    thumb_tip_body: str = DEFAULT_THUMB_TIP_BODY,
    index_tip_body: str = DEFAULT_INDEX_TIP_BODY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算闭合手型下的夹捏中心和两指尖位置。

    参数:
        mjcf_path: AR5+L6 MJCF 文件路径。
        hand_targets: 手部主动关节目标，单位 rad；函数内部会展开 mimic follower。
        hand_base_body: 手掌基座 body 名称。
        thumb_tip_body: 拇指 tip body 名称。
        index_tip_body: 食指 tip body 名称。
    返回:
        ``(pinch_center, thumb_tip, index_tip)``，三者都是手掌基座坐标系下的
        shape ``(3,)`` 位置数组，单位 m。
    """

    path = Path(mjcf_path)
    expanded_targets = expand_targets_with_mjcf_equalities(hand_targets, path)
    root = ET.parse(path).getroot()
    thumb_chain = body_chain_between(root, hand_base_body, thumb_tip_body)
    index_chain = body_chain_between(root, hand_base_body, index_tip_body)
    thumb_tip = body_chain_local_transform(thumb_chain, expanded_targets)[:3, 3]
    index_tip = body_chain_local_transform(index_chain, expanded_targets)[:3, 3]
    pinch_center = 0.5 * (thumb_tip + index_tip)
    return pinch_center, thumb_tip, index_tip


def make_pinch_tcp(
    mjcf_path: str | Path,
    hand_targets: dict[str, float],
    *,
    parent_frame: str,
    frame_name: str = DEFAULT_PINCH_TCP_FRAME,
) -> TcpFrame:
    """创建位于闭合夹捏中心的 TCP frame。

    参数:
        mjcf_path: AR5+L6 MJCF 文件路径。
        hand_targets: 用于计算闭合几何的手部目标，单位 rad。
        parent_frame: TCP 固连到的父 frame 名称，通常是手掌基座 link。
        frame_name: 新 TCP frame 名称。
    返回:
        ``TcpFrame``，其 ``xyz`` 是夹捏中心相对 ``parent_frame`` 的偏移，单位 m。
    """

    pinch_center, _thumb_tip, _index_tip = fingertip_pinch_local_offset(mjcf_path, hand_targets)
    return TcpFrame.from_xyz_rpy(frame_name=frame_name, parent_frame=parent_frame, xyz=pinch_center)
