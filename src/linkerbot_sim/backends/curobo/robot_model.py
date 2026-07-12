"""cuRobo robot YAML/URDF materialization 与 TCP frame 解析。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboRobotConfig,
    CuroboTcpFrame,
)
from linkerbot_sim.utils.config import load_yaml

LINKERBOT_SIM_CACHE_ROOT_ENV = "LINKERBOT_SIM_CACHE_ROOT"


def materialize_curobo_config(
    config: CuroboConfig,
    *,
    cache_root: str | Path | None = None,
) -> CuroboConfig:
    """把 custom TCP fixed frames 写入缓存 URDF 并返回不可变新配置。"""

    robot = config.robot
    if not robot.custom_tcp_frames:
        return config
    if robot.urdf_path is None:
        raise ValueError("cuRobo custom TCP materialization requires urdf_path")
    output_path = _custom_tcp_urdf_path(robot, cache_root=cache_root)
    expected_content = _render_curobo_tcp_urdf(
        robot.urdf_path,
        robot.custom_tcp_frames,
    )
    expected_digest = hashlib.sha256(expected_content).digest()
    if not _cached_content_matches(output_path, expected_digest):
        _write_atomic_cache_content(
            output_path,
            expected_content,
            expected_digest=expected_digest,
        )
    if not _cached_content_matches(output_path, expected_digest):
        raise RuntimeError(
            f"failed to materialize valid cuRobo TCP cache: {output_path}"
        )
    materialized_robot = replace(robot, urdf_path=output_path, custom_tcp_frames=())
    return replace(config, robot=materialized_robot)


def materialized_robot_mapping(
    robot: CuroboRobotConfig,
    *,
    tool_frames: Sequence[str],
    asset_root_path: Path | None,
) -> dict[str, object]:
    """加载 robot YAML，并注入最终 URDF、asset root 与 tool frames。"""

    if robot.robot_config_path is None:
        raise ValueError("cuRobo robot mapping requires robot_config_path")
    mapping: dict[str, object] = load_yaml(robot.robot_config_path)
    wrapped = mapping.get("robot_cfg")
    if wrapped is None:
        robot_mapping = mapping
    elif isinstance(wrapped, Mapping):
        robot_mapping = dict(wrapped)
        mapping["robot_cfg"] = robot_mapping
    else:
        raise ValueError(
            f"cuRobo robot config {robot.robot_config_path} has invalid robot_cfg mapping"
        )
    kinematics = robot_mapping.get("kinematics")
    if not isinstance(kinematics, Mapping):
        raise ValueError(
            f"cuRobo robot config {robot.robot_config_path} must contain a kinematics mapping"
        )
    materialized_kinematics = dict(kinematics)
    robot_mapping["kinematics"] = materialized_kinematics
    if robot.urdf_path is not None:
        materialized_kinematics["urdf_path"] = str(robot.urdf_path.resolve())
        resolved_asset_root = asset_root_path or robot.urdf_path.parent
        materialized_kinematics["asset_root_path"] = str(resolved_asset_root.resolve())
    materialized_kinematics["tool_frames"] = [str(frame) for frame in tool_frames]
    return mapping


def write_curobo_tcp_urdf_with_frames(
    urdf_path: str | Path,
    output_urdf_path: str | Path,
    tcp_frames: Sequence[CuroboTcpFrame],
) -> Path:
    """写出追加多个 fixed TCP link/joint 的 URDF 副本。"""

    source_path = Path(urdf_path)
    output_path = Path(output_urdf_path)
    content = _render_curobo_tcp_urdf(source_path, tcp_frames)
    _write_atomic_cache_content(
        output_path,
        content,
        expected_digest=hashlib.sha256(content).digest(),
    )
    return output_path


def _render_curobo_tcp_urdf(
    source_path: Path,
    tcp_frames: Sequence[CuroboTcpFrame],
) -> bytes:
    """生成确定性的 URDF XML 字节，并把完整字节流作为缓存契约。

    缓存摘要覆盖源机器人正文和追加的全部 TCP frame，源文件任一处变化都会使缓存失效。
    """

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
            raise ValueError(
                f"TCP joint {joint_name!r} already exists in {source_path}"
            )
        ET.SubElement(root, "link", {"name": tcp.frame_name})
        tcp_joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "fixed"})
        ET.SubElement(tcp_joint, "parent", {"link": tcp.parent_frame})
        ET.SubElement(tcp_joint, "child", {"link": tcp.frame_name})
        xyz = " ".join(f"{float(value):.9g}" for value in tcp.xyz)
        rpy = " ".join(f"{float(value):.9g}" for value in tcp.rpy)
        ET.SubElement(tcp_joint, "origin", {"xyz": xyz, "rpy": rpy})
        link_names.add(tcp.frame_name)
        joint_names.add(joint_name)
    ET.indent(tree, space="  ")
    buffer = BytesIO()
    tree.write(
        buffer,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )
    return buffer.getvalue()


def _write_atomic_cache_content(
    output_path: Path,
    content: bytes,
    *,
    expected_digest: bytes,
) -> None:
    """将完全同步的缓存字节通过同目录原子替换发布到目标路径。

    临时文件先写入、``fsync`` 并校验摘要，再执行 ``os.replace``；任一步失败都会清理
    临时文件，避免读者观察到部分内容。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if not _cached_content_matches(temporary_path, expected_digest):
            raise RuntimeError(
                f"generated cuRobo TCP cache failed validation: {temporary_path}"
            )
        os.replace(temporary_path, output_path)
        _fsync_directory(output_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _cached_content_matches(
    path: Path,
    expected_digest: bytes,
) -> bool:
    """用 SHA-256 比较全部生成字节，校验范围包含源机器人正文。"""

    try:
        actual_digest = hashlib.sha256(path.read_bytes()).digest()
    except OSError:
        return False
    return actual_digest == expected_digest


def _fsync_directory(path: Path) -> None:
    """在临时文件已同步后同步父目录，使原子重命名本身持久化。"""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def default_tcp_frame_name(config: CuroboConfig) -> str | None:
    """返回显式默认 TCP，缺省时使用第一个 resolved tool frame。"""

    if config.robot.default_tcp_frame:
        return config.robot.default_tcp_frame
    frames = tuple(config.robot.resolved_tool_frames)
    return frames[0] if frames else None


def resolve_tcp_frame_name(
    context: object,
    *,
    tcp_frame_name: str | None = None,
    default_tcp_frame_name: str | None = None,
    label: str = "tcp_frame_name",
) -> str:
    """按显式值、调用方默认、context 配置默认的优先级解析 frame。"""

    config = getattr(context, "config", None)
    frame_name = (
        tcp_frame_name
        or default_tcp_frame_name
        or (default_tcp_frame_name_from_config(config) if config is not None else None)
    )
    if frame_name is None:
        raise ValueError(
            f"{label} is required because cuRobo config has no default frame"
        )
    frame = str(frame_name)
    validate_curobo_frame(context, frame, label=label)
    return frame


def default_tcp_frame_name_from_config(config: object) -> str | None:
    """按 default TCP、materialized tool frame 顺序读取 config 默认 frame。"""

    robot = getattr(config, "robot", None)
    if robot is None:
        return None
    value = getattr(robot, "default_tcp_frame", None)
    if value:
        return str(value)
    frames = tuple(getattr(robot, "resolved_tool_frames", ()) or ())
    return str(frames[0]) if frames else None


def validate_curobo_frame(
    context: object, frame_name: str, *, label: str = "frame"
) -> None:
    """校验非空 frame，并在 context 可枚举 frame 时检查其确实存在。"""

    if not str(frame_name):
        raise ValueError(f"{label} cannot be empty")
    frame_names = getattr(context, "frame_names", None)
    if callable(frame_names) and str(frame_name) not in set(frame_names()):
        raise ValueError(f"cuRobo frame {frame_name!r} not found")


def resolve_curobo_cache_dir(
    cache_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """按 runtime、项目环境变量、XDG、用户目录解析 cuRobo cache。"""

    values = os.environ if environ is None else environ
    if cache_root is not None:
        root = Path(cache_root).expanduser()
    elif values.get(LINKERBOT_SIM_CACHE_ROOT_ENV):
        root = Path(values[LINKERBOT_SIM_CACHE_ROOT_ENV]).expanduser()
    elif values.get("XDG_CACHE_HOME"):
        root = Path(values["XDG_CACHE_HOME"]).expanduser() / "linkerbot_sim"
    else:
        root = Path.home() / ".cache" / "linkerbot_sim"
    return root.resolve() / "curobo"


def _custom_tcp_urdf_path(
    robot: CuroboRobotConfig,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    """按 source URDF 内容和 custom TCP 定义生成可复用 cache 路径。"""

    assert robot.urdf_path is not None
    base = Path(robot.urdf_path)
    frames = [
        {
            "frame_name": frame.frame_name,
            "parent_frame": frame.parent_frame,
            "xyz": np.asarray(frame.xyz, dtype=float).reshape(3).tolist(),
            "rpy": np.asarray(frame.rpy, dtype=float).reshape(3).tolist(),
        }
        for frame in robot.custom_tcp_frames
    ]
    digest = hashlib.sha256(
        json.dumps(
            {
                "urdf_path": str(base.resolve()),
                "urdf_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "frames": frames,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return resolve_curobo_cache_dir(cache_root) / (
        f"{base.stem}_custom_tcps_{digest}.urdf"
    )


__all__ = [
    "default_tcp_frame_name",
    "materialize_curobo_config",
    "materialized_robot_mapping",
    "resolve_curobo_cache_dir",
    "resolve_tcp_frame_name",
    "validate_curobo_frame",
    "write_curobo_tcp_urdf_with_frames",
]
