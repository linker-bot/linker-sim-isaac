"""根据双机器人 root_pose 和左右单臂描述生成 cuMotion 双臂规划资产。

该模块只处理 cuMotion 使用的机械臂 URDF/XRDF，不生成 Isaac 资产，也不包含手部 DOF。左右
AR5+L6 在 Isaac 中仍作为两个 articulation 导入；这里把左右 AR5 单臂 URDF/XRDF 融合成
同一个 14-DOF 规划模型，使规划侧看到和执行侧一致的安装位姿与 C-space。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import yaml

from linkerbot_sim.assets.robot_loader import RootPoseConfig
from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class DualUrdfGenerationConfig:
    """双臂 cuMotion 资产生成输入。

    左右 URDF/XRDF 是单臂规划模型；``output_dir`` 是按 root pose 和资源路径生成缓存文件的
    目录。``robot_name`` 和 ``parent_link`` 只影响生成出的 planning model 命名，和 Isaac
    stage 的 ``/World`` prim 不是同一个概念。
    """

    left_xrdf_path: Path | None
    right_xrdf_path: Path | None
    left_urdf_path: Path
    right_urdf_path: Path
    output_dir: Path
    robot_name: str = "AR5V2_DUAL"
    parent_link: str = "world"
    left_base_link: str = "AR5V2_L_arm_base"
    right_base_link: str = "AR5V2_R_arm_base"
    left_mount_joint: str = "world_to_AR5V2_L_arm_base"
    right_mount_joint: str = "world_to_AR5V2_R_arm_base"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        section: str = "cumotion.dual_urdf",
    ) -> "DualUrdfGenerationConfig | None":
        """从 ``cumotion.dual_urdf`` 解析生成配置。"""

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{section} must be a mapping")
        return cls(
            left_xrdf_path=_optional_repo_path(data.get("left_xrdf_path")),
            right_xrdf_path=_optional_repo_path(data.get("right_xrdf_path")),
            left_urdf_path=repo_path(
                _required_value(data, "left_urdf_path", section=section)
            ),
            right_urdf_path=repo_path(
                _required_value(data, "right_urdf_path", section=section)
            ),
            output_dir=repo_path(data.get("output_dir", ".cache/cumotion")),
            robot_name=str(data.get("robot_name", cls.robot_name)),
            parent_link=str(data.get("parent_link", cls.parent_link)),
            left_base_link=str(data.get("left_base_link", cls.left_base_link)),
            right_base_link=str(data.get("right_base_link", cls.right_base_link)),
            left_mount_joint=str(
                data.get("left_mount_joint", cls.left_mount_joint)
            ),
            right_mount_joint=str(
                data.get("right_mount_joint", cls.right_mount_joint)
            ),
        )


@dataclass(frozen=True)
class PreparedDualCuMotionAssets:
    """最终交给 cuMotion 的资产摘要。

    单臂配置时 ``generated_assets=False``，路径就是 robot YAML 中声明的 URDF/XRDF。双臂配置时
    会先生成缓存 URDF/XRDF，``backend_config`` 已替换成这些缓存路径；左右 flange frame 需要由
    调用方按 side 显式选择。
    """

    backend_config: CuMotionConfig
    urdf_path: Path
    xrdf_path: Path
    flange_frames: dict[str, str]
    generated_assets: bool


def prepare_cumotion_config_from_robot_config(
    robot_config: Mapping[str, object],
    *,
    dual_root_poses: Mapping[str, RootPoseConfig] | None = None,
) -> PreparedDualCuMotionAssets:
    """从 robot YAML 解析最终传给 cuMotion 的配置。

    单臂配置直接解析 ``cumotion.xrdf_path/urdf_path/flange_frame``。双臂配置推荐写成
    ``cumotion.left/right`` 两份单臂式描述；本函数会按 env 提供的左右 ``root_pose`` 生成缓存 URDF，
    并把左右 XRDF 融合成 14-DOF XRDF。
    """

    dual_urdf_config = dual_urdf_generation_config_from_robot_config(robot_config)
    if dual_urdf_config is None:
        config = CuMotionConfig.from_mapping(robot_config)
        return PreparedDualCuMotionAssets(
            backend_config=config,
            urdf_path=Path(config.urdf_path),
            xrdf_path=Path(config.xrdf_path),
            flange_frames={},
            generated_assets=False,
        )

    if dual_root_poses is None:
        raise ValueError("dual cuMotion generation requires env robot root poses")
    generated_path = build_dual_arm_urdf_from_root_poses(
        dual_urdf_config,
        left_pose=_required_root_pose(dual_root_poses, "left"),
        right_pose=_required_root_pose(dual_root_poses, "right"),
    )
    settings = dict(_cumotion_settings(robot_config))
    generated_xrdf_path = _dual_xrdf_path_for_urdf(generated_path)
    if dual_urdf_config.left_xrdf_path is not None:
        generated_xrdf_path = build_dual_arm_xrdf(
            dual_urdf_config,
            output_path=generated_xrdf_path,
        )
    else:
        generated_xrdf_path = repo_path(_required_value(
            settings,
            "xrdf_path",
            section="cumotion",
        ))

    settings["xrdf_path"] = str(generated_xrdf_path)
    settings["urdf_path"] = str(generated_path)
    settings.pop("flange_frame", None)
    config = CuMotionConfig.from_mapping(
        {"cumotion": settings},
        require_flange_frame=False,
    )
    return PreparedDualCuMotionAssets(
        backend_config=config,
        urdf_path=generated_path,
        xrdf_path=generated_xrdf_path,
        flange_frames=_dual_flange_frames(settings),
        generated_assets=True,
    )


def dual_cumotion_config_from_sides(
    *,
    left: Mapping[str, object],
    right: Mapping[str, object],
    output_dir: str | Path = ".cache/cumotion",
    robot_name: str = DualUrdfGenerationConfig.robot_name,
    parent_link: str = DualUrdfGenerationConfig.parent_link,
    left_base_link: str = DualUrdfGenerationConfig.left_base_link,
    right_base_link: str = DualUrdfGenerationConfig.right_base_link,
    left_mount_joint: str = DualUrdfGenerationConfig.left_mount_joint,
    right_mount_joint: str = DualUrdfGenerationConfig.right_mount_joint,
) -> dict[str, object]:
    """把左右单臂 robot profile 组合成双臂 cuMotion 资源配置。"""

    left_cumotion = _side_cumotion_settings(left, side="left")
    right_cumotion = _side_cumotion_settings(right, side="right")
    return {
        "cumotion": {
            "left": {
                "xrdf_path": _side_required_value(
                    left_cumotion, "left", "xrdf_path"
                ),
                "urdf_path": _side_required_value(
                    left_cumotion, "left", "urdf_path"
                ),
                "flange_frame": _side_required_value(
                    left_cumotion, "left", "flange_frame"
                ),
            },
            "right": {
                "xrdf_path": _side_required_value(
                    right_cumotion, "right", "xrdf_path"
                ),
                "urdf_path": _side_required_value(
                    right_cumotion, "right", "urdf_path"
                ),
                "flange_frame": _side_required_value(
                    right_cumotion, "right", "flange_frame"
                ),
            },
            "output_dir": str(output_dir),
            "robot_name": robot_name,
            "parent_link": parent_link,
            "left_base_link": left_base_link,
            "right_base_link": right_base_link,
            "left_mount_joint": left_mount_joint,
            "right_mount_joint": right_mount_joint,
        }
    }


def _required_root_pose(
    root_poses: Mapping[str, RootPoseConfig], side: str
) -> RootPoseConfig:
    """从 env root_pose 映射中读取指定侧位姿，缺失时给出生成阶段错误。"""

    pose = root_poses.get(side)
    if pose is None:
        raise ValueError(f"dual cuMotion generation missing {side!r} root_pose")
    return pose


def dual_urdf_generation_config_from_robot_config(
    robot_config: Mapping[str, object]
) -> DualUrdfGenerationConfig | None:
    """从合并后的 robot config 中提取双臂 URDF/XRDF 生成配置。"""

    cumotion = robot_config.get("cumotion")
    if not isinstance(cumotion, Mapping):
        return None
    if "left" in cumotion or "right" in cumotion:
        return _dual_generation_config_from_side_mappings(cumotion)
    return DualUrdfGenerationConfig.from_mapping(
        cumotion.get("dual_urdf"),
        section="cumotion.dual_urdf",
    )


def build_dual_arm_urdf_from_root_poses(
    config: DualUrdfGenerationConfig,
    *,
    left_pose: RootPoseConfig,
    right_pose: RootPoseConfig,
) -> Path:
    """按左右 root_pose 写出缓存双臂 URDF。"""

    _require_file(config.left_urdf_path, label="URDF")
    _require_file(config.right_urdf_path, label="URDF")
    output_path = _cached_output_path(config, left_pose=left_pose, right_pose=right_pose)
    if output_path.is_file():
        return output_path

    left_root = ET.parse(config.left_urdf_path).getroot()
    right_root = ET.parse(config.right_urdf_path).getroot()
    dual_root = ET.Element("robot", {"name": config.robot_name})
    ET.SubElement(dual_root, "link", {"name": config.parent_link})

    existing_names: set[str] = {f"link:{config.parent_link}"}
    for source_root, source_path, base_link, mount_joint, pose in (
        (
            left_root,
            config.left_urdf_path,
            config.left_base_link,
            config.left_mount_joint,
            left_pose,
        ),
        (
            right_root,
            config.right_urdf_path,
            config.right_base_link,
            config.right_mount_joint,
            right_pose,
        ),
    ):
        _append_materials(dual_root, source_root, existing_names)
        _append_robot_tree_without_world_mount(
            dual_root,
            source_root,
            source_path=source_path,
            output_urdf_path=output_path,
            existing_names=existing_names,
            world_link_name=config.parent_link,
        )
        _append_mount_joint(
            dual_root,
            parent_link=config.parent_link,
            child_base_link=base_link,
            joint_name=mount_joint,
            pose=pose,
            existing_names=existing_names,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(dual_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def build_dual_arm_xrdf(
    config: DualUrdfGenerationConfig,
    *,
    output_path: Path | None = None,
) -> Path:
    """把左右单臂 XRDF 融合成同一个 14-DOF XRDF。"""

    if config.left_xrdf_path is None or config.right_xrdf_path is None:
        raise ValueError("left/right XRDF paths are required to generate dual XRDF")
    _require_file(config.left_xrdf_path, label="XRDF")
    _require_file(config.right_xrdf_path, label="XRDF")
    xrdf_path = output_path or _cached_output_path(config).with_suffix(".xrdf")
    if xrdf_path.is_file():
        return xrdf_path

    left = _load_xrdf(config.left_xrdf_path)
    right = _load_xrdf(config.right_xrdf_path)
    combined = _merge_xrdf_documents(left, right)
    xrdf_path.parent.mkdir(parents=True, exist_ok=True)
    with xrdf_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            combined,
            file,
            sort_keys=False,
            allow_unicode=False,
        )
    return xrdf_path


def _cached_output_path(
    config: DualUrdfGenerationConfig,
    *,
    left_pose: RootPoseConfig | None = None,
    right_pose: RootPoseConfig | None = None,
) -> Path:
    """根据输入资产路径和 root_pose 生成稳定缓存文件名。"""

    digest = hashlib.sha256(
        json.dumps(
            {
                "left_xrdf_path": None
                if config.left_xrdf_path is None
                else str(config.left_xrdf_path.resolve()),
                "right_xrdf_path": None
                if config.right_xrdf_path is None
                else str(config.right_xrdf_path.resolve()),
                "left_urdf_path": str(config.left_urdf_path.resolve()),
                "right_urdf_path": str(config.right_urdf_path.resolve()),
                "robot_name": config.robot_name,
                "parent_link": config.parent_link,
                "left_base_link": config.left_base_link,
                "right_base_link": config.right_base_link,
                "left_mount_joint": config.left_mount_joint,
                "right_mount_joint": config.right_mount_joint,
                "left_pose": None
                if left_pose is None
                else {"xyz": left_pose.xyz, "rpy": left_pose.rpy},
                "right_pose": None
                if right_pose is None
                else {"xyz": right_pose.xyz, "rpy": right_pose.rpy},
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return config.output_dir / f"{config.robot_name}_{digest}.urdf"


def _append_materials(
    target_root: ET.Element, source_root: ET.Element, existing_names: set[str]
) -> None:
    """把源 URDF 顶层 material 复制到目标 URDF，并按名称去重。"""

    for material in source_root.findall("material"):
        name = material.get("name")
        key = f"material:{name}" if name else f"material:{id(material)}"
        if key in existing_names:
            continue
        target_root.append(_clone_element(material))
        existing_names.add(key)


def _append_robot_tree_without_world_mount(
    target_root: ET.Element,
    source_root: ET.Element,
    *,
    source_path: Path,
    output_urdf_path: Path,
    existing_names: set[str],
    world_link_name: str,
) -> None:
    """复制源机器人 link/joint 树，跳过源文件里的 world 固定挂载。"""

    for child in list(source_root):
        if child.tag == "material":
            continue
        if child.tag == "link" and child.get("name") == world_link_name:
            continue
        if _is_world_fixed_joint(child, world_link_name=world_link_name):
            continue
        name = child.get("name")
        key = f"{child.tag}:{name}" if name else f"{child.tag}:{id(child)}"
        if key in existing_names:
            raise ValueError(f"Duplicate URDF element name while merging: {key}")
        cloned = _clone_element(child)
        _rewrite_mesh_filenames(
            cloned,
            source_dir=source_path.parent,
            output_dir=output_urdf_path.parent,
        )
        target_root.append(cloned)
        existing_names.add(key)


def _append_mount_joint(
    target_root: ET.Element,
    *,
    parent_link: str,
    child_base_link: str,
    joint_name: str,
    pose: RootPoseConfig,
    existing_names: set[str],
) -> None:
    """在融合 URDF 中追加 parent_link 到单臂 base_link 的 fixed mount joint。"""

    key = f"joint:{joint_name}"
    if key in existing_names:
        raise ValueError(f"Duplicate mount joint name: {joint_name}")
    joint = ET.SubElement(target_root, "joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_base_link})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": _format_vec(pose.xyz),
            "rpy": _format_vec(pose.rpy),
        },
    )
    existing_names.add(key)


def _is_world_fixed_joint(element: ET.Element, *, world_link_name: str) -> bool:
    """判断 URDF 元素是否是源文件中连接 world link 的 fixed joint。"""

    if element.tag != "joint" or element.get("type") != "fixed":
        return False
    parent = element.find("parent")
    return parent is not None and parent.get("link") == world_link_name


def _clone_element(element: ET.Element) -> ET.Element:
    """深拷贝 XML 元素，避免修改源 URDF parse tree。"""

    return ET.fromstring(ET.tostring(element, encoding="unicode"))


def _rewrite_mesh_filenames(
    element: ET.Element, *, source_dir: Path, output_dir: Path
) -> None:
    """把相对 mesh 路径改写为相对缓存 URDF 输出目录的路径。"""

    for child in element.iter():
        filename = child.get("filename")
        if not filename or "://" in filename or Path(filename).is_absolute():
            continue
        source_mesh_path = (source_dir / filename).resolve()
        child.set("filename", os.path.relpath(source_mesh_path, output_dir.resolve()))


def _format_vec(values: tuple[float, float, float]) -> str:
    """按 URDF origin 需要的空格分隔格式输出三维向量。"""

    return " ".join(f"{float(value):.9g}" for value in values)


def _cumotion_settings(robot_config: Mapping[str, object]) -> Mapping[str, object]:
    """读取 robot config 顶层 cumotion 分组。"""

    settings = robot_config.get("cumotion")
    if not isinstance(settings, Mapping):
        raise ValueError("cuMotion config must be a mapping")
    return settings


def _side_cumotion_settings(
    robot_config: Mapping[str, object], *, side: str
) -> Mapping[str, object]:
    """读取单侧 robot profile 的 cumotion 资源配置，并拒绝嵌套双臂格式。"""

    settings = _cumotion_settings(robot_config)
    for key in ("left", "right"):
        if key in settings:
            raise ValueError(
                f"{side} robot profile must be a single-articulation robot config"
            )
    return settings


def _dual_generation_config_from_side_mappings(
    cumotion: Mapping[str, object],
) -> DualUrdfGenerationConfig:
    """从 cumotion.left/right 简写结构构造双臂生成配置。"""

    left = _required_mapping(cumotion, "left", section="cumotion")
    right = _required_mapping(cumotion, "right", section="cumotion")
    return DualUrdfGenerationConfig(
        left_xrdf_path=repo_path(_side_required_value(left, "left", "xrdf_path")),
        right_xrdf_path=repo_path(_side_required_value(right, "right", "xrdf_path")),
        left_urdf_path=repo_path(_side_required_value(left, "left", "urdf_path")),
        right_urdf_path=repo_path(_side_required_value(right, "right", "urdf_path")),
        output_dir=repo_path(cumotion.get("output_dir", ".cache/cumotion")),
        robot_name=str(cumotion.get("robot_name", DualUrdfGenerationConfig.robot_name)),
        parent_link=str(cumotion.get("parent_link", DualUrdfGenerationConfig.parent_link)),
        left_base_link=str(
            cumotion.get("left_base_link", DualUrdfGenerationConfig.left_base_link)
        ),
        right_base_link=str(
            cumotion.get("right_base_link", DualUrdfGenerationConfig.right_base_link)
        ),
        left_mount_joint=str(
            cumotion.get("left_mount_joint", DualUrdfGenerationConfig.left_mount_joint)
        ),
        right_mount_joint=str(
            cumotion.get("right_mount_joint", DualUrdfGenerationConfig.right_mount_joint)
        ),
    )


def _dual_flange_frames(cumotion_settings: Mapping[str, object]) -> dict[str, str]:
    """读取双臂配置中的左右 flange frame 名称。"""

    frames: dict[str, str] = {}
    for side in ("left", "right"):
        side_settings = _required_mapping(cumotion_settings, side, section="cumotion")
        value = _side_required_value(side_settings, side, "flange_frame")
        frames[side] = str(value)
    return frames


def _dual_xrdf_path_for_urdf(urdf_path: Path) -> Path:
    """根据缓存 URDF 路径推导同目录同名 XRDF 路径。"""

    return urdf_path.with_suffix(".xrdf")


def _load_xrdf(path: Path) -> Mapping[str, object]:
    """读取 XRDF YAML，并要求顶层是 mapping。"""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"XRDF must be a mapping: {path}")
    return data


def _merge_xrdf_documents(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    """融合左右单臂 XRDF，生成双臂 C-space 和默认关节配置。"""

    left_cspace = _required_mapping(left, "cspace", section="left XRDF")
    right_cspace = _required_mapping(right, "cspace", section="right XRDF")
    left_joint_names = _required_sequence(
        left_cspace, "joint_names", section="left XRDF cspace"
    )
    right_joint_names = _required_sequence(
        right_cspace, "joint_names", section="right XRDF cspace"
    )

    combined: dict[str, object] = {
        "format": left.get("format", right.get("format", "xrdf")),
        "format_version": left.get(
            "format_version", right.get("format_version", 2.0)
        ),
        "default_joint_positions": _merge_default_joint_positions(left, right),
        "cspace": _merge_xrdf_cspace(left_cspace, right_cspace),
    }
    combined_tool_frames = _merge_unique_sequences(
        _optional_sequence(left, "tool_frames"),
        _optional_sequence(right, "tool_frames"),
    )
    if combined_tool_frames:
        combined["tool_frames"] = combined_tool_frames

    _validate_cspace_width(
        combined["cspace"],
        expected_width=len(left_joint_names) + len(right_joint_names),
    )
    return combined


def _merge_default_joint_positions(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, float]:
    """合并左右 XRDF default_joint_positions，并拒绝重复关节名。"""

    merged: dict[str, float] = {}
    for label, document in (("left", left), ("right", right)):
        values = document.get("default_joint_positions", {})
        if not isinstance(values, Mapping):
            raise ValueError(f"{label} XRDF default_joint_positions must be a mapping")
        for name, value in values.items():
            key = str(name)
            if key in merged:
                raise ValueError(f"Duplicate XRDF default joint position: {key}")
            merged[key] = float(value)
    return merged


def _merge_xrdf_cspace(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    """合并左右 XRDF cspace 字段；列表字段拼接，标量字段必须一致或单侧缺省。"""

    merged: dict[str, object] = {
        "joint_names": [
            *_required_sequence(left, "joint_names", section="left XRDF cspace"),
            *_required_sequence(right, "joint_names", section="right XRDF cspace"),
        ]
    }
    for key in sorted((set(left) | set(right)) - {"joint_names"}):
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, list) and isinstance(right_value, list):
            merged[key] = [*left_value, *right_value]
        elif left_value is None:
            merged[key] = right_value
        elif right_value is None:
            merged[key] = left_value
        elif left_value == right_value:
            merged[key] = left_value
        else:
            raise ValueError(f"Cannot merge XRDF cspace field {key!r}")
    return merged


def _validate_cspace_width(cspace: object, *, expected_width: int) -> None:
    """校验合并后 cspace 的列表字段长度等于左右关节数之和。"""

    if not isinstance(cspace, Mapping):
        raise ValueError("merged XRDF cspace must be a mapping")
    for key, value in cspace.items():
        if key == "joint_names":
            continue
        if isinstance(value, list) and len(value) != expected_width:
            raise ValueError(
                f"merged XRDF cspace.{key} expected {expected_width} values, "
                f"got {len(value)}"
            )


def _optional_repo_path(value: object) -> Path | None:
    """把可选路径解析为仓库路径；None 原样保留。"""

    if value is None:
        return None
    return repo_path(value)


def _required_mapping(
    data: Mapping[str, object], key: str, *, section: str
) -> Mapping[str, object]:
    """读取必填 mapping 字段，并把错误定位到 section.key。"""

    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{section}.{key} must be a mapping")
    return value


def _required_sequence(
    data: Mapping[str, object], key: str, *, section: str
) -> list[object]:
    """读取必填序列字段；XRDF 中这里要求 YAML list。"""

    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{section}.{key} must be a sequence")
    return list(value)


def _optional_sequence(data: Mapping[str, object], key: str) -> list[object]:
    """读取可选 XRDF list 字段；缺省时返回空列表。"""

    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"XRDF {key} must be a sequence")
    return list(value)


def _merge_unique_sequences(left: list[object], right: list[object]) -> list[object]:
    """按出现顺序合并两个序列，并用字符串化值去重。"""

    merged: list[object] = []
    seen: set[str] = set()
    for value in (*left, *right):
        key = str(value)
        if key in seen:
            continue
        merged.append(value)
        seen.add(key)
    return merged


def _side_required_value(
    data: Mapping[str, object], side: str, key: str
) -> object:
    """读取 cumotion.<side> 下的必填字段。"""

    value = data.get(key)
    if value is None:
        raise ValueError(f"cumotion.{side}.{key} is required")
    return value


def _required_value(
    data: Mapping[str, object], key: str, *, section: str
) -> object:
    """读取必填字段，并把错误定位到 section.key。"""

    value = data.get(key)
    if value is None:
        raise ValueError(f"{section}.{key} is required")
    return value


def _require_file(path: Path, *, label: str) -> None:
    """在生成 URDF/XRDF 前确认输入资源文件存在。"""

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
