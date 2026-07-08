"""从 scene 选中的左右 robot profile 推导双臂规划语义。

这个模块只负责提取“规划层需要知道的机器人语义”，不创建 Isaac runtime，
也不创建 cuMotion context。双臂 runtime 的机器人事实来源是 env profile 中声明的
left/right robot profile；每个 robot profile 再通过 ``cumotion`` 小节描述该侧
cuMotion 使用的 XRDF、URDF 和 flange frame。

这里刻意不再读取 ``configs/dual_arm`` 这类独立 profile，原因是：

- 左右单臂 XRDF 已经定义了该侧 cuMotion C-space 的 ``joint_names``。
- 双臂 XRDF/context 本身也是由左右单臂 cuMotion 资源合并得到的。
- flange frame 和默认 TCP frame 已经是单臂 robot profile 的机器人资源语义。

因此 selected-side 规划所需的左右 C-space 分区可以直接从各自 XRDF 推导，
默认 TCP frame 可以直接从各自 robot profile 的
``cumotion.default_tcp_frame`` / ``cumotion.flange_frame`` 读取。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class DualArmRobotSemantics:
    """双臂规划层需要的最小语义集合。

    ``left_arm_joints`` / ``right_arm_joints`` 是左右机械臂在各自单臂 XRDF
    C-space 中的关节顺序。双臂 cuMotion context 暴露的是融合后的 C-space，
    ``DualArmJointPartitions`` 会用这两组名字在融合关节列表里定位左右分区。

    ``left_default_tcp_frame`` / ``right_default_tcp_frame`` 是 selected-side 动作省略
    ``tcp_frame_name`` 时使用的默认 frame。若 robot profile 没有写 ``default_tcp_frame``，
    则回退到对应侧 ``flange_frame``。
    """

    left_arm_joints: tuple[str, ...]
    right_arm_joints: tuple[str, ...]
    left_flange_frame: str
    right_flange_frame: str
    left_default_tcp_frame: str
    right_default_tcp_frame: str


def dual_arm_semantics_from_robot_configs(
    side_robot_configs: Mapping[str, Mapping[str, object]],
) -> DualArmRobotSemantics:
    """从左右 robot config 推导双臂规划语义。

    参数 ``side_robot_configs`` 来自 ``DualRobotAppRuntime.side_robot_configs``，
    其 key 应包含 ``"left"`` 和 ``"right"``。每侧 config 必须提供：

    - ``cumotion.xrdf_path``：读取 ``cspace.joint_names``，作为该侧机械臂关节名。
    - ``cumotion.flange_frame``：没有默认 TCP 时使用的法兰 frame，也是 custom TCP 的缺省 parent。

    返回值不保存原始 config，避免后续执行层再次依赖配置结构细节。
    """

    left = _side_robot_config(side_robot_configs, "left")
    right = _side_robot_config(side_robot_configs, "right")
    return DualArmRobotSemantics(
        left_arm_joints=_side_xrdf_cspace_joint_names(left, "left"),
        right_arm_joints=_side_xrdf_cspace_joint_names(right, "right"),
        left_flange_frame=_side_robot_flange_frame(left, "left"),
        right_flange_frame=_side_robot_flange_frame(right, "right"),
        left_default_tcp_frame=_side_default_tcp_frame(left, "left"),
        right_default_tcp_frame=_side_default_tcp_frame(right, "right"),
    )


def _side_robot_config(
    side_robot_configs: Mapping[str, Mapping[str, object]], side: str
) -> Mapping[str, object]:
    """读取某一侧 robot config，并在 scene 未声明该侧机器人时给出明确错误。"""

    robot_config = side_robot_configs.get(side)
    if not isinstance(robot_config, Mapping):
        raise ValueError(f"dual robot runtime missing {side!r} robot config")
    return robot_config


def _side_robot_cumotion_config(
    robot_config: Mapping[str, object], side: str
) -> Mapping[str, object]:
    """读取 robot config 的 ``cumotion`` 小节。

    双臂语义只依赖 cuMotion 相关资源，所以这里不读取 ``robot.asset_path``、
    ``robot.prim_path`` 等 Isaac 导入字段。若 robot profile 缺少 ``cumotion``，
    说明该机器人不能参与 cuMotion 双臂规划，应尽早失败。
    """

    cumotion = robot_config.get("cumotion")
    if not isinstance(cumotion, Mapping):
        raise ValueError(f"{side} robot config must contain cumotion mapping")
    return cumotion


def _side_xrdf_cspace_joint_names(
    robot_config: Mapping[str, object], side: str
) -> tuple[str, ...]:
    """从某一侧 robot profile 指向的 XRDF 中读取 C-space 关节顺序。

    单臂也使用这份 XRDF 作为 cuMotion C-space 定义；双臂 selected-side 规划
    只是在融合 C-space 中定位“哪些关节属于左臂/右臂”。因此这里直接复用
    单臂 XRDF 的 ``cspace.joint_names``，避免再维护一份容易漂移的双臂关节表。
    """

    cumotion = _side_robot_cumotion_config(robot_config, side)
    xrdf_value = cumotion.get("xrdf_path")
    if not isinstance(xrdf_value, (str, Path)):
        raise ValueError(f"{side} robot cumotion.xrdf_path must be a path string")

    # robot profile 中的路径通常是仓库相对路径；repo_path 同时兼容绝对路径，
    # 让测试和运行时都能使用同一套配置解析逻辑。
    xrdf_path = repo_path(xrdf_value)
    if not xrdf_path.is_file():
        raise FileNotFoundError(f"{side} robot XRDF file not found: {xrdf_path}")

    xrdf = load_yaml(xrdf_path)

    # XRDF 的 cspace.joint_names 是 cuMotion 规划关节顺序的权威来源。
    # 这里保留严格校验：缺失、类型错误或空列表都会让后续分区计算没有意义。
    cspace = xrdf.get("cspace")
    if not isinstance(cspace, Mapping):
        raise ValueError(f"{side} robot XRDF must contain cspace mapping: {xrdf_path}")
    value = cspace.get("joint_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{side} robot XRDF cspace.joint_names must be a sequence: {xrdf_path}"
        )

    # 统一转成 str tuple，调用侧只需要稳定、不可变的关节名序列。
    # 名字是否存在于融合 C-space 中由 DualArmJointPartitions 继续校验。
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError(f"{side} robot XRDF cspace.joint_names cannot be empty")
    return joints


def _side_robot_flange_frame(robot_config: Mapping[str, object], side: str) -> str:
    """读取某一侧 robot profile 中声明的 cuMotion flange frame。

    自定义 TCP frame 的父 frame 是机器人资源语义，应该随 robot profile 走，而不是随动作
    脚本或双臂 profile 另存一份。
    """

    cumotion = _side_robot_cumotion_config(robot_config, side)
    value = cumotion.get("flange_frame")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{side} robot cumotion.flange_frame cannot be empty")
    return value


def _side_default_tcp_frame(robot_config: Mapping[str, object], side: str) -> str:
    """读取单侧默认 TCP frame，缺省时回退到 flange frame。"""

    cumotion = _side_robot_cumotion_config(robot_config, side)
    value = cumotion.get("default_tcp_frame")
    if value is None or not str(value):
        return _side_robot_flange_frame(robot_config, side)
    return str(value)
