"""机器人部件分类工具。

本模块按资产命名规范中的 ``<single-system-name>_<category>_<local-name>`` 格式把关节/刚体
名称分组。``category`` 使用 ``arm``、``hand`` 等稳定字段，因此控制器、USD/PhysX 覆盖和
solver 设置不需要绑定具体设备型号。

职责边界:
    * 解析 profile 中的精确名称分组，并在名称未显式列出时按 token 做轻量推断。
    * 不读取资产文件，也不检查名称是否真的存在于 articulation；这由资产/运行时层负责。
    * 只把 ``arm``/``hand`` 作为当前控制参数分组，其它已知 token 仍会回退到 ``default``。
    * 未知或不符合规范的名称返回 ``default``，避免第三方 USD prim 或临时调试对象破坏导入流程。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


KNOWN_COMPONENTS = frozenset({"arm", "hand", "gripper", "sensor", "tool", "base"})


@dataclass(frozen=True)
class ComponentNameGroups:
    """显式分配给规范机器人部件的实体名称集合。

    ``arm``、``hand`` 和 ``default`` 保存互不重复的精确关节或刚体名称；tuple 与冻结
    dataclass 使映射在资产导入生命周期内保持稳定。该对象只表达 profile 中的显式归属，
    不自行应用名称 token 推断。
    """

    arm: tuple[str, ...] = ()
    hand: tuple[str, ...] = ()
    default: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        label: str,
        default_key: str,
    ) -> "ComponentNameGroups | None":
        """严格解析 arm、hand 和调用方指定的默认名称分组。

        参数:
            data: 分组 mapping；``None`` 表示 profile 没有声明该类分组。
            label: 完整配置路径，用于错误信息。
            default_key: YAML 中映射到 ``default`` 的键，例如关节的 ``passive``。
        返回:
            冻结的名称分组；输入为 ``None`` 时返回 ``None``。
        异常:
            ValueError: 结构、名称类型、重复项、未知键或跨分组重复归属不合法。
        """

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        allowed = {"arm", "hand", default_key}
        unsupported = set(data) - allowed
        if unsupported:
            names = ", ".join(sorted(str(name) for name in unsupported))
            paths = ", ".join(f"{label}.{name}" for name in sorted(unsupported))
            raise ValueError(
                f"{label} contains unsupported keys: {names} (full paths: {paths})"
            )
        groups = {
            "arm": _exact_name_sequence(data.get("arm", ()), f"{label}.arm"),
            "hand": _exact_name_sequence(data.get("hand", ()), f"{label}.hand"),
            "default": _exact_name_sequence(
                data.get(default_key, ()), f"{label}.{default_key}"
            ),
        }
        owners: dict[str, str] = {}
        duplicates: list[str] = []
        for component, names in groups.items():
            for name in names:
                if name in owners:
                    duplicates.append(name)
                else:
                    owners[name] = component
        if duplicates:
            raise ValueError(
                f"{label} assigns names to multiple components: {sorted(duplicates)}"
            )
        return cls(**groups)

    def explicit_component(self, name: str) -> str | None:
        """返回名称的显式部件归属，不应用命名约定推断。

        未在任何 tuple 中声明时返回 ``None``，让调用方区分“显式 default”和“未声明”。
        """

        if name in self.arm:
            return "arm"
        if name in self.hand:
            return "hand"
        if name in self.default:
            return "default"
        return None


@dataclass(frozen=True)
class RobotComponentMapping:
    """关节和刚体的部件归属解析器。

    ``joints`` 与 ``rigid_bodies`` 分别保存 profile 的精确名称分组；任一分组为 ``None``
    表示没有显式配置。公开解析方法始终先查精确归属，再按规范名称 token 推断，最终保证
    返回 ``arm``、``hand`` 或 ``default``。
    """

    joints: ComponentNameGroups | None = None
    rigid_bodies: ComponentNameGroups | None = None

    @classmethod
    def from_profile(cls, data: Mapping[str, object]) -> "RobotComponentMapping":
        """从完整机器人 profile 读取规范部件映射。

        参数:
            data: 包含可选 ``joint_groups`` 和 ``rigid_body_groups`` 的 profile。
        返回:
            冻结的映射对象；两个字段缺失时仍返回空映射对象。
        异常:
            ValueError: 任一显式分组不符合严格名称 schema。
        """

        return cls(
            joints=ComponentNameGroups.from_mapping(
                data.get("joint_groups"),
                label="joint_groups",
                default_key="passive",
            ),
            rigid_bodies=ComponentNameGroups.from_mapping(
                data.get("rigid_body_groups"),
                label="rigid_body_groups",
                default_key="default",
            ),
        )

    def joint_component(self, name: str) -> str:
        """解析关节归属，顺序为显式分组优先、规范名称推断其次。

        参数:
            name: profile 或 articulation 中的精确关节名。
        返回:
            ``arm``、``hand`` 或 ``default``；不修改映射对象。
        """

        exact = (
            self.joints.explicit_component(name) if self.joints is not None else None
        )
        return exact if exact is not None else component_for_name(name)

    def rigid_body_component(self, name: str) -> str:
        """解析刚体归属，顺序为显式分组优先、规范名称推断其次。

        参数:
            name: profile 或 USD 中的精确刚体名。
        返回:
            ``arm``、``hand`` 或 ``default``；不修改映射对象。
        """

        exact = (
            self.rigid_bodies.explicit_component(name)
            if self.rigid_bodies is not None
            else None
        )
        return exact if exact is not None else component_for_name(name)

    def explicit_rigid_body_component(self, name: str) -> str | None:
        """仅返回刚体的显式归属，供 USD 祖先向后代传播策略。

        未配置刚体分组或名称未列出时返回 ``None``；刻意不做 token 推断，避免把推断结果
        当作用户明确声明并继续向整棵子树传播。
        """

        if self.rigid_bodies is None:
            return None
        return self.rigid_bodies.explicit_component(name)


def _exact_name_sequence(value: object, label: str) -> tuple[str, ...]:
    """解析严格、非空且不重复的实体名称序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(names)


def component_token_from_name(name: str) -> str | None:
    """从规范实体名中提取部件字段。

    命名规范要求内部实体名形如
    ``<single-system-name>_<category>_<local-name>``，其中单体系统名自身可能
    带侧别字段。因此这里从左到右查找已知 category token，而不是匹配具体
    设备型号前缀。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        首个已知 category token；名称不符合规范时返回 ``None``。
    副作用:
        无。
    """

    # 从左到右查找已知 category，而不是假设型号前缀长度固定；这样 AR5V2_L、L6V1_L 等
    # 不同系统名都能复用同一分类逻辑。
    tokens = [token for token in str(name).split("_") if token]
    for token in tokens:
        if token in KNOWN_COMPONENTS:
            return token
    return None


def is_arm_name(name: str) -> bool:
    """判断名称是否属于机械臂。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否包含规范 category ``arm``。
    """

    return component_token_from_name(name) == "arm"


def is_hand_name(name: str) -> bool:
    """判断名称是否属于灵巧手。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否包含规范 category ``hand``。
    """

    return component_token_from_name(name) == "hand"


def component_for_name(name: str) -> str:
    """按名称返回 ``arm``、``hand`` 或 ``default``。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        分类字符串；未知名称返回 ``default``，调用方可使用回退参数。
    """

    component = component_token_from_name(name)
    return component if component in {"arm", "hand"} else "default"
