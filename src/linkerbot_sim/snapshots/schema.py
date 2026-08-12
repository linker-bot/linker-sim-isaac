"""运行时无关的 canonical simulation snapshot schema。

机器人使用 array，且每个 entry 同时携带会话级 ``robot_id`` 和稳定 ``label``。
schema 不绑定 Mirror/Kaleidoscope 的具体 Isaac 对象。所有数值数组在
构造时统一转为有限 ``float`` 的 numpy 副本，避免调用方后续修改输入数组而悄悄改变已校验
快照。

坐标与 shape 约定如下：单对象根位姿为 ``(3,)`` 和 ``(4,)``，四元数顺序固定为
``wxyz``；多刚体字段为 ``(body_count, 3)`` / ``(body_count, 4)``。对象位置保存于
metadata 指定的 local frame，adapter 负责在 runtime 边界转换，schema 本身不猜测
world offset。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


SCENE_SNAPSHOT_SCHEMA = "linkerbot.scene-snapshot.v1"


@dataclass(frozen=True)
class SnapshotMetadata:
    """描述 snapshot 来源的元数据。

    ``coordinate_frame`` 记录位姿数组使用的坐标约定：replicated env 通常保存
    ``env-local``，MirrorSceneResources 保存 ``scene-local``。恢复时 adapter 会根据目标 runtime
    再转换成需要的 world/env 坐标。
    """

    source_runtime: str = ""
    source_env_id: int | None = None
    step: int | None = None
    time_s: float | None = None
    coordinate_frame: str = "local"
    info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """直接构造与 JSON 解析使用同一有限数值约束。"""

        source_env_id = _optional_int(self.source_env_id, "source_env_id")
        step = _optional_int(self.step, "step")
        time_s = _optional_float(self.time_s, "time_s")
        object.__setattr__(self, "source_env_id", source_env_id)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "info", dict(self.info))

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "SnapshotMetadata":
        """从 JSON-compatible mapping 解析元数据。

        当前协议允许客户端省略 metadata；此时使用空来源信息。该字段只提供诊断上下文，
        不参与机器人/对象兼容性匹配，因此缺失不应阻止恢复。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("snapshot metadata must be a JSON object")
        return cls(
            source_runtime=str(data.get("source_runtime", "")),
            source_env_id=_optional_int(data.get("source_env_id"), "source_env_id"),
            step=_optional_int(data.get("step"), "step"),
            time_s=_optional_float(data.get("time_s"), "time_s"),
            coordinate_frame=str(data.get("coordinate_frame", "local")),
            info=_mapping_or_empty(data.get("info"), "metadata.info"),
        )

    def as_dict(self) -> dict[str, object]:
        """转换成 JSON-compatible dict，用于 TCP/WebSocket/JSONL 直接传输。"""

        result: dict[str, object] = {
            "source_runtime": self.source_runtime,
            "coordinate_frame": self.coordinate_frame,
            "info": dict(self.info),
        }
        if self.source_env_id is not None:
            result["source_env_id"] = int(self.source_env_id)
        if self.step is not None:
            result["step"] = int(self.step)
        if self.time_s is not None:
            result["time_s"] = float(self.time_s)
        return result


@dataclass(frozen=True)
class RobotSnapshot:
    """单个机器人的逻辑关节状态。

    ``joint_names`` 决定 ``joint_positions`` 和 ``joint_velocities`` 的一维 shape 与元素
    顺序；两者都必须是 ``(len(joint_names),)``。这里刻意只保存 command joints：它们
    是交互控制实际会写入的关节集合，也能避开不同 URDF/USD 中非受控 DOF 的差异。

    ``command_targets`` 是可选控制缓存而非观测状态。提供时必须同时给出
    ``command_joint_names``，并保持同样的一一对应顺序；恢复器据此避免“物理位置已恢复，
    下一控制步却被旧 target 拉回”的跳变。
    """

    label: str
    robot_id: int
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    robot_profile: str | None = None
    asset_fingerprint: str | None = None
    command_joint_names: tuple[str, ...] = ()
    command_targets: np.ndarray | None = None

    def __post_init__(self) -> None:
        """校验并标准化机器人数组。

        快照一旦构造成功，后续兼容性检查就可以假设关节名唯一、位置/速度维度一致。
        这里提前失败能让错误出现在最靠近输入的地方。
        """

        label = str(self.label)
        if not label:
            raise ValueError("RobotSnapshot.label cannot be empty")
        robot_id = _required_nonnegative_int(self.robot_id, "RobotSnapshot.robot_id")
        joint_names = _string_tuple(self.joint_names, "joint_names")
        if not joint_names:
            raise ValueError("RobotSnapshot.joint_names cannot be empty")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("RobotSnapshot.joint_names contains duplicates")
        positions = _vector(self.joint_positions, len(joint_names), "joint_positions")
        velocities = _vector(
            self.joint_velocities, len(joint_names), "joint_velocities"
        )
        command_names = _string_tuple(self.command_joint_names, "command_joint_names")
        if len(set(command_names)) != len(command_names):
            raise ValueError("RobotSnapshot.command_joint_names contains duplicates")
        targets = None
        if self.command_targets is not None:
            if not command_names:
                raise ValueError("command_targets requires command_joint_names")
            targets = _vector(
                self.command_targets,
                len(command_names),
                "command_targets",
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "robot_id", robot_id)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_velocities", velocities)
        object.__setattr__(self, "command_joint_names", command_names)
        object.__setattr__(self, "command_targets", targets)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RobotSnapshot":
        """从 JSON-compatible mapping 解析机器人快照。"""

        if not isinstance(data, Mapping):
            raise ValueError("robot snapshot must be a JSON object")
        _reject_unknown_keys(
            data,
            {
                "label",
                "robot_id",
                "robot_profile",
                "asset_fingerprint",
                "joint_names",
                "joint_positions",
                "joint_velocities",
                "command_joint_names",
                "command_targets",
            },
            "robot snapshot",
        )
        return cls(
            label=_required_str(data.get("label"), "label"),
            robot_id=_required_nonnegative_int(data.get("robot_id"), "robot_id"),
            robot_profile=_optional_str(data.get("robot_profile")),
            asset_fingerprint=_optional_str(data.get("asset_fingerprint")),
            joint_names=tuple(str(item) for item in data.get("joint_names", ())),
            joint_positions=np.asarray(data.get("joint_positions", ()), dtype=float),
            joint_velocities=np.asarray(data.get("joint_velocities", ()), dtype=float),
            command_joint_names=tuple(
                str(item) for item in data.get("command_joint_names", ())
            ),
            command_targets=(
                None
                if data.get("command_targets") is None
                else np.asarray(data.get("command_targets"), dtype=float)
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """转换成 JSON-compatible dict。

        numpy 数组在这里被显式转成 list，保证响应能直接被 ``json.dumps`` 处理。
        """

        result: dict[str, object] = {
            "label": self.label,
            "robot_id": int(self.robot_id),
            "joint_names": list(self.joint_names),
            "joint_positions": self.joint_positions.astype(float).tolist(),
            "joint_velocities": self.joint_velocities.astype(float).tolist(),
            "command_joint_names": list(self.command_joint_names),
        }
        if self.robot_profile is not None:
            result["robot_profile"] = self.robot_profile
        if self.asset_fingerprint is not None:
            result["asset_fingerprint"] = self.asset_fingerprint
        if self.command_targets is not None:
            result["command_targets"] = self.command_targets.astype(float).tolist()
        return result


@dataclass(frozen=True)
class ObjectSnapshot:
    """单个 runtime object 的逻辑位姿状态。

    根对象位姿分别使用 ``(3,)`` 的 local-frame 平移和 ``(4,)`` 的 ``wxyz`` 单位四元数；
    线/角速度若存在也都是 ``(3,)``。dynamic-chain 一类对象还会保存每个 child
    rigid body 的位姿。这样恢复绳子、链条等多刚体对象时，不会只恢复 root 而丢失
    PhysX 中每段刚体的真实状态。

    Newton dynamic-chain 还可携带 owner 的精确广义状态。这里的五字段
    ``generalized_signature/q_names/qd_names/q/qd`` 是一个原子协议：必须全部存在或全部
    缺席。signature 描述 body/joint 拓扑、关节类型与 quaternion/twist ABI，names 则固定
    q/qd 每一列的含义；只有数值 shape 相同并不足以证明两份状态可以安全互换。

    ``generalized_world_origin`` 不属于这五字段身份本身，它记录 world-frame FREE-root q 的
    来源原点，供 replicated env 间恢复时只重定位平移坐标。body fields 始终保留为跨 backend、
    旧快照或部分 body mapping 的 maximal-coordinate fallback。
    """

    name: str
    positions_local: np.ndarray
    orientations_wxyz: np.ndarray
    object_profile: str | None = None
    linear_velocities: np.ndarray | None = None
    angular_velocities: np.ndarray | None = None
    body_names: tuple[str, ...] = ()
    body_positions_local: np.ndarray | None = None
    body_orientations_wxyz: np.ndarray | None = None
    body_linear_velocities: np.ndarray | None = None
    body_angular_velocities: np.ndarray | None = None
    generalized_signature: tuple[str, ...] = ()
    generalized_q_names: tuple[str, ...] = ()
    generalized_qd_names: tuple[str, ...] = ()
    generalized_q: np.ndarray | None = None
    generalized_qd: np.ndarray | None = None
    generalized_world_origin: np.ndarray | None = None

    def __post_init__(self) -> None:
        """校验并标准化对象位姿数组。

        四元数会逐个归一化；如果保存了 ``body_names``，则 body pose 必须同时存在且
        shape 分别为 ``(body_count, 3)``、``(body_count, 4)``，否则恢复 dynamic-chain
        时无法知道每段刚体应该写回到哪里。所有数组都会复制，确保 frozen dataclass 的
        逻辑不可变性不被外部 numpy 引用绕过。广义状态的五字段也在此一次性校验，避免
        adapter 读到“有 q/qd、却没有可验证列身份”的半份 owner state 后发生静默错列写入。
        """

        name = str(self.name)
        if not name:
            raise ValueError("ObjectSnapshot.name cannot be empty")
        position = _vector(self.positions_local, 3, "positions_local")
        orientation = _quat(self.orientations_wxyz, "orientations_wxyz")
        linear = _optional_vector(self.linear_velocities, 3, "linear_velocities")
        angular = _optional_vector(self.angular_velocities, 3, "angular_velocities")
        body_names = _string_tuple(self.body_names, "body_names")
        if len(set(body_names)) != len(body_names):
            raise ValueError("ObjectSnapshot.body_names contains duplicates")
        body_count = len(body_names)
        body_positions = _optional_matrix(
            self.body_positions_local,
            (body_count, 3),
            "body_positions_local",
            required=body_count > 0,
        )
        body_orientations = _optional_quat_matrix(
            self.body_orientations_wxyz,
            body_count,
            "body_orientations_wxyz",
            required=body_count > 0,
        )
        body_linear = _optional_matrix(
            self.body_linear_velocities,
            (body_count, 3),
            "body_linear_velocities",
        )
        body_angular = _optional_matrix(
            self.body_angular_velocities,
            (body_count, 3),
            "body_angular_velocities",
        )
        generalized_signature = _string_tuple(
            self.generalized_signature,
            "generalized_signature",
        )
        generalized_q_names = _string_tuple(
            self.generalized_q_names,
            "generalized_q_names",
        )
        generalized_qd_names = _string_tuple(
            self.generalized_qd_names,
            "generalized_qd_names",
        )
        generalized_presence = (
            bool(generalized_signature),
            bool(generalized_q_names),
            bool(generalized_qd_names),
            self.generalized_q is not None,
            self.generalized_qd is not None,
        )
        # 这是一个不可拆分的 owner-state envelope；不允许按字段做向后兼容猜测。
        if any(generalized_presence) and not all(generalized_presence):
            raise ValueError(
                "ObjectSnapshot generalized_signature, generalized_q_names, "
                "generalized_qd_names, generalized_q, and generalized_qd must "
                "be provided together"
            )
        if len(set(generalized_q_names)) != len(generalized_q_names):
            raise ValueError("ObjectSnapshot.generalized_q_names contains duplicates")
        if len(set(generalized_qd_names)) != len(generalized_qd_names):
            raise ValueError("ObjectSnapshot.generalized_qd_names contains duplicates")
        generalized_q = (
            None
            if not all(generalized_presence)
            else _vector(
                self.generalized_q,
                len(generalized_q_names),
                "generalized_q",
            )
        )
        generalized_qd = (
            None
            if not all(generalized_presence)
            else _vector(
                self.generalized_qd,
                len(generalized_qd_names),
                "generalized_qd",
            )
        )
        if self.generalized_world_origin is not None and generalized_q is None:
            raise ValueError(
                "generalized_world_origin requires a complete generalized state"
            )
        generalized_world_origin = _optional_vector(
            self.generalized_world_origin,
            3,
            "generalized_world_origin",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "positions_local", position)
        object.__setattr__(self, "orientations_wxyz", orientation)
        object.__setattr__(self, "linear_velocities", linear)
        object.__setattr__(self, "angular_velocities", angular)
        object.__setattr__(self, "body_names", body_names)
        object.__setattr__(self, "body_positions_local", body_positions)
        object.__setattr__(self, "body_orientations_wxyz", body_orientations)
        object.__setattr__(self, "body_linear_velocities", body_linear)
        object.__setattr__(self, "body_angular_velocities", body_angular)
        object.__setattr__(self, "generalized_signature", generalized_signature)
        object.__setattr__(self, "generalized_q_names", generalized_q_names)
        object.__setattr__(self, "generalized_qd_names", generalized_qd_names)
        object.__setattr__(self, "generalized_q", generalized_q)
        object.__setattr__(self, "generalized_qd", generalized_qd)
        object.__setattr__(
            self,
            "generalized_world_origin",
            generalized_world_origin,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ObjectSnapshot":
        """从 JSON-compatible mapping 解析对象快照。"""

        if not isinstance(data, Mapping):
            raise ValueError("object snapshot must be a JSON object")
        return cls(
            name=str(data.get("name", "")),
            object_profile=_optional_str(data.get("object_profile")),
            positions_local=np.asarray(data.get("positions_local", ()), dtype=float),
            orientations_wxyz=np.asarray(
                data.get("orientations_wxyz", ()), dtype=float
            ),
            linear_velocities=_optional_array(data.get("linear_velocities")),
            angular_velocities=_optional_array(data.get("angular_velocities")),
            body_names=tuple(str(item) for item in data.get("body_names", ())),
            body_positions_local=_optional_array(data.get("body_positions_local")),
            body_orientations_wxyz=_optional_array(data.get("body_orientations_wxyz")),
            body_linear_velocities=_optional_array(data.get("body_linear_velocities")),
            body_angular_velocities=_optional_array(
                data.get("body_angular_velocities")
            ),
            generalized_signature=tuple(
                str(item) for item in data.get("generalized_signature", ())
            ),
            generalized_q_names=tuple(
                str(item) for item in data.get("generalized_q_names", ())
            ),
            generalized_qd_names=tuple(
                str(item) for item in data.get("generalized_qd_names", ())
            ),
            generalized_q=_optional_array(data.get("generalized_q")),
            generalized_qd=_optional_array(data.get("generalized_qd")),
            generalized_world_origin=_optional_array(
                data.get("generalized_world_origin")
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """转换成 JSON-compatible dict，并保留可选的多刚体位姿字段。"""

        result: dict[str, object] = {
            "name": self.name,
            "positions_local": self.positions_local.astype(float).tolist(),
            "orientations_wxyz": self.orientations_wxyz.astype(float).tolist(),
            "body_names": list(self.body_names),
        }
        if self.object_profile is not None:
            result["object_profile"] = self.object_profile
        _put_optional_array(result, "linear_velocities", self.linear_velocities)
        _put_optional_array(result, "angular_velocities", self.angular_velocities)
        _put_optional_array(result, "body_positions_local", self.body_positions_local)
        _put_optional_array(
            result,
            "body_orientations_wxyz",
            self.body_orientations_wxyz,
        )
        _put_optional_array(
            result,
            "body_linear_velocities",
            self.body_linear_velocities,
        )
        _put_optional_array(
            result,
            "body_angular_velocities",
            self.body_angular_velocities,
        )
        if self.generalized_q is not None:
            result.update(
                {
                    "generalized_signature": list(self.generalized_signature),
                    "generalized_q_names": list(self.generalized_q_names),
                    "generalized_qd_names": list(self.generalized_qd_names),
                    "generalized_q": self.generalized_q.astype(float).tolist(),
                    "generalized_qd": self.generalized_qd.astype(float).tolist(),
                }
            )
        _put_optional_array(
            result,
            "generalized_world_origin",
            self.generalized_world_origin,
        )
        return result


@dataclass(frozen=True)
class SceneSnapshot:
    """一个 scene/env 实例的完整逻辑快照。

    replicated runtime 下表示某一个 env，MirrorSceneResources 下表示当前 scene。这个对象是跨 runtime 交换的
    唯一数据结构，所有 get/set/clone API 都应以它或它的 JSON dict 为边界。外层 mapping
    key 与内部稳定 label/name 必须一致；机器人 ``robot_id`` 也必须唯一，防止兼容性映射
    在恢复目标选择上出现歧义。
    """

    robots: Mapping[str, RobotSnapshot]
    objects: Mapping[str, ObjectSnapshot] = field(default_factory=dict)
    metadata: SnapshotMetadata = field(default_factory=SnapshotMetadata)
    schema: str = SCENE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        """校验 schema discriminator，并保证 mapping key 与稳定 label 一致。"""

        if str(self.schema) != SCENE_SNAPSHOT_SCHEMA:
            raise ValueError(
                f"unsupported snapshot schema: {self.schema!r}; "
                f"expected {SCENE_SNAPSHOT_SCHEMA!r}"
            )
        robots = _robot_mapping(self.robots)
        objects = _object_mapping(self.objects)
        object.__setattr__(self, "robots", robots)
        object.__setattr__(self, "objects", objects)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SceneSnapshot":
        """从 canonical JSON-compatible mapping 解析完整快照。"""

        if not isinstance(data, Mapping):
            raise ValueError("simulation snapshot must be a JSON object")
        _reject_unknown_keys(
            data,
            {"schema", "metadata", "robots", "objects"},
            "simulation snapshot",
        )
        if "schema" not in data:
            raise ValueError("snapshot.schema is required")
        schema = str(data["schema"])
        if schema != SCENE_SNAPSHOT_SCHEMA:
            raise ValueError(
                f"unsupported snapshot schema: {schema!r}; expected {SCENE_SNAPSHOT_SCHEMA!r}"
            )
        robots_data = data.get("robots", {})
        objects_data = data.get("objects", {})
        if isinstance(robots_data, Mapping):
            raise ValueError("snapshot.robots must be an array")
        if not isinstance(robots_data, (list, tuple)):
            raise ValueError("snapshot.robots must be an array")
        if not isinstance(objects_data, Mapping):
            raise ValueError("snapshot.objects must be a JSON object")
        robots = {}
        for index, value in enumerate(robots_data):
            if not isinstance(value, Mapping):
                raise ValueError(f"snapshot.robots[{index}] must be an object")
            robot = RobotSnapshot.from_mapping(value)
            if robot.label in robots:
                raise ValueError(f"duplicate snapshot robot label: {robot.label!r}")
            robots[robot.label] = robot
        objects = {
            str(name): ObjectSnapshot.from_mapping(_with_default_name(name, value))
            for name, value in objects_data.items()
        }
        return cls(
            schema=schema,
            metadata=SnapshotMetadata.from_mapping(data.get("metadata")),
            robots=robots,
            objects=objects,
        )

    def as_dict(self) -> dict[str, object]:
        """转换成 JSON-compatible dict，作为交互协议中的标准 snapshot payload。"""

        return {
            "schema": self.schema,
            "metadata": self.metadata.as_dict(),
            "robots": [robot.as_dict() for robot in self.robots.values()],
            "objects": {name: obj.as_dict() for name, obj in self.objects.items()},
        }


@dataclass(frozen=True)
class SnapshotRestoreResult:
    """snapshot restore 操作返回给交互客户端的摘要。

    ``partial`` 表示有些 snapshot entry 没有被恢复，常见原因是非 strict 模式下只找
    到了部分同名关节/对象；调用方可以据此决定是否接受这次恢复。
    """

    accepted: bool
    event: str = "snapshot_restored"
    robots: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    env_ids: tuple[int, ...] = ()
    partial: bool = False
    message: str = ""

    def as_dict(self) -> dict[str, object]:
        """转换成 JSON-compatible restore 结果。"""

        result: dict[str, object] = {
            "event": self.event,
            "accepted": bool(self.accepted),
            "robots": list(self.robots),
            "objects": list(self.objects),
            "env_ids": [int(env_id) for env_id in self.env_ids],
            "partial": bool(self.partial),
        }
        if self.message:
            result["message"] = self.message
        return result


def _robot_mapping(values: Mapping[str, RobotSnapshot]) -> dict[str, RobotSnapshot]:
    """校验 ``SceneSnapshot.robots``，并返回普通 dict 副本。

    key 必须与 ``RobotSnapshot.label`` 一致，避免恢复时把状态写入错误机器人。
    """

    if not isinstance(values, Mapping):
        raise ValueError("SceneSnapshot.robots must be a mapping")
    result = {}
    robot_ids: set[int] = set()
    labels: set[str] = set()
    for label, robot in values.items():
        if not isinstance(robot, RobotSnapshot):
            raise ValueError("SceneSnapshot.robots values must be RobotSnapshot")
        key = str(label)
        if key != robot.label:
            raise ValueError(
                f"robot mapping key {key!r} does not match label {robot.label!r}"
            )
        result[key] = robot
        if robot.robot_id in robot_ids:
            raise ValueError(f"duplicate snapshot robot_id: {robot.robot_id}")
        robot_ids.add(robot.robot_id)
        if robot.label in labels:
            raise ValueError(f"duplicate snapshot robot label: {robot.label!r}")
        labels.add(robot.label)
    return result


def _object_mapping(values: Mapping[str, ObjectSnapshot]) -> dict[str, ObjectSnapshot]:
    """校验 ``SceneSnapshot.objects``，并返回普通 dict 副本。

    object 名字同样要求外层 key 与内部 ``ObjectSnapshot.name`` 一致，保证对象恢复按名字
    匹配时没有歧义。
    """

    if not isinstance(values, Mapping):
        raise ValueError("SceneSnapshot.objects must be a mapping")
    result = {}
    for name, obj in values.items():
        if not isinstance(obj, ObjectSnapshot):
            raise ValueError("SceneSnapshot.objects values must be ObjectSnapshot")
        key = str(name)
        if key != obj.name:
            raise ValueError(
                f"object mapping key {key!r} does not match name {obj.name!r}"
            )
        result[key] = obj
    return result


def _with_default_name(name: object, value: object) -> Mapping[str, object]:
    """给 object 子 payload 补默认 ``name``。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"snapshot.objects.{name} must be a JSON object")
    result = dict(value)
    result.setdefault("name", str(name))
    return result


def _string_tuple(values: object, label: str) -> tuple[str, ...]:
    """把任意可迭代名字序列转换成非空字符串 tuple。"""

    try:
        result = tuple(str(item) for item in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence") from exc
    if any(not item for item in result):
        raise ValueError(f"{label} cannot contain empty names")
    return result


def _vector(values: object, width: int, label: str) -> np.ndarray:
    """把输入校验为固定长度一维有限 float 向量，并返回独立副本。"""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != int(width):
        raise ValueError(f"{label} must have shape ({int(width)},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return array.astype(float, copy=True)


def _optional_vector(
    values: object | None, width: int, label: str
) -> np.ndarray | None:
    """读取可选固定长度向量；字段缺失时保留 ``None``。"""

    if values is None:
        return None
    return _vector(values, width, label)


def _optional_matrix(
    values: object | None,
    shape: tuple[int, int],
    label: str,
    *,
    required: bool = False,
) -> np.ndarray | None:
    """读取可选固定 shape 矩阵。

    ``required=True`` 用于 body pose：只要 body_names 存在，矩阵就必须存在且维度匹配。
    """

    if values is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return array.astype(float, copy=True)


def _quat(values: object, label: str) -> np.ndarray:
    """读取并归一化单个 wxyz 四元数，拒绝零四元数。"""

    quat = _vector(values, 4, label)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError(f"{label} quaternion must be non-zero")
    return quat / norm


def _optional_quat_matrix(
    values: object | None,
    body_count: int,
    label: str,
    *,
    required: bool = False,
) -> np.ndarray | None:
    """读取并逐行归一化 ``(body_count, 4)`` 的可选 wxyz 四元数矩阵。"""

    if values is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (int(body_count), 4):
        raise ValueError(f"{label} must have shape ({int(body_count)}, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{label} contains a zero quaternion")
    return (array / norms[:, None]).astype(float, copy=True)


def _optional_array(value: object | None) -> np.ndarray | None:
    """把可选 JSON 数组转换成 float ndarray，不做 shape 约束。"""

    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _put_optional_array(
    result: dict[str, object],
    key: str,
    value: np.ndarray | None,
) -> None:
    """把可选 ndarray 放入 JSON dict；``None`` 字段不输出。"""

    if value is not None:
        result[key] = np.asarray(value, dtype=float).tolist()


def _optional_int(value: object, label: str) -> int | None:
    """读取可选整数字段，并把类型错误转成带字段名的 ``ValueError``。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _required_nonnegative_int(value: object, label: str) -> int:
    """读取必填非负整数，拒绝 bool 和隐式缺失。"""

    if value is None:
        raise ValueError(f"{label} is required")
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _optional_float(value: object, label: str) -> float | None:
    """读取可选浮点字段，并把类型错误转成带字段名的 ``ValueError``。"""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_str(value: object | None) -> str | None:
    """读取可选字符串字段；``None`` 保持为缺失语义。"""

    if value is None:
        return None
    return str(value)


def _required_str(value: object | None, label: str) -> str:
    """读取必填非空字符串。"""

    if value is None or not str(value):
        raise ValueError(f"{label} is required")
    return str(value)


def _reject_unknown_keys(
    data: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    """拒绝 canonical schema 之外的字段，而不是静默忽略。"""

    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {unknown}")


def _mapping_or_empty(value: object, label: str) -> Mapping[str, object]:
    """读取可选 JSON object；缺失时返回空 dict。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)
