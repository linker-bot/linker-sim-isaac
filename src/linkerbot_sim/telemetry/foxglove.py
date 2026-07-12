"""Foxglove 遥测输出封装。

该模块提供非 ROS 的 Foxglove SDK 接入：可以写离线 MCAP，也可以开启本地 WebSocket server
给 Foxglove 实时连接。SDK 采用懒加载，未安装 ``foxglove-sdk`` 时不会影响项目其它模块导入。

职责边界:
    * 把项目中的关节状态、点云式 marker 和线段 marker 转换成 Foxglove 消息。
    * 负责外部遥测数据出口，不负责 CSV 数值日志；CSV 仍放在 ``linkerbot_sim.logging``。
    * 不参与控制闭环，不改变机器人或 world 状态。
    * 不负责采样频率控制；调用方决定何时写一帧遥测数据。

时间/单位约定:
    遥测数据采用仿真时间戳（秒转纳秒）写入；位置单位为 m，关节角单位为 rad。任何发送
    失败都应由调用方在调试层处理，而不改变机器人运动逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from linkerbot_sim.utils.config import require_loopback_host
from linkerbot_sim.utils.output_paths import OutputPathPlan, plan_output_file


DEFAULT_SPHERE_MARKER_RADIUS_M = 0.02
DEFAULT_SPHERE_MARKER_COLOR_RGBA = (0.1, 0.45, 1.0, 1.0)
DEFAULT_LINE_MARKER_THICKNESS_M = 0.01
DEFAULT_LINE_MARKER_COLOR_RGBA = (0.0, 0.8, 0.25, 1.0)


def prepare_mcap_output(
    path: str | Path | None,
    *,
    existing_file_policy: str,
) -> OutputPathPlan | None:
    """预检一个 MCAP 输出目标，不打开文件或修改目录。

    MCAP writer 不支持安全追加，因此明确拒绝 ``resume``；其余已有文件策略交给统一输出
    路径规划器，待所有输出一起通过校验后再提交文件系统修改。
    """

    if path is None:
        return None
    if existing_file_policy == "resume":
        raise ValueError(
            "runtime.output.mcap_existing_file_policy='resume' is unsupported: "
            "Foxglove MCAP cannot append to an existing recording"
        )
    return plan_output_file(path, policy=existing_file_policy)


def _ns_time(time_s: float | None = None) -> int:
    """把秒转换为 Foxglove 使用的纳秒时间戳。

    参数:
        time_s: 可选秒时间；为空时返回 0，表示由 sink 使用当前时间。
    返回:
        int 纳秒时间戳。
    """

    # Foxglove SDK 的 log_time 使用 ns；传 None 时返回 0，调用处会让 sink 使用当前时间。
    if time_s is None:
        return 0
    return int(round(float(time_s) * 1_000_000_000))


def _timestamp(time_s: float | None, messages):
    """构造 Foxglove ``Timestamp``。

    参数:
        time_s: 可选秒时间。
        messages: ``foxglove.messages`` 模块。
    返回:
        ``Timestamp`` 实例。
    """

    ns = _ns_time(time_s)
    return messages.Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)


def _vector3(values, messages):
    """构造 Foxglove ``Vector3``。

    参数:
        values: 长度 3 的数值序列。
        messages: ``foxglove.messages`` 模块。
    返回:
        ``Vector3`` 实例。
    """

    x, y, z = np.asarray(values, dtype=float).reshape(3)
    return messages.Vector3(x=float(x), y=float(y), z=float(z))


def _color(rgba, messages):
    """构造 Foxglove ``Color``。

    参数:
        rgba: 长度 4 的 ``(r, g, b, a)``。
        messages: ``foxglove.messages`` 模块。
    返回:
        ``Color`` 实例。
    """

    r, g, b, a = np.asarray(rgba, dtype=float).reshape(4)
    return messages.Color(r=float(r), g=float(g), b=float(b), a=float(a))


def _pose(position, messages):
    """构造仅包含平移的 Foxglove ``Pose``。

    参数:
        position: 长度 3 的位置，单位 m。
        messages: ``foxglove.messages`` 模块。
    返回:
        ``Pose`` 实例，orientation 为单位四元数。
    """

    return messages.Pose(
        position=_vector3(position, messages),
        orientation=messages.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _load_foxglove():
    """导入 Foxglove SDK。

    参数:
        无。
    返回:
        ``(foxglove, foxglove.messages)``。
    """

    # Foxglove 是可选遥测能力，导入失败只在用户真正创建 sink 时暴露，避免影响仿真核心测试。
    try:
        import foxglove
        from foxglove import messages
    except ImportError as exc:
        raise ImportError(
            "Foxglove telemetry requires the optional dependency 'foxglove-sdk'. "
            "Install it with: pip install foxglove-sdk"
        ) from exc
    return foxglove, messages


def _load_foxglove_channels():
    """导入 Foxglove well-known message channel 封装。"""

    try:
        from foxglove import channels
    except ImportError as exc:
        raise ImportError(
            "Foxglove telemetry requires the optional dependency 'foxglove-sdk'. "
            "Install it with: pip install foxglove-sdk"
        ) from exc
    return channels


@dataclass(frozen=True)
class FoxgloveTopicConfig:
    """Foxglove topic 名称配置。

    输入字段:
        joint_states: 关节状态 topic。
        scene: 3D scene marker topic。
    输出:
        传给 ``FoxgloveLogger`` 后用于创建 channel。
    """

    joint_states: str = "/joint_states"
    scene: str = "/scene"
    state: str = "/linkerbot/state"


class FoxgloveLogger:
    """Foxglove MCAP/WebSocket 遥测出口。

    输入:
        sink: Foxglove SDK sink 上下文，例如 ``open_mcap`` 或 ``start_server`` 的返回值。
        topics: topic 名称配置。
    输出:
        ``log_joint_state`` 和 ``log_scene_*`` 方法会向 Foxglove channel 写消息。
    """

    def __init__(self, sink: Any, *, topics: FoxgloveTopicConfig | None = None) -> None:
        """创建 Foxglove 遥测出口。

        参数:
            sink: Foxglove sink/context。
            topics: 可选 topic 名称配置。
        返回:
            无返回值。
        """

        self.foxglove, self.messages = _load_foxglove()
        self.channels = _load_foxglove_channels()
        self.sink = sink
        self.topics = topics or FoxgloveTopicConfig()
        # SDK 的 typed channel 会绑定 Foxglove well-known protobuf schema/encoding，
        # 后续可以直接 log Foxglove message 对象。
        self.joint_channel = self.channels.JointStatesChannel(self.topics.joint_states)
        self.scene_channel = self.channels.SceneUpdateChannel(self.topics.scene)
        self.state_channel = self.foxglove.Channel(
            self.topics.state,
            message_encoding="json",
        )

    @classmethod
    def open_mcap(
        cls,
        path: str | Path,
        *,
        topics: FoxgloveTopicConfig | None = None,
    ) -> "FoxgloveLogger":
        """创建写入离线 MCAP 的 logger。

        参数:
            path: 输出 MCAP 文件路径。
            topics: 可选 topic 名称配置。
        返回:
            ``FoxgloveLogger`` 实例。
        """

        foxglove, _messages = _load_foxglove()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            foxglove.open_mcap(output_path, allow_overwrite=False),
            topics=topics,
        )

    @classmethod
    def open_live_server(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        name: str = "linkerbot-sim",
        topics: FoxgloveTopicConfig | None = None,
    ) -> "FoxgloveLogger":
        """创建本地 WebSocket server logger。

        参数:
            host: 监听地址。
            port: 监听端口。
            name: Foxglove 中显示的 server 名称。
            topics: 可选 topic 名称配置。
        返回:
            ``FoxgloveLogger`` 实例，可用 Foxglove 连接 ``ws://host:port``。
        """

        host = require_loopback_host(host, label="host")
        foxglove, _messages = _load_foxglove()
        return cls(
            foxglove.start_server(name=name, host=host, port=int(port)), topics=topics
        )

    def close(self) -> None:
        """关闭 Foxglove sink。

        不同 Foxglove sink 可能是 context object、server handle 或轻量 mock；只有暴露
        ``close`` 方法时才调用它，因此本方法可安全用于测试替身和上下文管理器清理。
        """

        close = getattr(self.sink, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "FoxgloveLogger":
        """进入上下文管理器。

        返回:
            ``self``。
        """

        return self

    def __exit__(self, *_exc_info) -> None:
        """退出上下文时关闭 sink。

        参数:
            *_exc_info: 上下文异常信息。
        返回:
            无返回值。
        """

        self.close()

    def log_joint_state(
        self,
        *,
        joint_names: list[str] | tuple[str, ...],
        positions,
        velocities=None,
        efforts=None,
        time_s: float | None = None,
    ) -> None:
        """写入 Foxglove ``JointStates`` 消息。

        参数:
            joint_names: 关节名顺序。
            positions: 关节位置数组，单位 rad。
            velocities: 可选关节速度数组，单位 rad/s。
            efforts: 可选关节力/力矩数组。
            time_s: 可选日志时间，单位 s。
        返回:
            无返回值。
        """

        # 先做长度校验再构造消息，避免 Foxglove 中出现关节名和值错位的难查问题。
        positions_array = np.asarray(positions, dtype=float).reshape(-1)
        if positions_array.size != len(joint_names):
            raise ValueError(
                f"positions expected {len(joint_names)} values, got {positions_array.size}"
            )
        velocities_array = (
            None
            if velocities is None
            else np.asarray(velocities, dtype=float).reshape(-1)
        )
        efforts_array = (
            None if efforts is None else np.asarray(efforts, dtype=float).reshape(-1)
        )
        if velocities_array is not None and velocities_array.size != len(joint_names):
            raise ValueError(
                f"velocities expected {len(joint_names)} values, got {velocities_array.size}"
            )
        if efforts_array is not None and efforts_array.size != len(joint_names):
            raise ValueError(
                f"efforts expected {len(joint_names)} values, got {efforts_array.size}"
            )

        joints = []
        for index, name in enumerate(joint_names):
            joints.append(
                self.messages.JointState(
                    name=str(name),
                    position=float(positions_array[index]),
                    velocity=None
                    if velocities_array is None
                    else float(velocities_array[index]),
                    effort=None
                    if efforts_array is None
                    else float(efforts_array[index]),
                )
            )
        msg = self.messages.JointStates(
            timestamp=_timestamp(time_s, self.messages), joints=joints
        )
        self.joint_channel.log(
            msg, log_time=None if time_s is None else _ns_time(time_s)
        )

    def log_state_json(
        self,
        state: Mapping[str, object],
        *,
        time_s: float | None = None,
    ) -> None:
        """写入项目完整状态 JSON 快照。

        参数:
            state: JSON 可序列化的状态字典。
            time_s: 可选日志时间，单位 s。
        返回:
            无返回值。
        """

        self.state_channel.log(
            dict(state), log_time=None if time_s is None else _ns_time(time_s)
        )

    def log_scene_spheres(
        self,
        *,
        entity_id: str,
        positions,
        frame_id: str = "world",
        radius: float = DEFAULT_SPHERE_MARKER_RADIUS_M,
        color=DEFAULT_SPHERE_MARKER_COLOR_RGBA,
        time_s: float | None = None,
    ) -> None:
        """写入一组球形 marker。

        参数:
            entity_id: Scene entity 唯一 ID。
            positions: shape ``(N, 3)`` 的位置数组，单位 m。
            frame_id: 坐标系名称。
            radius: 球半径，单位 m。
            color: RGBA 颜色。
            time_s: 可选日志时间，单位 s。
        返回:
            无返回值。
        """

        # reshape(-1, 3) 允许调用方传单个点或点列表；每个点转换成独立 sphere primitive。
        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        spheres = [
            self.messages.SpherePrimitive(
                pose=_pose(point, self.messages),
                size=self.messages.Vector3(
                    x=float(radius * 2.0), y=float(radius * 2.0), z=float(radius * 2.0)
                ),
                color=_color(color, self.messages),
            )
            for point in points
        ]
        self._log_scene_entity(
            entity_id=entity_id, frame_id=frame_id, spheres=spheres, time_s=time_s
        )

    def log_scene_line_strip(
        self,
        *,
        entity_id: str,
        points,
        frame_id: str = "world",
        thickness: float = DEFAULT_LINE_MARKER_THICKNESS_M,
        color=DEFAULT_LINE_MARKER_COLOR_RGBA,
        time_s: float | None = None,
    ) -> None:
        """写入一条 3D polyline。

        参数:
            entity_id: Scene entity 唯一 ID。
            points: shape ``(N, 3)`` 的点数组，单位 m。
            frame_id: 坐标系名称。
            thickness: 线宽，单位 m。
            color: RGBA 颜色。
            time_s: 可选日志时间，单位 s。
        返回:
            无返回值。
        """

        # LineStrip 按输入顺序连接点，适合画 TCP 轨迹、绳体中心线或调试路径。
        point_array = np.asarray(points, dtype=float).reshape(-1, 3)
        line = self.messages.LinePrimitive(
            type=self.messages.LinePrimitiveLineType.LineStrip,
            points=[_vector3(point, self.messages) for point in point_array],
            thickness=float(thickness),
            color=_color(color, self.messages),
        )
        self._log_scene_entity(
            entity_id=entity_id, frame_id=frame_id, lines=[line], time_s=time_s
        )

    def _log_scene_entity(
        self, *, entity_id: str, frame_id: str, time_s: float | None, **primitive_lists
    ) -> None:
        """写入单个 Foxglove scene entity。

        参数:
            entity_id: Scene entity 唯一 ID。
            frame_id: 坐标系名称。
            time_s: 可选日志时间，单位 s。
            **primitive_lists: ``SceneEntity`` 支持的 primitive 列表字段。
        返回:
            无返回值。
        """

        # 每次写一个 SceneEntity，entity_id 稳定时 Foxglove 会更新同一可视对象，而不是无限累积。
        entity = self.messages.SceneEntity(
            timestamp=_timestamp(time_s, self.messages),
            frame_id=frame_id,
            id=entity_id,
            **primitive_lists,
        )
        msg = self.messages.SceneUpdate(entities=[entity])
        self.scene_channel.log(
            msg, log_time=None if time_s is None else _ns_time(time_s)
        )
