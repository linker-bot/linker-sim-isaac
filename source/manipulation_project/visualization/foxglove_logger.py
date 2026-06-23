"""Foxglove 可视化日志封装。

该模块提供非 ROS 的 Foxglove SDK 接入：可以写离线 MCAP，也可以开启本地
WebSocket server 给 Foxglove 实时连接。SDK 采用懒加载，未安装 ``foxglove-sdk``
时不会影响项目其它模块导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ns_time(time_s: float | None = None) -> int:
    """把秒转换为 Foxglove 使用的纳秒时间戳。

    参数:
        time_s: 可选秒时间；为空时返回 0，表示由 sink 使用当前时间。
    返回:
        int 纳秒时间戳。
    """

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

    try:
        import foxglove
        from foxglove import messages
    except ImportError as exc:
        raise ImportError(
            "Foxglove visualization requires the optional dependency 'foxglove-sdk'. "
            "Install it with: pip install foxglove-sdk"
        ) from exc
    return foxglove, messages


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


class FoxgloveLogger:
    """Foxglove MCAP/WebSocket 日志器。

    输入:
        sink: Foxglove SDK sink 上下文，例如 ``open_mcap`` 或 ``start_server`` 的返回值。
        topics: topic 名称配置。
    输出:
        ``log_joint_state`` 和 ``log_scene_*`` 方法会向 Foxglove channel 写消息。
    """

    def __init__(self, sink: Any, *, topics: FoxgloveTopicConfig | None = None) -> None:
        """创建 Foxglove logger。

        参数:
            sink: Foxglove sink/context。
            topics: 可选 topic 名称配置。
        返回:
            无返回值。
        """

        self.foxglove, self.messages = _load_foxglove()
        self.sink = sink
        self.topics = topics or FoxgloveTopicConfig()
        self.joint_channel = self.foxglove.Channel(
            self.topics.joint_states,
            schema=self.messages.JointStates.get_schema(),
        )
        self.scene_channel = self.foxglove.Channel(
            self.topics.scene,
            schema=self.messages.SceneUpdate.get_schema(),
        )

    @classmethod
    def open_mcap(
        cls,
        path: str | Path,
        *,
        allow_overwrite: bool = True,
        topics: FoxgloveTopicConfig | None = None,
    ) -> "FoxgloveLogger":
        """创建写入离线 MCAP 的 logger。

        参数:
            path: 输出 MCAP 文件路径。
            allow_overwrite: 是否允许覆盖已有文件。
            topics: 可选 topic 名称配置。
        返回:
            ``FoxgloveLogger`` 实例。
        """

        foxglove, _messages = _load_foxglove()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(foxglove.open_mcap(output_path, allow_overwrite=allow_overwrite), topics=topics)

    @classmethod
    def open_live_server(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        name: str = "linkerhand-simulation",
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

        foxglove, _messages = _load_foxglove()
        return cls(foxglove.start_server(name=name, host=host, port=int(port)), topics=topics)

    def close(self) -> None:
        """关闭 Foxglove sink。

        参数:
            无。
        返回:
            无返回值。
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

        positions_array = np.asarray(positions, dtype=float).reshape(-1)
        if positions_array.size != len(joint_names):
            raise ValueError(f"positions expected {len(joint_names)} values, got {positions_array.size}")
        velocities_array = None if velocities is None else np.asarray(velocities, dtype=float).reshape(-1)
        efforts_array = None if efforts is None else np.asarray(efforts, dtype=float).reshape(-1)
        if velocities_array is not None and velocities_array.size != len(joint_names):
            raise ValueError(f"velocities expected {len(joint_names)} values, got {velocities_array.size}")
        if efforts_array is not None and efforts_array.size != len(joint_names):
            raise ValueError(f"efforts expected {len(joint_names)} values, got {efforts_array.size}")

        joints = []
        for index, name in enumerate(joint_names):
            joints.append(
                self.messages.JointState(
                    name=str(name),
                    position=float(positions_array[index]),
                    velocity=None if velocities_array is None else float(velocities_array[index]),
                    effort=None if efforts_array is None else float(efforts_array[index]),
                )
            )
        msg = self.messages.JointStates(timestamp=_timestamp(time_s, self.messages), joints=joints)
        self.joint_channel.log(msg, log_time=None if time_s is None else _ns_time(time_s))

    def log_scene_spheres(
        self,
        *,
        entity_id: str,
        positions,
        frame_id: str = "world",
        radius: float = 0.02,
        color=(0.1, 0.45, 1.0, 1.0),
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

        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        spheres = [
            self.messages.SpherePrimitive(
                pose=_pose(point, self.messages),
                size=self.messages.Vector3(x=float(radius * 2.0), y=float(radius * 2.0), z=float(radius * 2.0)),
                color=_color(color, self.messages),
            )
            for point in points
        ]
        self._log_scene_entity(entity_id=entity_id, frame_id=frame_id, spheres=spheres, time_s=time_s)

    def log_scene_line_strip(
        self,
        *,
        entity_id: str,
        points,
        frame_id: str = "world",
        thickness: float = 0.01,
        color=(0.0, 0.8, 0.25, 1.0),
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

        point_array = np.asarray(points, dtype=float).reshape(-1, 3)
        line = self.messages.LinePrimitive(
            type=self.messages.LinePrimitiveLineType.LineStrip,
            points=[_vector3(point, self.messages) for point in point_array],
            thickness=float(thickness),
            color=_color(color, self.messages),
        )
        self._log_scene_entity(entity_id=entity_id, frame_id=frame_id, lines=[line], time_s=time_s)

    def _log_scene_entity(self, *, entity_id: str, frame_id: str, time_s: float | None, **primitive_lists) -> None:
        """写入单个 Foxglove scene entity。

        参数:
            entity_id: Scene entity 唯一 ID。
            frame_id: 坐标系名称。
            time_s: 可选日志时间，单位 s。
            **primitive_lists: ``SceneEntity`` 支持的 primitive 列表字段。
        返回:
            无返回值。
        """

        entity = self.messages.SceneEntity(
            timestamp=_timestamp(time_s, self.messages),
            frame_id=frame_id,
            id=entity_id,
            **primitive_lists,
        )
        msg = self.messages.SceneUpdate(entities=[entity])
        self.scene_channel.log(msg, log_time=None if time_s is None else _ns_time(time_s))
