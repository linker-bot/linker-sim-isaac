"""tiled interactive Foxglove/MCAP telemetry helpers.

本模块只处理纯数据 payload 到 Foxglove 输出的转换，不读取 Isaac stage，也不参与控制闭环。
调用方负责在安全的仿真主线程读出 state，再把 JSON-compatible response 交给这里发布。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.telemetry.foxglove import FoxgloveLogger, FoxgloveTopicConfig


@dataclass(frozen=True)
class TiledTelemetryConfig:
    """tiled telemetry 输出配置。

    ``selected_env_ids`` 用于限制交互调试时发布的 env。标准 JointStates 只发布第一个
    selected env，完整 JSON 仍保留本次 state response 中的所有 selected env 数据。
    """

    selected_env_ids: tuple[int, ...] = (0,)
    publish_decimation: int = 1
    topic_prefix: str = "/tiled"
    include_full_batch_json: bool = True
    include_standard_joint_states: bool = True

    def __post_init__(self) -> None:
        """校验配置字段，避免 telemetry 线程里才暴露低级 shape 错误。"""

        if not self.selected_env_ids:
            raise ValueError("selected_env_ids cannot be empty")
        if len(set(self.selected_env_ids)) != len(self.selected_env_ids):
            raise ValueError("selected_env_ids cannot contain duplicates")
        if any(env_id < 0 for env_id in self.selected_env_ids):
            raise ValueError("selected_env_ids cannot contain negative values")
        if int(self.publish_decimation) < 1:
            raise ValueError("publish_decimation must be >= 1")
        if not self.topic_prefix.startswith("/"):
            raise ValueError("topic_prefix must start with '/'")

    @classmethod
    def from_env_ids(
        cls,
        env_ids: str | Sequence[int] | None,
        *,
        publish_decimation: int = 1,
        topic_prefix: str = "/tiled",
        include_full_batch_json: bool = True,
        include_standard_joint_states: bool = True,
    ) -> "TiledTelemetryConfig":
        """从 CLI 字符串或整数序列创建配置。"""

        if env_ids is None:
            selected = (0,)
        elif isinstance(env_ids, str):
            selected = _parse_env_ids(env_ids)
        else:
            selected = tuple(int(item) for item in env_ids)
        return cls(
            selected_env_ids=selected,
            publish_decimation=int(publish_decimation),
            topic_prefix=topic_prefix.rstrip("/") or "/tiled",
            include_full_batch_json=bool(include_full_batch_json),
            include_standard_joint_states=bool(include_standard_joint_states),
        )


class TiledInteractiveTelemetrySink:
    """把 tiled interactive state response 写入 Foxglove live server 或 MCAP。"""

    def __init__(
        self,
        loggers: Sequence[FoxgloveLogger],
        *,
        config: TiledTelemetryConfig,
    ) -> None:
        """保存一个或多个 Foxglove logger。"""

        if not loggers:
            raise ValueError("at least one FoxgloveLogger is required")
        self.loggers = tuple(loggers)
        self.config = config
        self.last_published_step: int | None = None

    @classmethod
    def open(
        cls,
        *,
        config: TiledTelemetryConfig,
        live_host: str = "127.0.0.1",
        live_port: int | None = None,
        mcap_path: str | Path | None = None,
    ) -> "TiledInteractiveTelemetrySink | None":
        """按可选 live/MCAP 输出创建 sink；没有输出目标时返回 ``None``。"""

        if live_port is None and mcap_path is None:
            return None
        topics = _topics_for_config(config)
        loggers: list[FoxgloveLogger] = []
        if live_port is not None:
            loggers.append(
                FoxgloveLogger.open_live_server(
                    host=live_host,
                    port=int(live_port),
                    name="linkerbot-tiled-interactive",
                    topics=topics,
                )
            )
        if mcap_path is not None:
            loggers.append(FoxgloveLogger.open_mcap(mcap_path, topics=topics))
        return cls(loggers, config=config)

    def close(self) -> None:
        """关闭所有底层 Foxglove sink。"""

        for logger in self.loggers:
            logger.close()

    def publish_interactive_state(
        self,
        state_response: Mapping[str, object],
        *,
        event: str,
        trigger_response: Mapping[str, object] | None = None,
    ) -> bool:
        """发布一帧 interactive state response。

        返回 ``True`` 表示本次实际写入了 Foxglove；``False`` 表示被 decimation 跳过。
        """

        step = int(state_response.get("step", 0))
        if not self._should_publish(step=step, event=event):
            return False
        payload = _json_payload(
            state_response,
            event=event,
            trigger_response=trigger_response,
        )
        time_s = float(state_response.get("time_s", 0.0))
        joint_state = _selected_joint_state_arrays(
            state_response,
            selected_env_id=self.config.selected_env_ids[0],
        )
        scene_markers = _selected_scene_markers(
            state_response,
            selected_env_id=self.config.selected_env_ids[0],
        )
        for logger in self.loggers:
            if self.config.include_full_batch_json:
                logger.log_state_json(payload, time_s=time_s)
            if self.config.include_standard_joint_states and joint_state is not None:
                names, positions, velocities = joint_state
                logger.log_joint_state(
                    joint_names=names,
                    positions=positions,
                    velocities=velocities,
                    time_s=time_s,
                )
            _publish_scene_markers(
                logger,
                scene_markers,
                selected_env_id=self.config.selected_env_ids[0],
                time_s=time_s,
            )
        self.last_published_step = step
        return True

    def _should_publish(self, *, step: int, event: str) -> bool:
        """根据事件类型和 decimation 判断是否发布。"""

        if event in {"reset", "set_state"}:
            return True
        if self.last_published_step == step and event != "state":
            return False
        return step % int(self.config.publish_decimation) == 0


def _topics_for_config(config: TiledTelemetryConfig) -> FoxgloveTopicConfig:
    """根据 topic_prefix 和 selected env 创建 Foxglove topic 配置。"""

    prefix = config.topic_prefix.rstrip("/") or "/tiled"
    selected = int(config.selected_env_ids[0])
    return FoxgloveTopicConfig(
        joint_states=f"{prefix}/env_{selected:03d}/joint_states",
        scene=f"{prefix}/env_{selected:03d}/scene",
        state=f"{prefix}/state",
    )


def _json_payload(
    state_response: Mapping[str, object],
    *,
    event: str,
    trigger_response: Mapping[str, object] | None,
) -> dict[str, object]:
    """构造写入 `/tiled/state` 的 JSON payload。"""

    payload = {
        "event": str(event),
        "step": int(state_response.get("step", 0)),
        "time_s": float(state_response.get("time_s", 0.0)),
        "env_ids": list(state_response.get("env_ids", [])),
        "state": state_response.get("state", {}),
    }
    if trigger_response is not None:
        payload["trigger"] = {
            key: value
            for key, value in trigger_response.items()
            if key not in {"state", "joint_positions"}
        }
    return payload


def _selected_joint_state_arrays(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """从 interactive state response 中提取一个 env 的标准 JointStates 数组。"""

    state = state_response.get("state")
    if not isinstance(state, Mapping):
        return None
    row_index = _selected_row_index(state_response, selected_env_id=selected_env_id)
    if row_index is None:
        return None
    robots = state.get("robots")
    if isinstance(robots, Mapping):
        return _robot_joint_state_arrays(robots, row_index=row_index)
    if "joint_positions" in state:
        positions = np.asarray(state["joint_positions"], dtype=float)
        if positions.ndim != 2 or row_index >= positions.shape[0]:
            return None
        names = [f"command_{index}" for index in range(positions.shape[1])]
        velocities = np.zeros(positions.shape[1], dtype=float)
        return names, positions[row_index], velocities
    return None


def _selected_row_index(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> int | None:
    """把 env id 映射到 state response 的行索引。"""

    env_ids = tuple(int(item) for item in state_response.get("env_ids", ()))
    if not env_ids:
        return None
    try:
        return env_ids.index(int(selected_env_id))
    except ValueError:
        return None


def _robot_joint_state_arrays(
    robots: Mapping[str, object],
    *,
    row_index: int,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """把 robots mapping 展平成 Foxglove JointStates 数组。"""

    names: list[str] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for robot_name, robot_state in robots.items():
        if not isinstance(robot_state, Mapping):
            continue
        joint_positions = robot_state.get("joint_positions")
        joint_names = tuple(str(name) for name in robot_state.get("joint_names", ()))
        if joint_positions is None:
            continue
        q = np.asarray(joint_positions, dtype=float)
        if q.ndim != 2 or row_index >= q.shape[0]:
            continue
        if not joint_names:
            joint_names = tuple(f"joint_{index}" for index in range(q.shape[1]))
        dq = _robot_velocity_row(robot_state, row_index=row_index, width=q.shape[1])
        names.extend(f"{robot_name}/{joint_name}" for joint_name in joint_names)
        positions.append(q[row_index])
        velocities.append(dq)
    if not names:
        return None
    return names, np.concatenate(positions), np.concatenate(velocities)


def _robot_velocity_row(
    robot_state: Mapping[str, object],
    *,
    row_index: int,
    width: int,
) -> np.ndarray:
    """读取一个 robot 的 selected env 速度；缺省时用 0 填充。"""

    values = robot_state.get("joint_velocities")
    if values is None:
        return np.zeros(int(width), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or row_index >= array.shape[0] or array.shape[1] != int(width):
        return np.zeros(int(width), dtype=float)
    return array[row_index]


def _selected_scene_markers(
    state_response: Mapping[str, object],
    *,
    selected_env_id: int,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """从 state response 中提取 selected env 的 object/TCP marker 点。"""

    state = state_response.get("state")
    if not isinstance(state, Mapping):
        return {"objects": [], "tcps": []}
    row_index = _selected_row_index(state_response, selected_env_id=selected_env_id)
    markers: dict[str, list[tuple[str, np.ndarray]]] = {"objects": [], "tcps": []}
    objects = state.get("objects")
    if isinstance(objects, Mapping):
        for object_name, object_state in objects.items():
            if not isinstance(object_state, Mapping):
                continue
            point = _marker_row_for_env(
                object_state.get("positions_world"),
                object_state=object_state,
                fallback_row_index=row_index,
                selected_env_id=selected_env_id,
            )
            if point is not None:
                markers["objects"].append((str(object_name), point))
    robots = state.get("robots")
    if isinstance(robots, Mapping):
        for robot_name, robot_state in robots.items():
            if not isinstance(robot_state, Mapping):
                continue
            point = _marker_row_for_env(
                robot_state.get("tcp_positions_world"),
                object_state=robot_state,
                fallback_row_index=row_index,
                selected_env_id=selected_env_id,
            )
            if point is not None:
                markers["tcps"].append((str(robot_name), point))
    point = _marker_row_for_env(
        state.get("tcp_positions_world"),
        object_state=state,
        fallback_row_index=row_index,
        selected_env_id=selected_env_id,
    )
    if point is not None:
        markers["tcps"].append(("debug", point))
    return markers


def _marker_row_for_env(
    values: object,
    *,
    object_state: Mapping[str, object],
    fallback_row_index: int | None,
    selected_env_id: int,
) -> np.ndarray | None:
    """按 object/robot 自己的 env_ids 读取 marker row；没有 env_ids 时使用顶层行。"""

    if "env_ids" not in object_state:
        return _marker_row(values, row_index=fallback_row_index)
    row_index = _row_index_from_env_ids(
        object_state.get("env_ids"),
        selected_env_id=selected_env_id,
    )
    return _marker_row(values, row_index=row_index)


def _marker_row(values: object, *, row_index: int | None) -> np.ndarray | None:
    """读取 marker position 的 selected row。"""

    if values is None or row_index is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or row_index >= array.shape[0]:
        return None
    return array[row_index].copy()


def _row_index_from_env_ids(
    env_ids: object,
    *,
    selected_env_id: int,
) -> int | None:
    """从局部 env_ids 字段推导 selected env 的行索引。"""

    if env_ids is None:
        return None
    try:
        values = tuple(int(item) for item in env_ids)
    except TypeError:
        return None
    try:
        return values.index(int(selected_env_id))
    except ValueError:
        return None


def _publish_scene_markers(
    logger: FoxgloveLogger,
    markers: Mapping[str, list[tuple[str, np.ndarray]]],
    *,
    selected_env_id: int,
    time_s: float,
) -> None:
    """发布 selected env object/TCP marker；fake logger 缺方法时直接跳过。"""

    log_scene_spheres = getattr(logger, "log_scene_spheres", None)
    if not callable(log_scene_spheres):
        return
    env_prefix = f"env_{int(selected_env_id):03d}"
    for object_name, point in markers.get("objects", []):
        log_scene_spheres(
            entity_id=f"{env_prefix}/object/{object_name}",
            positions=[point],
            frame_id="world",
            radius=0.025,
            color=(0.95, 0.72, 0.20, 1.0),
            time_s=time_s,
        )
    for robot_name, point in markers.get("tcps", []):
        log_scene_spheres(
            entity_id=f"{env_prefix}/tcp/{robot_name}",
            positions=[point],
            frame_id="world",
            radius=0.018,
            color=(0.1, 0.72, 0.95, 1.0),
            time_s=time_s,
        )


def _parse_env_ids(value: str) -> tuple[int, ...]:
    """解析逗号分隔的 env ids。"""

    if not value.strip():
        raise ValueError("env id list cannot be empty")
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())
