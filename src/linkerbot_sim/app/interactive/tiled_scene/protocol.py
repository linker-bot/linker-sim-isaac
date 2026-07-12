"""Tiled Scene 交互协议的 type 路由、public robot ID 边界与错误响应。

外部协议只暴露稳定的整数 ``robot_id``，runtime 内部仍以场景 label 索引机器人。入口在
调用 runtime 前把 ID 转为 label，出口装饰器再递归恢复为 ID；这条边界禁止内部 label
从状态、轨迹或规划响应中泄漏。协议错误统一转成 JSON-compatible ``rejected``，不会
越过 transport 连接边界。
"""

from __future__ import annotations

from collections.abc import Mapping

from linkerbot_sim.app.interactive.tiled_scene.action_messages import parse_tiled_action
from linkerbot_sim.app.interactive.tiled_scene.message_utils import (
    json_number,
    json_numeric_array,
    optional_json_integer,
    optional_json_string,
    reject_unknown_fields,
    strict_optional_bool,
)
from linkerbot_sim.app.interactive.tiled_scene.selectors import (
    message_env_id,
    message_env_ids,
    message_required_env_ids,
    message_robot_names,
    message_single_robot_name,
    message_source_env_id,
    message_target_env_ids,
    robot_id,
    robot_name_for_id,
)
from linkerbot_sim.tiled.control.adapter import TiledIKRequestRejected


_CONTROL_FIELDS: dict[str, frozenset[str]] = {
    "status": frozenset({"type"}),
    "reset": frozenset({"type", "env_ids"}),
    "get_state": frozenset({"type", "env_ids", "fields"}),
    "set_state": frozenset({"type", "env_ids", "state"}),
    "get_snapshot": frozenset({"type", "env_id"}),
    "set_snapshot": frozenset({"type", "snapshot", "env_ids", "label_map", "strict"}),
    "clone_state": frozenset({"type", "source_env_id", "target_env_ids", "strict"}),
    "step_trajectory": frozenset(
        {"type", "env_ids", "robot_id", "robot_ids", "decimation"}
    ),
    "trajectory_status": frozenset({"type", "env_ids", "robot_id"}),
    "clear_trajectory": frozenset({"type", "env_ids", "robot_id"}),
    "planner_status": frozenset({"type", "wait_timeout_s"}),
    "cancel_plan": frozenset({"type", "request_id", "env_ids", "robot_id"}),
    "clear_completed": frozenset({"type", "request_id", "request_ids"}),
    "quit": frozenset({"type"}),
}

_ENV_IDS_REQUIRED_MESSAGE_TYPES = frozenset(
    {
        "reset",
        "get_state",
        "set_state",
        "set_snapshot",
        "load_trajectory",
        "step_trajectory",
        "trajectory_status",
        "clear_trajectory",
        "hand",
        "plan",
        "step",
    }
)


def _robot_id_response_boundary(handler):
    """装饰 runtime handler，确保所有正常及拒绝响应都经过 public ID 出口。"""

    def wrapped(
        message: Mapping[str, object],
        runtime: object,
        **kwargs: object,
    ) -> dict[str, object]:
        """执行 handler，并在返回 transport 前统一应用 public identity 转换。"""

        return _public_robot_identities(handler(message, runtime, **kwargs), runtime)

    return wrapped


@_robot_id_response_boundary
def handle_tiled_interactive_message(
    message: Mapping[str, object],
    runtime: object,
) -> dict[str, object]:
    """同步执行一条 canonical tiled 消息并返回 JSON-compatible response。

    调用方必须是拥有 runtime 的主仿真线程；本函数本身不提供线程切换。所有 selector、
    字段白名单和 public/internal identity 转换都在触发 runtime 副作用前完成。预期的 IK
    拒绝会保留机器可读失败信息，其余异常在消息边界收敛为拒绝响应。
    """

    try:
        message_type = _message_type(message)
        _validate_control_fields(message, message_type=message_type)
        _require_env_ids(message, message_type=message_type)
        if message_type == "status":
            return runtime.status()
        if message_type == "reset":
            return runtime.reset(env_ids=message_required_env_ids(message))
        if message_type == "get_state":
            return runtime.get_state(
                env_ids=message_required_env_ids(message),
                fields=_message_fields(message),
            )
        if message_type == "set_state":
            state = message.get("state")
            if not isinstance(state, Mapping):
                raise ValueError("set_state.state must be a JSON object")
            return runtime.set_state(
                _runtime_state_from_public_message(state, runtime=runtime),
                env_ids=message_required_env_ids(message),
            )
        if message_type == "get_snapshot":
            return runtime.get_snapshot(env_id=message_env_id(message))
        if message_type == "set_snapshot":
            snapshot = message.get("snapshot")
            if not isinstance(snapshot, Mapping):
                raise ValueError("set_snapshot.snapshot must be a JSON object")
            return runtime.set_snapshot(
                snapshot,
                env_ids=message_required_env_ids(message),
                label_map=_message_label_map(message),
                strict=_message_strict(message),
            )
        if message_type == "clone_state":
            return runtime.clone_state(
                source_env_id=message_source_env_id(message),
                target_env_ids=message_target_env_ids(message),
                strict=_message_strict(message),
            )
        if message_type == "load_trajectory":
            return runtime.load_trajectory(
                message,
                env_ids=message_required_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "step_trajectory":
            return runtime.step_trajectory(
                env_ids=message_required_env_ids(message),
                robot_names=message_robot_names(message, runtime=runtime),
                decimation=optional_json_integer(
                    message,
                    "decimation",
                    label="step_trajectory.decimation",
                ),
            )
        if message_type == "trajectory_status":
            return runtime.trajectory_status(
                env_ids=message_required_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "clear_trajectory":
            return runtime.clear_trajectory(
                env_ids=message_required_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "hand":
            return runtime.submit_hand_motion(
                message,
                env_ids=message_required_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "plan":
            return runtime.submit_plan(
                message,
                env_ids=message_required_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "planner_status":
            return runtime.planner_status(
                wait_timeout_s=(
                    0.0
                    if "wait_timeout_s" not in message
                    else json_number(
                        message["wait_timeout_s"],
                        label="planner_status.wait_timeout_s",
                    )
                ),
            )
        if message_type == "cancel_plan":
            return runtime.cancel_plan(
                request_id=optional_json_string(
                    message,
                    "request_id",
                    label="cancel_plan.request_id",
                ),
                env_ids=message_env_ids(message),
                robot_name=message_single_robot_name(message, runtime=runtime),
            )
        if message_type == "clear_completed":
            return runtime.clear_completed(request_ids=_message_request_ids(message))
        if message_type == "quit":
            if runtime.quit_event is not None:
                runtime.quit_event.set()
            return {"event": "quit", "accepted": True}
        action = parse_tiled_action(
            message,
            planner_defaults=getattr(runtime, "planner_request_defaults", None),
            command_defaults=getattr(runtime, "command_defaults", None),
        )
        return runtime.step_action(
            action,
            env_ids=message_required_env_ids(message),
            robot_names=message_robot_names(message, runtime=runtime),
        )
    except TiledIKRequestRejected as exc:
        return {
            "event": "rejected",
            "accepted": False,
            "code": "ik_failure",
            "error": str(exc),
            "failure_policy": exc.failure_policy,
            "failed_env_ids": list(exc.failed_env_ids),
        }
    except Exception as exc:
        return {"event": "rejected", "error": str(exc)}


def _public_robot_identities(value: object, runtime: object):
    """递归把内部 robot label 字段替换为 session robot IDs。

    转换返回新的 dict/list，不修改 runtime 返回对象；tuple 也规范化为 JSON 可编码 list。
    仅在结构明确表示 robot mapping 时改写键，普通业务 mapping 保持原有含义。
    """

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == "robot" and isinstance(item, str):
                result["robot_id"] = _robot_id_for_name(runtime, item)
            elif key in {"info", "trajectory"} and _is_robot_mapping(item, runtime):
                result[str(key)] = [
                    {
                        "robot_id": _robot_id_for_name(runtime, str(name)),
                        **_public_robot_identities(payload, runtime),
                    }
                    for name, payload in item.items()
                ]
            elif key == "cleared" and _is_robot_mapping(item, runtime):
                result["cleared"] = [
                    {
                        "robot_id": _robot_id_for_name(runtime, str(name)),
                        "env_ids": _public_robot_identities(payload, runtime),
                    }
                    for name, payload in item.items()
                ]
            elif key == "robots" and isinstance(item, Mapping):
                result["robots"] = [
                    {
                        "robot_id": _robot_id_for_name(runtime, str(name)),
                        **_public_robot_identities(payload, runtime),
                    }
                    for name, payload in item.items()
                ]
            elif (
                key == "robots"
                and isinstance(item, list)
                and all(isinstance(name, str) for name in item)
            ):
                result["robot_ids"] = [
                    _robot_id_for_name(runtime, name) for name in item
                ]
            else:
                result[str(key)] = _public_robot_identities(item, runtime)
        return result
    if isinstance(value, list):
        return [_public_robot_identities(item, runtime) for item in value]
    if isinstance(value, tuple):
        return [_public_robot_identities(item, runtime) for item in value]
    return value


def _is_robot_mapping(value: object, runtime: object) -> bool:
    """判断 mapping 的全部 key 是否都是当前 runtime 的内部 robot label。"""

    if not isinstance(value, Mapping) or not value:
        return False
    names = set(str(name) for name in getattr(runtime, "robot_names", ()))
    scene = getattr(runtime, "scene", None)
    mapping = getattr(scene, "robot_id_by_label", None)
    if isinstance(mapping, Mapping):
        names.update(str(name) for name in mapping)
    return bool(names) and all(str(key) in names for key in value)


def _robot_id_for_name(runtime: object, robot_name: str) -> int:
    """把 response 中的内部 robot label 转回 public ID。"""

    scene = getattr(runtime, "scene", None)
    mapping = getattr(scene, "robot_id_by_label", None)
    if isinstance(mapping, Mapping) and robot_name in mapping:
        return int(mapping[robot_name])
    names = tuple(str(name) for name in getattr(runtime, "robot_names", ()))
    try:
        return names.index(str(robot_name))
    except ValueError as exc:
        raise ValueError(
            f"runtime returned unknown robot label {robot_name!r}"
        ) from exc


def _runtime_state_from_public_message(
    state: Mapping[str, object],
    *,
    runtime: object,
) -> dict[str, object]:
    """把 public ``robots[]`` state rows 转成 runtime label-keyed map。

    每个 ``robot_id`` 必须唯一且属于当前 session。所有行完成结构和数值校验后，结果才
    交给 runtime；本函数只变换数据形状，不读取或修改物理状态。
    """

    reject_unknown_fields(
        state,
        {"robots", "episode_steps", "episode_ids"},
        label="set_state.state",
    )
    result: dict[str, object] = {}
    for field in ("episode_steps", "episode_ids"):
        if field in state:
            result[field] = _json_integer_array(
                state[field], label=f"set_state.state.{field}"
            )
    if "robots" not in state:
        return result
    robots = state["robots"]
    if isinstance(robots, Mapping):
        raise ValueError(
            "set_state.state.robots must be an array with robot_id; "
            "label-keyed robot maps are internal"
        )
    if not isinstance(robots, list):
        raise ValueError("set_state.state.robots must be an array")
    by_name: dict[str, object] = {}
    seen_ids: set[int] = set()
    for index, item in enumerate(robots):
        if not isinstance(item, Mapping):
            raise ValueError(f"set_state.state.robots[{index}] must be an object")
        reject_unknown_fields(
            item,
            {"robot_id", "joint_positions", "joint_velocities"},
            label=f"set_state.state.robots[{index}]",
        )
        if "robot_id" not in item:
            raise ValueError(f"set_state.state.robots[{index}].robot_id is required")
        item_robot_id = robot_id(item["robot_id"])
        if item_robot_id in seen_ids:
            raise ValueError("set_state.state.robots cannot contain duplicate robot_id")
        seen_ids.add(item_robot_id)
        label = robot_name_for_id(runtime, item_robot_id)
        payload = {}
        for field in ("joint_positions", "joint_velocities"):
            if field not in item:
                continue
            json_numeric_array(
                item[field],
                label=f"set_state.state.robots[{index}].{field}",
            )
            payload[field] = item[field]
        by_name[label] = payload
    result["robots"] = by_name
    return result


def _json_integer_array(value: object, *, label: str) -> list[int]:
    """读取非负 JSON 整数数组，不接受标量、bool 或数值转换。"""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise ValueError(f"{label}[{index}] must be a JSON integer")
        if item < 0:
            raise ValueError(f"{label}[{index}] must be nonnegative")
    return value


def _message_label_map(message: Mapping[str, object]) -> dict[str, str] | None:
    """读取 snapshot source label 到 runtime target label 的显式映射。"""

    if "label_map" not in message:
        return None
    value = message["label_map"]
    if not isinstance(value, Mapping):
        raise ValueError("label_map must be a JSON object")
    result: dict[str, str] = {}
    for source, target in value.items():
        if not isinstance(source, str) or not source:
            raise ValueError("label_map keys must be non-empty strings")
        if not isinstance(target, str) or not target:
            raise ValueError(f"label_map[{source!r}] must be a non-empty string")
        result[source] = target
    return result


def _message_type(message: Mapping[str, object]) -> str:
    """读取必填命令 discriminator，不把其他值转换成字符串。"""

    value = message.get("type")
    if not isinstance(value, str) or not value:
        raise ValueError("type must be a non-empty string")
    return value


def _validate_control_fields(
    message: Mapping[str, object],
    *,
    message_type: str,
) -> None:
    """对非 payload 控制命令应用当前字段白名单。"""

    allowed = _CONTROL_FIELDS.get(message_type)
    if allowed is not None:
        reject_unknown_fields(message, allowed, label=message_type)


def _require_env_ids(message: Mapping[str, object], *, message_type: str) -> None:
    """要求每个作用于环境的命令显式携带环境 selector。

    ``cancel_plan`` 按 request ID 取消时已能唯一定位，只有按环境/机器人筛选时才要求
    ``env_ids``，避免无意中把缺省范围扩大到所有环境。
    """

    requires_env_ids = message_type in _ENV_IDS_REQUIRED_MESSAGE_TYPES or (
        message_type == "cancel_plan" and "request_id" not in message
    )
    if requires_env_ids and "env_ids" not in message:
        raise ValueError(f"{message_type}.env_ids is required")


def _message_strict(message: Mapping[str, object]) -> bool:
    """读取 snapshot/clone 的 strict 开关，缺省为完整一致性校验。"""

    return strict_optional_bool(
        message,
        "strict",
        default=True,
        label=str(message.get("type", "message")),
    )


def _message_fields(message: Mapping[str, object]) -> tuple[str, ...] | None:
    """读取 ``get_state.fields``，并冻结成便于 runtime 使用的 tuple。"""

    if "fields" not in message:
        return None
    fields = message["fields"]
    if not isinstance(fields, list):
        raise ValueError("fields must be a list of strings")
    if any(not isinstance(item, str) or not item for item in fields):
        raise ValueError("fields must contain non-empty strings")
    return tuple(fields)


def _message_request_ids(
    message: Mapping[str, object],
) -> str | tuple[str, ...] | None:
    """统一读取 clear-completed 的单个或多个 request ID selector。"""

    if "request_id" in message:
        if "request_ids" in message:
            raise ValueError("request_id and request_ids cannot be combined")
        return optional_json_string(
            message,
            "request_id",
            label="clear_completed.request_id",
        )
    if "request_ids" not in message:
        return None
    values = message["request_ids"]
    if not isinstance(values, list):
        raise ValueError("request_ids must be a list of strings")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"clear_completed.request_ids[{index}] must be a non-empty string"
            )
        result.append(item)
    return tuple(result)


__all__ = ["handle_tiled_interactive_message"]
