"""Strict, versioned Mirror JSON envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from types import MappingProxyType

from linkerbot_sim.mirror.motion.owner import MIRROR_V1_MOTION_OPERATIONS


MIRROR_PROTOCOL_V1 = "linkerbot.mirror.v1"
MIRROR_PROTOCOL_V2 = "linkerbot.mirror.v2"
MIRROR_PROTOCOL_V3 = "linkerbot.mirror.v3"
# Keep the historical import stable for v1 clients.
MIRROR_PROTOCOL = MIRROR_PROTOCOL_V1
MIRROR_V1_OPERATIONS = frozenset(
    set(MIRROR_V1_MOTION_OPERATIONS)
    | {
        "runtime.reset",
        "state.get",
        "state.set",
        "snapshot.get",
        "snapshot.set",
        "runtime.status",
        "queue.cancel",
        "queue.cancel_current",
        "runtime.estop",
        "runtime.quit",
    }
)
MIRROR_V2_OPERATIONS = frozenset(
    set(MIRROR_V1_OPERATIONS)
    | {
        "control.get_mode",
        "control.set_mode",
        "motion.joint_effort",
    }
)
MIRROR_V3_OPERATIONS = frozenset(
    set(MIRROR_V2_OPERATIONS)
    | {
        "control.get_hybrid_parameters",
        "control.set_hybrid_parameters",
        "control.tare_wrench",
        "motion.hybrid_force_position",
    }
)
_OPERATIONS_BY_PROTOCOL = {
    MIRROR_PROTOCOL_V1: MIRROR_V1_OPERATIONS,
    MIRROR_PROTOCOL_V2: MIRROR_V2_OPERATIONS,
    MIRROR_PROTOCOL_V3: MIRROR_V3_OPERATIONS,
}


def _freeze_json(value: object, *, label: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise ValueError(f"{label} 不能包含 NaN/Infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} 的 JSON object key 必须是字符串")
            result[key] = _freeze_json(item, label=f"{label}.{key}")
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _freeze_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{label} 包含非 JSON 类型 {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class MirrorRequest:
    protocol: str
    request_id: str
    operation: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.protocol not in _OPERATIONS_BY_PROTOCOL:
            raise ValueError(
                "protocol 必须是 "
                f"{MIRROR_PROTOCOL_V1!r}、{MIRROR_PROTOCOL_V2!r} "
                f"或 {MIRROR_PROTOCOL_V3!r}，"
                f"得到 {self.protocol!r}"
            )
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id 必须是非空字符串")
        if self.request_id.strip() != self.request_id:
            raise ValueError("request_id 首尾不能包含空白")
        if self.operation not in _OPERATIONS_BY_PROTOCOL[self.protocol]:
            raise ValueError(
                f"协议 {self.protocol!r} 不支持 Mirror operation: {self.operation!r}"
            )
        frozen = _freeze_json(self.arguments, label="arguments")
        if not isinstance(frozen, Mapping):
            raise ValueError("arguments 必须是 JSON object")
        object.__setattr__(self, "arguments", frozen)

    def arguments_dict(self) -> dict[str, object]:
        return _thaw_json(self.arguments)  # type: ignore[return-value]


@dataclass(frozen=True)
class MirrorError:
    code: str
    message: str
    details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("MirrorError code/message 必须非空")
        if self.details is not None:
            frozen = _freeze_json(self.details, label="error.details")
            if not isinstance(frozen, Mapping):
                raise ValueError("error.details 必须是 mapping")
            object.__setattr__(self, "details", frozen)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"code": self.code, "message": self.message}
        if self.details is not None:
            result["details"] = _thaw_json(self.details)
        return result


@dataclass(frozen=True)
class MirrorResponse:
    request_id: str
    ok: bool
    result: object | None = None
    error: MirrorError | None = None
    protocol: str = MIRROR_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol not in _OPERATIONS_BY_PROTOCOL:
            raise ValueError("MirrorResponse protocol 非法")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("MirrorResponse.request_id 必须非空")
        if self.ok == (self.error is not None):
            raise ValueError("成功响应只能有 result，失败响应必须有 error")
        if self.ok:
            object.__setattr__(
                self, "result", _freeze_json(self.result, label="result")
            )

    @classmethod
    def success(
        cls,
        request_id: str,
        result: object,
        *,
        protocol: str = MIRROR_PROTOCOL,
    ) -> "MirrorResponse":
        return cls(request_id=request_id, ok=True, result=result, protocol=protocol)

    @classmethod
    def failure(
        cls,
        request_id: str,
        *,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        protocol: str = MIRROR_PROTOCOL,
    ) -> "MirrorResponse":
        return cls(
            request_id=request_id,
            ok=False,
            error=MirrorError(code=code, message=message, details=details),
            protocol=protocol,
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "protocol": self.protocol,
            "request_id": self.request_id,
            "ok": self.ok,
        }
        if self.ok:
            result["result"] = _thaw_json(self.result)
        else:
            assert self.error is not None
            result["error"] = self.error.as_dict()
        return result


def request_from_mapping(value: Mapping[str, object]) -> MirrorRequest:
    if not isinstance(value, Mapping):
        raise ValueError("Mirror request 必须是 JSON object")
    unknown = sorted(set(value) - {"protocol", "request_id", "operation", "arguments"})
    missing = sorted({"protocol", "request_id", "operation", "arguments"} - set(value))
    if missing:
        raise ValueError(f"Mirror request 缺少字段: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Mirror request 包含未知字段: {', '.join(unknown)}")
    protocol = value["protocol"]
    request_id = value["request_id"]
    operation = value["operation"]
    arguments = value["arguments"]
    if not isinstance(protocol, str):
        raise ValueError("protocol 必须是字符串")
    if not isinstance(request_id, str):
        raise ValueError("request_id 必须是字符串")
    if not isinstance(operation, str):
        raise ValueError("operation 必须是字符串")
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments 必须是 JSON object")
    return MirrorRequest(
        protocol=protocol,
        request_id=request_id,
        operation=operation,
        arguments=arguments,
    )


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object 包含重复字段 {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"JSON 不能包含 {value}")


def decode_request(payload: str | bytes) -> MirrorRequest:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="strict")
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("request payload 必须是非空 UTF-8 JSON")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"无效 Mirror JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Mirror request 顶层必须是 JSON object")
    return request_from_mapping(value)


def encode_response(response: MirrorResponse) -> str:
    return json.dumps(
        response.as_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


__all__ = [
    "MIRROR_PROTOCOL",
    "MIRROR_PROTOCOL_V1",
    "MIRROR_PROTOCOL_V2",
    "MIRROR_PROTOCOL_V3",
    "MIRROR_V1_OPERATIONS",
    "MIRROR_V2_OPERATIONS",
    "MIRROR_V3_OPERATIONS",
    "MirrorError",
    "MirrorRequest",
    "MirrorResponse",
    "decode_request",
    "encode_response",
    "request_from_mapping",
]
