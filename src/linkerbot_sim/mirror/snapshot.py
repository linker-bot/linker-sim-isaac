"""Mirror versioned snapshot 的 capture/restore 边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass


def _json_snapshot(value: object) -> dict[str, object]:
    """把 canonical snapshot object 转成 owned JSON mapping。"""

    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    serializer = getattr(value, "as_dict", None)
    if not callable(serializer):
        serializer = getattr(value, "to_mapping", None)
    if not callable(serializer):
        raise RuntimeError(
            "snapshot adapter 必须返回 mapping 或可序列化 snapshot object"
        )
    result = serializer()
    if not isinstance(result, Mapping):
        raise RuntimeError("snapshot serializer 必须返回 mapping")
    return deepcopy(dict(result))


def _result_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    serializer = getattr(value, "as_dict", None)
    return serializer() if callable(serializer) else value


@dataclass
class MirrorSnapshotService:
    """拥有 Snapshot use case，但不拥有 session 或 engine handle。"""

    capture: Callable[[], object]
    restore: Callable[..., object]

    def capture_snapshot(self) -> dict[str, object]:
        return _json_snapshot(self.capture())

    def restore_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        label_map: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> object:
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot 必须是 mapping")
        copied_map = None if label_map is None else dict(label_map)
        result = self.restore(
            deepcopy(dict(snapshot)),
            label_map=copied_map,
            strict=bool(strict),
        )
        return _result_mapping(result)


__all__ = ["MirrorSnapshotService"]
