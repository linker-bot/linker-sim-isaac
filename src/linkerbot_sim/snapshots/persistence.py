"""SceneSnapshot 的显式 JSON 冷存储边界。"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .schema import SceneSnapshot


def validate_scene_snapshot(value: object) -> SceneSnapshot:
    """严格校验 typed snapshot 或 JSON-compatible mapping。

    返回值始终是拥有自身 NumPy 数组的 ``SceneSnapshot``；未知字段、错误 schema、非有限
    数值和 shape 不匹配都由 schema 构造器在任何文件/仿真写入前拒绝。
    """

    if isinstance(value, SceneSnapshot):
        # 重新经过 mapping 构造，避免调用方把同一 mutable NumPy backing 当作存储副本。
        return SceneSnapshot.from_mapping(value.as_dict())
    if not isinstance(value, Mapping):
        raise ValueError("scene snapshot 必须是 SceneSnapshot 或 JSON object")
    return SceneSnapshot.from_mapping(value)


def load_scene_snapshot(path: str | Path) -> SceneSnapshot:
    """从 UTF-8 JSON 文件加载并严格校验一个 SceneSnapshot。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无效 SceneSnapshot JSON: {source}: {exc}") from exc
    return validate_scene_snapshot(payload)


def save_scene_snapshot(
    snapshot: SceneSnapshot | Mapping[str, object],
    path: str | Path,
    *,
    replace: bool = False,
) -> Path:
    """原子写入 SceneSnapshot，默认拒绝覆盖已有文件。

    临时文件与目标位于同一目录，完成 ``fsync`` 后才 ``os.replace``，避免进程中断留下
    半份 JSON。该函数是显式冷边界，不应在 Kaleidoscope rollout 热路径调用。
    """

    parsed = validate_scene_snapshot(snapshot)
    destination = Path(path)
    if destination.exists() and not replace:
        raise FileExistsError(f"SceneSnapshot 已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            parsed.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not replace:
            raise FileExistsError(f"SceneSnapshot 已存在: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "load_scene_snapshot",
    "save_scene_snapshot",
    "validate_scene_snapshot",
]
