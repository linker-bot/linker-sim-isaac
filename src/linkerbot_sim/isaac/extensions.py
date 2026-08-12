"""Kit 已启用扩展闭包的完整枚举工具。

物理 owner 的排他性不能依靠一张手工维护的扩展名白名单。Kit 的依赖求解可能通过新的
中间扩展带入 PhysX/Newton stage-update 插件，因此这里读取 extension manager 的完整
enabled closure，并把名称、版本和解析路径规范成与 Kit 版本无关的轻量记录。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnabledKitExtension:
    """一个实际启用的 Kit 扩展。"""

    name: str
    version: str
    path: str | None
    extension_id: str | None = None


def enumerate_enabled_kit_extensions(
    extension_manager: object | None = None,
) -> tuple[EnabledKitExtension, ...]:
    """枚举当前 Kit 的完整 enabled-extension closure。

    `get_extensions()` 是完成闭包证明所必需的接口。若运行时只提供按名称查询的 API，
    就无法证明未知的依赖 carrier 没有被启用，因此必须 fail closed，不能退化成固定名称
    逐个探测。
    """

    manager = extension_manager or _active_extension_manager()
    getter = getattr(manager, "get_extensions", None)
    if not callable(getter):
        raise RuntimeError(
            "Kit extension manager cannot enumerate the enabled extension closure"
        )
    try:
        raw_extensions = tuple(getter())
    except BaseException as exc:
        raise RuntimeError("failed to enumerate Kit extension closure") from exc

    result: dict[str, EnabledKitExtension] = {}
    for raw in raw_extensions:
        extension_id, details = _extension_entry(raw, manager=manager)
        name = _extension_name(details, extension_id=extension_id)
        if not name or not _extension_is_enabled(
            manager,
            details=details,
            name=name,
            extension_id=extension_id,
        ):
            continue
        package = details.get("package", {})
        if not isinstance(package, Mapping):
            package = {}
        version = _normalize_extension_version(
            package.get("version", details.get("version", "unknown"))
        )
        path = _extension_path(manager, details=details, extension_id=extension_id)
        result[name] = EnabledKitExtension(
            name=name,
            version=version,
            path=path,
            extension_id=extension_id,
        )
    return tuple(result[name] for name in sorted(result))


def _normalize_extension_version(value: object) -> str:
    """把不同 Kit 版本返回的扩展版本统一成可比较字符串。

    Isaac Sim 6 的 extension manager 在部分安装方式下会把版本报告为
    ``(major, minor, patch, prerelease, platform)``，而不是 manifest 中的
    ``"major.minor.patch"``。直接 ``str(tuple)`` 会制造虚假的版本不匹配，例如把实际
    Warp 1.13.0 报成 ``"(1, 13, 0, '', 'lx64')"``。元组/列表只取前三个数值分量；
    字符串和其它 Kit 自带版本对象仍保留其稳定文本表示。
    """

    if isinstance(value, (tuple, list)):
        numeric: list[str] = []
        for component in value[:3]:
            if isinstance(component, bool):
                break
            if isinstance(component, int):
                numeric.append(str(component))
                continue
            if isinstance(component, str) and component.isdigit():
                numeric.append(component)
                continue
            break
        if numeric:
            return ".".join(numeric)
    return str(value)


def _active_extension_manager() -> object:
    """延迟取得 extension manager，避免 pure import 启动 Kit。"""

    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()
    except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError("Kit extension manager is unavailable") from exc
    if manager is None:
        raise RuntimeError("Kit extension manager is unavailable")
    return manager


def _extension_entry(
    raw: object,
    *,
    manager: object,
) -> tuple[str | None, Mapping[str, object]]:
    """兼容 Kit 返回 mapping 或 ``(id, mapping)`` 两种 entry 形状。"""

    extension_id: str | None = None
    details: Mapping[str, object]
    if isinstance(raw, Mapping):
        details = raw
        candidate = raw.get("id", raw.get("extension_id"))
        extension_id = None if candidate is None else str(candidate)
    elif (
        isinstance(raw, (tuple, list)) and len(raw) == 2 and isinstance(raw[1], Mapping)
    ):
        extension_id = None if raw[0] is None else str(raw[0])
        details = raw[1]
    else:
        extension_id = str(raw)
        details = {}

    # 有些 Kit 版本的 get_extensions() 只返回 id；再向 manager 读取完整 manifest。
    if extension_id is not None:
        get_details = getattr(manager, "get_extension_dict", None)
        if callable(get_details):
            try:
                resolved = get_details(extension_id)
            except BaseException:
                resolved = None
            if isinstance(resolved, Mapping):
                merged = dict(resolved)
                merged.update(details)
                details = merged
    return extension_id, details


def _extension_name(
    details: Mapping[str, object],
    *,
    extension_id: str | None,
) -> str | None:
    package = details.get("package", {})
    if not isinstance(package, Mapping):
        package = {}
    candidate = details.get("name", package.get("name"))
    if candidate is not None and str(candidate).strip():
        return str(candidate).strip()
    if extension_id is None:
        return None
    # Kit extension id 通常为 ``name-version+build``。名称本身可能含连字符，因此只在
    # 最后一个 ``-`` 后确实以数字开头时剥离版本。
    prefix, separator, suffix = extension_id.rpartition("-")
    return prefix if separator and suffix[:1].isdigit() else extension_id


def _extension_is_enabled(
    manager: object,
    *,
    details: Mapping[str, object],
    name: str,
    extension_id: str | None,
) -> bool:
    enabled = details.get("enabled")
    if isinstance(enabled, bool):
        return enabled
    predicate = getattr(manager, "is_extension_enabled", None)
    if callable(predicate):
        for candidate in (extension_id, name):
            if candidate is None:
                continue
            try:
                if bool(predicate(candidate)):
                    return True
            except BaseException:
                continue
        return False
    enabled_id = getattr(manager, "get_enabled_extension_id", None)
    if callable(enabled_id):
        try:
            return bool(enabled_id(name))
        except BaseException:
            return False
    raise RuntimeError(
        "Kit extension metadata does not expose whether an extension is enabled"
    )


def _extension_path(
    manager: object,
    *,
    details: Mapping[str, object],
    extension_id: str | None,
) -> str | None:
    getter = getattr(manager, "get_extension_path", None)
    value: object | None = None
    if callable(getter) and extension_id is not None:
        try:
            value = getter(extension_id)
        except BaseException:
            value = None
    if not value:
        value = details.get("path")
    if not value:
        return None
    try:
        return str(Path(str(value)).resolve())
    except (OSError, RuntimeError):
        return str(value)


__all__ = ["EnabledKitExtension", "enumerate_enabled_kit_extensions"]
