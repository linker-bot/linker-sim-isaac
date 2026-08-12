"""项目固定携带的 cuRobo 后端资源路径。

这些文件是与锁定 cuRobo 版本共同发布的实现资源，不是用户可选择的 profile。
统一在此解析路径，可以避免后端把 ``configs/`` 误当成第三方库的资源目录。
"""

from __future__ import annotations

from pathlib import Path


_TASK_ROOT = Path(__file__).resolve().parent / "task"


def curobo_task_resource_path(relative_path: str) -> str:
    """返回一个经过目录逃逸和存在性校验的 task 资源绝对路径。"""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("cuRobo task resource path must be a non-empty string")
    root = _TASK_ROOT.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"cuRobo task resource escapes backend root: {relative_path!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"cuRobo task resource does not exist: {path}")
    return str(path)


__all__ = ["curobo_task_resource_path"]
