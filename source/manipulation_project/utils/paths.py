"""manipulation 包使用的仓库相对路径。

这些常量让代码在脚本目录、IDE 或命令行不同启动位置下都能找到 assets/configs/logs。
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPO_ROOT / "assets"
CONFIGS_ROOT = REPO_ROOT / "configs"
LOGS_ROOT = REPO_ROOT / "logs"


def repo_path(value: str | Path) -> Path:
    """把路径解析为绝对路径。

    参数:
        value: 绝对路径、用户路径或仓库相对路径。
    返回:
        ``Path``；绝对路径原样返回，相对路径拼到 ``REPO_ROOT`` 下。
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path
