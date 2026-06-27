"""manipulation 包使用的仓库相对路径。

这些常量让代码在脚本目录、IDE 或命令行不同启动位置下都能找到 ``assets``、``configs`` 和
``logs``。``REPO_ROOT`` 通过当前文件位置推导，不依赖进程工作目录；这对 VS Code、pytest
和脚本入口从不同 cwd 启动尤其重要。

职责边界:
    * 提供仓库根目录及常用子目录常量。
    * 把用户路径、绝对路径和仓库相对路径统一解析为 ``Path``。
    * 不检查路径是否存在；是否要求文件/目录存在由具体调用方决定。
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

    # expanduser 支持 ``~/...``；相对路径一律拼到 REPO_ROOT，避免脚本从 scripts/ 或测试目录
    # 启动时出现不同解析结果。
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path
