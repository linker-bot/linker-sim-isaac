"""tiled scene 构建中无 Isaac 状态所有权的轻量共享辅助函数。"""


def _print_status(status_prefix: str | None, message: str) -> None:
    """按可选前缀打印可 grep 的构建状态。"""

    if status_prefix is None:
        return
    print(f"{status_prefix}_{message}", flush=True)
