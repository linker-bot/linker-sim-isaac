"""CSV 写入的小封装。

仿真循环里通常每个 physics step 都会写一行日志。这个类在 ``csv.DictWriter``
外面补了目录创建、可选禁用写入和周期 flush，避免各个 logger 重复处理文件细节。
"""

from __future__ import annotations

import csv
from pathlib import Path


class CsvWriter:
    """带周期刷盘能力的 ``DictWriter`` 包装。

    输入:
        path: CSV 文件路径；为 ``None`` 时表示禁用日志输出。
        fieldnames: 固定 CSV 列顺序，便于后续脚本按列名读取。
        flush_interval_rows: 每写入多少行 flush 一次，最小值会钳制为 1。
    输出:
        实例持有可选文件句柄和 ``csv.DictWriter``；写入行为通过 ``write`` 触发。
    """

    def __init__(self, path: str | Path | None, fieldnames: list[str], *, flush_interval_rows: int = 1) -> None:
        """初始化 CSV writer 并写入表头。

        参数:
            path: 输出路径或 ``None``。
            fieldnames: CSV 列名列表。
            flush_interval_rows: 自动 flush 的行数间隔。
        返回:
            无返回值；副作用是创建父目录、打开文件并写表头。
        """

        self.path = None if path is None else Path(path)
        self.file = None
        self.writer = None
        self.flush_interval_rows = max(1, int(flush_interval_rows))
        self.rows_written = 0
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        """写入一行字典数据，并按设置的行数间隔 flush。

        参数:
            row: ``列名 -> 值`` 的映射，列名应与 ``fieldnames`` 对齐。
        返回:
            无返回值；禁用日志时直接返回。
        """

        if self.writer is None:
            return
        self.writer.writerow(row)
        self.rows_written += 1
        if self.file is not None and self.rows_written % self.flush_interval_rows == 0:
            self.file.flush()

    def close(self) -> None:
        """关闭底层文件句柄。

        参数:
            无。
        返回:
            无返回值；可重复调用。
        """

        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

    def __enter__(self) -> "CsvWriter":
        """进入上下文管理器。

        返回:
            ``self``，便于 ``with CsvWriter(...) as writer`` 使用。
        """

        return self

    def __exit__(self, *_exc_info) -> None:
        """退出上下文时关闭文件。

        参数:
            *_exc_info: Python 上下文管理器传入的异常信息，当前不特殊处理。
        返回:
            无返回值。
        """

        self.close()
