"""CSV 写入的小封装。

仿真循环里通常每个 physics step 都会写一行日志。这个类在 ``csv.DictWriter`` 外面补了
目录创建、可选禁用写入和周期 flush，避免各个 logger 重复处理文件细节。

职责边界:
    * 只负责“按固定列名写字典行”这一件事，不解释行内字段的物理含义。
    * ``path is None`` 表示禁用实际文件 I/O，但调用方仍可沿用同一 logger 流程。
    * 不吞掉 I/O 异常：磁盘路径、权限或编码问题应尽早暴露，防止用户误以为实验日志已保存。

性能约定:
    频繁 flush 会显著拖慢 Isaac 仿真；因此调用方可以把 ``flush_interval_rows`` 设置为
    “约几十毫秒仿真时间对应的行数”。关闭文件时仍会由 Python 文件对象写出剩余缓冲区。
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

        # ``None`` 是显式的“不开日志”模式。保留 writer 接口可让上层代码无需在每次
        # 写日志前判断路径是否存在，从而减少仿真循环中的分支噪声。
        self.path = None if path is None else Path(path)
        self.file = None
        self.writer = None
        self.flush_interval_rows = max(1, int(flush_interval_rows))
        self.rows_written = 0
        if self.path is None:
            return
        # 日志目录通常带有任务名/时间戳，多层目录不存在是正常情况；在这里统一创建，
        # 调用方只需要决定输出文件名。
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

        # 禁用日志时保持 no-op，而不是让调用方传入空 writer 或特殊对象。
        if self.writer is None:
            return
        self.writer.writerow(row)
        self.rows_written += 1
        # 周期 flush 在“数据安全”和“仿真速度”之间折中：长时间 GUI 调试时即使中断进程，
        # 最近一小段数据也更可能已经落盘。
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
