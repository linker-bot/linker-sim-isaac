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

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
    plan_output_file,
    validate_existing_data_policy,
)


@dataclass(frozen=True)
class CsvOutputPlan:
    """打开文件前可批量收集的 CSV 路径与表头预检结果。

    ``path_plan`` 保存请求路径、解析路径和已存在数据策略；``fieldnames`` 固化预检时的列
    顺序；``append`` 表示 resume 模式下目标在预检时已存在。对象冻结，应用路径计划与创建
    writer 之间不得更换字段或策略。
    """

    path_plan: OutputPathPlan
    fieldnames: tuple[str, ...]
    append: bool


def plan_csv_output(
    path: str | Path,
    fieldnames: list[str],
    *,
    existing_data_policy: str,
    timestamped_run_name: str | None = None,
    path_plan: OutputPathPlan | None = None,
) -> CsvOutputPlan:
    """在不修改文件系统的前提下预检一个 CSV 目标。

    参数:
        path: 调用方请求的 CSV 文件路径。
        fieldnames: 固定列名及顺序。
        existing_data_policy: 目标已存在时的 error/resume 等当前策略。
        timestamped_run_name: 时间戳目录策略使用的可选运行名。
        path_plan: 可选的上游统一路径计划；提供时必须与本请求完全一致。
    返回:
        冻结的 :class:`CsvOutputPlan`；resume 目标会先验证表头和所有记录完整性。
    异常:
        ValueError: 策略、外部计划、现有 CSV 表头或记录结构不合法。
        OSError: resume 目标无法读取或检查。
    副作用:
        仅查询路径并可能读取现有 CSV，不创建、删除、截断或追加文件。
    """

    policy = validate_existing_data_policy(
        existing_data_policy,
        label="csv existing_data_policy",
    )
    requested = Path(path).expanduser()
    plan = (
        plan_output_file(
            requested,
            policy=policy,
            run_name=timestamped_run_name,
        )
        if path_plan is None
        else path_plan
    )
    if plan.requested_path != requested or plan.policy != policy or plan.kind != "file":
        raise ValueError("CSV path plan does not match the requested path and policy")
    append = policy == "resume" and plan.existed_at_preflight
    if append:
        _validate_resumable_csv(plan.resolved_path, fieldnames)
    return CsvOutputPlan(
        path_plan=plan,
        fieldnames=tuple(fieldnames),
        append=append,
    )


def apply_csv_output_plans(plans: Sequence[CsvOutputPlan]) -> None:
    """在打开任一 CSV 前统一应用一组已预检路径计划。

    参数:
        plans: 已完成字段和 resume 内容校验的计划序列。
    返回:
        成功时返回 ``None``。
    异常:
        ValueError/OSError: 路径计划过期、互相冲突或无法应用。
    副作用:
        可能创建目录、移动/清理目标或准备时间戳路径，具体取决于路径策略；不打开 CSV。
    """

    apply_output_path_plans([plan.path_plan for plan in plans])


class CsvWriter:
    """带周期刷盘能力的 ``DictWriter`` 包装。

    输入:
        path: CSV 文件路径；为 ``None`` 时表示禁用日志输出。
        fieldnames: 固定 CSV 列顺序，便于后续脚本按列名读取。
        flush_interval_rows: 每写入多少行 flush 一次，最小值会钳制为 1。
    输出:
        实例持有可选文件句柄和 ``csv.DictWriter``；写入行为通过 ``write`` 触发。
    生命周期:
        构造时完成路径计划应用并打开文件；调用 ``close`` 或退出上下文后释放句柄。
        ``path=None`` 时整个生命周期保持禁用 no-op 状态。实例不是线程安全 writer。
    """

    def __init__(
        self,
        path: str | Path | None,
        fieldnames: list[str],
        *,
        flush_interval_rows: int = 1,
        existing_data_policy: str = "error",
        timestamped_run_name: str | None = None,
        output_plan: CsvOutputPlan | None = None,
        paths_applied: bool = False,
    ) -> None:
        """初始化 CSV writer 并写入表头。

        参数:
            path: 输出路径或 ``None``。
            fieldnames: CSV 列名列表。
            flush_interval_rows: 自动 flush 的行数间隔。
            existing_data_policy: 目标已存在时采用的当前输出策略。
            timestamped_run_name: 时间戳输出策略使用的可选运行名。
            output_plan: 可选预检计划；必须与路径、字段和策略一致。
            paths_applied: 上层是否已批量应用计划，避免重复执行路径变更。
        异常:
            ValueError: 策略、计划、字段或 resume 内容不一致。
            OSError: 目录/文件无法准备、创建或打开。
        副作用:
            path 非空时可能应用路径策略、打开文件，并在非 resume 模式写入表头。
        """

        # ``None`` 是显式的“不开日志”模式。保留 writer 接口可让上层代码无需在每次
        # 写日志前判断路径是否存在，从而减少仿真循环中的分支噪声。
        policy = validate_existing_data_policy(
            existing_data_policy,
            label="csv existing_data_policy",
        )
        self.path = None if path is None else Path(path)
        self.file = None
        self.writer = None
        self.flush_interval_rows = max(1, int(flush_interval_rows))
        self.rows_written = 0
        if self.path is None:
            return
        prepared = output_plan or plan_csv_output(
            self.path,
            fieldnames,
            existing_data_policy=policy,
            timestamped_run_name=timestamped_run_name,
        )
        if prepared.fieldnames != tuple(fieldnames):
            raise ValueError(
                "CSV output plan fieldnames do not match writer fieldnames"
            )
        if prepared.path_plan.requested_path != self.path.expanduser():
            raise ValueError("CSV output plan path does not match writer path")
        if prepared.path_plan.policy != policy:
            raise ValueError("CSV output plan policy does not match writer policy")
        if not paths_applied:
            apply_csv_output_plans((prepared,))
        self.path = prepared.path_plan.resolved_path
        self.file = self.path.open(
            "a" if prepared.append else "x",
            newline="",
            encoding="utf-8",
        )
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not prepared.append:
            self.writer.writeheader()

    def write(self, row: dict) -> None:
        """写入一行字典数据，并按设置的行数间隔 flush。

        参数:
            row: ``列名 -> 值`` 的映射，列名应与 ``fieldnames`` 对齐。
        返回:
            无返回值；禁用日志时直接返回。
        异常:
            ValueError: 行包含 ``DictWriter`` 不接受的额外字段。
            OSError: 写入或周期 flush 失败。
        副作用:
            追加一条 CSV 记录，更新 ``rows_written``，并可能刷新文件缓冲区。
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

        该方法是幂等的；禁用日志或已经关闭时直接返回。关闭后 ``write`` 会退化为 no-op，
        因此上下文管理器和显式 ``finally`` 中重复调用都是安全的。

        副作用:
            关闭文件句柄并清空 ``file``/``writer`` 引用；底层 close 错误会原样传播。
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


def _validate_resumable_csv(path: Path, fieldnames: list[str]) -> None:
    """在追加前拒绝表头不一致、畸形记录和字段数不完整的现有 CSV。"""

    _require_terminated_csv_record(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"CSV resume target is empty and has no header: {path}"
            ) from exc
        except csv.Error as exc:
            raise ValueError(f"CSV resume target is malformed: {path}") from exc
        if header != fieldnames:
            raise ValueError(
                f"CSV resume header does not match configured fields: {path}"
            )
        try:
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(fieldnames):
                    raise ValueError(
                        f"CSV resume row {row_number} has {len(row)} fields; "
                        f"expected {len(fieldnames)}: {path}"
                    )
        except csv.Error as exc:
            raise ValueError(
                f"CSV resume target is malformed near line {reader.line_num}: {path}"
            ) from exc


def _require_terminated_csv_record(path: Path) -> None:
    """拒绝未以换行结束的末条记录，防止追加内容粘连成同一行。"""

    size = path.stat().st_size
    if size == 0:
        return
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        if stream.read(1) not in {b"\n", b"\r"}:
            raise ValueError(
                f"CSV resume target has an unterminated final record: {path}"
            )
