"""文件型 runtime 输出共享的路径预检与变更规则。

输出启动分为只读 ``plan`` 和有副作用的 ``apply``。plan 记录预检时目标是否存在、目标类型
和既有数据策略；批量 apply 前会再次验证所有 plan，并拒绝解析后相同或互相包含的目标，
防止两个 sink 在同一文件树内 truncate/resume。最终目标不能是符号链接，时间戳目录名也
只能是单个安全路径片段。

该模块缩小了预检与执行之间的竞态窗口，但不提供跨进程锁，也不把多个文件系统操作包装成
真正事务：批量应用中途发生 I/O 错误时，较早的 plan 可能已经生效。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Literal, TypeAlias, cast


ExistingDataPolicy: TypeAlias = Literal[
    "error", "truncate", "resume", "timestamped_dir"
]
EXISTING_DATA_POLICIES = frozenset({"error", "truncate", "resume", "timestamped_dir"})
OutputPathKind: TypeAlias = Literal["file", "directory"]


@dataclass(frozen=True)
class OutputPathPlan:
    """只读路径预检结果，可在完整输出集合通过校验后应用。

    ``requested_path`` 保留用户输入经 ``expanduser`` 后的形式，``resolved_path`` 是实际
    创建/删除目标；``timestamped_dir`` 下两者会不同。``existed_at_preflight`` 用于 apply
    阶段发现目标存在性在两阶段之间发生变化。
    """

    requested_path: Path
    resolved_path: Path
    kind: OutputPathKind
    policy: ExistingDataPolicy
    existed_at_preflight: bool


def validate_existing_data_policy(
    value: object,
    *,
    label: str = "existing_data_policy",
) -> ExistingDataPolicy:
    """校验既有数据策略并收窄为 ``ExistingDataPolicy``。

    策略语义：``error`` 要求目标不存在；``truncate`` 删除后重建；``resume`` 复用已有
    同类型目标；``timestamped_dir`` 在一次 run 的时间戳子目录中创建新目标。
    """

    if not isinstance(value, str) or value not in EXISTING_DATA_POLICIES:
        choices = "|".join(sorted(EXISTING_DATA_POLICIES))
        raise ValueError(f"{label} must be one of {choices}")
    return cast(ExistingDataPolicy, value)


def timestamped_run_name(now: datetime | None = None) -> str:
    """返回可排序的 UTC run 目录名。

    naive ``datetime`` 按 UTC 解释；aware 值先转换为 UTC。调用方应在一次多输出启动中只
    生成一次并共享该名称，保证各 sink 落入同一逻辑 run。
    """

    instant = datetime.now(timezone.utc) if now is None else now
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def plan_output_file(
    path: str | Path,
    *,
    policy: object,
    run_name: str | None = None,
) -> OutputPathPlan:
    """只读预检文件目标，不创建、截断或打开文件。"""

    return _plan_output_path(
        path,
        kind="file",
        policy=validate_existing_data_policy(policy),
        run_name=run_name,
    )


def plan_output_directory(
    path: str | Path,
    *,
    policy: object,
    run_name: str | None = None,
) -> OutputPathPlan:
    """只读预检目录目标，不创建或删除目录。"""

    return _plan_output_path(
        path,
        kind="directory",
        policy=validate_existing_data_policy(policy),
        run_name=run_name,
    )


def apply_output_path_plans(
    plans: tuple[OutputPathPlan, ...] | list[OutputPathPlan],
) -> None:
    """重新校验全部目标，再按输入顺序执行文件系统变更。

    校验阶段无副作用，但多个 plan 的应用不是文件系统事务。I/O 失败可能使较早 plan 已
    生效而较晚 plan 尚未执行；调用方应在 apply 之后才打开 sink，并对打开失败做资源清理。
    """

    snapshot = validate_output_path_plans(plans)
    for plan in snapshot:
        _apply_plan(plan)


def validate_output_path_plans(
    plans: tuple[OutputPathPlan, ...] | list[OutputPathPlan],
) -> tuple[OutputPathPlan, ...]:
    """不修改文件系统地重新校验完整输出集合。

    返回 tuple 快照，避免调用方在校验结束与后续 apply 循环之间修改原 list。
    """

    snapshot = tuple(plans)
    _reject_duplicate_targets(snapshot)
    for plan in snapshot:
        _revalidate_plan(plan)
    return snapshot


def _plan_output_path(
    path: str | Path,
    *,
    kind: OutputPathKind,
    policy: ExistingDataPolicy,
    run_name: str | None,
) -> OutputPathPlan:
    """根据目标类型与既有数据策略构造单个路径 plan。"""

    requested = Path(path).expanduser()
    if not requested.name:
        raise ValueError("output path must name a file or directory")
    resolved = requested
    if policy == "timestamped_dir":
        name = timestamped_run_name() if run_name is None else _safe_run_name(run_name)
        resolved = (
            requested / name
            if kind == "directory"
            else requested.parent / name / requested.name
        )
    _validate_existing_kind(resolved, kind=kind)
    existed = _lexists(resolved)
    if policy in {"error", "timestamped_dir"} and existed:
        raise FileExistsError(f"output {kind} already exists: {resolved}")
    _validate_parent(resolved)
    return OutputPathPlan(
        requested_path=requested,
        resolved_path=resolved,
        kind=kind,
        policy=policy,
        existed_at_preflight=existed,
    )


def _validate_existing_kind(path: Path, *, kind: OutputPathKind) -> None:
    """校验已存在最终目标的类型，并拒绝最终目标本身是符号链接。"""

    if not _lexists(path):
        return
    if path.is_symlink():
        raise ValueError(f"output {kind} must not be a symbolic link: {path}")
    if kind == "file" and not path.is_file():
        raise ValueError(f"output path must be a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise ValueError(f"output path must be a directory: {path}")


def _validate_parent(path: Path) -> None:
    """确认最近的已存在祖先可作为目录，避免在普通文件下创建输出。"""

    parent = path.parent
    while not _lexists(parent):
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent
    if _lexists(parent) and not parent.is_dir():
        raise ValueError(f"output parent must be a directory: {parent}")


def _safe_run_name(value: str) -> str:
    """校验 run name 是非空单路径片段，禁止目录穿越。"""

    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("timestamped output run_name must be a safe directory name")
    return value


def _reject_duplicate_targets(plans: tuple[OutputPathPlan, ...]) -> None:
    """拒绝 canonical 化后相同、祖先或后代关系的批量输出目标。

    ``resolve(strict=False)`` 用于比较路径别名和已有符号链接指向；若允许嵌套目标，一个
    directory truncate 可能删除另一个 sink 的 file target，因此也必须拒绝。
    """

    owners: dict[Path, OutputPathKind] = {}
    for plan in plans:
        try:
            target = plan.resolved_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"cannot canonicalize output path: {plan.resolved_path}"
            ) from exc
        previous = owners.get(target)
        if previous is not None:
            raise ValueError(
                "multiple outputs resolve to the same path: "
                f"{plan.resolved_path} ({previous}, {plan.kind})"
            )
        for existing in owners:
            if target.is_relative_to(existing) or existing.is_relative_to(target):
                raise ValueError(
                    "overlapping output paths are not allowed in one startup: "
                    f"{existing} and {target}"
                )
        owners[target] = plan.kind


def _revalidate_plan(plan: OutputPathPlan) -> None:
    """在 apply 前检测目标类型或存在性相对预检时刻的变化。"""

    _validate_existing_kind(plan.resolved_path, kind=plan.kind)
    exists_now = _lexists(plan.resolved_path)
    if plan.policy in {"error", "timestamped_dir"} and exists_now:
        raise FileExistsError(
            f"output path changed after preflight: {plan.resolved_path}"
        )
    if plan.policy in {"truncate", "resume"} and (
        exists_now != plan.existed_at_preflight
    ):
        raise RuntimeError(f"output path changed after preflight: {plan.resolved_path}")
    _validate_parent(plan.resolved_path)


def _apply_plan(plan: OutputPathPlan) -> None:
    """应用单个已复检 plan 的 truncate/create/resume 文件系统变更。"""

    target = plan.resolved_path
    if plan.policy == "truncate" and _lexists(target):
        if plan.kind == "directory":
            shutil.rmtree(target)
        else:
            target.unlink()
    if plan.kind == "directory":
        target.mkdir(parents=True, exist_ok=plan.policy == "resume")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)


def _lexists(path: Path) -> bool:
    """判断目录项是否存在，并把悬空符号链接也视为已占用目标。"""

    return path.exists() or path.is_symlink()


__all__ = [
    "EXISTING_DATA_POLICIES",
    "ExistingDataPolicy",
    "OutputPathPlan",
    "apply_output_path_plans",
    "plan_output_directory",
    "plan_output_file",
    "timestamped_run_name",
    "validate_existing_data_policy",
    "validate_output_path_plans",
]
