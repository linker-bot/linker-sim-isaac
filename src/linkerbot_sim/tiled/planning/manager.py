"""tiled planning 请求队列、批处理、future 取消和线程池生命周期。

主线程只提交由 numpy 数组组成的冻结请求，并在 polling 点收集结果；planner backend 在
线程池中运行，不得访问 Isaac stage 或 articulation。manager 保持请求 FIFO，只合并连续
且同构的请求；超大请求按 env 行切分并原子合并，任何 chunk 失败都会让整个外部请求失败。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
import math
from typing import cast

import numpy as np

from linkerbot_sim.tiled.planning.batching import (
    planning_batch_key,
    request_problem_count,
)
from linkerbot_sim.tiled.planning.linear_backend import LinearJointPlannerBackend
from linkerbot_sim.tiled.planning.types import (
    TiledPlannerBackend,
    TiledPlanningRequest,
    TiledPlanningResult,
)


class TiledPlannerManager:
    """把冻结的 numpy 请求分组后交给后台 planner workers。

    manager 的可变容器由调用它的 runtime 主线程拥有，worker 只接收不可再修改的请求并
    返回结果。取消正在执行的批次采用协作语义：只有批次内全部请求都取消时才尝试取消
    future；否则保留计算并在收集阶段把相应 request 改写为 ``CANCELLED``。
    """

    def __init__(
        self,
        *,
        backend: TiledPlannerBackend | None = None,
        max_workers: int = 2,
        max_pending_requests: int | None = 64,
        max_completed_results: int | None = 256,
        max_batch_problems: int = 64,
        oversize_request_policy: str = "split",
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        """校验资源上限并创建固定大小的 planner 线程池。

        ``max_pending_requests`` 限制 queued 与 in-flight 请求总数；
        ``max_completed_results`` 限制可查询结果缓存，设为零表示只从 ``collect_ready``
        返回一次而不缓存；``None`` 表示对应上限不启用。
        """

        if int(max_workers) < 1:
            raise ValueError("max_workers must be positive")
        if max_pending_requests is not None and int(max_pending_requests) < 1:
            raise ValueError("max_pending_requests must be positive or None")
        if max_completed_results is not None and int(max_completed_results) < 0:
            raise ValueError("max_completed_results must be non-negative or None")
        if int(max_batch_problems) < 1:
            raise ValueError("max_batch_problems must be positive")
        oversize_policy = str(oversize_request_policy).strip()
        if oversize_policy not in {"split", "reject"}:
            raise ValueError("oversize_request_policy must be one of: reject, split")
        if isinstance(shutdown_timeout_s, bool) or (
            not math.isfinite(float(shutdown_timeout_s))
            or float(shutdown_timeout_s) < 0.0
        ):
            raise ValueError("shutdown_timeout_s must be a non-negative finite number")
        self.backend = backend or LinearJointPlannerBackend()
        self.max_pending_requests = (
            None if max_pending_requests is None else int(max_pending_requests)
        )
        self.max_completed_results = (
            None if max_completed_results is None else int(max_completed_results)
        )
        self.max_batch_problems = int(max_batch_problems)
        self.oversize_request_policy = oversize_policy
        self.max_workers = int(max_workers)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="tiled-planner"
        )
        self._futures: dict[str, Future[tuple[TiledPlanningResult, ...]]] = {}
        self._future_request_ids: dict[str, tuple[str, ...]] = {}
        self._request_batches: dict[str, str] = {}
        self._requests: dict[str, TiledPlanningRequest] = {}
        self._queued: dict[str, TiledPlanningRequest] = {}
        self._immediate_results: dict[str, TiledPlanningResult] = {}
        self._completed: dict[str, TiledPlanningResult] = {}
        self._cancelled: set[str] = set()
        self._next_batch_index = 0
        self._rejected_requests = 0
        self._split_requests = 0
        self._evicted_completed_results = 0
        self._shutdown_requested = False
        self._shutdown_timed_out = False
        self._executor_shutdown = False

    def submit(self, request: TiledPlanningRequest) -> str:
        """把请求加入待派发 FIFO，并返回其 request ID。

        该方法不启动 backend；派发发生在下一次 ``collect_ready``。重复 ID、关闭后的提交、
        pending 超限或 ``reject`` 策略下的超大请求会在修改队列前抛出异常。
        """

        request_id = str(request.request_id)
        if self._shutdown_requested:
            self._rejected_requests += 1
            raise RuntimeError("planner manager is shutting down")
        if request_id in self._requests or request_id in self._completed:
            self._rejected_requests += 1
            raise ValueError(f"duplicate planning request_id: {request_id}")
        problem_count = request_problem_count(request)
        if (
            problem_count > self.max_batch_problems
            and self.oversize_request_policy == "reject"
        ):
            self._rejected_requests += 1
            raise ValueError(
                f"planning request {request_id!r} has {problem_count} problems, "
                f"exceeding max_batch_problems={self.max_batch_problems}"
            )
        if (
            self.max_pending_requests is not None
            and len(self._requests) >= self.max_pending_requests
        ):
            self._rejected_requests += 1
            raise RuntimeError(
                "too many pending planning requests: "
                f"{len(self._requests)} >= {self.max_pending_requests}"
            )
        self._requests[request_id] = request
        self._queued[request_id] = request
        if problem_count > self.max_batch_problems:
            self._split_requests += 1
        return request_id

    def collect_ready(
        self, *, timeout_s: float = 0.0
    ) -> tuple[TiledPlanningResult, ...]:
        """派发 queued 请求，并收集已结束 future 的逐 request 结果。

        queued 阶段取消产生的即时结果先入 completed cache，随后才派发剩余请求。
        ``timeout_s`` 只等待至少一个已有 future 完成，不等待全部任务，也不会阻塞新提交。
        backend 异常和返回数量错误在这里转换为每个 request 的失败结果。
        """

        ready: list[TiledPlanningResult] = []
        for request_id, result in list(self._immediate_results.items()):
            self._immediate_results.pop(request_id, None)
            self._store_completed(request_id, result)
            ready.append(result)
        self._dispatch_pending()
        if timeout_s > 0.0 and self._futures:
            wait(
                tuple(self._futures.values()),
                timeout=float(timeout_s),
                return_when=FIRST_COMPLETED,
            )
        for batch_id, future in list(self._futures.items()):
            if not future.done():
                continue
            request_ids = self._future_request_ids.pop(batch_id)
            self._futures.pop(batch_id)
            batch_results = self._future_results(future=future, request_ids=request_ids)
            for request_id, result in zip(request_ids, batch_results, strict=True):
                request = self._requests.pop(request_id)
                self._request_batches.pop(request_id, None)
                if request_id in self._cancelled:
                    result = TiledPlanningResult.failed(
                        request,
                        status="CANCELLED",
                        message="planning request was cancelled",
                    )
                    self._cancelled.discard(request_id)
                self._store_completed(request_id, result)
                ready.append(result)
        return tuple(ready)

    def cancel(self, request_id: str) -> dict[str, object]:
        """取消 queued 或 in-flight 的单个请求并返回接受状态。

        queued 请求可确定地移出队列并生成取消结果。in-flight 请求仅记录取消意图；共享
        future 仍可能继续运行，最终结果会在 ``collect_ready`` 时被替换为取消结果。
        completed 或未知 ID 不改变任何状态。
        """

        key = str(request_id)
        queued = self._queued.pop(key, None)
        if queued is not None:
            self._requests.pop(key, None)
            self._immediate_results[key] = TiledPlanningResult.failed(
                queued,
                status="CANCELLED",
                message="planning request was cancelled before dispatch",
            )
            return {
                "request_id": key,
                "accepted": True,
                "status": "cancelled",
                "future_cancelled": False,
            }
        batch_id = self._request_batches.get(key)
        future = None if batch_id is None else self._futures.get(batch_id)
        if batch_id is None or future is None:
            return {
                "request_id": key,
                "accepted": False,
                "status": "not_found" if key not in self._completed else "completed",
            }
        self._cancelled.add(key)
        batch_request_ids = self._future_request_ids.get(batch_id, ())
        future_cancelled = (
            future.cancel()
            if batch_request_ids
            and all(item in self._cancelled for item in batch_request_ids)
            else False
        )
        return {
            "request_id": key,
            "accepted": True,
            "status": "cancel_requested",
            "future_cancelled": bool(future_cancelled),
        }

    def cancel_matching(
        self,
        *,
        robot_name: str | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        """按 robot/env 取消请求，防止 reset/set_state 后加载过期轨迹。"""

        env_set = (
            None
            if env_ids is None
            else {int(env_id) for env_id in np.asarray(env_ids, dtype=int).reshape(-1)}
        )
        cancelled = []
        for request_id, request in list(self._requests.items()):
            if robot_name is not None and request.robot_name != str(robot_name):
                continue
            if env_set is not None and not (set(request.env_ids) & env_set):
                continue
            cancelled.append(self.cancel(request_id))
        return cancelled

    def status(self) -> dict[str, object]:
        """只读取队列、容量和关闭状态，不派发或消费规划结果。

        Ready result 必须由 runtime 的显式 collection 边界统一收集，才能同时执行
        ``load_on_success``。如果状态查询在这里调用 ``collect_ready()``，普通 ``status``
        就可能先消费结果，导致后续 playback 自动载入永久丢失。
        """

        pending = []
        for request_id, request in sorted(self._queued.items()):
            pending.append(
                {
                    "request_id": request_id,
                    "robot": request.robot_name,
                    "env_ids": list(request.env_ids),
                    "queued": True,
                    "running": False,
                    "cancel_requested": request_id in self._cancelled,
                }
            )
        for request_id, batch_id in sorted(self._request_batches.items()):
            request = self._requests[request_id]
            future = self._futures[batch_id]
            pending.append(
                {
                    "request_id": request_id,
                    "robot": request.robot_name,
                    "env_ids": list(request.env_ids),
                    "queued": False,
                    "running": bool(future.running()),
                    "cancel_requested": request_id in self._cancelled,
                }
            )
        return {
            "pending": pending,
            "pending_count": len(pending),
            "completed_count": len(self._completed),
            "max_pending_requests": self.max_pending_requests,
            "max_completed_results": self.max_completed_results,
            "max_batch_problems": self.max_batch_problems,
            "oversize_request_policy": self.oversize_request_policy,
            "max_workers": self.max_workers,
            "running_batch_count": sum(
                1 for future in self._futures.values() if future.running()
            ),
            "rejected_requests": self._rejected_requests,
            "split_requests": self._split_requests,
            "evicted_completed_results": self._evicted_completed_results,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_timed_out": self._shutdown_timed_out,
            "queued_request_ids": sorted(self._queued),
            "running_request_ids": sorted(
                request_id
                for request_id, batch_id in self._request_batches.items()
                if self._futures[batch_id].running()
            ),
            "live_request_ids": sorted(self._requests),
            "completed": [
                result.to_json() for _, result in sorted(self._completed.items())
            ],
        }

    def clear_completed(
        self, request_ids: str | Sequence[str] | None = None
    ) -> dict[str, object]:
        """清理 completed cache，并分别报告已清理和未找到的 IDs。

        参数为 ``None`` 时清空全部缓存；该操作不会影响 queued 或 in-flight 请求。
        """

        if request_ids is None:
            cleared = list(self._completed)
            self._completed.clear()
            return {"cleared": cleared, "missing": [], "count": len(cleared)}
        keys = (
            (request_ids,)
            if isinstance(request_ids, str)
            else tuple(str(item) for item in request_ids)
        )
        cleared: list[str] = []
        missing: list[str] = []
        for key in keys:
            if key in self._completed:
                self._completed.pop(key, None)
                cleared.append(key)
            else:
                missing.append(key)
        return {"cleared": cleared, "missing": missing, "count": len(cleared)}

    def shutdown(self, *, wait_timeout_s: float | None = None) -> dict[str, object]:
        """拒绝新提交、取消未启动工作，并有界等待正在执行的 future。

        Python 线程不能被强制终止。超时后 executor 只执行非阻塞 shutdown，仍存活请求继续
        保留在内部映射和返回状态中，资源所有者可据此判定进程是否满足干净退出条件。
        重复调用不会重复关闭 executor。
        """

        timeout = (
            self.shutdown_timeout_s if wait_timeout_s is None else float(wait_timeout_s)
        )
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("wait_timeout_s must be a non-negative finite number")
        self._shutdown_requested = True

        # queued requests 尚未提交给 executor，可以确定地转成取消结果。
        for request_id in tuple(self._queued):
            self.cancel(request_id)
        self.collect_ready()

        for future in tuple(self._futures.values()):
            future.cancel()
        if self._futures:
            wait(tuple(self._futures.values()), timeout=timeout)
        self.collect_ready()

        live_request_ids = sorted(self._requests)
        if live_request_ids:
            self._shutdown_timed_out = True
        if not self._executor_shutdown:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor_shutdown = True
        return {
            "shutdown_timed_out": bool(live_request_ids),
            "live_request_ids": live_request_ids,
        }

    def _dispatch_pending(self) -> None:
        """按 FIFO batch group 派发请求，并登记 future 与 request 的双向映射。

        映射必须在返回主线程前完整建立，收集和取消路径依赖它们把一个 batch future 还原
        成多个外部 request。
        """

        if self._shutdown_requested or not self._queued:
            return
        supports_plan_many = callable(getattr(self.backend, "plan_many", None))
        while self._queued:
            request_ids = self._next_dispatch_request_ids(
                supports_plan_many=supports_plan_many
            )
            if not request_ids:  # pragma: no cover
                return
            requests = tuple(self._queued.pop(request_id) for request_id in request_ids)
            batch_id = self._new_batch_id(request_ids)
            future = self._executor.submit(self._run_backend_many, requests)
            self._futures[batch_id] = future
            self._future_request_ids[batch_id] = request_ids
            for request_id in request_ids:
                self._request_batches[request_id] = batch_id

    def _next_dispatch_request_ids(
        self, *, supports_plan_many: bool
    ) -> tuple[str, ...]:
        """选择下一个连续同构请求组，绝不越过异构请求重排 FIFO。

        batch key 缺失或 backend 不支持 ``plan_many`` 时只选择队首。累计 problem 数不会
        超过配置上限；单个超大请求例外地独占 batch，随后由切分路径处理。
        """

        first_id, first_request = next(iter(self._queued.items()))
        if not supports_plan_many:
            return (first_id,)
        first_key = planning_batch_key(first_request)
        if first_key is None:
            return (first_id,)
        selected: list[str] = []
        total_problems = 0
        for request_id, request in list(self._queued.items()):
            if planning_batch_key(request) != first_key:
                break
            problem_count = request_problem_count(request)
            if selected and total_problems + problem_count > self.max_batch_problems:
                break
            selected.append(request_id)
            total_problems += problem_count
            if total_problems >= self.max_batch_problems:
                break
        return tuple(selected or (first_id,))

    def _new_batch_id(self, request_ids: tuple[str, ...]) -> str:
        """为 single request 或 merged group 生成仅供 manager 内部使用的 batch ID。"""

        if len(request_ids) == 1:
            return f"single:{request_ids[0]}"
        self._next_batch_index += 1
        return f"batch:{self._next_batch_index}"

    def _run_backend_many(
        self, requests: tuple[TiledPlanningRequest, ...]
    ) -> tuple[TiledPlanningResult, ...]:
        """在 worker 中调用批量 backend，并把单个超大请求转入有界切分路径。"""

        if any(
            request_problem_count(request) > self.max_batch_problems
            for request in requests
        ):
            if len(requests) != 1:
                raise ValueError("oversized request must be dispatched in isolation")
            return (self._run_split_request(requests[0]),)

        return self._invoke_backend_many(requests)

    def _invoke_backend_many(
        self, requests: tuple[TiledPlanningRequest, ...]
    ) -> tuple[TiledPlanningResult, ...]:
        """调用 backend，并再次防御任何绕过 problem 上限的内部路径。

        backend 可实现 ``plan_many``，否则逐 request 调用 ``plan``。返回数量必须与输入严格
        一致，避免 zip 截断导致某个请求永久留在 pending 状态。
        """

        problem_count = sum(request_problem_count(request) for request in requests)
        if problem_count > self.max_batch_problems:
            raise ValueError(
                "planner backend invocation would exceed max_batch_problems: "
                f"{problem_count} > {self.max_batch_problems}"
            )

        plan_many = getattr(self.backend, "plan_many", None)
        if callable(plan_many):
            batch_planner = cast(
                Callable[
                    [tuple[TiledPlanningRequest, ...]],
                    Sequence[TiledPlanningResult],
                ],
                plan_many,
            )
            results = tuple(batch_planner(requests))
        else:
            results = tuple(self.backend.plan(request) for request in requests)
        if len(results) != len(requests):
            raise ValueError(
                "planner backend returned "
                f"{len(results)} results for {len(requests)} requests"
            )
        return results

    def _run_split_request(self, request: TiledPlanningRequest) -> TiledPlanningResult:
        """按 env 行有界执行超大请求，并恢复其唯一外部身份。

        chunks 按原顺序串行执行。任一 chunk 失败后不发布部分轨迹，整个请求以原 request
        ID 失败；全部成功才合并时间轴和逐 env 数组。
        """

        chunks = _split_planning_request(
            request, max_batch_problems=self.max_batch_problems
        )
        chunk_results: list[TiledPlanningResult] = []
        for chunk in chunks:
            result = self._invoke_backend_many((chunk,))[0]
            chunk_results.append(result)
            if not result.success:
                return TiledPlanningResult.failed(
                    request,
                    status=result.status,
                    message=(
                        "oversized planning request failed atomically in "
                        f"chunk {len(chunk_results)}/{len(chunks)}: {result.message}"
                    ),
                )
        return _merge_split_results(request, tuple(chunk_results))

    def _future_results(
        self,
        *,
        future: Future[tuple[TiledPlanningResult, ...]],
        request_ids: tuple[str, ...],
    ) -> tuple[TiledPlanningResult, ...]:
        """把 cancelled/exception/shape mismatch future 统一成逐 request failure result。"""

        requests = tuple(self._requests[request_id] for request_id in request_ids)
        if future.cancelled():
            return tuple(
                TiledPlanningResult.failed(
                    request,
                    status="CANCELLED",
                    message="planning future was cancelled",
                )
                for request in requests
            )
        try:
            results = tuple(future.result())
        except Exception as exc:  # pragma: no cover
            return tuple(
                TiledPlanningResult.failed(
                    request, status=type(exc).__name__, message=str(exc)
                )
                for request in requests
            )
        if len(results) != len(requests):
            return tuple(
                TiledPlanningResult.failed(
                    request,
                    status="INVALID_BACKEND_RESULTS",
                    message=(
                        "planner backend returned "
                        f"{len(results)} results for {len(requests)} requests"
                    ),
                )
                for request in requests
            )
        return results

    def _store_completed(self, request_id: str, result: TiledPlanningResult) -> None:
        """按配置保留 completed summary，并以插入顺序淘汰最旧结果。"""

        if self.max_completed_results == 0:
            self._evicted_completed_results += 1
            return
        self._completed[request_id] = result
        if self.max_completed_results is None:
            return
        while len(self._completed) > self.max_completed_results:
            oldest = next(iter(self._completed))
            self._completed.pop(oldest, None)
            self._evicted_completed_results += 1


def _split_planning_request(
    request: TiledPlanningRequest,
    *,
    max_batch_problems: int,
) -> tuple[TiledPlanningRequest, ...]:
    """按 env 行切分请求，并同步切分所有逐行 segment goals。

    chunk ID 仅供 backend 调试，不能泄漏为外部结果身份。非逐行元数据通过 dataclass
    ``replace`` 原样共享，numpy 行视图由请求不可变约定保护。
    """

    chunks: list[TiledPlanningRequest] = []
    for part_index, start in enumerate(
        range(0, request_problem_count(request), int(max_batch_problems)), start=1
    ):
        stop = min(start + int(max_batch_problems), request_problem_count(request))
        segments = tuple(
            replace(
                segment,
                goal_positions=(
                    None
                    if segment.goal_positions is None
                    else segment.goal_positions[start:stop]
                ),
            )
            for segment in request.segments
        )
        chunks.append(
            replace(
                request,
                request_id=f"{request.request_id}#split-{part_index}",
                env_ids=request.env_ids[start:stop],
                current_positions=request.current_positions[start:stop],
                goal_positions=(
                    None
                    if request.goal_positions is None
                    else request.goal_positions[start:stop]
                ),
                segments=segments,
            )
        )
    return tuple(chunks)


def _merge_split_results(
    request: TiledPlanningRequest,
    results: tuple[TiledPlanningResult, ...],
) -> TiledPlanningResult:
    """校验并把全部成功 chunks 原子合并成单一外部结果。

    每个 chunk 的 env ID 必须覆盖原请求中的相邻区间，时间采样必须完全一致；不满足这些
    条件说明 backend 违反批处理契约，返回明确失败而不是拼接错位轨迹。
    """

    if not results:
        return TiledPlanningResult.failed(
            request,
            status="INVALID_SPLIT_RESULTS",
            message="oversized planning request produced no chunk results",
        )
    times = results[0].times
    offset = 0
    for result in results:
        expected_env_ids = request.env_ids[offset : offset + len(result.env_ids)]
        if (
            result.robot_name != request.robot_name
            or result.env_ids != expected_env_ids
            or result.joint_names != request.joint_names
            or result.times.shape != times.shape
            or not np.allclose(result.times, times)
        ):
            return TiledPlanningResult.failed(
                request,
                status="INVALID_SPLIT_RESULTS",
                message=(
                    "oversized planning request chunks returned inconsistent "
                    "robot, env, joint, or time structure"
                ),
            )
        offset += len(result.env_ids)
    if offset != len(request.env_ids):
        return TiledPlanningResult.failed(
            request,
            status="INVALID_SPLIT_RESULTS",
            message="oversized planning request chunks did not cover every env",
        )
    return TiledPlanningResult(
        request_id=request.request_id,
        robot_name=request.robot_name,
        env_ids=request.env_ids,
        success=True,
        status="SUCCESS",
        message=f"planning request completed in {len(results)} bounded chunks",
        times=times,
        positions=np.concatenate([result.positions for result in results], axis=0),
        joint_names=request.joint_names,
        source=request.source,
        load_on_success=request.load_on_success,
        replace=request.replace,
    )


__all__ = ["TiledPlannerManager"]
