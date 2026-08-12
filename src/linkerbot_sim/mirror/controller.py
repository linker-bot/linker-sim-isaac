"""Mirror v1 请求到主线程 use case 的唯一控制器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event

from linkerbot_sim.controllers.control_mode import (
    ControlModeGenerationConflict,
    ControlModeIncompatibleError,
    ControlModeLockedError,
    ControlModeRollbackError,
    ControlModeSwitchError,
    require_expected_generation,
)
from linkerbot_sim.mirror.control_mode import MirrorControlModeService
from linkerbot_sim.mirror.hybrid_parameters import (
    HYBRID_PARAMETER_FIELDS,
    HybridNotConfiguredError,
    HybridParameterGenerationConflict,
    HybridParameterOutOfRange,
    HybridParameterService,
)
from linkerbot_sim.mirror.interface.admission import (
    MirrorAdmissionError,
    MirrorAdmissionQueue,
    RuntimeEstoppedError,
)
from linkerbot_sim.mirror.interface.protocol import MirrorRequest, MirrorResponse
from linkerbot_sim.mirror.motion.hybrid_executor import HybridExecutionError
from linkerbot_sim.mirror.motion.owner import (
    MIRROR_V3_MOTION_OPERATIONS,
    MirrorMotionOwner,
)
from linkerbot_sim.mirror.reset import MirrorResetService
from linkerbot_sim.mirror.snapshot import MirrorSnapshotService
from linkerbot_sim.mirror.state import MirrorStateService
from linkerbot_sim.snapshots.transactions import RuntimeMutationRejected


_OUT_OF_BAND_OPERATIONS = frozenset(
    {"queue.cancel", "queue.cancel_current", "runtime.estop", "runtime.quit"}
)


def _exact_arguments(
    arguments: Mapping[str, object],
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
    operation: str,
) -> None:
    missing = sorted(set(required) - set(arguments))
    unknown = sorted(set(arguments) - set(allowed))
    if missing:
        raise ValueError(f"{operation} 缺少 arguments: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{operation} 包含未知 arguments: {', '.join(unknown)}")


def _json_result(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_result(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_result(item) for item in value]
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        return _json_result(serializer())
    return {"value": str(value)}


@dataclass
class MirrorController:
    """只在 runtime owner thread 执行状态、物理与 planner 操作。

    cancel/estop/quit 只改变线程安全 admission 标志，可由 ingress 立即处理；它们不直接
    触碰 Isaac handle。长 motion 每个 tick 查询 ``should_cancel``，从而得到有界取消。
    """

    admission: MirrorAdmissionQueue
    motion: MirrorMotionOwner
    state: MirrorStateService
    snapshots: MirrorSnapshotService
    reset_service: MirrorResetService
    control_mode: MirrorControlModeService | None = None
    hybrid_parameters: HybridParameterService | None = None
    status_provider: Callable[[], Mapping[str, object]] | None = None
    _quit: Event = field(default_factory=Event, init=False, repr=False)

    def bind_status_provider(
        self, provider: Callable[[], Mapping[str, object]]
    ) -> None:
        if self.status_provider is not None:
            raise RuntimeError("MirrorController status provider 已绑定")
        self.status_provider = provider

    @property
    def quit_requested(self) -> bool:
        return self._quit.is_set()

    def request_quit(self) -> None:
        """供 stdin EOF/进程信号设置退出意图，不访问 engine handle。"""

        self._quit.set()

    def submit_and_wait(
        self,
        request: MirrorRequest,
        *,
        timeout_s: float,
    ) -> MirrorResponse:
        if request.operation in _OUT_OF_BAND_OPERATIONS:
            try:
                self.admission.reserve_immediate(request.request_id)
            except MirrorAdmissionError as exc:
                return MirrorResponse.failure(
                    request.request_id,
                    code=exc.code,
                    message=str(exc),
                    protocol=request.protocol,
                )
            response = self._dispatch(request)
            self.admission.complete_immediate(response)
            return response
        try:
            self.admission.submit(request)
        except MirrorAdmissionError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                protocol=request.protocol,
            )
        return self.admission.wait_response(request.request_id, timeout_s=timeout_s)

    def process_next(self, *, timeout_s: float = 0.0) -> MirrorResponse | None:
        request = self.admission.claim(timeout_s=timeout_s)
        if request is None:
            return None
        response = self._dispatch(request)
        self.admission.complete(response)
        return response

    def dispatch(self, request: MirrorRequest) -> MirrorResponse:
        """同步测试/embedded API；调用方必须已经位于 runtime owner thread。"""

        return self._dispatch(request)

    def _dispatch(self, request: MirrorRequest) -> MirrorResponse:
        try:
            result = self._execute(request)
            return MirrorResponse.success(
                request.request_id,
                _json_result(result),
                protocol=request.protocol,
            )
        except MirrorAdmissionError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                protocol=request.protocol,
            )
        except ControlModeGenerationConflict as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="control_generation_conflict",
                message=str(exc),
                details={"expected": exc.expected, "actual": exc.actual},
                protocol=request.protocol,
            )
        except HybridParameterGenerationConflict as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                details={"expected": exc.expected, "actual": exc.actual},
                protocol=request.protocol,
            )
        except HybridParameterOutOfRange as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                details=exc.details,
                protocol=request.protocol,
            )
        except HybridNotConfiguredError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                protocol=request.protocol,
            )
        except HybridExecutionError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code=exc.code,
                message=str(exc),
                details=exc.details or None,
                protocol=request.protocol,
            )
        except ControlModeIncompatibleError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="control_mode_incompatible",
                message=str(exc),
                details=exc.details,
                protocol=request.protocol,
            )
        except ControlModeRollbackError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="control_mode_rollback_failed",
                message=str(exc),
                details={"fatal": True},
                protocol=request.protocol,
            )
        except ControlModeSwitchError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="control_mode_switch_failed",
                message=str(exc),
                protocol=request.protocol,
            )
        except (ControlModeLockedError, RuntimeMutationRejected) as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="runtime_mutation_rejected",
                message=str(exc),
                protocol=request.protocol,
            )
        except ValueError as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="invalid_arguments",
                message=str(exc),
                protocol=request.protocol,
            )
        except Exception as exc:
            return MirrorResponse.failure(
                request.request_id,
                code="operation_failed",
                message=str(exc),
                details={"exception_type": type(exc).__name__},
                protocol=request.protocol,
            )

    def _execute(self, request: MirrorRequest) -> object:
        operation = request.operation
        arguments = request.arguments_dict()
        if operation == "control.get_mode":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return self._control_mode_service().get_mode()
        if operation == "control.set_mode":
            _exact_arguments(
                arguments,
                allowed={"mode", "expected_generation"},
                required={"mode"},
                operation=operation,
            )
            if self.admission.status().estopped:
                raise RuntimeEstoppedError(
                    "Mirror 已 estop；reset 前拒绝 control.set_mode"
                )
            expected_generation = (
                require_expected_generation(arguments["expected_generation"])
                if "expected_generation" in arguments
                else None
            )
            return self._control_mode_service().set_mode(
                arguments["mode"],  # type: ignore[arg-type]
                expected_generation=expected_generation,
            )
        if operation == "control.get_hybrid_parameters":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return self._hybrid_parameter_service().get_state()
        if operation == "control.set_hybrid_parameters":
            _exact_arguments(
                arguments,
                allowed=set(HYBRID_PARAMETER_FIELDS) | {"expected_generation"},
                operation=operation,
            )
            if self.admission.status().estopped:
                raise RuntimeEstoppedError(
                    "Mirror is estopped; reset is required before hybrid tuning"
                )
            expected_generation = arguments.pop("expected_generation", None)
            return self._hybrid_parameter_service().set_parameters(
                arguments,
                expected_generation=expected_generation,  # type: ignore[arg-type]
            )
        if operation == "control.tare_wrench":
            if self.admission.status().estopped:
                raise RuntimeEstoppedError(
                    "Mirror is estopped; reset is required before wrench tare"
                )
            return self.motion.tare_wrench(
                arguments,
                request_id=request.request_id,
                should_cancel=lambda: self.admission.should_cancel(request.request_id),
            )
        if operation in MIRROR_V3_MOTION_OPERATIONS:
            return self.motion.execute(
                operation,
                arguments,
                request_id=request.request_id,
                should_cancel=lambda: self.admission.should_cancel(request.request_id),
                protocol=request.protocol,
            )
        if operation == "runtime.reset":
            _exact_arguments(
                arguments,
                allowed={"clear_queue", "hold_after_reset"},
                operation=operation,
            )
            if (
                arguments.get("clear_queue", True) is not True
                and arguments.get("clear_queue", True) is not False
            ):
                raise ValueError("runtime.reset.clear_queue 必须是 boolean")
            if (
                arguments.get("hold_after_reset", True) is not True
                and arguments.get("hold_after_reset", True) is not False
            ):
                raise ValueError("runtime.reset.hold_after_reset 必须是 boolean")
            cancelled = ()
            if bool(arguments.get("clear_queue", True)):
                cancelled = self.admission.cancel_pending(
                    code="reset_queue_cleared",
                    message="请求因 runtime.reset 清空队列",
                )
            result = self.reset_service.reset(
                hold_after_reset=bool(arguments.get("hold_after_reset", True))
            )
            self.admission.clear_estop()
            return {
                "reset": _json_result(result),
                "cancelled_request_ids": list(cancelled),
            }
        if operation == "state.get":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return self.state.get_state()
        if operation == "state.set":
            _exact_arguments(
                arguments,
                allowed={"state", "strict"},
                required={"state"},
                operation=operation,
            )
            state = arguments["state"]
            strict = arguments.get("strict", True)
            if not isinstance(state, Mapping):
                raise ValueError("state.set.state 必须是 object")
            if type(strict) is not bool:
                raise ValueError("state.set.strict 必须是 boolean")
            # state mutation 必须留在 owner thread；急停只允许继续读取状态，不能通过
            # state.set 改写仿真或隐式解除 latch。普通 ingress 在 admission 层也执行同一检查。
            if self.admission.status().estopped:
                raise RuntimeEstoppedError("Mirror 已 estop；reset 前拒绝 state.set")
            return self.state.set_state(state, strict=strict)
        if operation == "snapshot.get":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return self.snapshots.capture_snapshot()
        if operation == "snapshot.set":
            _exact_arguments(
                arguments,
                allowed={"snapshot", "label_map", "strict"},
                required={"snapshot"},
                operation=operation,
            )
            snapshot = arguments["snapshot"]
            label_map = arguments.get("label_map")
            strict = arguments.get("strict", True)
            if not isinstance(snapshot, Mapping):
                raise ValueError("snapshot.set.snapshot 必须是 object")
            if label_map is not None and not isinstance(label_map, Mapping):
                raise ValueError("snapshot.set.label_map 必须是 object 或 null")
            if type(strict) is not bool:
                raise ValueError("snapshot.set.strict 必须是 boolean")
            return self.snapshots.restore_snapshot(
                snapshot,
                label_map=label_map,  # type: ignore[arg-type]
                strict=strict,
            )
        if operation == "runtime.status":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            runtime_status = (
                {} if self.status_provider is None else dict(self.status_provider())
            )
            runtime_status["queue"] = self.admission.status().as_dict()
            return runtime_status
        if operation == "queue.cancel":
            _exact_arguments(
                arguments,
                allowed={"request_id"},
                required={"request_id"},
                operation=operation,
            )
            target = arguments["request_id"]
            if not isinstance(target, str) or not target:
                raise ValueError("queue.cancel.request_id 必须非空")
            return {"cancelled": self.admission.cancel(target), "request_id": target}
        if operation == "queue.cancel_current":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return {"cancelled": self.admission.cancel_current()}
        if operation == "runtime.estop":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            return {"cancelled_request_ids": list(self.admission.estop())}
        if operation == "runtime.quit":
            _exact_arguments(arguments, allowed=set(), operation=operation)
            self._quit.set()
            return {"quitting": True}
        raise ValueError(f"不支持的 operation: {operation!r}")

    def _control_mode_service(self) -> MirrorControlModeService:
        if self.control_mode is None:
            raise RuntimeError("Mirror control-mode service is not configured")
        return self.control_mode

    def _hybrid_parameter_service(self) -> HybridParameterService:
        if self.hybrid_parameters is None:
            raise HybridNotConfiguredError(
                "hybrid force/position control is not configured for this runtime"
            )
        return self.hybrid_parameters

    def close(self) -> bool:
        self._quit.set()
        return self.admission.close()


__all__ = ["MirrorController"]
