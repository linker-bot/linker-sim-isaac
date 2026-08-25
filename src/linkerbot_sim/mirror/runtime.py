"""Mirror 产品资源图与严格逆依赖关闭状态机。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from threading import get_ident

from linkerbot_sim.configuration.modes.mirror import MirrorConfig
from linkerbot_sim.mirror.collision import MirrorCollisionOwner
from linkerbot_sim.mirror.controller import MirrorController
from linkerbot_sim.mirror.control_mode import MirrorControlModeService
from linkerbot_sim.mirror.lifecycle import close_result_stopped
from linkerbot_sim.mirror.motion import MirrorMotionOwner
from linkerbot_sim.mirror.rendering import RenderCoordinator
from linkerbot_sim.mirror.reset import MirrorResetService
from linkerbot_sim.mirror.snapshot import MirrorSnapshotService
from linkerbot_sim.mirror.state import MirrorStateService
from linkerbot_sim.mirror.timing import WallClockStepSynchronizer


@dataclass(frozen=True)
class MirrorCloseReport:
    stopped: bool
    completed_phases: tuple[str, ...]
    live_resources: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "stopped": self.stopped,
            "completed_phases": list(self.completed_phases),
            "live_resources": list(self.live_resources),
            "errors": list(self.errors),
        }


def _resource_name(resource: object) -> str:
    explicit = getattr(resource, "resource_name", None)
    return str(explicit) if explicit else type(resource).__name__


@dataclass
class MirrorRuntime:
    """一个现实映像、一个 session、一个明确的资源所有权根。

    Session 是 App/stage/concrete physics runtime 的唯一 closer；Mirror 的其它对象只能借用
    它们，并必须在 session 之前停止。该类型故意没有 ``world`` 属性，避免把 Newton
    runtime 伪装成 Isaac World。
    """

    config: MirrorConfig
    session: object
    controller: MirrorController
    state_service: MirrorStateService
    snapshot_service: MirrorSnapshotService
    reset_service: MirrorResetService
    motion: MirrorMotionOwner
    collision: MirrorCollisionOwner
    control_mode: MirrorControlModeService | None = None
    rendering: RenderCoordinator | None = None
    ingress: list[object] = field(default_factory=list)
    outputs: tuple[object, ...] = ()
    controllers: tuple[object, ...] = ()
    views: tuple[object, ...] = ()
    scene_resources: object | None = field(default=None, repr=False)
    step_synchronizer: WallClockStepSynchronizer = field(init=False, repr=False)
    fatal_error: str | None = field(default=None, init=False)
    _owner_thread_id: int = field(default_factory=get_ident, init=False, repr=False)
    _closed_resource_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _completed_phases: list[str] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        physics_runtime = getattr(self.session, "physics_runtime", None)
        if physics_runtime is None:
            raise ValueError("MirrorRuntime.session must own a physics_runtime")
        self.step_synchronizer = WallClockStepSynchronizer(
            enabled=self.config.control.sync_simulation_to_wall_clock
        )
        bind_step_synchronizer = getattr(
            self.motion.backend, "bind_step_synchronizer", None
        )
        if callable(bind_step_synchronizer):
            bind_step_synchronizer(self.step_synchronizer)
        bind_runtime_owner = getattr(self.motion.backend, "bind_runtime_owner", None)
        if callable(bind_runtime_owner):
            bind_runtime_owner(self)
        if self.control_mode is None:
            self.control_mode = MirrorControlModeService(
                initial_mode=self.config.control.mode,
                bindings=(),
            )
        if self.controller.control_mode is None:
            self.controller.control_mode = self.control_mode
        elif self.controller.control_mode is not self.control_mode:
            raise ValueError("Mirror controller/runtime control-mode services differ")
        self.control_mode.bind_runtime(self)
        self.controller.bind_status_provider(self.status)

    @property
    def physics_runtime(self) -> object:
        return self.session.physics_runtime

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def physics_dt_s(self) -> float:
        get_dt = getattr(self.physics_runtime, "get_physics_dt", None)
        dt = (
            float(get_dt())
            if callable(get_dt)
            else 1.0 / float(self.config.scene.physics_frequency_hz)
        )
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("physics runtime returned an invalid get_physics_dt")
        return dt

    def attach_ingress(self, resource: object) -> None:
        self._require_owner_thread("attach_ingress")
        if self._closed:
            raise RuntimeError("MirrorRuntime is closed")
        if any(item is resource for item in self.ingress):
            raise ValueError("ingress resource is already registered")
        self.ingress.append(resource)

    def step(self, *, render: bool = False) -> None:
        self._require_open("step")
        self._require_owner_thread("step")
        step = getattr(self.physics_runtime, "step", None)
        if not callable(step):
            raise RuntimeError("physics runtime is missing step")
        self.step_synchronizer.before_step(self.physics_dt_s)
        # 物理步只推进一次；显式渲染由 coordinator 在 step 后处理。
        step(render=False)
        self.collision.mark_dirty()
        # camera annotator 只能在本 tick 的 render transaction 完成后读取；状态与日志也在
        # 同一个 post-step 边界采样，保证每个 physics tick 最多写入一次。
        if render:
            if self.rendering is None:
                raise RuntimeError(
                    "the current Mirror config does not enable rendering"
                )
            # 内部 physics step 只推进 renderer；下面的 scene observer 按 camera frequency
            # 采样并发布一次。显式 runtime.render() 才立即返回 camera frame。
            self.rendering.render_only()
        observe = getattr(self.scene_resources, "observe_after_step", None)
        if callable(observe):
            observe(phase="idle")

    def render(self) -> object:
        self._require_open("render")
        self._require_owner_thread("render")
        if self.rendering is None:
            raise RuntimeError("the current Mirror config does not enable rendering")
        return self.rendering.render_frame()

    def get_state(self) -> dict[str, object]:
        self._require_open("get_state")
        self._require_owner_thread("get_state")
        return self.state_service.get_state()

    def set_state(self, state: Mapping[str, object], *, strict: bool = True) -> object:
        self._require_open("set_state")
        self._require_owner_thread("set_state")
        result = self.state_service.set_state(state, strict=strict)
        self.collision.mark_dirty()
        return result

    def capture_snapshot(self) -> dict[str, object]:
        self._require_open("capture_snapshot")
        self._require_owner_thread("capture_snapshot")
        return self.snapshot_service.capture_snapshot()

    def restore_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        label_map: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> object:
        self._require_open("restore_snapshot")
        self._require_owner_thread("restore_snapshot")
        result = self.snapshot_service.restore_snapshot(
            snapshot,
            label_map=label_map,
            strict=strict,
        )
        self.collision.mark_dirty()
        return result

    def reset(self, *, hold_after_reset: bool = True) -> object:
        self._require_open("reset")
        self._require_owner_thread("reset")
        self.step_synchronizer.rebase()
        result = self.reset_service.reset(hold_after_reset=hold_after_reset)
        self.collision.mark_dirty()
        return result

    def get_control_mode(self):
        """Return runtime control state without touching engine handles."""

        if self._closed:
            raise RuntimeError(
                "MirrorRuntime is closed; cannot perform get_control_mode"
            )
        self._require_owner_thread("get_control_mode")
        assert self.control_mode is not None
        return self.control_mode.get_mode()

    def set_control_mode(
        self,
        mode: str,
        *,
        expected_generation: int | None = None,
    ):
        """Switch all robots at an owner-thread between-motion boundary."""

        self._require_open("set_control_mode")
        self._require_owner_thread("set_control_mode")
        if self.controller.admission.status().estopped:
            raise RuntimeError(
                "Mirror is e-stopped; set_control_mode is rejected before reset"
            )
        assert self.control_mode is not None
        return self.control_mode.set_mode(
            mode,  # type: ignore[arg-type]
            expected_generation=expected_generation,
        )

    def request_quit(self) -> None:
        """Expose the thread-safe fail-stop quit signal to transactions."""

        self.controller.request_quit()

    def status(self) -> dict[str, object]:
        physics = self.physics_runtime
        result: dict[str, object] = {
            "mode": "mirror",
            "closed": self._closed,
            "physics": {
                "backend": str(getattr(physics, "backend", "unknown")),
                "kind": str(getattr(physics, "kind", "unknown")),
                "execution": str(getattr(physics, "execution", "unknown")),
            },
            "collision": self.collision.status(),
            "control": (
                {}
                if self.control_mode is None
                else self.control_mode.get_mode().as_dict()
            ),
            "fatal_error": self.fatal_error,
            "shutdown": {
                "completed_phases": list(self._completed_phases),
            },
        }
        hybrid_status = getattr(self.motion.backend, "hybrid_status", None)
        if callable(hybrid_status):
            result["hybrid_control"] = dict(hybrid_status())
        status = getattr(self.scene_resources, "status", None)
        if callable(status):
            result["scene"] = dict(status())
        return result

    def close(self) -> MirrorCloseReport:
        """按 ingress→outputs/camera/planner→controllers/views→session 重试关闭。"""

        self._require_owner_thread("close")
        if self._closed:
            return MirrorCloseReport(
                stopped=True,
                completed_phases=tuple(self._completed_phases),
            )
        phase_resources = (
            ("ingress", (*self.ingress, self.controller)),
            (
                "outputs_camera_planner",
                (*self.outputs, self.rendering, self.motion, self.collision),
            ),
            ("controllers_views", (*self.controllers, *self.views)),
        )
        for phase, resources in phase_resources:
            if phase in self._completed_phases:
                continue
            report = self._close_phase(phase, resources)
            if report is not None:
                return report
            self._completed_phases.append(phase)

        if "session" not in self._completed_phases:
            try:
                # 只有这里能关闭 App/stage/physics runtime。
                self.session.close()
            except BaseException as exc:
                return MirrorCloseReport(
                    stopped=False,
                    completed_phases=tuple(self._completed_phases),
                    live_resources=("IsaacSession",),
                    errors=(f"{type(exc).__name__}: {exc}",),
                )
            self._completed_phases.append("session")
        self._closed = True
        return MirrorCloseReport(
            stopped=True,
            completed_phases=tuple(self._completed_phases),
        )

    def _close_phase(
        self,
        phase: str,
        resources: Sequence[object | None],
    ) -> MirrorCloseReport | None:
        live: list[str] = []
        errors: list[str] = []
        seen: set[int] = set()
        for resource in resources:
            if resource is None:
                continue
            identity = id(resource)
            if identity in seen or identity in self._closed_resource_ids:
                continue
            seen.add(identity)
            callback = getattr(resource, "close", None)
            if not callable(callback):
                self._closed_resource_ids.add(identity)
                continue
            try:
                stopped = close_result_stopped(callback())
            except BaseException as exc:
                live.append(_resource_name(resource))
                errors.append(
                    f"{_resource_name(resource)}: {type(exc).__name__}: {exc}"
                )
                continue
            if stopped:
                self._closed_resource_ids.add(identity)
            else:
                live.append(_resource_name(resource))
        if live or errors:
            return MirrorCloseReport(
                stopped=False,
                completed_phases=tuple(self._completed_phases),
                live_resources=tuple(sorted(set(live))),
                errors=tuple(errors),
            )
        return None

    def _require_open(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"MirrorRuntime is closed; cannot perform {operation}")
        if self.fatal_error is not None:
            raise RuntimeError(
                f"MirrorRuntime is fail-stopped; cannot perform {operation}: {self.fatal_error}"
            )

    def _require_owner_thread(self, operation: str) -> None:
        if get_ident() != self._owner_thread_id:
            raise RuntimeError(f"{operation} must run on the Mirror owner thread")


__all__ = ["MirrorCloseReport", "MirrorRuntime"]
