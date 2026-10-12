"""只属于 Mirror 冷边界的相机与 physics-to-USD 协调器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time

from linkerbot_sim.mirror.lifecycle import close_result_stopped


# Match SyntheticData next_render_simulation_async: asynchronous RTX can need
# more app updates than SWH sensor waits when several products initialize together.
_MAX_CAPTURE_UPDATES = 150


def _close_resource(resource: object) -> bool:
    callback = getattr(resource, "close", None)
    if not callable(callback):
        return True
    return close_result_stopped(callback())


@dataclass
class CameraBundle:
    """MirrorRuntime 独占的 camera handles 与输出 sink。

    Newton manager 只执行 physics-to-USD 同步；它不注册、启动或关闭 camera。
    Bundle 在 session 之前关闭，确保 render product/worker 不会访问已销毁的 stage。
    """

    cameras: tuple[object, ...] = ()
    output: object | None = None
    capture_hook: Callable[[Sequence[object]], object] | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def capture(self, cameras: Sequence[object]) -> object:
        if self._closed:
            raise RuntimeError("CameraBundle is closed")
        if self.capture_hook is not None:
            return self.capture_hook(cameras)
        frames: dict[str, object] = {}
        for index, camera in enumerate(cameras):
            getter = getattr(camera, "get_current_frame", None)
            if not callable(getter):
                continue
            name = str(getattr(camera, "name", f"camera_{index}"))
            frames[name] = getter(clone=True)
        return frames

    def close(self) -> bool:
        if self._closed:
            return True
        # 先停止 sink/worker，阻止新消费；再逆序释放 render products。
        if self.output is not None and not _close_resource(self.output):
            return False
        for camera in reversed(self.cameras):
            if not _close_resource(camera):
                return False
        self._closed = True
        return True


@dataclass
class RenderCoordinator:
    """把 Newton 的 D2H/USD 同步限制在显式 Mirror render cadence。"""

    physics_runtime: object
    cameras: CameraBundle | None = None
    gui_frequency_hz: float | None = None
    _next_gui_at: float = field(default=0.0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _snapshot_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._activate(())
        self._gui_serviced()

    def render_frame(
        self, camera_ids: Sequence[str] | None = None, *, capture: bool = True
    ) -> object:
        """Capture a new frozen snapshot for selected cameras, without publishing outputs."""

        self._require_open()
        if type(capture) is not bool:
            raise TypeError("render capture must be a boolean")
        cameras = self._select_cameras(camera_ids)
        targets = self._render_targets(cameras)
        render = getattr(self.physics_runtime, "render", None)
        if not callable(render):
            raise RuntimeError("physics runtime is missing the render contract")
        render_tick = render
        render_update = getattr(self.physics_runtime, "render_update", None)
        pre_render = getattr(self.physics_runtime, "pre_render", None)
        if callable(render_update) and callable(pre_render):
            pre_render()
            render_tick = render_update
        self._snapshot_index += 1
        try:
            for camera in cameras:
                invalidate = getattr(camera, "invalidate_render_capture", None)
                if callable(invalidate):
                    invalidate()
            # Selected products share renderer updates for one frozen snapshot.
            # Keep each product's history budget and native completion barrier.
            self._activate(cameras)
            self._render_group(render_tick, targets)
            # Expose identities only after every selected product has completed.
            for camera in cameras:
                finish = getattr(camera, "finish_render_capture", None)
                if callable(finish):
                    finish(
                        snapshot_index=self._snapshot_index,
                        physics_time_s=getattr(
                            self.physics_runtime, "simulation_time", None
                        ),
                    )
            result = (
                {}
                if not capture or self.cameras is None
                else self.cameras.capture(cameras)
            )
            self._activate(())
            self._gui_serviced()
            return result
        except BaseException as error:
            # A partial group or readback must not leave a successful mixed capture.
            for camera in cameras:
                invalidate = getattr(camera, "invalidate_render_capture", None)
                if callable(invalidate):
                    invalidate()
            try:
                # Restore the idle set even when capture or readback fails.
                self._activate(())
            except BaseException as cleanup_error:
                error.add_note(f"camera deactivation also failed: {cleanup_error}")
            raise

    def after_physics_step(self) -> None:
        """Acquire due outputs once; the canonical scene observer publishes them."""

        self._require_open()
        output = None if self.cameras is None else self.cameras.output
        observer = getattr(output, "observer", None)
        due = (
            ()
            if observer is None
            else observer.due_camera_ids(float(self.physics_runtime.simulation_time))
        )
        if due:
            self.render_frame(due, capture=False)
        else:
            self.service_gui()

    def gui_wait_timeout(self, maximum_s: float) -> float:
        """Bound admission waits by the next GUI deadline; headless does not wake."""

        if self.gui_frequency_hz is None:
            return maximum_s
        return min(maximum_s, max(0.0, self._next_gui_at - time.monotonic()))

    def service_gui(self) -> None:
        """Publish and display current state without a sensor barrier or physics step."""

        self._require_open()
        if self.gui_frequency_hz is None or time.monotonic() < self._next_gui_at:
            return
        physics_time = getattr(self.physics_runtime, "simulation_time", None)
        self.physics_runtime.render()
        if getattr(self.physics_runtime, "simulation_time", None) != physics_time:
            raise RuntimeError("GUI rendering advanced physics time")
        self._gui_serviced()

    def _gui_serviced(self) -> None:
        if self.gui_frequency_hz is not None:
            self._next_gui_at = time.monotonic() + 1.0 / self.gui_frequency_hz

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RenderCoordinator is closed")

    def _select_cameras(self, camera_ids: Sequence[str] | None) -> tuple[object, ...]:
        cameras = () if self.cameras is None else self.cameras.cameras
        if camera_ids is None:
            return cameras
        if isinstance(camera_ids, (str, bytes)) or not isinstance(camera_ids, Sequence):
            raise TypeError("camera_ids must be a sequence of camera names")
        if not camera_ids or any(not isinstance(name, str) for name in camera_ids):
            raise ValueError("camera_ids must contain at least one camera name")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_ids must not contain duplicate names")
        by_name = {camera.name: camera for camera in cameras}
        unknown = set(camera_ids) - by_name.keys()
        if unknown:
            raise ValueError(f"unknown camera_ids: {sorted(unknown)}")
        return tuple(by_name[name] for name in camera_ids)

    def _activate(self, selected: tuple[object, ...]) -> None:
        if self.cameras is None:
            return
        error: BaseException | None = None
        for camera in self.cameras.cameras:
            try:
                camera.set_render_active(any(camera is item for item in selected))
            except BaseException as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"additional camera activation failure: {exc}")
        if error is not None:
            raise error

    def invalidate_captures(self) -> None:
        if self.cameras is not None:
            for camera in self.cameras.cameras:
                invalidate = getattr(camera, "invalidate_render_capture", None)
                if callable(invalidate):
                    invalidate()

    def _render_targets(
        self, cameras: tuple[object, ...]
    ) -> tuple[tuple[object, int], ...]:
        targets = []
        for camera in cameras:
            count = getattr(camera, "render_update_count", 1)
            if type(count) is not int or count < 1:
                raise RuntimeError(
                    "camera render_update_count must be a positive integer"
                )
            targets.append((camera, count))
        return tuple(targets)

    def _render_group(
        self,
        render: Callable[[], object],
        targets: tuple[tuple[object, int], ...],
    ) -> None:
        cameras = tuple(camera for camera, _count in targets)
        for camera in cameras:
            begin = getattr(camera, "begin_render_capture", None)
            if callable(begin):
                begin()
        count = max((count for _target, count in targets), default=1)
        physics_time = getattr(self.physics_runtime, "simulation_time", None)
        pending = []
        # Native completion may lag several renderer updates. Bound warmup instead
        # of advancing physics or returning old annotator data under a new index.
        limit = max(count, _MAX_CAPTURE_UPDATES)
        for update in range(limit):
            render()
            if getattr(self.physics_runtime, "simulation_time", None) != physics_time:
                raise RuntimeError("camera rendering advanced physics time")
            if update + 1 < count:
                continue
            pending = [
                camera
                for camera in cameras
                if callable(ready := getattr(camera, "render_capture_ready", None))
                and not ready()
            ]
            if not pending:
                return
        names = [str(getattr(camera, "name", "unknown")) for camera in pending]
        details = {
            name: status()
            for name, camera in zip(names, pending)
            if callable(status := getattr(camera, "render_capture_status", None))
        }
        raise RuntimeError(
            f"camera fresh-frame timeout after {limit} renderer updates: {names}; {details}"
        )

    def close(self) -> bool:
        if self._closed:
            return True
        if self.cameras is not None and not self.cameras.close():
            return False
        self._closed = True
        return True


__all__ = ["CameraBundle", "RenderCoordinator"]
