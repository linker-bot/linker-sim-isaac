"""只属于 Mirror 冷边界的相机与 physics-to-USD 协调器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from linkerbot_sim.mirror.lifecycle import close_result_stopped


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

    def capture(self) -> object:
        if self._closed:
            raise RuntimeError("CameraBundle is closed")
        if self.capture_hook is not None:
            return self.capture_hook(self.cameras)
        frames: dict[str, object] = {}
        for index, camera in enumerate(self.cameras):
            getter = getattr(camera, "get_current_frame", None)
            if not callable(getter):
                continue
            name = str(getattr(camera, "name", f"camera_{index}"))
            frames[name] = getter(clone=True)
        publisher = getattr(self.output, "publish", None)
        if callable(publisher):
            publisher(frames)
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
    _closed: bool = field(default=False, init=False, repr=False)

    def render_frame(self, *, capture: bool = True) -> object:
        """执行一次 render transaction，并可在显式调用边界返回当前帧。

        Physics step/timeline 使用 :meth:`render_only`，随后由有频率和背压策略的
        ``CameraFrameObserver`` 统一采样；只有显式 ``MirrorRuntime.render()`` 需要这里
        立即 capture，避免一次物理 tick 对同一 camera 做两次 readback。
        """

        if self._closed:
            raise RuntimeError("RenderCoordinator is closed")
        if type(capture) is not bool:
            raise TypeError("render capture must be a boolean")
        render = getattr(self.physics_runtime, "render", None)
        if not callable(render):
            raise RuntimeError("physics runtime is missing the render contract")
        targets = self._render_targets()
        selectors = tuple(
            getattr(target, "set_render_active", None) for target, _count in targets
        )
        selected_render = len(targets) > 1 and (
            any(callable(selector) for selector in selectors)
            or any(count > 1 for _target, count in targets)
        )
        if selected_render and not all(callable(selector) for selector in selectors):
            raise RuntimeError(
                "an independent per-camera render budget requires each camera to implement set_render_active"
            )

        # Newton 暴露纯 renderer tick，因此每个 Mirror frame 只发布一次物理快照，
        # 后续 camera budget/轮转不再重复 CUDA synchronize 和 USD transform 写入。PhysX
        # 没有这个拆分合同，继续使用其 World.render() 完成一次完整 transaction。
        render_tick = render
        render_update = getattr(self.physics_runtime, "render_update", None)
        pre_render = getattr(self.physics_runtime, "pre_render", None)
        if callable(render_update) and callable(pre_render):
            pre_render()
            render_tick = render_update
        if not selected_render:
            # camera 只声明“同一快照需要多少次 renderer update”，不取得 App 或 physics 的
            # 所有权。Newton 的隐藏 SyntheticData product 需要连续四次完整 transaction
            # 才能越过 Kit 的三帧 history；PhysX CameraSensor 没有该声明，默认只渲染一次。
            # 重复调用期间 concrete runtime 不推进 physics time，因而每次看到的是同一快照。
            count = max((count for _target, count in targets), default=1)
            for _ in range(count):
                render_tick()
        else:
            self._render_selected_targets(render_tick, targets)
        return {} if not capture or self.cameras is None else self.cameras.capture()

    def render_only(self) -> None:
        """推进 renderer 但不读取 camera；供 completed physics-step 边界调用。"""

        self.render_frame(capture=False)

    def _render_targets(self) -> tuple[tuple[object, int], ...]:
        if self.cameras is None:
            return ()
        targets: list[tuple[object, int]] = []
        for camera in self.cameras.cameras:
            target = getattr(camera, "camera", camera)
            count = getattr(target, "render_update_count", 1)
            if type(count) is not int or count < 1:
                raise RuntimeError("camera render_update_count must be a positive integer")
            targets.append((target, count))
        return tuple(targets)

    @staticmethod
    def _render_selected_targets(
        render: Callable[[], object],
        targets: tuple[tuple[object, int], ...],
    ) -> None:
        """在同一物理快照上逐个推进 direct camera 的隐藏 render product。

        多个 SyntheticData viewport 若同时 active，会在为 camera A 预热的四次 Kit update 中
        一并推进 camera B，后续 capture 无法再证明每份数据来自各自完整的连续预算。因此多相机
        必须全部实现选择接口；任何异常都先恢复 active 状态，避免下一帧永久遗漏某个 viewport。
        """

        selectors = [getattr(target, "set_render_active") for target, _ in targets]

        primary_error: BaseException | None = None
        try:
            for selected_index, (_target, count) in enumerate(targets):
                for index, selector in enumerate(selectors):
                    selector(index == selected_index)
                for _ in range(count):
                    render()
        except BaseException as exc:
            primary_error = exc

        restore_error: BaseException | None = None
        for selector in selectors:
            try:
                selector(True)
            except BaseException as exc:
                if restore_error is None:
                    restore_error = exc
                else:
                    restore_error.add_note(
                        "additional camera activation failure: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if primary_error is not None:
            if restore_error is not None:
                primary_error.add_note(
                    "camera activation restore also failed: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
            raise primary_error.with_traceback(primary_error.__traceback__)
        if restore_error is not None:
            raise restore_error

    def close(self) -> bool:
        if self._closed:
            return True
        if self.cameras is not None and not self.cameras.close():
            return False
        self._closed = True
        return True


__all__ = ["CameraBundle", "RenderCoordinator"]
