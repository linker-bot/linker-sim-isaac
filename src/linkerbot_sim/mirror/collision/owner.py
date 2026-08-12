"""实时碰撞采样、registry 与 planning context 的 Mirror 生命周期边界。"""

from __future__ import annotations

from dataclasses import dataclass, field

from linkerbot_sim.mirror.lifecycle import close_result_stopped


@dataclass
class MirrorCollisionOwner:
    registry: object | None = None
    contexts: tuple[object, ...] = ()
    _closed: bool = field(default=False, init=False, repr=False)

    def status(self) -> dict[str, object]:
        metrics = getattr(self.registry, "metrics", None)
        return {
            "enabled": self.registry is not None,
            "contexts": len(self.contexts),
            "registry": dict(metrics()) if callable(metrics) else {},
        }

    def mark_dirty(self) -> None:
        callback = getattr(self.registry, "mark_dirty", None)
        if callable(callback):
            callback()

    def close(self) -> bool:
        if self._closed:
            return True
        for context in reversed(self.contexts):
            callback = getattr(context, "close", None)
            if not callable(callback):
                continue
            if not close_result_stopped(callback()):
                return False
        callback = getattr(self.registry, "close", None)
        if callable(callback) and not close_result_stopped(callback()):
            return False
        self._closed = True
        return True


__all__ = ["MirrorCollisionOwner"]
