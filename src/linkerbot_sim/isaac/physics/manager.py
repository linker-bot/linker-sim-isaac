"""PhysicsRuntime 的进程级唯一 owner 注册表。

PhysX runtime 拥有 Isaac ``World``，Newton runtime 拥有自己的
Model/State/Control/Solver；两者只共享 :class:`PhysicsRuntime` 窄合同。本模块不
制造 World facade，也不为扩展托管的 Newton 制造第三种 owner 形状。

注册表刻意只允许一个活动 owner。项目 Newton 与扩展托管的 physics world 若同时存在，
两者都会尝试推进状态和处理销毁顺序，结果既可能一帧推进两次，也可能在 Kit teardown 时
释放仍被 CUDA kernel 引用的资源；因此这里把“双 owner”视为初始化错误，而不是自动选一个。
"""

from __future__ import annotations

from threading import RLock

from linkerbot_sim.isaac.physics.runtime import PhysicsRuntime


_registry_lock = RLock()
# 这是“进程物理 owner”注册表，不是 manager 缓存。保持强引用能确保 session 构造完成后
# owner 不会提前析构。关闭期间继续保留 identity，可阻止另一个 session 趁 teardown 尚未
# 成功时安装新 owner；只有 close() 正常返回后才从注册表摘除。
_active_manager: PhysicsRuntime | None = None


def install_physics_manager(manager: PhysicsRuntime) -> PhysicsRuntime:
    """安装进程级唯一 manager，并拒绝与已有 manager 形成双重所有权。

    Kit physics extension 的排他性由启动层先行校验；本注册表负责第二道进程内边界，
    防止两个 Newton session（或其它项目 manager）同时成为 API 分派目标。
    """

    if not isinstance(manager, PhysicsRuntime):
        raise TypeError("manager does not implement the PhysicsRuntime protocol")
    global _active_manager
    with _registry_lock:
        if _active_manager is not None and _active_manager is not manager:
            raise RuntimeError(
                "a physics manager is already active: "
                f"backend={_active_manager.backend!r}, "
                f"execution={_active_manager.execution!r}"
            )
        _active_manager = manager
    return manager


def active_physics_manager(*, required: bool = True) -> PhysicsRuntime | None:
    """返回已安装 owner；需要物理能力的路径在 owner 缺失时立即失败。"""

    with _registry_lock:
        manager = _active_manager
    if manager is None and required:
        raise RuntimeError("no physics manager is active")
    return manager


def release_physics_manager(
    manager: PhysicsRuntime | None = None,
    *,
    close: bool = True,
) -> None:
    """只释放调用方期望的 owner，并按需关闭其资源。

    ``close`` 在锁外执行，但 active identity 在成功前保持不变。旧实现先清空 registry，
    close 失败后既丢失 owner，又允许新 session 安装到半关闭 CUDA 进程；现在失败会保留
    owner，下一次精确 release 可重试。manager 参数用于阻止调用方误关其它 session。
    """

    global _active_manager
    with _registry_lock:
        active = _active_manager
        if manager is not None and active is not None and active is not manager:
            raise RuntimeError("refusing to release a different active physics manager")
        if active is None:
            return
        if not close:
            _active_manager = None
            return

    # close 回调可能触发 view 析构并查询 registry，因此不能持有全局锁；同时 registry
    # 保留 active identity，使重入 install 仍然 fail closed。
    active.close()
    with _registry_lock:
        if _active_manager is not active:
            raise RuntimeError("active physics manager changed during close")
        _active_manager = None


__all__ = [
    "active_physics_manager",
    "install_physics_manager",
    "release_physics_manager",
]
