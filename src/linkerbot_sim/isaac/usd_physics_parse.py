"""Bound the known OpenUSD 25.11 collider parser race to asset loading."""

from contextlib import contextmanager
import ctypes


def _set_pinned_work_limit(limit: int) -> None:
    """Call the pinned Work implementation when Kit's env lock blocks its wrapper.

    This exported 25.11 function takes one unsigned integer (no C++ objects). Load
    it from the *already imported* Kit extension, never another installed USD.
    No symbols, binaries, environment variables or third-party files are patched.
    """
    from pxr.Work import _work

    library = ctypes.CDLL(_work.__file__)
    setter = getattr(
        library,
        "_ZN35pxrInternal_v0_25_11__pxrReserved__28WorkImpl_SetConcurrencyLimitEj",
    )
    setter.argtypes = [ctypes.c_uint]
    setter.restype = None
    setter(limit)


@contextmanager
def serialized_usd_physics_parse():
    """Serialize the pinned parser, verify the effective limit, and restore it.

    OpenUSD 25.11 _FinalizeCollisionDescs concurrently appends to each rigid
    body's collider vector (upstream fix ed857d77c95b, OpenUSD #4002). Kit sets
    PXR_WORK_THREAD_LIMIT, which makes the public setter ignore its argument.
    The version-pinned implementation bypasses only that wrapper. TBB concurrency
    is limited during synchronous owner-thread loading, never during simulation.
    """
    from pxr import Usd, Work

    if Usd.GetVersion() != (0, 25, 11):
        yield
        return
    previous = Work.GetConcurrencyLimit()
    if previous == 1:
        yield
        return
    Work.SetConcurrencyLimit(1)
    native_override = Work.GetConcurrencyLimit() != 1
    try:
        if native_override:
            _set_pinned_work_limit(1)
        if Work.GetConcurrencyLimit() != 1:
            raise RuntimeError("cannot serialize the OpenUSD 25.11 physics parser")
        yield
    finally:
        if native_override:
            _set_pinned_work_limit(previous)
        else:
            Work.SetConcurrencyLimit(previous)
        if Work.GetConcurrencyLimit() != previous:
            raise RuntimeError("failed to restore USD Work concurrency after parsing")
