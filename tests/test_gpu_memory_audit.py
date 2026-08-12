from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.gpu_memory_audit import (
    CudaNvmlMemoryProbe,
    GpuMemoryAuditError,
    GpuMemoryAuditor,
    GpuMemoryBudgetExceeded,
    GpuMemoryLimits,
    GpuMemoryProbeError,
    GpuMemorySample,
    MIB_BYTES,
)


_GIB = 1024 * MIB_BYTES
_PID = 4242
_UUID = "GPU-00000000-1111-2222-3333-444444444444"
_LIMITS = GpuMemoryLimits(
    max_simulator_process_mib=16 * 1024,
    min_free_floor_mib=4 * 1024,
    min_free_fraction_after_warmup=0.20,
    max_steady_growth_mib=128,
)


def _sample(
    *,
    free_bytes: int = 20 * _GIB,
    process_used_bytes: int = 8 * _GIB,
    process_visible: bool = True,
    torch_allocated_bytes: int = 2 * _GIB,
    torch_reserved_bytes: int = 3 * _GIB,
) -> GpuMemorySample:
    return GpuMemorySample(
        cuda_device=0,
        device_uuid=_UUID,
        pid=_PID,
        total_bytes=32 * _GIB,
        free_bytes=free_bytes,
        process_used_bytes=process_used_bytes,
        process_visible=process_visible,
        torch_allocated_bytes=torch_allocated_bytes,
        torch_reserved_bytes=torch_reserved_bytes,
    )


class _SequenceProbe:
    def __init__(self, *results: GpuMemorySample | Exception) -> None:
        self.results = list(results)
        self.calls: list[tuple[int, int]] = []

    def sample(self, *, cuda_device: int, pid: int) -> GpuMemorySample:
        self.calls.append((cuda_device, pid))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _prelaunch_sample() -> GpuMemorySample:
    return _sample(
        free_bytes=28 * _GIB,
        process_used_bytes=0,
        process_visible=False,
        torch_allocated_bytes=0,
        torch_reserved_bytes=0,
    )


def test_explicit_lifecycle_reports_all_values_in_mib() -> None:
    final = _sample(process_used_bytes=8 * _GIB + 64 * MIB_BYTES)
    probe = _SequenceProbe(
        _prelaunch_sample(),
        _sample(),
        _sample(),
        final,
    )
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=probe,
        pid=_PID,
    )

    prelaunch = auditor.capture_prelaunch()
    post_warmup = auditor.capture_post_warmup()
    baseline = auditor.capture_steady_baseline()
    steady_final = auditor.capture_steady_final()

    assert [report.phase for report in auditor.reports] == [
        "prelaunch",
        "post_warmup",
        "steady_baseline",
        "steady_final",
    ]
    assert probe.calls == [(0, _PID)] * 4
    assert prelaunch.passed and post_warmup.passed and baseline.passed
    report = steady_final.as_dict()
    assert report["memory_unit"] == "MiB"
    assert report["bytes_per_mib"] == 1_048_576
    assert report["total_mib"] == 32 * 1024
    assert report["process_used_mib"] == 8 * 1024 + 64
    assert report["torch_allocated_mib"] == 2 * 1024
    assert report["torch_reserved_mib"] == 3 * 1024
    assert report["process_growth_mib"] == 64
    assert report["violations"] == []


def test_prelaunch_does_not_apply_after_warmup_fraction() -> None:
    # 5 GiB 高于 floor，但在 32 GiB 设备上低于 warmup 后要求的 20%。
    probe = _SequenceProbe(
        _sample(
            free_bytes=5 * _GIB,
            process_used_bytes=0,
            process_visible=False,
            torch_allocated_bytes=0,
            torch_reserved_bytes=0,
        ),
        _sample(free_bytes=5 * _GIB),
    )
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=probe,
        pid=_PID,
    )

    assert auditor.capture_prelaunch().passed
    with pytest.raises(GpuMemoryBudgetExceeded) as captured:
        auditor.capture_post_warmup()

    assert captured.value.report.phase == "post_warmup"
    assert captured.value.report.as_dict()["free_fraction"] == 0.15625
    assert any(
        "min_free_fraction_after_warmup" in item
        for item in captured.value.report.violations
    )


def test_post_warmup_reports_floor_fraction_and_process_max_together() -> None:
    probe = _SequenceProbe(
        _prelaunch_sample(),
        _sample(
            free_bytes=3 * _GIB,
            process_used_bytes=17 * _GIB,
            torch_reserved_bytes=3 * _GIB,
        ),
    )
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=probe,
        pid=_PID,
    )
    auditor.capture_prelaunch()

    with pytest.raises(GpuMemoryBudgetExceeded) as captured:
        auditor.capture_post_warmup()

    violations = captured.value.report.violations
    assert len(violations) == 3
    assert any("free_mib=3072.0" in item for item in violations)
    assert any("free_fraction=" in item for item in violations)
    assert any("process_used_mib=17408.0" in item for item in violations)


def test_steady_growth_uses_whole_process_nvml_memory() -> None:
    probe = _SequenceProbe(
        _prelaunch_sample(),
        _sample(),
        _sample(process_used_bytes=8 * _GIB),
        _sample(process_used_bytes=8 * _GIB + 129 * MIB_BYTES),
    )
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=probe,
        pid=_PID,
    )
    auditor.capture_prelaunch()
    auditor.capture_post_warmup()
    auditor.capture_steady_baseline()

    with pytest.raises(GpuMemoryBudgetExceeded) as captured:
        auditor.capture_steady_final()

    report = captured.value.report
    assert report.process_growth_bytes == 129 * MIB_BYTES
    assert report.as_dict()["process_growth_mib"] == 129
    assert report.violations == (
        "process_growth_mib=129.0 exceeds max_steady_growth_mib=128",
    )


def test_stage_order_and_probe_failures_fail_closed() -> None:
    probe = _SequenceProbe(RuntimeError("NVML unavailable"))
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=probe,
        pid=_PID,
    )

    with pytest.raises(GpuMemoryAuditError, match="expected 'prelaunch'"):
        auditor.capture_post_warmup()
    assert probe.calls == []

    with pytest.raises(GpuMemoryProbeError, match="NVML unavailable"):
        auditor.capture_prelaunch()
    assert probe.calls == [(0, _PID)]
    assert auditor.reports == ()


@pytest.mark.parametrize(
    ("changed", "message"),
    (
        ({"pid": _PID + 1}, "wrong pid"),
        ({"cuda_device": 1}, "wrong CUDA device"),
        ({"free_bytes": 33 * _GIB}, "free_bytes cannot exceed"),
        ({"torch_reserved_bytes": 9 * _GIB}, "Torch reserved memory exceeds"),
    ),
)
def test_invalid_probe_samples_fail_closed(
    changed: dict[str, object],
    message: str,
) -> None:
    sample = replace(_prelaunch_sample(), **changed)
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=_SequenceProbe(sample),
        pid=_PID,
    )

    with pytest.raises(GpuMemoryProbeError, match=message):
        auditor.capture_prelaunch()


def test_device_identity_cannot_change_after_launch() -> None:
    changed_device = replace(_sample(), device_uuid="GPU-different")
    auditor = GpuMemoryAuditor(
        cuda_device=0,
        limits=_LIMITS,
        probe=_SequenceProbe(_prelaunch_sample(), changed_device),
        pid=_PID,
    )
    auditor.capture_prelaunch()

    with pytest.raises(GpuMemoryProbeError, match="UUID changed"):
        auditor.capture_post_warmup()


class _FakeNvml:
    class AlreadyInitializedError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.events: list[object] = []

    def init_v2(self) -> None:
        self.events.append("init")

    def shutdown(self) -> None:
        self.events.append("shutdown")

    def device_get_handle_by_uuid(self, value: str) -> str:
        self.events.append(("uuid_lookup", value))
        return "handle"

    def device_get_uuid(self, handle: str) -> str:
        assert handle == "handle"
        return _UUID

    def device_get_memory_info_v2(self, handle: str) -> object:
        assert handle == "handle"
        return SimpleNamespace(total=32 * _GIB, free=20 * _GIB)

    def device_get_compute_running_processes_v3(self, handle: str) -> list[object]:
        assert handle == "handle"
        return [
            SimpleNamespace(pid=_PID, used_gpu_memory=8 * _GIB),
            SimpleNamespace(pid=99, used_gpu_memory=1 * _GIB),
        ]

    def device_get_graphics_running_processes_v3(self, handle: str) -> list[object]:
        assert handle == "handle"
        return [SimpleNamespace(pid=_PID, used_gpu_memory=7 * _GIB)]


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_properties(index: int) -> object:
        assert index == 0
        return SimpleNamespace(uuid="00000000-1111-2222-3333-444444444444")

    @staticmethod
    def memory_allocated(index: int) -> int:
        assert index == 0
        return 2 * _GIB

    @staticmethod
    def memory_reserved(index: int) -> int:
        assert index == 0
        return 3 * _GIB


def test_cuda_nvml_probe_maps_uuid_and_deduplicates_process_tables() -> None:
    nvml = _FakeNvml()
    probe = CudaNvmlMemoryProbe(
        nvml_module=nvml,
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
    )

    sample = probe.sample(cuda_device=0, pid=_PID)

    assert sample.device_uuid == _UUID
    assert sample.process_visible is True
    assert sample.process_used_bytes == 8 * _GIB
    assert sample.torch_allocated_bytes == 2 * _GIB
    assert sample.torch_reserved_bytes == 3 * _GIB
    assert nvml.events == [
        "init",
        ("uuid_lookup", "00000000-1111-2222-3333-444444444444"),
        "shutdown",
    ]


def test_cuda_nvml_probe_rejects_unavailable_process_memory() -> None:
    nvml = _FakeNvml()
    nvml.device_get_compute_running_processes_v3 = lambda _handle: [
        SimpleNamespace(pid=_PID, used_gpu_memory=(1 << 64) - 1)
    ]
    probe = CudaNvmlMemoryProbe(
        nvml_module=nvml,
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
    )

    with pytest.raises(GpuMemoryProbeError, match="cannot report GPU memory"):
        probe.sample(cuda_device=0, pid=_PID)

    assert nvml.events[-1] == "shutdown"
