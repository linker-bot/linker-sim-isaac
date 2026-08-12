"""PhysX CUDA 显存预算的显式冷边界审计。

模块本身不会采样，也不会挂接训练热路径。调用方必须在启动前、warmup 后和稳态
工作负载两端显式调用 :class:`GpuMemoryAuditor`；每个方法只读取一次注入的 probe。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Protocol, runtime_checkable


MIB_BYTES = 1 << 20
AuditPhase = Literal[
    "prelaunch",
    "post_warmup",
    "steady_baseline",
    "steady_final",
]


class GpuMemoryAuditError(RuntimeError):
    """显存审计无法给出可信通过结论。"""


class GpuMemoryProbeError(GpuMemoryAuditError):
    """NVML 或 Torch 采样失败。"""


class GpuMemoryBudgetExceeded(GpuMemoryAuditError):
    """一次有效采样违反了配置预算。"""

    def __init__(self, report: "GpuMemoryAuditReport") -> None:
        self.report = report
        detail = "; ".join(report.violations)
        super().__init__(f"GPU memory budget rejected {report.phase}: {detail}")


@dataclass(frozen=True, slots=True)
class GpuMemoryLimits:
    """从 ``GpuMemoryBudget`` 复制出的不可变审计阈值。"""

    max_simulator_process_mib: int
    min_free_floor_mib: int
    min_free_fraction_after_warmup: float
    max_steady_growth_mib: int

    def __post_init__(self) -> None:
        for name in ("max_simulator_process_mib", "min_free_floor_mib"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer MiB value")
        fraction = self.min_free_fraction_after_warmup
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise TypeError("min_free_fraction_after_warmup must be numeric")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("min_free_fraction_after_warmup must be in (0, 1]")
        if (
            type(self.max_steady_growth_mib) is not int
            or self.max_steady_growth_mib < 0
        ):
            raise ValueError("max_steady_growth_mib must be a non-negative integer")

    @classmethod
    def from_budget(cls, budget: object) -> "GpuMemoryLimits":
        """读取 typed config；字段缺失或类型错误时不采用默认值。"""

        try:
            return cls(
                max_simulator_process_mib=getattr(budget, "max_simulator_process_mib"),
                min_free_floor_mib=getattr(budget, "min_free_floor_mib"),
                min_free_fraction_after_warmup=getattr(
                    budget, "min_free_fraction_after_warmup"
                ),
                max_steady_growth_mib=getattr(budget, "max_steady_growth_mib"),
            )
        except AttributeError as exc:
            raise TypeError(
                "budget must expose the complete GpuMemoryBudget contract"
            ) from exc


@dataclass(frozen=True, slots=True)
class GpuMemorySample:
    """一个设备和进程的原始 byte 计数。

    ``process_used_bytes`` 来自 NVML，因此包含 PhysX、Kit、Torch 和其它原生 CUDA
    allocator；两个 Torch 字段只描述当前进程中的 caching allocator。
    """

    cuda_device: int
    device_uuid: str
    pid: int
    total_bytes: int
    free_bytes: int
    process_used_bytes: int
    process_visible: bool
    torch_allocated_bytes: int
    torch_reserved_bytes: int

    @property
    def device_used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def free_fraction(self) -> float:
        return self.free_bytes / self.total_bytes


@dataclass(frozen=True, slots=True)
class GpuMemoryAuditReport:
    """单次门禁结果；所有对外显存数值都显式标注为 MiB。"""

    phase: AuditPhase
    sample: GpuMemorySample
    limits: GpuMemoryLimits
    process_growth_bytes: int | None
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        """返回适合 JSON marker 的稳定、单位明确报告。"""

        sample = self.sample
        growth_mib = (
            None
            if self.process_growth_bytes is None
            else _mib(self.process_growth_bytes)
        )
        return {
            "phase": self.phase,
            "passed": self.passed,
            "cuda_device": sample.cuda_device,
            "device_uuid": sample.device_uuid,
            "pid": sample.pid,
            "process_visible": sample.process_visible,
            "memory_unit": "MiB",
            "bytes_per_mib": MIB_BYTES,
            "total_mib": _mib(sample.total_bytes),
            "free_mib": _mib(sample.free_bytes),
            "device_used_mib": _mib(sample.device_used_bytes),
            "free_fraction": round(sample.free_fraction, 6),
            "process_used_mib": _mib(sample.process_used_bytes),
            "torch_allocated_mib": _mib(sample.torch_allocated_bytes),
            "torch_reserved_mib": _mib(sample.torch_reserved_bytes),
            "process_growth_mib": growth_mib,
            "limits": {
                "max_simulator_process_mib": (self.limits.max_simulator_process_mib),
                "min_free_floor_mib": self.limits.min_free_floor_mib,
                "min_free_fraction_after_warmup": (
                    self.limits.min_free_fraction_after_warmup
                ),
                "max_steady_growth_mib": self.limits.max_steady_growth_mib,
            },
            "violations": list(self.violations),
        }


@runtime_checkable
class GpuMemoryProbe(Protocol):
    """可替换的单次采样边界；实现不得缓存或启动后台轮询。"""

    def sample(self, *, cuda_device: int, pid: int) -> GpuMemorySample: ...


class CudaNvmlMemoryProbe:
    """使用官方 ``cuda.bindings.nvml`` 和 Torch allocator 计数采样。

    Torch 的逻辑 CUDA device 先通过不可变 UUID 映射到 NVML handle，因此设置
    ``CUDA_VISIBLE_DEVICES`` 时也不会误读另一个物理设备。
    """

    def __init__(
        self,
        *,
        nvml_module: object | None = None,
        torch_module: object | None = None,
    ) -> None:
        self._nvml_module = nvml_module
        self._torch_module = torch_module

    def sample(self, *, cuda_device: int, pid: int) -> GpuMemorySample:
        if type(cuda_device) is not int or cuda_device < 0:
            raise GpuMemoryProbeError("cuda_device must be a non-negative integer")
        if type(pid) is not int or pid <= 0:
            raise GpuMemoryProbeError("pid must be a positive integer")
        try:
            nvml = self._load_nvml()
            torch = self._load_torch()
            cuda = torch.cuda
            if not bool(cuda.is_available()):
                raise RuntimeError("Torch reports that CUDA is unavailable")
            device_count = int(cuda.device_count())
            if cuda_device >= device_count:
                raise RuntimeError(
                    f"selected cuda:{cuda_device} is outside device_count={device_count}"
                )
            torch_uuid = str(cuda.get_device_properties(cuda_device).uuid)
            if not torch_uuid:
                raise RuntimeError("Torch did not expose a CUDA device UUID")
            return self._sample_initialized_nvml(
                nvml=nvml,
                cuda=cuda,
                cuda_device=cuda_device,
                torch_uuid=torch_uuid,
                pid=pid,
            )
        except GpuMemoryProbeError:
            raise
        except Exception as exc:
            raise GpuMemoryProbeError(
                f"failed to sample cuda:{cuda_device} for pid={pid}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _sample_initialized_nvml(
        self,
        *,
        nvml: object,
        cuda: object,
        cuda_device: int,
        torch_uuid: str,
        pid: int,
    ) -> GpuMemorySample:
        initialized_here = False
        primary_error: Exception | None = None
        try:
            try:
                nvml.init_v2()
                initialized_here = True
            except Exception as exc:
                already_initialized = getattr(nvml, "AlreadyInitializedError", None)
                if already_initialized is None or not isinstance(
                    exc, already_initialized
                ):
                    raise

            handle = nvml.device_get_handle_by_uuid(torch_uuid)
            device_uuid = str(nvml.device_get_uuid(handle))
            memory = nvml.device_get_memory_info_v2(handle)
            process_used, process_visible = _current_process_memory(
                nvml=nvml,
                handle=handle,
                pid=pid,
            )
            return GpuMemorySample(
                cuda_device=cuda_device,
                device_uuid=device_uuid,
                pid=pid,
                total_bytes=int(memory.total),
                free_bytes=int(memory.free),
                process_used_bytes=process_used,
                process_visible=process_visible,
                torch_allocated_bytes=int(cuda.memory_allocated(cuda_device)),
                torch_reserved_bytes=int(cuda.memory_reserved(cuda_device)),
            )
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            if initialized_here:
                try:
                    nvml.shutdown()
                except Exception as shutdown_error:
                    if primary_error is not None:
                        primary_error.add_note(
                            "NVML shutdown also failed: "
                            f"{type(shutdown_error).__name__}: {shutdown_error}"
                        )
                    else:
                        raise

    def _load_nvml(self) -> object:
        if self._nvml_module is None:
            from cuda.bindings import nvml

            self._nvml_module = nvml
        return self._nvml_module

    def _load_torch(self) -> object:
        if self._torch_module is None:
            import torch

            self._torch_module = torch
        return self._torch_module


class GpuMemoryAuditor:
    """按固定阶段执行显存门禁，并拒绝缺失或乱序采样。"""

    def __init__(
        self,
        *,
        cuda_device: int,
        limits: GpuMemoryLimits,
        probe: GpuMemoryProbe | None = None,
        pid: int | None = None,
    ) -> None:
        if type(cuda_device) is not int or cuda_device < 0:
            raise ValueError("cuda_device must be a non-negative integer")
        if not isinstance(limits, GpuMemoryLimits):
            raise TypeError("limits must be GpuMemoryLimits")
        selected_pid = os.getpid() if pid is None else pid
        if type(selected_pid) is not int or selected_pid <= 0:
            raise ValueError("pid must be a positive integer")
        selected_probe = CudaNvmlMemoryProbe() if probe is None else probe
        if not isinstance(selected_probe, GpuMemoryProbe):
            raise TypeError("probe must implement GpuMemoryProbe")

        self.cuda_device = cuda_device
        self.pid = selected_pid
        self.limits = limits
        self._probe = selected_probe
        self._state = "created"
        self._reports: list[GpuMemoryAuditReport] = []
        self._device_uuid: str | None = None
        self._total_bytes: int | None = None
        self._steady_baseline: GpuMemoryAuditReport | None = None

    @property
    def reports(self) -> tuple[GpuMemoryAuditReport, ...]:
        return tuple(self._reports)

    def capture_prelaunch(self) -> GpuMemoryAuditReport:
        """启动 Kit 前验证设备可读、显存 floor 和当前进程上限。"""

        self._require_state("created", phase="prelaunch")
        report = self._capture(
            phase="prelaunch",
            require_fraction=False,
            require_process_visible=False,
        )
        self._state = "prelaunch"
        return report

    def capture_post_warmup(self) -> GpuMemoryAuditReport:
        """warmup 后验收 free floor/fraction 与 simulator PID 上限。"""

        self._require_state("prelaunch", phase="post_warmup")
        report = self._capture(
            phase="post_warmup",
            require_fraction=True,
            require_process_visible=True,
        )
        self._state = "post_warmup"
        return report

    def capture_steady_baseline(self) -> GpuMemoryAuditReport:
        """稳态工作负载开始前记录进程级 NVML 基线。"""

        self._require_state("post_warmup", phase="steady_baseline")
        report = self._capture(
            phase="steady_baseline",
            require_fraction=True,
            require_process_visible=True,
        )
        self._steady_baseline = report
        self._state = "steady_baseline"
        return report

    def capture_steady_final(self) -> GpuMemoryAuditReport:
        """稳态工作负载结束后验收容量与进程显存增长。"""

        self._require_state("steady_baseline", phase="steady_final")
        assert self._steady_baseline is not None
        report = self._capture(
            phase="steady_final",
            require_fraction=True,
            require_process_visible=True,
            baseline=self._steady_baseline.sample,
        )
        self._state = "complete"
        return report

    def _capture(
        self,
        *,
        phase: AuditPhase,
        require_fraction: bool,
        require_process_visible: bool,
        baseline: GpuMemorySample | None = None,
    ) -> GpuMemoryAuditReport:
        try:
            sample = self._probe.sample(
                cuda_device=self.cuda_device,
                pid=self.pid,
            )
        except GpuMemoryAuditError:
            raise
        except Exception as exc:
            raise GpuMemoryProbeError(
                f"GPU memory probe failed during {phase}: {type(exc).__name__}: {exc}"
            ) from exc
        _validate_sample(
            sample,
            expected_cuda_device=self.cuda_device,
            expected_pid=self.pid,
        )
        self._validate_device_identity(sample)

        growth = (
            None
            if baseline is None
            else sample.process_used_bytes - baseline.process_used_bytes
        )
        violations = _budget_violations(
            sample=sample,
            limits=self.limits,
            require_fraction=require_fraction,
            require_process_visible=require_process_visible,
            process_growth_bytes=growth,
        )
        report = GpuMemoryAuditReport(
            phase=phase,
            sample=sample,
            limits=self.limits,
            process_growth_bytes=growth,
            violations=violations,
        )
        if violations:
            raise GpuMemoryBudgetExceeded(report)
        self._reports.append(report)
        return report

    def _validate_device_identity(self, sample: GpuMemorySample) -> None:
        if self._device_uuid is None:
            self._device_uuid = sample.device_uuid
            self._total_bytes = sample.total_bytes
            return
        if sample.device_uuid != self._device_uuid:
            raise GpuMemoryProbeError(
                "GPU device UUID changed between audit stages: "
                f"expected={self._device_uuid!r}, actual={sample.device_uuid!r}"
            )
        if sample.total_bytes != self._total_bytes:
            raise GpuMemoryProbeError(
                "GPU total memory changed between audit stages: "
                f"expected={self._total_bytes}, actual={sample.total_bytes} bytes"
            )

    def _require_state(self, expected: str, *, phase: AuditPhase) -> None:
        if self._state != expected:
            raise GpuMemoryAuditError(
                f"cannot capture {phase} while auditor state is {self._state!r}; "
                f"expected {expected!r}"
            )


def _current_process_memory(
    *,
    nvml: object,
    handle: object,
    pid: int,
) -> tuple[int, bool]:
    """合并 compute/graphics 列表；同一 PID 重复出现时不重复计费。"""

    matching: list[int] = []
    unavailable = (1 << 64) - 1
    getters = (
        nvml.device_get_compute_running_processes_v3,
        nvml.device_get_graphics_running_processes_v3,
    )
    for getter in getters:
        for process in getter(handle):
            if int(process.pid) != pid:
                continue
            used = int(process.used_gpu_memory)
            if used == unavailable:
                raise RuntimeError(
                    f"NVML cannot report GPU memory for simulator pid={pid}"
                )
            matching.append(used)
    if not matching:
        return 0, False
    return max(matching), True


def _validate_sample(
    sample: object,
    *,
    expected_cuda_device: int,
    expected_pid: int,
) -> None:
    if not isinstance(sample, GpuMemorySample):
        raise GpuMemoryProbeError("probe must return GpuMemorySample")
    if sample.cuda_device != expected_cuda_device:
        raise GpuMemoryProbeError(
            "probe sampled the wrong CUDA device: "
            f"expected={expected_cuda_device}, actual={sample.cuda_device}"
        )
    if sample.pid != expected_pid:
        raise GpuMemoryProbeError(
            f"probe sampled the wrong pid: expected={expected_pid}, actual={sample.pid}"
        )
    if not isinstance(sample.device_uuid, str) or not sample.device_uuid.strip():
        raise GpuMemoryProbeError("probe returned an empty device UUID")
    if type(sample.process_visible) is not bool:
        raise GpuMemoryProbeError("process_visible must be boolean")
    byte_fields = (
        "total_bytes",
        "free_bytes",
        "process_used_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
    )
    for name in byte_fields:
        value = getattr(sample, name)
        if type(value) is not int or value < 0:
            raise GpuMemoryProbeError(f"{name} must be a non-negative integer")
    if sample.total_bytes <= 0:
        raise GpuMemoryProbeError("total_bytes must be positive")
    if sample.free_bytes > sample.total_bytes:
        raise GpuMemoryProbeError("free_bytes cannot exceed total_bytes")
    if sample.process_used_bytes > sample.total_bytes:
        raise GpuMemoryProbeError("process_used_bytes cannot exceed total_bytes")
    if sample.torch_allocated_bytes > sample.torch_reserved_bytes:
        raise GpuMemoryProbeError(
            "torch_allocated_bytes cannot exceed torch_reserved_bytes"
        )
    if sample.torch_reserved_bytes > sample.process_used_bytes:
        raise GpuMemoryProbeError(
            "Torch reserved memory exceeds NVML process memory; process attribution "
            "is not trustworthy"
        )


def _budget_violations(
    *,
    sample: GpuMemorySample,
    limits: GpuMemoryLimits,
    require_fraction: bool,
    require_process_visible: bool,
    process_growth_bytes: int | None,
) -> tuple[str, ...]:
    violations: list[str] = []
    if sample.free_bytes < limits.min_free_floor_mib * MIB_BYTES:
        violations.append(
            f"free_mib={_mib(sample.free_bytes)} is below "
            f"min_free_floor_mib={limits.min_free_floor_mib}"
        )
    if (
        require_fraction
        and sample.free_fraction < limits.min_free_fraction_after_warmup
    ):
        violations.append(
            f"free_fraction={sample.free_fraction:.6f} is below "
            "min_free_fraction_after_warmup="
            f"{limits.min_free_fraction_after_warmup:.6f}"
        )
    if sample.process_used_bytes > limits.max_simulator_process_mib * MIB_BYTES:
        violations.append(
            f"process_used_mib={_mib(sample.process_used_bytes)} exceeds "
            "max_simulator_process_mib="
            f"{limits.max_simulator_process_mib}"
        )
    if require_process_visible and not sample.process_visible:
        violations.append(
            f"simulator pid={sample.pid} is not visible in NVML process tables"
        )
    if (
        process_growth_bytes is not None
        and process_growth_bytes > limits.max_steady_growth_mib * MIB_BYTES
    ):
        violations.append(
            f"process_growth_mib={_mib(process_growth_bytes)} exceeds "
            f"max_steady_growth_mib={limits.max_steady_growth_mib}"
        )
    return tuple(violations)


def _mib(value_bytes: int) -> float:
    return round(value_bytes / MIB_BYTES, 3)


__all__ = [
    "AuditPhase",
    "CudaNvmlMemoryProbe",
    "GpuMemoryAuditError",
    "GpuMemoryAuditReport",
    "GpuMemoryAuditor",
    "GpuMemoryBudgetExceeded",
    "GpuMemoryLimits",
    "GpuMemoryProbe",
    "GpuMemoryProbeError",
    "GpuMemorySample",
    "MIB_BYTES",
]
