from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.gpu_memory_audit import GpuMemorySample, MIB_BYTES
from scripts import smoke_physx_gpu_memory_budget as smoke


_GIB = 1024 * MIB_BYTES
_PID = 5151
_UUID = "GPU-smoke"


class _Probe:
    def __init__(self, samples: list[GpuMemorySample]) -> None:
        self.samples = samples
        self.calls = 0

    def sample(self, *, cuda_device: int, pid: int) -> GpuMemorySample:
        assert cuda_device == 0
        assert pid == _PID
        self.calls += 1
        return self.samples.pop(0)


class _Env:
    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        self.device = "cuda:0"
        self.num_envs = 2
        self.action_dim = 3
        self.reset_seeds: list[int] = []
        self.close_codes: list[int] = []
        self.events: list[str] = []
        self._capsys = capsys

    def reset(self, *, seed: int) -> tuple[object, dict[str, object]]:
        self.reset_seeds.append(seed)
        return object(), {}

    def close(self, *, exit_code: int) -> None:
        self.events.extend(self._capsys.readouterr().out.splitlines())
        self.close_codes.append(exit_code)
        self.events.append(f"CLOSED:{exit_code}")


def _sample(
    *,
    process_mib: int,
    free_mib: int = 20 * 1024,
    visible: bool = True,
) -> GpuMemorySample:
    torch_reserved = min(process_mib, 1024) * MIB_BYTES
    return GpuMemorySample(
        cuda_device=0,
        device_uuid=_UUID,
        pid=_PID,
        total_bytes=32 * _GIB,
        free_bytes=free_mib * MIB_BYTES,
        process_used_bytes=process_mib * MIB_BYTES,
        process_visible=visible,
        torch_allocated_bytes=torch_reserved // 2,
        torch_reserved_bytes=torch_reserved,
    )


def _config(*, engine: str = "physx", execution: str = "cuda") -> object:
    memory = SimpleNamespace(
        max_simulator_process_mib=16 * 1024,
        min_free_floor_mib=4 * 1024,
        min_free_fraction_after_warmup=0.20,
        max_steady_growth_mib=128,
    )
    return SimpleNamespace(
        cuda_device=0,
        physics=SimpleNamespace(engine=engine, execution=execution, memory=memory),
    )


def _passing_samples() -> list[GpuMemorySample]:
    return [
        _sample(process_mib=0, free_mib=28 * 1024, visible=False),
        _sample(process_mib=8 * 1024),
        _sample(process_mib=8 * 1024),
        _sample(process_mib=8 * 1024 + 64),
    ]


def test_smoke_marker_is_flushed_before_fast_shutdown_close(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _Env(capsys)
    probe = _Probe(_passing_samples())
    workloads: list[int] = []
    synchronizations: list[int] = []
    monkeypatch.setattr(smoke, "load_kaleidoscope_config", lambda _profile: _config())
    monkeypatch.setattr(smoke, "make_torch_env", lambda **_kwargs: env)
    monkeypatch.setattr(
        smoke,
        "_run_zero_action_steps",
        lambda _env, *, steps: workloads.append(steps),
    )
    monkeypatch.setattr(
        smoke,
        "_synchronize_cuda",
        lambda *, cuda_device: synchronizations.append(cuda_device),
    )

    result = smoke.run_smoke(
        profile="physx_cuda",
        num_envs=2,
        warmup_steps=3,
        steady_steps=5,
        probe=probe,
        pid=_PID,
    )

    assert probe.calls == 4
    assert workloads == [3, 5]
    assert synchronizations == [0, 0]
    assert env.reset_seeds == [123]
    assert env.close_codes == [0]
    assert env.events[0].startswith(smoke.SUCCESS_MARKER + " ")
    assert env.events[1] == "CLOSED:0"
    assert [entry["phase"] for entry in result["audits"]] == [
        "prelaunch",
        "post_warmup",
        "steady_baseline",
        "steady_final",
    ]


def test_smoke_budget_failure_closes_with_nonzero_and_has_no_success_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _Env(capsys)
    failing = _passing_samples()
    failing[1] = _sample(process_mib=8 * 1024, free_mib=1024)
    monkeypatch.setattr(smoke, "load_kaleidoscope_config", lambda _profile: _config())
    monkeypatch.setattr(smoke, "make_torch_env", lambda **_kwargs: env)
    monkeypatch.setattr(smoke, "_run_zero_action_steps", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(smoke, "_synchronize_cuda", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="GPU memory budget rejected post_warmup"):
        smoke.run_smoke(
            profile="physx_cuda",
            num_envs=2,
            warmup_steps=1,
            steady_steps=1,
            probe=_Probe(failing),
            pid=_PID,
        )

    assert env.close_codes == [1]
    assert env.events == ["CLOSED:1"]


def test_smoke_rejects_non_physx_profile_before_environment_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    monkeypatch.setattr(
        smoke,
        "load_kaleidoscope_config",
        lambda _profile: _config(engine="newton"),
    )
    monkeypatch.setattr(
        smoke,
        "make_torch_env",
        lambda **_kwargs: created.append(object()),
    )

    with pytest.raises(RuntimeError, match="requires.*PhysX CUDA"):
        smoke.run_smoke(
            profile="newton_cuda",
            num_envs=2,
            warmup_steps=1,
            steady_steps=1,
            probe=_Probe(_passing_samples()),
            pid=_PID,
        )

    assert created == []


def test_cli_rejects_nonpositive_workload() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--num-envs", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--warmup-steps", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--steady-steps", "0"])


def test_zero_action_workload_uses_same_step_across_episode_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions = object()
    # 这里只验证 smoke 的协议编排，不依赖 tensor 算法或 CUDA。向函数内的惰性 import
    # 注入最小替身，让该门禁在不安装 simulation/Torch extra 的纯开发环境也能执行。
    fake_torch = SimpleNamespace(
        float32=object(),
        zeros=lambda *_args, **_kwargs: actions,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _SameStepEnv:
        num_envs = 2
        action_dim = 3
        device = "cuda:0"

        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []
            self.generation = 0

        def begin_same_step(self) -> object:
            token = f"token-{self.generation}"
            self.events.append(("begin", token))
            return token

        def step_same_step(self, token: object, value: object) -> tuple[object, ...]:
            assert value is actions
            self.events.append(("step", token))
            # helper 不读取 done tensor，也不构造变长 env-id；reset 由 runtime 的
            # 固定 N 行 device mask 在 complete_same_step 内完成。
            return (object(), object(), object(), object(), {})

        def complete_same_step(self, token: object) -> object:
            self.events.append(("complete", token))
            self.generation += 1
            return object()

    env = _SameStepEnv()
    smoke._run_zero_action_steps(env, steps=2)

    assert env.events == [
        ("begin", "token-0"),
        ("step", "token-0"),
        ("complete", "token-0"),
        ("begin", "token-1"),
        ("step", "token-1"),
        ("complete", "token-1"),
    ]
