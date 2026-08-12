"""Kaleidoscope 的进程内 GPU 状态、快照和克隆 API。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from linkerbot_sim.controllers.control_mode import (
    ControlModeState,
    require_control_mode,
)
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.snapshot import KaleidoscopeEpisodeSnapshot
from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    normalize_env_ids,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch

StateWriter = Callable[["torch.Tensor", "torch.Tensor"], None]


@dataclass(frozen=True, slots=True)
class StateBinding:
    """一个 canonical 状态字段及其可选 engine write-through 回调。

    ``tensor`` 始终是 runtime-owned 的完整 ``(num_envs, ...)`` buffer。writer 接收 env ids 和
    已经完成结构/有限性预检的 K 行值；writer 不得保留输入 alias。
    """

    tensor: "torch.Tensor"
    writer: StateWriter | None = None
    finite: bool = True
    cloneable: bool = True


class KaleidoscopeStateAPI:
    """对一组 GPU canonical buffers 提供事务式批量状态操作。

    所有字段先完整预检，再执行 writer 和 canonical ``index_copy_``。如果任一 engine writer
    抛错，实例进入 fail-stop 状态；继续物理推进可能让 engine 与 canonical buffer 分叉，因此必须
    关闭并重建 runtime，不能用剩余 writer 猜测性“补偿”。
    """

    def __init__(
        self,
        bindings: Mapping[str, StateBinding],
        *,
        num_envs: int,
        rng_fields: Sequence[str] = (),
        compatibility_fingerprint: str = "unspecified",
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive int")
        if not bindings:
            raise ValueError("state bindings cannot be empty")
        normalized: dict[str, StateBinding] = {}
        for name, binding in bindings.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("state binding names must be non-empty strings")
            if name in normalized:
                raise ValueError(f"duplicate state binding {name!r}")
            require_cuda_tensor(
                binding.tensor,
                name=f"state[{name!r}]",
                leading_dim=num_envs,
            )
            normalized[name] = binding
        self.device = require_common_cuda_device(
            (binding.tensor for binding in normalized.values()),
            label="state bindings",
        )
        unknown_rng = set(rng_fields) - set(normalized)
        if unknown_rng:
            raise ValueError(f"unknown RNG state fields: {sorted(unknown_rng)}")
        self.num_envs = num_envs
        self._bindings = normalized
        self._rng_fields = frozenset(rng_fields)
        if (
            not isinstance(compatibility_fingerprint, str)
            or not compatibility_fingerprint.strip()
        ):
            raise ValueError("compatibility_fingerprint cannot be empty")

        def schema_fingerprint(*, include_position_reference: bool) -> str:
            schema_contract = {
                "configuration": compatibility_fingerprint,
                "fields": [
                    {
                        "name": name,
                        "shape": list(binding.tensor.shape[1:]),
                        "dtype": str(binding.tensor.dtype),
                        "finite": binding.finite,
                        "cloneable": binding.cloneable,
                        "engine_owned": binding.writer is not None,
                    }
                    for name, binding in normalized.items()
                    if include_position_reference or name != "robot.position_reference"
                ],
            }
            return hashlib.sha256(
                json.dumps(
                    schema_contract,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest()

        self.compatibility_fingerprint = schema_fingerprint(
            include_position_reference=True
        )
        self._schema1_compatibility_fingerprint = schema_fingerprint(
            include_position_reference=False
        )
        self._poisoned = False
        self._control_mode_provider: Callable[[], object] = lambda: ControlModeState(
            initial_mode="position",
            active_mode="position",
            generation=0,
            supported_modes=("position",),
        )
        self._control_mode_provider_bound = False

    def bind_control_mode_provider(self, provider: Callable[[], object]) -> None:
        if not callable(provider):
            raise TypeError("control mode provider must be callable")
        if self._control_mode_provider_bound:
            raise RuntimeError("state API control mode provider is already bound")
        self._mode_state(provider())
        self._control_mode_provider = provider
        self._control_mode_provider_bound = True

    @property
    def poisoned(self) -> bool:
        """writer 失败后返回 True；poisoned runtime 禁止继续读写。"""

        return self._poisoned

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    def get_state(
        self,
        env_ids: "torch.Tensor | None" = None,
        *,
        fields: Sequence[str] | None = None,
        clone: bool = True,
    ) -> dict[str, "torch.Tensor"]:
        """读取所选行；默认返回 owned tensor，显式 ``clone=False`` 才返回 view/index 结果。"""

        self._require_healthy()
        ids = self._ids(env_ids)
        names = self._field_names(fields)
        result: dict[str, torch.Tensor] = {}
        full_selection = env_ids is None
        for name in names:
            source = self._bindings[name].tensor
            if full_selection:
                result[name] = source.clone() if clone else source
            else:
                # index_select 已返回 owned tensor；clone=True 不需要再复制一次。
                result[name] = source.index_select(0, ids)
        return result

    def set_state(
        self,
        state: Mapping[str, "torch.Tensor"],
        env_ids: "torch.Tensor | None" = None,
    ) -> None:
        """事务式写入所选字段；unknown/missing shape 在任何 engine 写入前失败。"""

        self._require_healthy()
        ids = self._ids(env_ids)
        prepared = self._preflight(state, ids)
        try:
            # 先调用 engine writer，再提交 canonical buffer。若 writer 失败，canonical 不伪装成
            # 成功状态；runtime 标记 poisoned 并要求重建。
            for name, value in prepared.items():
                writer = self._bindings[name].writer
                if writer is not None:
                    writer(ids, value)
            for name, value in prepared.items():
                self._bindings[name].tensor.index_copy_(0, ids, value)
        except BaseException:
            self._poisoned = True
            raise

    def snapshot(
        self,
        env_ids: "torch.Tensor | None" = None,
        *,
        fields: Sequence[str] | None = None,
    ) -> KaleidoscopeEpisodeSnapshot:
        """捕获 owned CUDA snapshot；不经过 NumPy/CPU serialization。"""

        ids = self._ids(env_ids)
        selected = self.get_state(ids, fields=fields, clone=True)
        active_mode, generation = self._mode_state(self._control_mode_provider())
        return KaleidoscopeEpisodeSnapshot(
            env_ids=ids.clone(),
            fields=selected,
            compatibility_fingerprint=self.compatibility_fingerprint,
            control_mode=active_mode,
            control_generation=generation,
        )

    def restore_snapshot(
        self,
        snapshot: KaleidoscopeEpisodeSnapshot,
        *,
        target_env_ids: "torch.Tensor | None" = None,
    ) -> None:
        """把快照恢复到原 env 或等宽 target selector。"""

        self._require_healthy()
        if not isinstance(snapshot, KaleidoscopeEpisodeSnapshot):
            raise TypeError("snapshot must be a KaleidoscopeEpisodeSnapshot")
        active_mode, _generation = self._mode_state(self._control_mode_provider())
        snapshot_mode: ControlMode = (
            "position"
            if snapshot.schema_version == 1
            else require_control_mode(
                snapshot.control_mode,
                label="snapshot.control_mode",
            )
        )
        if snapshot_mode != active_mode:
            raise ValueError(
                f"snapshot control mode {snapshot_mode!r} does not match "
                f"runtime mode {active_mode!r}"
            )
        compatible_fingerprints = {self.compatibility_fingerprint}
        if snapshot.schema_version == 1:
            compatible_fingerprints.add(self._schema1_compatibility_fingerprint)
        if snapshot.compatibility_fingerprint not in compatible_fingerprints:
            raise ValueError(
                "snapshot is incompatible with this Kaleidoscope state/configuration "
                "fingerprint"
            )
        if snapshot.device != self.device:
            raise ValueError(
                f"snapshot must live on {self.device}, got {snapshot.device}"
            )
        targets = (
            snapshot.env_ids if target_env_ids is None else self._ids(target_env_ids)
        )
        if targets.numel() != snapshot.count:
            raise ValueError("target_env_ids length must match snapshot count")
        fields = dict(snapshot.fields)
        if snapshot.schema_version == 1 and "robot.position_reference" not in fields:
            target = fields.get("robot.target")
            if target is not None and "robot.position_reference" in self._bindings:
                fields["robot.position_reference"] = target
        self.set_state(fields, targets)

    def clone_state(
        self,
        source_env_ids: "torch.Tensor",
        target_env_ids: "torch.Tensor",
        *,
        include_rng: bool = True,
        fields: Sequence[str] | None = None,
    ) -> None:
        """在 GPU 内把 K 个 source env 克隆到 K 个 target env。

        source/target 不允许重叠。先 clone 全部 source 行，再进入 writer 阶段，保证多个字段和
        engine setter 看见同一逻辑时刻。``include_rng`` 默认复制 logical RNG key/counter，确保
        克隆后的首次 rollout 可复现；关闭时明确排除登记的 RNG 字段。
        """

        self._require_healthy()
        sources = self._ids(source_env_ids)
        targets = self._ids(target_env_ids)
        if sources.numel() != targets.numel():
            raise ValueError("source_env_ids and target_env_ids must have equal length")
        import torch

        overlap = torch.isin(sources, targets)
        torch._assert_async(
            torch.all(~overlap),
            "source_env_ids and target_env_ids cannot overlap",
        )
        names = self._field_names(fields)
        if not include_rng:
            names = tuple(name for name in names if name not in self._rng_fields)
        names = tuple(name for name in names if self._bindings[name].cloneable)
        payload = {
            # index_select 先完整物化所有 source 行，随后 writer 即使改写 target 也不会
            # 影响其它字段或尚未提交的 source payload。
            name: self._bindings[name].tensor.index_select(0, sources)
            for name in names
        }
        self.set_state(payload, targets)

    def _ids(self, env_ids: "torch.Tensor | None") -> "torch.Tensor":
        return normalize_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _field_names(self, fields: Sequence[str] | None) -> tuple[str, ...]:
        if fields is None:
            return tuple(self._bindings)
        names = tuple(fields)
        if not names:
            raise ValueError("fields cannot be empty")
        if len(set(names)) != len(names):
            raise ValueError("fields cannot contain duplicates")
        unknown = set(names) - set(self._bindings)
        if unknown:
            raise KeyError(f"unknown state fields: {sorted(unknown)}")
        return names

    def _preflight(
        self,
        state: Mapping[str, "torch.Tensor"],
        ids: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        import torch

        if not state:
            raise ValueError("state payload cannot be empty")
        unknown = set(state) - set(self._bindings)
        if unknown:
            raise KeyError(f"unknown state fields: {sorted(unknown)}")
        prepared: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            binding = self._bindings[name]
            tensor = require_cuda_tensor(
                value,
                name=f"state payload {name!r}",
                leading_dim=ids.numel(),
                dtype=binding.tensor.dtype,
            )
            if tensor.device != self.device:
                raise ValueError(
                    f"state payload {name!r} must live on {self.device}, got {tensor.device}"
                )
            if tensor.shape[1:] != binding.tensor.shape[1:]:
                raise ValueError(
                    f"state payload {name!r} must have trailing shape "
                    f"{tuple(binding.tensor.shape[1:])}, got {tuple(tensor.shape[1:])}"
                )
            if binding.finite:
                assert_finite_async(tensor, name=f"state payload {name!r}")
            # writer 可能异步消费输入；contiguous owned copy 避免调用方在提交后改写 storage。
            prepared[name] = tensor.clone(memory_format=torch.contiguous_format)
        active_mode, _generation = self._mode_state(self._control_mode_provider())
        if active_mode == "position":
            target = prepared.get("robot.target")
            reference = prepared.get("robot.position_reference")
            if target is not None and reference is not None:
                if not torch.equal(target, reference):
                    raise ValueError(
                        "position-mode robot.target and robot.position_reference "
                        "must match"
                    )
            elif target is not None and "robot.position_reference" in self._bindings:
                prepared["robot.position_reference"] = target.clone()
            elif reference is not None and "robot.target" in self._bindings:
                prepared["robot.target"] = reference.clone()
        return prepared

    @staticmethod
    def _mode_state(value: object) -> tuple[ControlMode, int]:
        mode = require_control_mode(
            getattr(value, "active_mode", None),
            label="runtime active control mode",
        )
        generation = getattr(value, "generation", None)
        if type(generation) is not int or generation < 0:
            raise ValueError("runtime control generation must be a non-negative int")
        return mode, generation

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "Kaleidoscope state API is poisoned after a failed engine write; "
                "close and recreate the runtime"
            )


__all__ = ["KaleidoscopeStateAPI", "StateBinding", "StateWriter"]
