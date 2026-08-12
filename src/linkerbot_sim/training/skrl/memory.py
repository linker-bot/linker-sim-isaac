"""CUDA selector-only 的 skrl rollout memory。"""

from __future__ import annotations

import torch
from skrl.memories.torch import RandomMemory


class CudaRolloutMemory(RandomMemory):
    """绕过 skrl 2.1 的 CPU ``randperm`` 与 NumPy ``array_split``。"""

    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CudaRolloutMemory requires a CUDA device")
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=resolved,
            export=False,
            replacement=False,
        )
        self._index_buffer = torch.empty(
            memory_size * num_envs, device=resolved, dtype=torch.int64
        )

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor]]:
        """用同设备 randperm/tensor_split 生成固定数量 mini-batch。"""

        if sequence_length != 1:
            raise ValueError("Kaleidoscope PPO only supports sequence_length=1")
        size = len(self)
        if batch_size != size:
            raise ValueError("Kaleidoscope PPO samples the complete rollout")
        if mini_batches < 1 or size < mini_batches:
            raise ValueError(
                "mini_batches must be positive and no greater than rollout size"
            )
        if any(name not in self.tensors for name in names):
            missing = [name for name in names if name not in self.tensors]
            raise KeyError(f"rollout memory is missing tensors: {missing}")
        indexes = self._index_buffer[:size]
        torch.randperm(size, device=self.device, out=indexes)
        self.sampling_indexes = indexes
        batches = torch.tensor_split(indexes, mini_batches)
        return [
            [self.tensors_view[name].index_select(0, batch) for name in names]
            for batch in batches
        ]

    def sample_by_index(self, *args: object, **kwargs: object):
        """禁止回退到含 NumPy split 的 stock 路径。"""

        del args, kwargs
        raise RuntimeError(
            "use CudaRolloutMemory.sample; stock sample_by_index is disabled"
        )


__all__ = ["CudaRolloutMemory"]
