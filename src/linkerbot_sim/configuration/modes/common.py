"""所有产品模式共用的进程级计算设备配置。

mode root 是 CUDA 设备编号的唯一配置 owner。物理、Torch、cuRobo、渲染和训练框架
只能消费根配置派生出的设备，不得在各自 leaf profile 重复声明设备编号。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common import ConfigurationError, as_int, require_keys, strict_mapping


@dataclass(frozen=True)
class ComputeSettings:
    """mode root 持有的唯一 CUDA 设备事实。"""

    cuda_device: int

    def __post_init__(self) -> None:
        if type(self.cuda_device) is not int or self.cuda_device < 0:
            raise ConfigurationError("compute.cuda_device 必须是非负整数")

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "compute"
    ) -> "ComputeSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"cuda_device"}, label=label)
        return cls(
            cuda_device=as_int(
                mapping["cuda_device"], label=f"{label}.cuda_device", minimum=0
            )
        )


__all__ = ["ComputeSettings"]
