"""Construction-time CUDA projection of one robot controller profile."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from linkerbot_sim.controllers.runtime_projection import CommandRuntimeProjection
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.tensors import require_cuda_tensor


@dataclass(frozen=True, slots=True)
class PreparedKaleidoscopeControlRuntime:
    mode: ControlMode
    projection: CommandRuntimeProjection
    stiffness: torch.Tensor
    damping: torch.Tensor
    effort_limits: torch.Tensor
    drive_stiffness: torch.Tensor
    drive_damping: torch.Tensor
    implicit_indices: torch.Tensor
    explicit_indices: torch.Tensor

    def __post_init__(self) -> None:
        width = len(self.projection.joint_names)
        tensors = (
            self.stiffness,
            self.damping,
            self.effort_limits,
            self.drive_stiffness,
            self.drive_damping,
        )
        for value in tensors:
            tensor = require_cuda_tensor(
                value,
                name="prepared control parameter",
                ndim=1,
                dtype=torch.float32,
            )
            if tensor.shape != (width,):
                raise ValueError("prepared control parameter has the wrong width")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("prepared control parameters must share one CUDA device")
        for name in ("implicit_indices", "explicit_indices"):
            indices = require_cuda_tensor(
                getattr(self, name),
                name=f"prepared {name}",
                ndim=1,
                dtype=torch.int64,
            )
            if indices.device != tensors[0].device:
                raise ValueError(
                    "prepared control selectors must share parameter device"
                )

    @property
    def device(self) -> torch.device:
        return self.stiffness.device


def prepare_device_control_runtime(
    projection: CommandRuntimeProjection,
    *,
    mode: ControlMode,
    device: torch.device,
) -> PreparedKaleidoscopeControlRuntime:
    """Upload immutable projection arrays and selectors before runtime switching."""

    if tuple(projection.modes) != (mode,) * len(projection.joint_names):
        raise ValueError("controller projection modes must match requested global mode")

    def values(source: object) -> torch.Tensor:
        return torch.tensor(
            [float(value) for value in source],
            device=device,
            dtype=torch.float32,
        ).contiguous()

    implicit = torch.tensor(
        [
            index
            for index, method in enumerate(projection.methods)
            if method == "implicit"
        ],
        device=device,
        dtype=torch.int64,
    )
    explicit = torch.tensor(
        [
            index
            for index, method in enumerate(projection.methods)
            if method != "implicit"
        ],
        device=device,
        dtype=torch.int64,
    )
    return PreparedKaleidoscopeControlRuntime(
        mode=mode,
        projection=projection,
        stiffness=values(projection.stiffness),
        damping=values(projection.damping),
        effort_limits=values(projection.effort_limits),
        drive_stiffness=values(projection.drive_stiffness),
        drive_damping=values(projection.drive_damping),
        implicit_indices=implicit,
        explicit_indices=explicit,
    )


__all__ = [
    "PreparedKaleidoscopeControlRuntime",
    "prepare_device_control_runtime",
]
