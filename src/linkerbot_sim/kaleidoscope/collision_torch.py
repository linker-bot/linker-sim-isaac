"""Explicit synchronous CUDA-Torch boundary for articulated collision queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.kaleidoscope.collision_catalog import SceneCollisionTargetCatalog
from linkerbot_sim.kaleidoscope.collision_query import (
    ArticulatedCollisionQueryResult,
    CollisionQueryTarget,
)
from linkerbot_sim.kaleidoscope.collision_registry import (
    ArticulatedCollisionServiceRegistry,
)
from linkerbot_sim.utils.tensors import require_cuda_tensor

if TYPE_CHECKING:
    import torch


class TorchArticulatedCollisionQueryAdapter:
    """Bridge explicit CUDA inputs to the synchronous NumPy/PhysX owner path.

    A query intentionally synchronizes selected inputs to CPU because USD and
    the Python PhysX scene-query API are owner-thread operations. No copy or
    synchronization occurs until :meth:`query` is called.
    """

    def __init__(
        self,
        registry: ArticulatedCollisionServiceRegistry,
        *,
        num_envs: int,
        device: object,
        target_catalog: SceneCollisionTargetCatalog,
    ) -> None:
        import torch

        if not isinstance(registry, ArticulatedCollisionServiceRegistry):
            raise TypeError("registry must be ArticulatedCollisionServiceRegistry")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive int")
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("collision query adapter device must be CUDA")
        if not isinstance(target_catalog, SceneCollisionTargetCatalog):
            raise TypeError("target_catalog must be SceneCollisionTargetCatalog")
        if target_catalog.num_envs != num_envs:
            raise ValueError("target catalog num_envs does not match adapter")
        if target_catalog.robot_labels != registry.robot_labels:
            raise ValueError("target catalog robot labels do not match registry")
        self._registry = registry
        self._target_catalog = target_catalog
        self.num_envs = num_envs
        self.device = resolved_device
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def query(
        self,
        *,
        robot_label: str,
        joint_positions: "torch.Tensor",
        env_ids: "torch.Tensor",
        targets: tuple[CollisionQueryTarget, ...],
    ) -> ArticulatedCollisionQueryResult:
        """Synchronously query selected env rows and return owned CUDA tensors."""

        import torch

        self._require_open()
        command_names = self._registry.command_joint_names_for(robot_label)
        _validate_targets(targets)
        positions = require_cuda_tensor(
            joint_positions,
            name="collision joint_positions",
            ndim=3,
            dtype=torch.float32,
        )
        ids = require_cuda_tensor(
            env_ids,
            name="collision env_ids",
            ndim=1,
            dtype=torch.int64,
        )
        if positions.device != self.device or ids.device != self.device:
            raise ValueError(
                f"collision query tensors must live on adapter device {self.device}"
            )
        if positions.shape[0] != ids.numel():
            raise ValueError("joint_positions env rows must match env_ids length")
        if positions.shape[1] < 1:
            raise ValueError("joint_positions sample axis cannot be empty")
        if positions.shape[2] != len(command_names):
            raise ValueError(
                "joint_positions command width does not match robot joint order"
            )

        # This is the one intentional device synchronization in the explicit
        # collision-query slow path. The regular training step never calls it.
        host_positions = _owned_host_numpy(positions, dtype=np.float32)
        host_ids_array = _owned_host_numpy(ids, dtype=np.int64)
        host_ids = tuple(int(value) for value in host_ids_array.tolist())
        if not host_ids:
            raise ValueError("collision env_ids cannot be empty")
        if len(set(host_ids)) != len(host_ids):
            raise ValueError("collision env_ids cannot contain duplicates")
        if any(value < 0 or value >= self.num_envs for value in host_ids):
            raise ValueError("collision env_ids contains an out-of-range index")
        if not np.all(np.isfinite(host_positions)):
            raise ValueError("collision joint_positions contains non-finite values")
        aggregated = self._registry.query(
            robot_label=robot_label,
            joint_positions=host_positions,
            env_ids=host_ids,
            target_catalog=self._target_catalog,
            targets=targets,
        )
        expected_mask_shape = (positions.shape[0], positions.shape[1])
        if aggregated.collision_mask.shape != expected_mask_shape:
            raise RuntimeError(
                "collision service result shape does not match query input: "
                f"{aggregated.collision_mask.shape} != {expected_mask_shape}"
            )
        return ArticulatedCollisionQueryResult(
            collision_mask=torch.as_tensor(
                np.array(aggregated.collision_mask, copy=True),
                device=self.device,
                dtype=torch.bool,
            ).contiguous(),
            first_hit_sample=torch.as_tensor(
                np.array(aggregated.first_hit_sample, copy=True),
                device=self.device,
                dtype=torch.int64,
            ).contiguous(),
            first_hit_category=torch.as_tensor(
                np.array(aggregated.first_hit_category, copy=True),
                device=self.device,
                dtype=torch.int64,
            ).contiguous(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Torch articulated collision query adapter is closed")


def _owned_host_numpy(value: "torch.Tensor", *, dtype: np.dtype) -> np.ndarray:
    host = value.detach().to(device="cpu").contiguous().numpy()
    return np.array(host, dtype=dtype, copy=True, order="C")


def _validate_targets(targets: tuple[CollisionQueryTarget, ...]) -> None:
    if (
        not isinstance(targets, tuple)
        or not targets
        or any(not isinstance(target, CollisionQueryTarget) for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("targets must be a non-empty unique target tuple")


__all__ = ["TorchArticulatedCollisionQueryAdapter"]
