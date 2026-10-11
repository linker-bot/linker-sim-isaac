from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.kaleidoscope.collision_catalog import (
    SceneCollisionTargetCatalog,
    SceneCollisionTargetShape,
)
from linkerbot_sim.kaleidoscope.collision_hits import AggregatedCollisionQueryHits
from linkerbot_sim.kaleidoscope.collision_query import (
    ArticulatedCollisionQueryPort,
    CollisionHitCategory,
    CollisionQueryTarget,
)
from linkerbot_sim.kaleidoscope.collision_registry import (
    ArticulatedCollisionServiceRegistry,
)
from linkerbot_sim.kaleidoscope.collision_torch import (
    TorchArticulatedCollisionQueryAdapter,
)


class _Device:
    def __init__(self, value: object) -> None:
        self.name = value.name if isinstance(value, _Device) else str(value)
        self.type = self.name.split(":", maxsplit=1)[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Device) and self.name == other.name

    def __str__(self) -> str:
        return self.name


class _Tensor:
    def __init__(self, value: object, *, dtype: str, device: object) -> None:
        numpy_dtype = {
            "bool": np.bool_,
            "float32": np.float32,
            "int64": np.int64,
        }[dtype]
        self.array = np.asarray(value, dtype=numpy_dtype)
        self.dtype = dtype
        self.device = _Device(device)

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def numel(self) -> int:
        return int(self.array.size)

    def detach(self) -> _Tensor:
        return self

    def to(self, *, device: object) -> _Tensor:
        return _Tensor(self.array.copy(), dtype=self.dtype, device=device)

    def contiguous(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def tolist(self) -> list[object]:
        return self.array.tolist()


def _fake_torch() -> object:
    return SimpleNamespace(
        Tensor=_Tensor,
        bool="bool",
        device=_Device,
        float32="float32",
        int64="int64",
        as_tensor=lambda value, *, device, dtype: _Tensor(
            value,
            dtype=dtype,
            device=device,
        ),
    )


class _Service:
    def __init__(self, robot_label: str) -> None:
        self.robot_label = robot_label
        self.command_joint_names = ("joint_0", "joint_1")
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    def query(self, **kwargs: object) -> AggregatedCollisionQueryHits:
        self.calls.append(kwargs)
        return AggregatedCollisionQueryHits(
            collision_mask=np.asarray(
                ((False, True), (True, False)),
                dtype=np.bool_,
            ),
            first_hit_sample=np.asarray((1, 0), dtype=np.int64),
            first_hit_category=np.asarray(
                (
                    CollisionHitCategory.DYNAMIC_OBJECT,
                    CollisionHitCategory.TABLE,
                ),
                dtype=np.int64,
            ),
        )

    def close(self) -> None:
        self.close_count += 1


def _target(
    env_id: int,
    category: CollisionQueryTarget,
    source_name: str,
    *,
    owner_robot_label: str | None = None,
) -> SceneCollisionTargetShape:
    root = f"/World/envs/env_{env_id}/{source_name}"
    return SceneCollisionTargetShape(
        env_id=env_id,
        category=category,
        source_name=source_name,
        owner_robot_label=owner_robot_label,
        root_path=root,
        collision_path=f"{root}/collision",
        rigid_body_path=None if owner_robot_label is None else root,
    )


def _target_catalog() -> SceneCollisionTargetCatalog:
    shapes = []
    for env_id in range(2):
        shapes.extend(
            (
                _target(env_id, CollisionQueryTarget.TABLE, "workstation"),
                _target(env_id, CollisionQueryTarget.DYNAMIC_OBJECT, "Tblock"),
                _target(
                    env_id,
                    CollisionQueryTarget.OTHER_ROBOT,
                    "left",
                    owner_robot_label="left",
                ),
                _target(
                    env_id,
                    CollisionQueryTarget.OTHER_ROBOT,
                    "right",
                    owner_robot_label="right",
                ),
            )
        )
    return SceneCollisionTargetCatalog(
        num_envs=2,
        robot_labels=("left", "right"),
        shapes=tuple(shapes),
    )


def _adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    left = _Service("left")
    right = _Service("right")
    registry = ArticulatedCollisionServiceRegistry(
        robot_labels=("left", "right"),
        services=(left, right),
    )
    adapter = TorchArticulatedCollisionQueryAdapter(
        registry,
        num_envs=2,
        device=_Device("cuda:0"),
        target_catalog=_target_catalog(),
    )
    return adapter, left, right


def test_torch_adapter_syncs_explicit_query_and_returns_owned_cuda_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, left, right = _adapter(monkeypatch)
    positions = _Tensor(
        np.zeros((2, 2, 2), dtype=np.float32),
        dtype="float32",
        device="cuda:0",
    )
    env_ids = _Tensor((1, 0), dtype="int64", device="cuda:0")

    result = adapter.query(
        robot_label="right",
        joint_positions=positions,  # type: ignore[arg-type]
        env_ids=env_ids,  # type: ignore[arg-type]
        targets=(CollisionQueryTarget.TABLE,),
    )

    assert isinstance(adapter, ArticulatedCollisionQueryPort)
    assert left.calls == []
    assert len(right.calls) == 1
    host_positions = right.calls[0]["joint_positions"]
    assert isinstance(host_positions, np.ndarray)
    assert host_positions.dtype == np.float32
    assert host_positions.flags.c_contiguous is True
    assert right.calls[0]["env_ids"] == (1, 0)
    for tensor, dtype in (
        (result.collision_mask, "bool"),
        (result.first_hit_sample, "int64"),
        (result.first_hit_category, "int64"),
    ):
        assert isinstance(tensor, _Tensor)
        assert tensor.device == _Device("cuda:0")
        assert tensor.dtype == dtype
    assert result.collision_mask.array.tolist() == [
        [False, True],
        [True, False],
    ]


@pytest.mark.parametrize(
    ("positions", "env_ids", "message"),
    [
        (
            _Tensor(np.zeros((1, 1, 2)), dtype="float32", device="cpu"),
            _Tensor((0,), dtype="int64", device="cuda:0"),
            "must live on CUDA",
        ),
        (
            _Tensor(np.zeros((1, 1, 2)), dtype="float32", device="cuda:0"),
            _Tensor((0,), dtype="int64", device="cuda:1"),
            "adapter device",
        ),
        (
            _Tensor(np.zeros((1, 0, 2)), dtype="float32", device="cuda:0"),
            _Tensor((0,), dtype="int64", device="cuda:0"),
            "sample axis",
        ),
        (
            _Tensor(np.zeros((1, 1, 1)), dtype="float32", device="cuda:0"),
            _Tensor((0,), dtype="int64", device="cuda:0"),
            "command width",
        ),
        (
            _Tensor(np.zeros((2, 1, 2)), dtype="float32", device="cuda:0"),
            _Tensor((0, 0), dtype="int64", device="cuda:0"),
            "duplicates",
        ),
        (
            _Tensor(np.full((1, 1, 2), np.nan), dtype="float32", device="cuda:0"),
            _Tensor((0,), dtype="int64", device="cuda:0"),
            "non-finite",
        ),
        (
            _Tensor(np.zeros((1, 1, 2)), dtype="float32", device="cuda:0"),
            _Tensor((2,), dtype="int64", device="cuda:0"),
            "out-of-range",
        ),
    ],
)
def test_torch_adapter_rejects_invalid_cuda_contract_before_service(
    monkeypatch: pytest.MonkeyPatch,
    positions: _Tensor,
    env_ids: _Tensor,
    message: str,
) -> None:
    adapter, left, right = _adapter(monkeypatch)

    with pytest.raises((TypeError, ValueError), match=message):
        adapter.query(
            robot_label="left",
            joint_positions=positions,  # type: ignore[arg-type]
            env_ids=env_ids,  # type: ignore[arg-type]
            targets=(CollisionQueryTarget.TABLE,),
        )

    assert left.calls == []
    assert right.calls == []


def test_torch_adapter_close_owns_registry_and_blocks_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, left, right = _adapter(monkeypatch)
    adapter.close()
    adapter.close()

    assert adapter.closed is True
    assert left.close_count == 1
    assert right.close_count == 1
    with pytest.raises(RuntimeError, match="adapter is closed"):
        adapter.query(
            robot_label="left",
            joint_positions=_Tensor(
                np.zeros((1, 1, 2)),
                dtype="float32",
                device="cuda:0",
            ),  # type: ignore[arg-type]
            env_ids=_Tensor(  # type: ignore[arg-type]
                (0,),
                dtype="int64",
                device="cuda:0",
            ),
            targets=(CollisionQueryTarget.TABLE,),
        )
