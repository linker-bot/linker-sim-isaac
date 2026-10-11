from __future__ import annotations

import pytest

from linkerbot_sim.isaac.scene.pose import (
    PrimWorldTransform,
    read_prim_world_transforms,
)


IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class _Inspector:
    def __init__(self, values: tuple[object, ...]) -> None:
        self.values = values
        self.requests: list[tuple[str, ...]] = []

    def inspect(self, paths: tuple[str, ...]) -> tuple[object, ...]:
        self.requests.append(paths)
        return self.values


def _transform(path: str) -> PrimWorldTransform:
    return PrimWorldTransform(prim_path=path, matrix_row_major=IDENTITY)


def test_world_transform_reader_restores_requested_exact_path_order() -> None:
    first = _transform("/World/envs/env_0/left")
    second = _transform("/World/envs/env_1/left")
    inspector = _Inspector((second, first))

    result = read_prim_world_transforms(
        object(),
        (first.prim_path, second.prim_path),
        inspector=inspector,
    )

    assert inspector.requests == [(first.prim_path, second.prim_path)]
    assert result == (first, second)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (IDENTITY[:-1], "tuple of 16"),
        (IDENTITY[:3] + (1.0,) + IDENTITY[4:], "affine Gf"),
        (IDENTITY[:10] + (0.0,) + IDENTITY[11:], "invertible"),
    ],
)
def test_world_transform_value_rejects_invalid_matrix(
    matrix: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PrimWorldTransform(
            prim_path="/World/envs/env_0/left",
            matrix_row_major=matrix,
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((object(),), "invalid value"),
        ((_transform("/World/unrequested"),), "unrequested path"),
        (
            (
                _transform("/World/envs/env_0/left"),
                _transform("/World/envs/env_0/left"),
            ),
            "duplicate path",
        ),
        ((), "omitted paths"),
    ],
)
def test_world_transform_reader_rejects_inspector_identity_drift(
    values: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, RuntimeError), match=message):
        read_prim_world_transforms(
            object(),
            ("/World/envs/env_0/left",),
            inspector=_Inspector(values),
        )


def test_world_transform_reader_rejects_invalid_requested_paths() -> None:
    inspector = _Inspector(())
    with pytest.raises(ValueError, match="canonical absolute"):
        read_prim_world_transforms(object(), ("World/left",), inspector=inspector)
    with pytest.raises(ValueError, match="duplicates"):
        read_prim_world_transforms(
            object(),
            ("/World/left", "/World/left"),
            inspector=inspector,
        )
