from __future__ import annotations

from dataclasses import dataclass

import pytest

from linkerbot_sim.isaac.physics.scene_query import (
    PhysxOverlapHit,
    PhysxSceneQueryAdapter,
)


@dataclass(frozen=True)
class _FakeNativeHit:
    collision: str
    rigid_body: str
    protoIndex: int


class _FakeSceneQueryInterface:
    def __init__(self, hits: tuple[_FakeNativeHit, ...] = ()) -> None:
        self.hits = hits
        self.calls: list[tuple[int, int, bool]] = []
        self.callback_results: list[bool] = []

    def overlap_shape(
        self,
        path_0: int,
        path_1: int,
        report_fn: object,
        any_hit: bool = False,
    ) -> int:
        self.calls.append((path_0, path_1, any_hit))
        callback = report_fn
        for hit in self.hits:
            should_continue = callback(hit)  # type: ignore[operator]
            self.callback_results.append(should_continue)
            if not should_continue:
                break
        return len(self.callback_results)


def test_overlap_shape_encodes_path_and_preserves_exact_hit_identity() -> None:
    native_hits = (
        _FakeNativeHit(
            collision="/World/envs/env_0/table/collider",
            rigid_body="/World/envs/env_0/table",
            protoIndex=0xFFFFFFFF,
        ),
        _FakeNativeHit(
            collision="/World/envs/env_0/tblock/collider",
            rigid_body="/World/envs/env_0/tblock",
            protoIndex=7,
        ),
    )
    interface = _FakeSceneQueryInterface(native_hits)
    encoded_inputs: list[str] = []

    def encode_path(path: str) -> tuple[int, int]:
        encoded_inputs.append(path)
        return 101, 202

    adapter = PhysxSceneQueryAdapter(interface, path_encoder=encode_path)

    assert adapter.overlap_shape("/World/query/hand") == (
        PhysxOverlapHit(
            collision_path="/World/envs/env_0/table/collider",
            rigid_body_path="/World/envs/env_0/table",
            prototype_index=0xFFFFFFFF,
        ),
        PhysxOverlapHit(
            collision_path="/World/envs/env_0/tblock/collider",
            rigid_body_path="/World/envs/env_0/tblock",
            prototype_index=7,
        ),
    )
    assert encoded_inputs == ["/World/query/hand"]
    assert interface.calls == [(101, 202, False)]
    assert interface.callback_results == [True, True]


def test_overlap_shape_can_stop_callback_traversal_after_first_hit() -> None:
    interface = _FakeSceneQueryInterface(
        (
            _FakeNativeHit("/first", "/body_1", 1),
            _FakeNativeHit("/second", "/body_2", 2),
        )
    )
    adapter = PhysxSceneQueryAdapter(
        interface,
        path_encoder=lambda _path: (1, 2),
    )

    assert adapter.overlap_shape(
        "/World/query/hand",
        stop_after_first=True,
    ) == (PhysxOverlapHit("/first", "/body_1", 1),)
    assert interface.callback_results == [False]
    assert interface.calls == [(1, 2, False)]


def test_close_is_idempotent_and_rejects_future_queries() -> None:
    adapter = PhysxSceneQueryAdapter(
        _FakeSceneQueryInterface(),
        path_encoder=lambda _path: (1, 2),
    )

    adapter.close()
    adapter.close()

    assert adapter.closed is True
    with pytest.raises(RuntimeError, match="adapter is closed"):
        adapter.overlap_shape("/World/query/hand")


@pytest.mark.parametrize("prim_path", ["", "World/query", 123])
def test_overlap_shape_rejects_non_absolute_prim_paths(prim_path: object) -> None:
    adapter = PhysxSceneQueryAdapter(
        _FakeSceneQueryInterface(),
        path_encoder=lambda _path: (1, 2),
    )

    with pytest.raises(ValueError, match="absolute USD prim path"):
        adapter.overlap_shape(prim_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("encoded", [(), (1,), (1, 2, 3), (True, 2), (1, "2")])
def test_overlap_shape_fails_closed_on_invalid_encoded_path(
    encoded: tuple[object, ...],
) -> None:
    adapter = PhysxSceneQueryAdapter(
        _FakeSceneQueryInterface(),
        path_encoder=lambda _path: encoded,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(RuntimeError, match="encodeSdfPath returned invalid data"):
        adapter.overlap_shape("/World/query/hand")


def test_constructor_requires_native_overlap_shape_entry_point() -> None:
    with pytest.raises(TypeError, match="callable overlap_shape"):
        PhysxSceneQueryAdapter(object(), path_encoder=lambda _path: (1, 2))
