from __future__ import annotations

from dataclasses import dataclass

import pytest

from linkerbot_sim.isaac.physics.query_shadow import (
    PhysxQueryShadow,
    PhysxQueryShadowOwner,
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

SCALED_AND_TRANSLATED = (
    0.001,
    0.0,
    0.0,
    0.0,
    0.0,
    0.001,
    0.0,
    0.0,
    0.0,
    0.0,
    0.001,
    0.0,
    1.0,
    2.5,
    -3.0,
    1.0,
)


@dataclass
class _FakeShadowBackend:
    fail_create: bool = False

    def __post_init__(self) -> None:
        self.created_roots: list[str] = []
        self.created_shadows: list[tuple[str, str]] = []
        self.transforms: list[tuple[str, tuple[float, ...]]] = []
        self.removed: list[str] = []

    def create_root(self, root_path: str) -> None:
        self.created_roots.append(root_path)

    def create_shadow(
        self,
        source_path: str,
        shadow_path: str,
    ) -> tuple[str, ...]:
        self.created_shadows.append((source_path, shadow_path))
        if self.fail_create:
            raise RuntimeError("forced create failure")
        return (shadow_path,)

    def set_world_transform(
        self,
        shadow_path: str,
        matrix_row_major: tuple[float, ...],
    ) -> None:
        self.transforms.append((shadow_path, matrix_row_major))

    def remove_prim(self, prim_path: str) -> None:
        self.removed.append(prim_path)


def test_owner_creates_private_root_and_tracks_exact_source_identity() -> None:
    backend = _FakeShadowBackend()
    owner = PhysxQueryShadowOwner(object(), backend=backend)

    shadow = owner.create_shadow(
        name="left_palm_0",
        source_collision_path="/World/envs/env_0/left/palm/collision",
    )

    assert owner.root_path == "/World/__linkerbot_collision_query__"
    assert backend.created_roots == [owner.root_path]
    assert backend.created_shadows == [
        (
            "/World/envs/env_0/left/palm/collision",
            "/World/__linkerbot_collision_query__/left_palm_0",
        )
    ]
    assert shadow.prim_path == backend.created_shadows[0][1]
    assert shadow.source_collision_path == backend.created_shadows[0][0]
    assert shadow.query_prim_paths == (shadow.prim_path,)
    assert owner.shadows == (shadow,)


def test_owner_tracks_all_compound_convex_query_pieces() -> None:
    class _CompoundBackend(_FakeShadowBackend):
        def create_shadow(
            self,
            source_path: str,
            shadow_path: str,
        ) -> tuple[str, ...]:
            super().create_shadow(source_path, shadow_path)
            return (
                f"{shadow_path}/piece_0000",
                f"{shadow_path}/piece_0001",
            )

    owner = PhysxQueryShadowOwner(object(), backend=_CompoundBackend())

    shadow = owner.create_shadow(
        name="concave_palm",
        source_collision_path="/World/source/palm/collision",
    )

    assert shadow.query_prim_paths == (
        f"{shadow.prim_path}/piece_0000",
        f"{shadow.prim_path}/piece_0001",
    )


def test_owner_preserves_complete_world_affine_transform_without_a_step() -> None:
    backend = _FakeShadowBackend()
    owner = PhysxQueryShadowOwner(object(), backend=backend)
    shadow = owner.create_shadow(
        name="finger_distal",
        source_collision_path="/World/source/finger/collision",
    )

    owner.set_world_transform(
        shadow,
        matrix_row_major=SCALED_AND_TRANSLATED,
    )

    assert backend.transforms == [
        (
            shadow.prim_path,
            SCALED_AND_TRANSLATED,
        )
    ]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (IDENTITY[:-1], "exactly 16"),
        ((*IDENTITY[:-1], float("nan")), "must be finite"),
        ((1.0, *IDENTITY[1:3], 1.0, *IDENTITY[4:]), "affine Gf matrix"),
        ((*IDENTITY[:10], 0.0, *IDENTITY[11:]), "must be invertible"),
    ],
)
def test_owner_rejects_invalid_world_transform(
    matrix: tuple[float, ...],
    message: str,
) -> None:
    owner = PhysxQueryShadowOwner(object(), backend=_FakeShadowBackend())
    shadow = owner.create_shadow(
        name="palm",
        source_collision_path="/World/source/palm/collision",
    )

    with pytest.raises(ValueError, match=message):
        owner.set_world_transform(
            shadow,
            matrix_row_major=matrix,
        )


def test_owner_normalizes_affine_tail_within_float_tolerance() -> None:
    backend = _FakeShadowBackend()
    owner = PhysxQueryShadowOwner(object(), backend=backend)
    shadow = owner.create_shadow(
        name="palm",
        source_collision_path="/World/source/palm/collision",
    )
    approximate = list(IDENTITY)
    approximate[3] = 1e-13
    approximate[15] = 1.0 + 1e-13

    owner.set_world_transform(shadow, matrix_row_major=approximate)

    assert backend.transforms[0][1] == IDENTITY


def test_owner_rejects_duplicate_or_unsafe_shadow_names() -> None:
    owner = PhysxQueryShadowOwner(object(), backend=_FakeShadowBackend())
    owner.create_shadow(
        name="palm",
        source_collision_path="/World/source/palm/collision",
    )

    with pytest.raises(ValueError, match="already exists"):
        owner.create_shadow(
            name="palm",
            source_collision_path="/World/source/other/collision",
        )
    with pytest.raises(ValueError, match="must match"):
        owner.create_shadow(
            name="../other",
            source_collision_path="/World/source/other/collision",
        )


def test_partial_create_failure_removes_only_attempted_shadow() -> None:
    backend = _FakeShadowBackend(fail_create=True)
    owner = PhysxQueryShadowOwner(object(), backend=backend)

    with pytest.raises(RuntimeError, match="forced create failure"):
        owner.create_shadow(
            name="palm",
            source_collision_path="/World/source/palm/collision",
        )

    assert backend.removed == ["/World/__linkerbot_collision_query__/palm"]
    assert owner.shadows == ()


def test_remove_shadow_rejects_handles_from_another_owner() -> None:
    first = PhysxQueryShadowOwner(object(), backend=_FakeShadowBackend())
    second = PhysxQueryShadowOwner(
        object(),
        root_path="/World/__other_query__",
        backend=_FakeShadowBackend(),
    )
    shadow = first.create_shadow(
        name="palm",
        source_collision_path="/World/source/palm/collision",
    )

    with pytest.raises(ValueError, match="not owned"):
        second.remove_shadow(shadow)


def test_remove_and_close_have_explicit_idempotent_lifecycle() -> None:
    backend = _FakeShadowBackend()
    owner = PhysxQueryShadowOwner(object(), backend=backend)
    shadow = owner.create_shadow(
        name="palm",
        source_collision_path="/World/source/palm/collision",
    )

    owner.remove_shadow(shadow)
    assert backend.removed == [shadow.prim_path]
    assert owner.shadows == ()
    with pytest.raises(ValueError, match="no longer owned"):
        owner.set_world_transform(
            shadow,
            matrix_row_major=IDENTITY,
        )

    owner.close()
    owner.close()
    assert backend.removed == [shadow.prim_path, owner.root_path]
    assert owner.closed is True
    with pytest.raises(RuntimeError, match="owner is closed"):
        owner.create_shadow(
            name="other",
            source_collision_path="/World/source/other/collision",
        )


def test_shadow_handle_type_is_required() -> None:
    owner = PhysxQueryShadowOwner(object(), backend=_FakeShadowBackend())

    with pytest.raises(TypeError, match="PhysxQueryShadow"):
        owner.remove_shadow(object())  # type: ignore[arg-type]

    forged = PhysxQueryShadow(
        prim_path=f"{owner.root_path}/forged",
        source_collision_path="/World/source/forged",
        query_prim_paths=(f"{owner.root_path}/forged",),
        _owner_token=object(),
    )
    with pytest.raises(ValueError, match="not owned"):
        owner.remove_shadow(forged)
