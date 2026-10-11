from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import linkerbot_sim.kaleidoscope as kaleidoscope

from linkerbot_sim.kaleidoscope.collision_query import (
    ArticulatedCollisionQueryPort,
    ArticulatedCollisionQueryResult,
    ArticulatedCollisionQuerySpec,
    CollisionHitCategory,
    CollisionQueryTarget,
    CollisionQueryUnavailableError,
)
from linkerbot_sim.kaleidoscope.training_port import KaleidoscopeTrainingPort


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeQueryPort:
    def query(self, **_kwargs: object) -> ArticulatedCollisionQueryResult:
        marker = object()
        return ArticulatedCollisionQueryResult(marker, marker, marker)  # type: ignore[arg-type]

    def close(self) -> None:
        return None


def test_collision_query_categories_have_stable_wire_values() -> None:
    assert tuple(CollisionQueryTarget) == (
        CollisionQueryTarget.TABLE,
        CollisionQueryTarget.DYNAMIC_OBJECT,
        CollisionQueryTarget.OTHER_ROBOT,
    )
    assert [item.value for item in CollisionQueryTarget] == [
        "table",
        "dynamic_object",
        "other_robot",
    ]
    assert [(item.name, item.value) for item in CollisionHitCategory] == [
        ("NONE", 0),
        ("TABLE", 1),
        ("DYNAMIC_OBJECT", 2),
        ("OTHER_ROBOT", 3),
    ]


def test_collision_query_result_is_an_immutable_owned_container() -> None:
    collision_mask = object()
    first_hit_sample = object()
    first_hit_category = object()
    result = ArticulatedCollisionQueryResult(
        collision_mask=collision_mask,  # type: ignore[arg-type]
        first_hit_sample=first_hit_sample,  # type: ignore[arg-type]
        first_hit_category=first_hit_category,  # type: ignore[arg-type]
    )

    assert result.collision_mask is collision_mask
    assert result.first_hit_sample is first_hit_sample
    assert result.first_hit_category is first_hit_category
    with pytest.raises(FrozenInstanceError):
        result.collision_mask = object()  # type: ignore[misc]


def test_collision_query_spec_is_strict_and_immutable() -> None:
    spec = ArticulatedCollisionQuerySpec(
        robot_label="left",
        command_joint_names=("arm", "finger"),
        joint_lower=(-1.0, -0.5),
        joint_upper=(1.0, 0.5),
    )

    assert spec.command_dim == 2
    assert spec.command_joint_names == ("arm", "finger")
    assert spec.joint_lower == (-1.0, -0.5)
    assert spec.joint_upper == (1.0, 0.5)
    with pytest.raises(FrozenInstanceError):
        spec.robot_label = "right"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"robot_label": ""}, ValueError, "robot_label"),
        ({"command_joint_names": ["arm"]}, ValueError, "command_joint_names"),
        (
            {"command_joint_names": ("arm", "arm")},
            ValueError,
            "command_joint_names",
        ),
        ({"joint_lower": [-1.0]}, ValueError, "joint_lower"),
        ({"joint_lower": (-1,)}, TypeError, "joint_lower"),
        ({"joint_lower": (float("nan"),)}, ValueError, "finite"),
        ({"joint_upper": (float("inf"),)}, ValueError, "finite"),
        ({"joint_lower": (1.0,), "joint_upper": (1.0,)}, ValueError, "strictly"),
    ),
)
def test_collision_query_spec_rejects_ambiguous_metadata(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "robot_label": "left",
        "command_joint_names": ("arm",),
        "joint_lower": (-1.0,),
        "joint_upper": (1.0,),
    }
    values.update(kwargs)

    with pytest.raises(error, match=message):
        ArticulatedCollisionQuerySpec(**values)  # type: ignore[arg-type]


def test_collision_query_port_is_runtime_checkable_without_importing_torch() -> None:
    assert isinstance(_FakeQueryPort(), ArticulatedCollisionQueryPort)


def test_unavailable_error_preserves_backend_identity() -> None:
    error = CollisionQueryUnavailableError("newton_cuda")

    assert error.backend_kind == "newton_cuda"
    assert "newton_cuda" in str(error)


def test_collision_query_types_are_exported_by_lazy_product_facade() -> None:
    assert kaleidoscope.ArticulatedCollisionQuerySpec is ArticulatedCollisionQuerySpec
    assert (
        kaleidoscope.ArticulatedCollisionQueryResult is ArticulatedCollisionQueryResult
    )
    assert kaleidoscope.CollisionHitCategory is CollisionHitCategory
    assert kaleidoscope.CollisionQueryTarget is CollisionQueryTarget
    assert kaleidoscope.CollisionQueryUnavailableError is CollisionQueryUnavailableError
    assert not hasattr(
        KaleidoscopeTrainingPort,
        "query_articulated_collisions",
    )


@pytest.mark.parametrize("language", ("en", "zh-CN"))
def test_public_collision_query_contract_is_documented_bilingually(
    language: str,
) -> None:
    text = (
        REPO_ROOT / "docs" / language / "reference" / "kaleidoscope-api.md"
    ).read_text(encoding="utf-8")

    for token in (
        "supports_articulated_collision_queries",
        "articulated_collision_query_specs",
        "query_articulated_collisions",
        "ArticulatedCollisionQuerySpec",
        "command_joint_names",
        "command_dim",
        "joint_lower",
        "joint_upper",
        "CollisionQueryTarget.TABLE",
        "CollisionQueryTarget.DYNAMIC_OBJECT",
        "CollisionQueryTarget.OTHER_ROBOT",
        "CollisionQueryUnavailableError",
        "collision_mask",
        "first_hit_sample",
        "first_hit_category",
        "[K, S, J]" if language == "en" else "[K,S,J]",
        "physx_cuda",
        "newton_cuda",
        "KaleidoscopeTrainingPort",
    ):
        assert token in text
