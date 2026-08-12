from __future__ import annotations

from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.extensions import enumerate_enabled_kit_extensions
from linkerbot_sim.isaac.physics.exclusivity import (
    newton_forbidden_extensions,
    validate_newton_exclusivity,
)


def test_enumerates_complete_enabled_extension_closure() -> None:
    manager = SimpleNamespace(
        get_extensions=lambda: (
            {
                "id": "omni.warp.core-1.13.0",
                "name": "omni.warp.core",
                "enabled": True,
                "package": {"version": "1.13.0"},
                "path": "/kit/warp",
            },
            {
                "name": "disabled.extension",
                "enabled": False,
                "package": {"version": "1.0.0"},
            },
        ),
        get_extension_dict=lambda _extension_id: None,
        get_extension_path=lambda _extension_id: "/resolved/warp",
    )

    result = enumerate_enabled_kit_extensions(manager)

    assert tuple(item.name for item in result) == ("omni.warp.core",)
    assert result[0].version == "1.13.0"
    assert result[0].path == "/resolved/warp"


def test_normalizes_isaac_6_tuple_extension_version() -> None:
    """Isaac 6 的 tuple 版本必须与 manifest 字符串使用同一比较形式。"""

    manager = SimpleNamespace(
        get_extensions=lambda: (
            {
                "name": "omni.warp.core",
                "enabled": True,
                "package": {"version": (1, 13, 0, "", "lx64")},
            },
        )
    )

    result = enumerate_enabled_kit_extensions(manager)

    assert result[0].version == "1.13.0"


def test_extension_closure_audit_fails_if_manager_cannot_enumerate() -> None:
    with pytest.raises(RuntimeError, match="cannot enumerate"):
        enumerate_enabled_kit_extensions(SimpleNamespace())


def test_newton_runtime_rejects_unknown_prefixed_physx_carrier() -> None:
    manager = SimpleNamespace(
        get_extensions=lambda: (
            {
                "name": "omni.physx.future_carrier",
                "enabled": True,
                "package": {"version": "999.0"},
            },
        )
    )

    with pytest.raises(RuntimeError, match="omni.physx.future_carrier"):
        validate_newton_exclusivity(
            extension_manager=manager,
            stage=None,
            phase="test",
        )


def test_forbidden_extension_predicate_covers_both_physics_families() -> None:
    assert newton_forbidden_extensions(
        (
            "safe.extension",
            "omni.physics.stageupdate",
            "omni.physx.fabric",
            "isaacsim.physics.newton",
            "isaacsim.core.cloner",
        )
    ) == (
        "isaacsim.core.cloner",
        "isaacsim.physics.newton",
        "omni.physics.stageupdate",
        "omni.physx.fabric",
    )
