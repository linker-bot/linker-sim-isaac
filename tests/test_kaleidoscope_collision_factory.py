from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.kaleidoscope.collision_factory import (
    RobotCommandJointLimits,
    build_physx_collision_query_resources,
)


class _RobotCatalog:
    robot_labels = ("left", "right")
    num_envs = 2

    def shapes_for(self, *, env_id: int, robot_label: str) -> tuple[object, ...]:
        del env_id
        return (
            SimpleNamespace(
                rigid_body_path=f"/World/{robot_label}/Robot/finger",
            ),
        )


class _Owned:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.events.append(f"close:{self.name}")


class _Overlap(_Owned):
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        shadow: _Owned,
        query: _Owned,
    ) -> None:
        super().__init__(f"overlap:{label}", events)
        self.robot_label = label
        self.shadow = shadow
        self.query = query

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self.query.close()
        self.shadow.close()


class _Service(_Owned):
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        overlap: _Overlap,
        command_joint_names: tuple[str, ...],
    ) -> None:
        super().__init__(f"service:{label}", events)
        self.robot_label = label
        self.command_joint_names = command_joint_names
        self.overlap = overlap

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self.overlap.close()


class _Registry(_Owned):
    def __init__(
        self,
        labels: tuple[str, ...],
        services: tuple[_Service, ...],
        events: list[str],
    ) -> None:
        super().__init__("registry", events)
        self.robot_labels = labels
        self.services = services

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        for service in reversed(self.services):
            service.close()


class _Adapter(_Owned):
    def __init__(self, registry: _Registry, events: list[str]) -> None:
        super().__init__("adapter", events)
        self.registry = registry

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self.registry.close()


def _limits(label: str) -> RobotCommandJointLimits:
    return RobotCommandJointLimits(
        robot_label=label,
        command_joint_names=("joint_0", "joint_1"),
        lower=np.asarray((-1.0, -2.0), dtype=np.float32),
        upper=np.asarray((1.0, 2.0), dtype=np.float32),
    )


def _inputs() -> dict[str, object]:
    robots = tuple(
        SimpleNamespace(
            label=label,
            command_joint_names=("joint_0", "joint_1"),
        )
        for label in ("left", "right")
    )
    return {
        "stage": object(),
        "replicated_scene": SimpleNamespace(
            collision_isolation_strategy="env_ids",
            robots=robots,
        ),
        "robot_profiles": {"left": object(), "right": object()},
        "command_limits": {"left": _limits("left"), "right": _limits("right")},
        "dynamic_object_name": "tblock",
        "device": "cuda:0",
    }


def _factories(
    events: list[str],
    calls: dict[str, list[dict[str, object]]],
    *,
    fail_service_label: str | None = None,
    fail_adapter: bool = False,
) -> dict[str, object]:
    catalog = _RobotCatalog()
    target_catalog = SimpleNamespace(
        num_envs=2,
        robot_labels=catalog.robot_labels,
    )

    def record(name: str, values: dict[str, object]) -> None:
        calls.setdefault(name, []).append(values)

    def robot_catalog_factory(**kwargs: object) -> _RobotCatalog:
        record("robot_catalog", kwargs)
        return catalog

    def target_catalog_factory(**kwargs: object) -> object:
        record("target_catalog", kwargs)
        return target_catalog

    def kinematics_factory(**kwargs: object) -> object:
        record("kinematics", kwargs)
        return SimpleNamespace(command_joint_names=kwargs["command_joint_names"])

    def root_transform_factory(stage: object, **kwargs: object) -> object:
        record("root_transform", {"stage": stage, **kwargs})
        return SimpleNamespace(robot_label=kwargs["robot_label"])

    def shadow_owner_factory(stage: object, **kwargs: object) -> _Owned:
        record("shadow", {"stage": stage, **kwargs})
        return _Owned(f"shadow:{kwargs['root_path']}", events)

    def scene_query_factory() -> _Owned:
        label = ("left", "right")[len(calls.get("scene_query", []))]
        record("scene_query", {"label": label})
        return _Owned(f"query:{label}", events)

    def overlap_factory(**kwargs: object) -> _Overlap:
        record("overlap", kwargs)
        return _Overlap(
            str(kwargs["robot_label"]),
            events,
            shadow=kwargs["shadow_owner"],  # type: ignore[arg-type]
            query=kwargs["scene_query"],  # type: ignore[arg-type]
        )

    def service_factory(**kwargs: object) -> _Service:
        record("service", kwargs)
        label = str(kwargs["robot_label"])
        if label == fail_service_label:
            raise RuntimeError(f"injected {label} service failure")
        return _Service(
            label,
            events,
            overlap=kwargs["overlap_port"],  # type: ignore[arg-type]
            command_joint_names=kwargs["kinematics_spec"].command_joint_names,
        )

    def registry_factory(**kwargs: object) -> _Registry:
        record("registry", kwargs)
        return _Registry(
            kwargs["robot_labels"],  # type: ignore[arg-type]
            kwargs["services"],  # type: ignore[arg-type]
            events,
        )

    def adapter_factory(registry: _Registry, **kwargs: object) -> _Adapter:
        record("adapter", {"registry": registry, **kwargs})
        if fail_adapter:
            raise RuntimeError("injected adapter failure")
        return _Adapter(registry, events)

    return {
        "robot_catalog_factory": robot_catalog_factory,
        "target_catalog_factory": target_catalog_factory,
        "root_transform_factory": root_transform_factory,
        "kinematics_factory": kinematics_factory,
        "shadow_owner_factory": shadow_owner_factory,
        "scene_query_factory": scene_query_factory,
        "overlap_factory": overlap_factory,
        "service_factory": service_factory,
        "registry_factory": registry_factory,
        "adapter_factory": adapter_factory,
    }


def test_factory_composes_two_private_owners_and_closes_from_one_root() -> None:
    events: list[str] = []
    calls: dict[str, list[dict[str, object]]] = {}
    resources = build_physx_collision_query_resources(
        **_inputs(),
        **_factories(events, calls),
    )

    assert tuple(resources.kinematics_specs) == ("left", "right")
    assert tuple(resources.robot_root_transforms) == ("left", "right")
    assert resources.shadow_root_paths == {
        "left": "/World/__linkerbot_collision_query_robot_0000__",
        "right": "/World/__linkerbot_collision_query_robot_0001__",
    }
    assert len(set(resources.shadow_root_paths.values())) == 2
    assert tuple(resources.query_specs) == ("left", "right")
    assert resources.query_specs["left"].robot_label == "left"
    assert resources.query_specs["left"].command_joint_names == (
        "joint_0",
        "joint_1",
    )
    assert resources.query_specs["left"].command_dim == 2
    assert resources.query_specs["left"].joint_lower == (-1.0, -2.0)
    assert resources.query_specs["left"].joint_upper == (1.0, 2.0)
    with pytest.raises(TypeError):
        resources.query_specs["extra"] = (  # type: ignore[index]
            resources.query_specs["left"]
        )
    assert [call["env_ids"] for call in calls["root_transform"]] == [
        (0, 1),
        (0, 1),
    ]
    assert all(
        call["dtype"] == np.dtype(np.float32) for call in calls["root_transform"]
    )
    assert all(
        call["required_link_names"] == ("finger",) for call in calls["kinematics"]
    )
    assert all(
        call["joint_lower"].dtype == np.dtype(np.float32)
        and not call["joint_lower"].flags.writeable
        for call in calls["service"]
    )

    resources.close()
    resources.close()
    assert events == [
        "close:adapter",
        "close:registry",
        "close:service:right",
        "close:overlap:right",
        "close:query:right",
        "close:shadow:/World/__linkerbot_collision_query_robot_0001__",
        "close:service:left",
        "close:overlap:left",
        "close:query:left",
        "close:shadow:/World/__linkerbot_collision_query_robot_0000__",
    ]


def test_factory_rolls_back_current_and_previous_robot_in_reverse() -> None:
    events: list[str] = []
    calls: dict[str, list[dict[str, object]]] = {}

    with pytest.raises(RuntimeError, match="right service failure"):
        build_physx_collision_query_resources(
            **_inputs(),
            **_factories(events, calls, fail_service_label="right"),
        )

    assert events == [
        "close:overlap:right",
        "close:query:right",
        "close:shadow:/World/__linkerbot_collision_query_robot_0001__",
        "close:service:left",
        "close:overlap:left",
        "close:query:left",
        "close:shadow:/World/__linkerbot_collision_query_robot_0000__",
    ]


def test_factory_rolls_back_registry_when_adapter_construction_fails() -> None:
    events: list[str] = []
    calls: dict[str, list[dict[str, object]]] = {}

    with pytest.raises(RuntimeError, match="adapter failure"):
        build_physx_collision_query_resources(
            **_inputs(),
            **_factories(events, calls, fail_adapter=True),
        )

    assert events[0:2] == ["close:registry", "close:service:right"]
    assert events[-1] == (
        "close:shadow:/World/__linkerbot_collision_query_robot_0000__"
    )


def test_factory_rejects_joint_order_mismatch_before_native_resources() -> None:
    events: list[str] = []
    calls: dict[str, list[dict[str, object]]] = {}
    inputs = _inputs()
    inputs["command_limits"] = {
        "left": RobotCommandJointLimits(
            robot_label="left",
            command_joint_names=("joint_1", "joint_0"),
            lower=np.asarray((-1.0, -1.0), dtype=np.float32),
            upper=np.asarray((1.0, 1.0), dtype=np.float32),
        ),
        "right": _limits("right"),
    }

    with pytest.raises(ValueError, match="replicated joint order"):
        build_physx_collision_query_resources(
            **inputs,
            **_factories(events, calls),
        )

    assert "shadow" not in calls
    assert events == []


def test_command_limits_copy_and_freeze_real_float32_values() -> None:
    lower = np.asarray((-1.0, -2.0), dtype=np.float32)
    upper = np.asarray((1.0, 2.0), dtype=np.float32)
    limits = RobotCommandJointLimits(
        robot_label="left",
        command_joint_names=("joint_0", "joint_1"),
        lower=lower,
        upper=upper,
    )

    lower[0] = -9.0
    upper[0] = 9.0
    assert limits.lower.tolist() == [-1.0, -2.0]
    assert limits.upper.tolist() == [1.0, 2.0]
    assert not limits.lower.flags.writeable
    with pytest.raises(TypeError, match="float32"):
        RobotCommandJointLimits(
            robot_label="left",
            command_joint_names=("joint_0",),
            lower=np.asarray((-1.0,), dtype=np.float64),
            upper=np.asarray((1.0,), dtype=np.float64),
        )
