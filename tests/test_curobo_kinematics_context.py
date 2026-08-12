from __future__ import annotations

import ast
from pathlib import Path

import pytest

from linkerbot_sim.backends.curobo.kinematics.context import (
    CuroboKinematicsContext,
    create_kinematics_context,
    kinematics_config_from_robot_profile,
)
from linkerbot_sim.configuration.curobo import CuroboProfileSettings
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def load_robot_profile_by_name(name: str) -> RobotProfileSettings:
    path = ROOT / "configs" / "robots" / f"{name}.yaml"
    return RobotProfileSettings.from_mapping(load_yaml(path), source=str(path))


def test_strict_profile_builds_collision_free_kinematics_config() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    settings = CuroboProfileSettings.from_mapping(
        {
            "kinematics": {
                "max_batch_size": 512,
                "seed_count": 16,
                "collision_check": False,
                "use_cuda_graph": True,
            },
        }
    )
    config = kinematics_config_from_robot_profile(
        profile,
        settings=settings,
        cuda_device=2,
    )
    assert config.device.device == "cuda:2"
    assert config.device.tensor_dtype == "float32"
    assert config.device.collision_geometry_dtype == "float32"
    assert config.device.collision_gradient_dtype == "float32"
    assert config.device.collision_distance_dtype == "float32"
    assert config.task_bundle.name == "curobo_v0_8_default"
    assert config.robot.load_collision_spheres is False
    assert config.ik.num_seeds == 16
    assert config.ik.seed_solver_num_seeds == 16
    assert config.ik.max_batch_size == 512
    assert config.ik.self_collision_check is False
    assert config.ik.collision_cache == {}


def test_kinematics_factories_reject_raw_robot_profile_mapping() -> None:
    profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    settings = object()

    with pytest.raises(TypeError, match="robot_profile must be RobotProfileSettings"):
        kinematics_config_from_robot_profile(  # type: ignore[arg-type]
            {"robot": profile.name},
            settings=settings,
            cuda_device=0,
        )
    with pytest.raises(TypeError, match="robot_profile must be RobotProfileSettings"):
        create_kinematics_context(  # type: ignore[arg-type]
            robot_profile={"robot": profile.name},
            settings=settings,
            cuda_device=0,
        )


def test_kinematics_context_imports_no_motion_or_scene_runtime() -> None:
    path = ROOT / "src/linkerbot_sim/backends/curobo/kinematics/context.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("motion_planner" in name for name in imports)
    assert not any("collision" in name for name in imports)
    assert not any(name.endswith(".scene") for name in imports)


class _DestroyOnce:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    def destroy(self) -> None:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("transient destroy failure")


def test_context_close_retains_failed_owner_and_retries_only_incomplete_resource() -> (
    None
):
    context = object.__new__(CuroboKinematicsContext)
    solver = _DestroyOnce(fail_once=True)
    kinematics = _DestroyOnce()
    context._ik_solver = solver
    context.kinematics = kinematics
    context._kinematics_closed = False
    context._closing_started = False
    context._closed = False

    with pytest.raises(RuntimeError, match="transient destroy failure"):
        context.close()
    assert context._ik_solver is solver
    assert kinematics.calls == 1
    assert context._closed is False

    context.close()
    assert solver.calls == 2
    assert kinematics.calls == 1
    assert context._ik_solver is None
    assert context._closed is True
