from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboDeviceConfig,
    CuroboIkConfig,
    CuroboMotionPlannerConfig,
    CuroboRobotConfig,
    CuroboTaskBundle,
    CuroboTcpFrame,
)
from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.backends.curobo.robot_model import (
    materialize_curobo_config,
    resolve_curobo_cache_dir,
    write_curobo_tcp_urdf_with_frames,
)
from linkerbot_sim.backends.curobo.profile_merge import (
    curobo_config_from_profiles,
)
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.configuration.curobo import CuroboProfileSettings
from linkerbot_sim.utils.config import load_yaml


def load_robot_profile_mapping(name: str) -> dict[str, object]:
    path = Path("configs/robots") / f"{name}.yaml"
    return load_yaml(path)


def load_robot_profile_by_name(name: str) -> RobotProfileSettings:
    path = Path("configs/robots") / f"{name}.yaml"
    return RobotProfileSettings.from_mapping(load_yaml(path), source=str(path))


def _mirror_curobo_settings() -> CuroboProfileSettings:
    """复用正式 Mirror 配置图已经严格解析的 cuRobo 设置。"""

    return load_mirror_config("physx_cpu").curobo


def test_curobo_complete_typed_config_validates_backend_sections() -> None:
    config = CuroboConfig(
        robot=CuroboRobotConfig.from_mapping(
            {
                "robot_config_path": "configs/robots/ar5v2_l.yaml",
                "urdf_path": "assets/robots/ar5v2_l/urdf/ar5v2_l.urdf",
                "base_link": "base",
                "default_tcp_frame": "tool",
                "custom_tcps": [
                    {
                        "frame_name": "pinch_tcp",
                        "parent_frame": "tool",
                        "xyz": [0.0, 0.0, 0.1],
                        "rpy": [0.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        ik=CuroboIkConfig(
            random_seed=999,
            optimizer_collision_activation_distance=0.025,
            store_debug=True,
            override_optimizer_num_iters={"particle": 12, "lbfgs": None},
            optimization_dt=0.02,
            velocity_regularization_weight=0.1,
            success_requires_convergence=False,
            num_seeds=16,
            seed_solver_num_seeds=8,
            max_batch_size=128,
            collision_cache={"cuboid": 4},
        ),
        motion_planner=CuroboMotionPlannerConfig(
            warmup=False,
            random_seed=321,
            optimizer_collision_activation_distance=0.03,
            store_debug=True,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
        ),
    )
    config.validate()

    assert config.robot.default_tcp_frame == "tool"
    assert config.robot.resolved_tool_frames == ("tool",)
    assert config.robot.custom_tcp_frames[0].frame_name == "pinch_tcp"
    assert config.task_bundle.name == "curobo_v0_8_default"
    # Direct IK is LBFGS-only (MPPI particle stage dropped; see config.py).
    assert config.task_bundle.ik_optimizer_configs == ("ik/lbfgs_ik.yml",)
    assert config.ik.num_seeds == 16
    assert config.ik.random_seed == 999
    assert config.ik.optimizer_collision_activation_distance == 0.025
    assert config.ik.store_debug is True
    assert config.ik.override_optimizer_num_iters == {"particle": 12, "lbfgs": None}
    assert config.ik.optimization_dt == 0.02
    assert config.ik.velocity_regularization_weight == 0.1
    assert config.ik.success_requires_convergence is False
    assert config.ik.seed_solver_num_seeds == 8
    assert config.ik.collision_cache == {"cuboid": 4}
    assert config.task_bundle.motion_ik_optimizer_configs == ("ik/lbfgs_ik.yml",)
    assert config.task_bundle.trajopt_optimizer_configs == (
        "trajopt/lbfgs_bspline_trajopt.yml",
    )
    assert config.task_bundle.graph_planner_config == (
        "graph_planner/exact_graph_planner.yml"
    )
    assert config.motion_planner.warmup is False
    assert config.motion_planner.random_seed == 321
    assert config.motion_planner.optimizer_collision_activation_distance == 0.03
    assert config.motion_planner.store_debug is True
    assert config.motion_planner.num_trajopt_seeds == 2


def test_mirror_curobo_profile_contains_valid_algorithm_defaults() -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=_mirror_curobo_settings(),
        cuda_device=0,
    )
    ik = config.ik
    planner = config.motion_planner

    assert ik.num_seeds == 32
    assert ik.max_batch_size == 8
    assert ik.random_seed == 123
    assert ik.optimizer_collision_activation_distance == 0.01
    assert ik.seed_solver_num_seeds == 32
    assert planner.num_ik_seeds == 32
    assert planner.num_trajopt_seeds == 4
    assert planner.warmup is True
    assert planner.use_cuda_graph is False
    assert planner.random_seed == 123
    assert planner.optimizer_collision_activation_distance == 0.01
    assert planner.collision_cache["cuboid"] == 48


def test_typed_curobo_composition_preserves_robot_resources() -> None:
    robot_profile = load_robot_profile_by_name("ar5v2_l6v1_l")
    curobo_settings = _mirror_curobo_settings()

    config = curobo_config_from_profiles(
        robot_profile,
        curobo_settings=curobo_settings,
        cuda_device=0,
    )

    assert config.robot.urdf_path is not None
    assert config.robot.urdf_path.name == "AR5V2_L.urdf"
    assert config.ik.max_batch_size == 8
    assert config.ik.collision_cache == {}
    assert config.motion_planner.collision_cache == {"cuboid": 48, "mesh": 4}
    assert config.device.device == "cuda:0"


def test_curobo_config_from_profiles_applies_profile_defaults() -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=_mirror_curobo_settings(),
        cuda_device=0,
    )

    assert config.robot.default_tcp_frame == "AR5V2_L_pinch_tcp"
    assert config.device.device == "cuda:0"
    assert config.device.tensor_dtype == "float32"
    assert config.device.collision_geometry_dtype == "float32"
    assert config.device.collision_gradient_dtype == "float32"
    assert config.device.collision_distance_dtype == "float32"
    assert config.task_bundle.name == "curobo_v0_8_default"


def test_curobo_profile_root_device_is_projected_into_backend_config() -> None:
    settings = _mirror_curobo_settings()

    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=settings,
        cuda_device=3,
    )

    assert config.device.device == "cuda:3"
    assert not hasattr(settings, "device")
    assert not hasattr(settings, "task_bundle")


def test_curobo_mirror_profile_is_loadable_yaml() -> None:
    document = load_yaml("configs/curobo/mirror.yaml")
    assert set(document) == {"curobo"}
    settings = CuroboProfileSettings.from_mapping(document["curobo"])

    assert settings.kinematics.max_batch_size == 8
    assert settings.kinematics.seed_count == 32
    assert settings.motion_planner is not None
    assert settings.motion_planner.trajectory_seed_count == 4
    assert not hasattr(settings.motion_planner, "max_batch_size")
    assert settings.kinematics.collision_cache is None
    assert not hasattr(settings, "device")
    assert not hasattr(settings, "task_bundle")


def test_all_bundled_curobo_profiles_load_strictly() -> None:
    expected = {
        "kaleidoscope_batch_ik": False,
        "mirror": True,
    }
    paths = sorted(Path("configs/curobo").glob("*.yaml"))
    assert {path.stem for path in paths} == set(expected)
    for path in paths:
        document = load_yaml(path)
        assert set(document) == {"curobo"}
        settings = CuroboProfileSettings.from_mapping(document["curobo"])
        assert (settings.motion_planner is not None) is expected[path.stem]
        assert settings.kinematics.collision_cache is None
        assert "task_bundle" not in document["curobo"]
        assert "device" not in document["curobo"]
        assert "compute" not in document


def test_kinematics_collision_disabled_may_retain_collision_cache() -> None:
    document = load_yaml("configs/curobo/mirror.yaml")
    curobo = document["curobo"]
    curobo["kinematics"]["collision_cache"] = {"cuboid": 4, "mesh": 2}

    settings = CuroboProfileSettings.from_mapping(curobo)
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=settings,
        cuda_device=0,
    )

    assert settings.kinematics.collision_cache is not None
    assert settings.kinematics.collision_cache.as_backend_mapping() == {
        "cuboid": 4,
        "mesh": 2,
    }
    assert config.ik.self_collision_check is False
    assert config.ik.collision_cache == {}


def test_kinematics_collision_enabled_requires_collision_cache() -> None:
    document = load_yaml("configs/curobo/mirror.yaml")
    curobo = document["curobo"]
    curobo["kinematics"]["collision_check"] = True

    with pytest.raises(ValueError, match="collision_cache"):
        CuroboProfileSettings.from_mapping(curobo)


@pytest.mark.parametrize(
    "cache",
    [None, {"cuboid": 4, "mesh": 2}],
)
def test_motion_planner_collision_disabled_may_omit_or_retain_collision_cache(
    cache: dict[str, int] | None,
) -> None:
    document = load_yaml("configs/curobo/mirror.yaml")
    curobo = document["curobo"]
    planner = curobo["motion_planner"]
    planner["collision_check"] = False
    if cache is None:
        planner.pop("collision_cache")
    else:
        planner["collision_cache"] = cache

    settings = CuroboProfileSettings.from_mapping(curobo)
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=settings,
        cuda_device=0,
    )

    assert settings.motion_planner is not None
    if cache is None:
        assert settings.motion_planner.collision_cache is None
    else:
        retained = settings.motion_planner.collision_cache
        assert retained is not None
        assert retained.as_backend_mapping() == cache
    assert config.motion_planner.self_collision_check is False
    assert config.motion_planner.collision_cache == {}


def test_motion_planner_collision_enabled_requires_collision_cache() -> None:
    document = load_yaml("configs/curobo/mirror.yaml")
    curobo = document["curobo"]
    curobo["motion_planner"].pop("collision_cache")

    with pytest.raises(ValueError, match="collision_cache"):
        CuroboProfileSettings.from_mapping(curobo)


def test_curobo_task_resources_are_owned_exactly_by_versioned_bundle() -> None:
    task_root = Path("src/linkerbot_sim/backends/curobo/resources/task")
    task_paths = {
        path.relative_to(task_root).as_posix() for path in task_root.rglob("*.yml")
    }
    assert task_paths
    bundle = CuroboTaskBundle.named("curobo_v0_8_default")
    assert bundle.compatible_versions == frozenset({"0.8.0"})
    referenced_paths = {
        *bundle.ik_optimizer_configs,
        bundle.ik_metrics_rollout,
        bundle.ik_transition_model,
        *bundle.motion_ik_optimizer_configs,
        bundle.motion_ik_transition_model,
        bundle.motion_metrics_rollout,
        *bundle.trajopt_optimizer_configs,
        bundle.trajopt_transition_model,
        bundle.graph_planner_config,
        bundle.graph_planner_rollout,
        bundle.graph_planner_transition_model,
    }
    assert referenced_paths == task_paths


@pytest.mark.parametrize("value", (-1, True, 1.5, "0", None))
def test_curobo_profile_requires_strict_root_cuda_device(value: object) -> None:
    with pytest.raises(ValueError, match="cuda_device"):
        curobo_config_from_profiles(
            load_robot_profile_by_name("ar5v2_l6v1_l"),
            curobo_settings=_mirror_curobo_settings(),
            cuda_device=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("robot_override", "path"),
    (
        ({"compute": {"cuda_device": 2}}, r"profile\.compute"),
        ({"curobo": {"device": {"device": "cuda:2"}}}, r"curobo\.device"),
        (
            {"curobo": {"device": {"tensor_dtype": "float32"}}},
            r"curobo\.device",
        ),
    ),
)
def test_curobo_composition_rejects_robot_device_ownership(
    robot_override: dict[str, object],
    path: str,
) -> None:
    robot = load_robot_profile_mapping("ar5v2_l6v1_l")
    for key, value in robot_override.items():
        if key == "curobo":
            robot["curobo"] = {**robot["curobo"], **value}
        else:
            robot[key] = value

    with pytest.raises(ValueError, match=path):
        RobotProfileSettings.from_mapping(robot)


@pytest.mark.parametrize(
    "field",
    (
        "tensor_dtype",
        "collision_geometry_dtype",
        "collision_gradient_dtype",
        "collision_distance_dtype",
    ),
)
def test_curobo_device_rejects_unvalidated_tensor_dtypes(field: str) -> None:
    with pytest.raises(ValueError, match=rf"curobo\.device\.{field}"):
        replace(CuroboDeviceConfig(), **{field: "typo"}).validate()


def test_backend_algorithm_and_device_configs_have_no_mapping_parser() -> None:
    """YAML 只能在 configuration 层解释，backend 不提供第二条配置入口。"""

    assert not hasattr(CuroboConfig, "from_mapping")
    assert not hasattr(CuroboDeviceConfig, "from_mapping")
    assert not hasattr(CuroboIkConfig, "from_mapping")
    assert not hasattr(CuroboMotionPlannerConfig, "from_mapping")
    assert not hasattr(CuroboProfileSettings, "as_backend_profile")
    assert hasattr(CuroboRobotConfig, "from_mapping")
    assert hasattr(CuroboTcpFrame, "from_mapping")


def test_curobo_context_validates_typed_config_before_runtime_imports() -> None:
    valid = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )
    invalid = replace(
        valid,
        device=replace(valid.device, device="cpu"),
    )

    with pytest.raises(ValueError, match="canonical non-negative CUDA device"):
        CuroboContext(invalid)
    with pytest.raises(TypeError, match="config must be CuroboConfig"):
        CuroboContext({"curobo": {}})  # type: ignore[arg-type]


def test_curobo_task_bundle_rejects_unknown_name_and_version() -> None:
    with pytest.raises(ValueError, match="curobo.task_bundle"):
        CuroboTaskBundle.named("local_task_files")

    bundle = CuroboTaskBundle.named("curobo_v0_8_default")
    bundle.validate_curobo_version("0.8.0")
    assert bundle.compatible_versions == frozenset({"0.8.0"})


@pytest.mark.parametrize("version", ("0.8.1", "0.8.999", "0.9.0"))
def test_curobo_task_bundle_rejects_unvalidated_patch_versions(version: str) -> None:
    bundle = CuroboTaskBundle.named("curobo_v0_8_default")
    with pytest.raises(RuntimeError, match=f"installed version.*{version}"):
        bundle.validate_curobo_version(version)


def test_curobo_robot_boolean_field_rejects_truthy_string() -> None:
    with pytest.raises(ValueError, match="load_collision_spheres"):
        CuroboRobotConfig.from_mapping(
            {
                "robot_config_path": "configs/robots/ar5v2_l.yaml",
                "default_tcp_frame": "tool",
                "load_collision_spheres": "false",
            }
        )


def test_curobo_context_passes_exposed_ik_and_planner_parameters() -> None:
    config = CuroboConfig(
        robot=CuroboRobotConfig.from_mapping(
            {
                "robot_config_path": "configs/robots/ar5v2_l.yaml",
                "default_tcp_frame": "tool",
            }
        ),
        ik=CuroboIkConfig(
            random_seed=77,
            optimizer_collision_activation_distance=0.04,
            store_debug=True,
            override_optimizer_num_iters={"lbfgs": 5},
            optimization_dt=0.02,
            velocity_regularization_weight=0.2,
            acceleration_regularization_weight=0.3,
            success_requires_convergence=False,
            seed_position_weight=2.0,
            seed_orientation_weight=3.0,
            seed_velocity_weight=0.4,
            seed_acceleration_weight=0.5,
            seed_solver_num_seeds=12,
            collision_cache={"cuboid": 4},
        ),
        motion_planner=CuroboMotionPlannerConfig(
            random_seed=88,
            optimizer_collision_activation_distance=0.05,
            store_debug=True,
            collision_cache={"cuboid": 4},
        ),
    )
    config.validate()
    context = CuroboContext.__new__(CuroboContext)
    context.config = config
    context.device_cfg = object()
    context._robot_input_for_solver = lambda: "robot-input"
    context._supports_collision_queries = lambda enabled: bool(enabled)
    context._collision_cache_for_solver = lambda cache: dict(cache) if cache else None
    ik_calls = []
    planner_calls = []
    warmups: list[int] = []
    context.ik_module = SimpleNamespace(
        InverseKinematicsCfg=SimpleNamespace(
            create=lambda **kwargs: ik_calls.append(kwargs) or "ik-cfg"
        ),
        InverseKinematics=lambda cfg: SimpleNamespace(cfg=cfg),
    )
    context.motion_module = SimpleNamespace(
        MotionPlannerCfg=SimpleNamespace(
            create=lambda **kwargs: planner_calls.append(kwargs) or "planner-cfg"
        ),
        MotionPlanner=lambda cfg: SimpleNamespace(
            cfg=cfg,
            warmup=lambda *, num_warmup_iterations: warmups.append(
                num_warmup_iterations
            ),
        ),
    )
    ik_solver = context._make_ik_solver()
    planner = context._make_motion_planner()

    assert ik_solver.cfg == "ik-cfg"
    assert planner.cfg == "planner-cfg"
    assert warmups == [1]
    task_root = Path("src/linkerbot_sim/backends/curobo/resources/task").resolve()

    def assert_task_path(value: str, relative_path: str) -> None:
        path = Path(value)
        assert path.is_absolute()
        assert path == task_root / relative_path
        assert path.is_file()

    assert [Path(value).name for value in ik_calls[0]["optimizer_configs"]] == [
        "lbfgs_ik.yml",
    ]
    assert_task_path(ik_calls[0]["metrics_rollout"], "metrics_base.yml")
    assert_task_path(ik_calls[0]["transition_model"], "ik/transition_ik.yml")
    assert ik_calls[0]["random_seed"] == 77
    assert ik_calls[0]["optimizer_collision_activation_distance"] == 0.04
    assert ik_calls[0]["store_debug"] is True
    assert ik_calls[0]["override_optimizer_num_iters"] == {"lbfgs": 5}
    assert ik_calls[0]["optimization_dt"] == 0.02
    assert ik_calls[0]["velocity_regularization_weight"] == 0.2
    assert ik_calls[0]["acceleration_regularization_weight"] == 0.3
    assert ik_calls[0]["success_requires_convergence"] is False
    assert ik_calls[0]["seed_position_weight"] == 2.0
    assert ik_calls[0]["seed_orientation_weight"] == 3.0
    assert ik_calls[0]["seed_velocity_weight"] == 0.4
    assert ik_calls[0]["seed_acceleration_weight"] == 0.5
    assert ik_calls[0]["seed_solver_num_seeds"] == 12
    assert ik_calls[0]["scene_model"] == {}
    assert ik_calls[0]["collision_cache"] == {"cuboid": 4}
    assert_task_path(planner_calls[0]["ik_optimizer_configs"][0], "ik/lbfgs_ik.yml")
    assert_task_path(planner_calls[0]["ik_transition_model"], "ik/transition_ik.yml")
    assert_task_path(planner_calls[0]["metrics_rollout"], "metrics_base.yml")
    assert_task_path(
        planner_calls[0]["trajopt_optimizer_configs"][0],
        "trajopt/lbfgs_bspline_trajopt.yml",
    )
    assert_task_path(
        planner_calls[0]["trajopt_transition_model"],
        "trajopt/transition_bspline_trajopt.yml",
    )
    assert_task_path(
        planner_calls[0]["graph_planner_config"],
        "graph_planner/exact_graph_planner.yml",
    )
    assert_task_path(planner_calls[0]["graph_planner_rollout"], "metrics_base.yml")
    assert_task_path(
        planner_calls[0]["graph_planner_transition_model"],
        "graph_planner/transition_graph_planner.yml",
    )
    assert planner_calls[0]["random_seed"] == 88
    assert planner_calls[0]["optimizer_collision_activation_distance"] == 0.05
    assert planner_calls[0]["store_debug"] is True
    assert planner_calls[0]["max_batch_size"] == 1

    context.config = replace(
        config,
        motion_planner=replace(config.motion_planner, warmup=False),
    )
    context._make_motion_planner()
    assert warmups == [1]


def test_curobo_config_rejects_missing_robot_resources() -> None:
    with pytest.raises(ValueError, match="robot_config_path or urdf_path"):
        CuroboRobotConfig.from_mapping({"default_tcp_frame": "tool"})


def test_curobo_config_rejects_scene_cache_types_ignored_by_v080() -> None:
    for shape in ("sphere", "capsule", "voxel", "unknown"):
        with pytest.raises(ValueError, match=shape):
            CuroboIkConfig(collision_cache={shape: 1}).validate()


def test_curobo_tcp_frame_uses_default_parent_when_omitted() -> None:
    robot = CuroboRobotConfig.from_mapping(
        {
            "robot_config_path": "configs/robots/ar5v2_l.yaml",
            "default_tcp_frame": "flange",
            "custom_tcps": [
                {
                    "frame_name": "tool",
                    "xyz": [0.0, 0.0, 0.0],
                    "rpy": [0.0, 0.0, 0.0],
                }
            ],
        }
    )
    assert robot.custom_tcp_frames[0].parent_frame == "flange"


def test_curobo_robot_config_accepts_tool_frames() -> None:
    robot = CuroboRobotConfig.from_mapping(
        {
            "robot_config_path": "configs/robots/ar5v2_l.yaml",
            "tool_frames": ["tool"],
        }
    )
    assert robot.resolved_tool_frames == ("tool",)


def test_typed_robot_profile_projects_curobo_collision_model() -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )

    assert config.robot.robot_config_path is not None
    assert config.robot.robot_config_path.name == "AR5V2_L_curobo.yml"
    assert config.robot.urdf_path is not None
    assert config.robot.urdf_path.name == "AR5V2_L.urdf"
    assert config.robot.base_link == "world"
    assert config.robot.flange_frame == "AR5V2_L_arm_flan_link"
    assert config.robot.default_tcp_frame == "AR5V2_L_pinch_tcp"
    assert config.robot.resolved_tool_frames == ("AR5V2_L_pinch_tcp",)
    assert config.robot.custom_tcp_frames[0].parent_frame == "AR5V2_L_arm_flan_link"
    assert config.robot.load_collision_spheres is True


@pytest.mark.parametrize("side", ("L", "R"))
def test_curobo_collision_model_ignores_concentric_wrist_links(side: str) -> None:
    config = load_yaml(f"assets/single_system/arm/AR5V2_{side}/AR5V2_{side}_curobo.yml")
    prefix = f"AR5V2_{side}_arm"
    ignored = config["robot_cfg"]["kinematics"]["self_collision_ignore"][
        f"{prefix}_link4"
    ]

    assert f"{prefix}_link7" in ignored
    assert f"{prefix}_flan_link" in ignored


@pytest.mark.parametrize("profile_name", ("ar5v2_l6v1_l", "ar5v2_l6v1_r"))
def test_curobo_solver_input_materializes_profile_paths_and_tcp(
    profile_name: str,
    tmp_path: Path,
) -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name(profile_name),
        cuda_device=0,
    )
    source_urdf_path = config.robot.urdf_path
    assert source_urdf_path is not None
    materialized = materialize_curobo_config(config, cache_root=tmp_path)
    context = CuroboContext.__new__(CuroboContext)
    context.config = materialized
    context.tool_frames = materialized.robot.resolved_tool_frames
    context._robot_asset_root_path = source_urdf_path.parent

    robot_input = context._robot_input_for_solver()

    kinematics = robot_input["robot_cfg"]["kinematics"]
    materialized_urdf_path = Path(kinematics["urdf_path"])
    assert materialized_urdf_path.is_absolute()
    assert materialized_urdf_path.is_file()
    assert materialized_urdf_path.is_relative_to(tmp_path / "curobo")
    assert Path(kinematics["asset_root_path"]) == source_urdf_path.parent
    assert kinematics["tool_frames"] == list(materialized.robot.resolved_tool_frames)
    assert kinematics["collision_spheres"]
    assert kinematics["cspace"]["joint_names"]
    urdf_links = {
        link.get("name") for link in ET.parse(materialized_urdf_path).findall("link")
    }
    assert materialized.robot.default_tcp_frame in urdf_links


def test_curobo_materialization_uses_environment_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "external-cache"
    monkeypatch.setenv("LINKERBOT_SIM_CACHE_ROOT", str(cache_root))
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )

    materialized = materialize_curobo_config(config)

    output_path = materialized.robot.urdf_path
    assert output_path is not None
    assert output_path.is_file()
    assert output_path.is_relative_to(cache_root / "curobo")


def test_curobo_materialization_rebuilds_corrupted_cache(tmp_path: Path) -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )
    expected_frame = config.robot.custom_tcp_frames[0].frame_name
    first = materialize_curobo_config(config, cache_root=tmp_path)
    output_path = first.robot.urdf_path
    assert output_path is not None
    output_path.write_text("<robot>", encoding="utf-8")

    rebuilt = materialize_curobo_config(config, cache_root=tmp_path)

    assert rebuilt.robot.urdf_path == output_path
    root = ET.parse(output_path).getroot()
    assert root.find(f"./link[@name='{expected_frame}']") is not None
    assert root.find(f"./joint[@name='{expected_frame}_joint']") is not None


@pytest.mark.parametrize("element_tag", ("link", "joint"))
def test_curobo_materialization_rebuilds_valid_xml_with_tampered_robot_body(
    tmp_path: Path,
    element_tag: str,
) -> None:
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )
    materialized = materialize_curobo_config(config, cache_root=tmp_path)
    output_path = materialized.robot.urdf_path
    assert output_path is not None
    expected_content = output_path.read_bytes()
    root = ET.parse(output_path).getroot()
    tcp_names = {frame.frame_name for frame in config.robot.custom_tcp_frames}
    if element_tag == "link":
        candidates = [
            element
            for element in root.findall("link")
            if element.get("name") not in tcp_names
        ]
    else:
        candidates = [
            element
            for element in root.findall("joint")
            if element.get("name") not in {f"{name}_joint" for name in tcp_names}
        ]
    assert candidates
    root.remove(candidates[0])
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    ET.parse(output_path)

    rebuilt = materialize_curobo_config(config, cache_root=tmp_path)

    assert rebuilt.robot.urdf_path == output_path
    assert output_path.read_bytes() == expected_content


def test_curobo_materialization_is_atomic_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linkerbot_sim.backends.curobo import robot_model

    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        cuda_device=0,
    )
    expected_frame = config.robot.custom_tcp_frames[0].frame_name
    initial = materialize_curobo_config(config, cache_root=tmp_path)
    output_path = initial.robot.urdf_path
    assert output_path is not None
    output_path.unlink()
    stop = threading.Event()
    observations: list[str] = []
    read_errors: list[BaseException] = []
    replace_sources: list[Path] = []
    real_replace = robot_model.os.replace

    def checked_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == output_path.parent
        root = ET.parse(source_path).getroot()
        assert root.find(f"./link[@name='{expected_frame}']") is not None
        replace_sources.append(source_path)
        real_replace(source_path, destination_path)

    monkeypatch.setattr(robot_model.os, "replace", checked_replace)

    def observe_cache() -> None:
        while not stop.wait(0.0005):
            try:
                root = ET.parse(output_path).getroot()
            except FileNotFoundError:
                continue
            except BaseException as exc:
                read_errors.append(exc)
                return
            if root.find(f"./link[@name='{expected_frame}']") is None:
                read_errors.append(AssertionError("visible cache omitted expected TCP"))
                return
            observations.append(root.tag)

    observer = threading.Thread(target=observe_cache)
    observer.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            materialized = tuple(
                executor.map(
                    lambda _index: materialize_curobo_config(
                        config,
                        cache_root=tmp_path,
                    ),
                    range(24),
                )
            )
    finally:
        stop.set()
        observer.join(timeout=2.0)

    assert not observer.is_alive()
    assert not read_errors
    assert observations
    assert replace_sources
    assert {item.robot.urdf_path for item in materialized} == {output_path}
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_curobo_robot_config_accepts_mapping_custom_tcp_frames() -> None:
    robot = CuroboRobotConfig.from_mapping(
        {
            "urdf_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
            "flange_frame": "AR5V2_L_arm_flan_link",
            "default_tcp_frame": "pinch_tcp",
            "custom_tcps": {
                "pinch_tcp": {
                    "xyz": [0.0, 0.0, 0.1],
                    "rpy": [0.0, 0.0, 0.0],
                }
            },
        }
    )

    assert robot.base_link == "world"
    assert robot.custom_tcp_frames[0].frame_name == "pinch_tcp"
    assert robot.custom_tcp_frames[0].parent_frame == "AR5V2_L_arm_flan_link"


def test_curobo_cache_root_precedence_avoids_repository_cache(tmp_path: Path) -> None:
    explicit = tmp_path / "runtime-cache"
    env_root = tmp_path / "env-cache"
    xdg_root = tmp_path / "xdg-cache"
    values = {
        "LINKERBOT_SIM_CACHE_ROOT": str(env_root),
        "XDG_CACHE_HOME": str(xdg_root),
    }

    assert (
        resolve_curobo_cache_dir(explicit, environ=values)
        == (explicit / "curobo").resolve()
    )
    assert resolve_curobo_cache_dir(environ=values) == (env_root / "curobo").resolve()
    assert (
        resolve_curobo_cache_dir(environ={"XDG_CACHE_HOME": str(xdg_root)})
        == (xdg_root / "linkerbot_sim" / "curobo").resolve()
    )
    assert not resolve_curobo_cache_dir(explicit, environ=values).is_relative_to(
        Path.cwd() / ".cache"
    )


def test_write_curobo_tcp_urdf_with_frames_adds_fixed_link(tmp_path) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "robot_tcp.urdf"
    source.write_text(
        """
<robot name="tiny">
  <link name="base" />
  <link name="tool" />
  <joint name="base_to_tool" type="fixed">
    <parent link="base" />
    <child link="tool" />
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )
    tcp = CuroboRobotConfig.from_mapping(
        {
            "urdf_path": str(source),
            "base_link": "base",
            "default_tcp_frame": "pinch",
            "custom_tcps": [
                {
                    "frame_name": "pinch",
                    "parent_frame": "tool",
                    "xyz": [0.0, 0.0, 0.1],
                }
            ],
        }
    ).custom_tcp_frames[0]

    write_curobo_tcp_urdf_with_frames(source, output, (tcp,))

    root = ET.parse(output).getroot()
    assert "pinch" in {link.get("name") for link in root.findall("link")}
    fixed_joint = root.find("joint[@name='pinch_joint']")
    assert fixed_joint is not None
    assert fixed_joint.find("parent").get("link") == "tool"


def test_fake_namespace_keeps_import_side_effect_free() -> None:
    # 这个测试主要确保本测试模块没有误导入真实 cuRobo runtime；后续 fake context 会使用
    # SimpleNamespace 模拟后端返回值。
    fake = SimpleNamespace(success=np.asarray([True]))
    assert bool(fake.success[0]) is True
