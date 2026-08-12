from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import shutil
from pathlib import Path

import pytest

from linkerbot_sim.backends.curobo.profile_merge import curobo_config_from_profiles
from linkerbot_sim.configuration import (
    KaleidoscopeConfig,
    KaleidoscopeEnvironmentSettings,
    MirrorConfig,
    load_kaleidoscope_config,
    load_mirror_config,
    semantic_config_fingerprint,
    semantic_config_payload,
)
from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.tasks.kaleidoscope import JointControlActionSettings
from linkerbot_sim.configuration.catalog import load_yaml_mapping
from linkerbot_sim.configuration.common import deep_freeze_configuration
from linkerbot_sim.configuration.physics import (
    NewtonCpuSettings,
    NewtonCudaSettings,
    PhysxCpuSettings,
    PhysxCudaSettings,
    physics_settings_from_mapping,
)
from linkerbot_sim.configuration.objects import RigidObjectProfileConfig
from linkerbot_sim.controllers.projection import (
    joint_control_settings,
    robot_usd_override_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_CONFIG_GROUPS = (
    "modes",
    "scenes",
    "physics",
    "tasks",
    "control",
    "curobo",
    "planning",
    "outputs",
    "training",
    "objects",
    "robots",
    "controllers",
)


def _isolated_configs(tmp_path: Path) -> Path:
    """复制新 schema 配置，使负向测试不会改动仓库 canonical 文件。"""

    root = tmp_path / "configs"
    root.mkdir(parents=True)
    for group in NEW_CONFIG_GROUPS:
        source = REPO_ROOT / "configs" / group
        if source.exists():
            shutil.copytree(source, root / group)
    return root


def _replace(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    assert old in content
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def _assert_controller_projection_reuses_profile_cache(
    config: MirrorConfig | KaleidoscopeConfig,
) -> None:
    profiles = config.controller_bundles[config.default_controller_bundle]
    first = joint_control_settings(profiles)
    second = joint_control_settings(profiles)
    assert first.default is first.arm
    assert first.arm is second.arm
    assert first.hand is second.hand

    first_overrides = robot_usd_override_configs(profiles)
    assert first_overrides["default"] is first_overrides["arm"]


def test_canonical_mirror_profiles_build_strict_backend_union() -> None:
    physx = load_mirror_config()
    newton = load_mirror_config("newton_cuda")

    assert isinstance(physx, MirrorConfig)
    assert isinstance(physx.physics, PhysxCpuSettings)
    assert physx.physics.engine == "physx"
    assert physx.physics.execution == "cpu"
    assert not hasattr(physx.physics, "cuda_device")
    assert physx.compute.cuda_device == 0
    assert physx.torch_device == "cuda:0"
    assert physx.scene.scene_id == "scene3"
    assert physx.scene.planning_startup == "prewarm"
    # 仓库 profile 自带与视觉地板对齐的解析 Plane，不能再创建第二个默认 ground。
    assert physx.scene.add_ground is False
    assert physx.scene.ground_height == 0.0
    assert {item.name for item in physx.scene.objects} == {
        "warehouse",
        "workstation",
        "Tblock",
    }
    assert physx.outputs.camera.enabled is True
    assert physx.profiles.control == "mirror"
    assert physx.profiles.curobo == "mirror"
    assert physx.profiles.planning == "mirror"
    assert physx.curobo.kinematics.max_batch_size == 8
    assert physx.curobo.motion_planner is not None
    assert not hasattr(physx.curobo.motion_planner, "max_batch_size")
    assert physx.planning.request_defaults.avoid_collisions is True
    assert physx.planning.request_defaults.sample_dt_s == 0.02
    assert physx.default_controller_bundle == "physx"
    assert physx.control.interface.admission_capacity == 32
    assert physx.control.interface.terminal_history_capacity == 256
    assert physx.control.sync_simulation_to_wall_clock is True
    assert all(item.resolved_profile is not None for item in physx.scene.robots)
    assert tuple(physx.controller_bundles) == ("physx",)
    assert physx.sources["robot.left_arm"].name == "ar5v2_l6v1_l.yaml"
    assert physx.sources["robot.right_arm"].name == "ar5v2_l6v1_r.yaml"
    assert physx.sources["controller.physx.arm"].name == "arm_controller.yaml"
    assert physx.sources["controller.physx.hand"].name == "hand_controller.yaml"

    assert isinstance(newton.physics, NewtonCudaSettings)
    assert newton.physics.engine == "newton"
    assert newton.physics.execution == "cuda"
    assert newton.compute.cuda_device == 0
    assert not hasattr(newton.physics, "cuda_device")
    assert not hasattr(newton.physics, "world_count")
    assert newton.profiles.control == "mirror"
    assert newton.profiles.curobo == "mirror"
    assert newton.profiles.planning == "mirror"
    assert newton.default_controller_bundle == "newton"
    assert newton.sources["control"].name == "mirror.yaml"
    _assert_controller_projection_reuses_profile_cache(physx)
    _assert_controller_projection_reuses_profile_cache(newton)


def test_canonical_newton_cpu_leaf_is_strictly_parseable() -> None:
    """CPU leaf 只包含 Mirror Newton CPU runtime 实际消费的字段。"""

    path = REPO_ROOT / "configs" / "physics" / "newton" / "cpu.yaml"
    document = load_yaml_mapping(path)
    physics = physics_settings_from_mapping(document["physics"], label=str(path))

    assert isinstance(physics, NewtonCpuSettings)
    assert physics.engine == "newton"
    assert physics.execution == "cpu"
    assert not hasattr(physics, "solver_type")
    assert not hasattr(physics, "use_cuda_graph")
    assert not hasattr(physics, "cuda_device")
    assert not hasattr(physics, "world_count")


def test_newton_cpu_leaf_rejects_cuda_only_graph_setting() -> None:
    path = REPO_ROOT / "configs" / "physics" / "newton" / "cpu.yaml"
    document = load_yaml_mapping(path)
    raw = dict(document["physics"])  # type: ignore[arg-type]
    raw["use_cuda_graph"] = False

    with pytest.raises(ConfigurationError, match="use_cuda_graph"):
        physics_settings_from_mapping(raw, label=str(path))


def test_newton_cpu_leaf_rejects_newton_contact_pipeline() -> None:
    path = REPO_ROOT / "configs" / "physics" / "newton" / "cpu.yaml"
    document = load_yaml_mapping(path)
    raw = dict(document["physics"])  # type: ignore[arg-type]
    raw["contact_pipeline"] = "newton"

    with pytest.raises(ConfigurationError, match="contact_pipeline=newton"):
        physics_settings_from_mapping(raw, label=str(path))


def test_mirror_interface_terminal_history_cannot_be_smaller_than_admission(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    control_path = root / "control" / "mirror.yaml"
    _replace(
        control_path,
        "    terminal_history_capacity: 256",
        "    terminal_history_capacity: 16",
    )

    with pytest.raises(ConfigurationError, match="terminal_history_capacity"):
        load_mirror_config(configs_root=root)


def test_mirror_wall_clock_sync_requires_a_strict_boolean(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    control_path = root / "control" / "mirror.yaml"
    _replace(
        control_path,
        "  sync_simulation_to_wall_clock: true",
        "  sync_simulation_to_wall_clock: 1",
    )

    with pytest.raises(ConfigurationError, match="sync_simulation_to_wall_clock"):
        load_mirror_config(configs_root=root)


def test_mirror_rejects_unproven_physx_cuda_pipeline(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "mirror" / "physx_cpu.yaml"
    _replace(mode_path, "physics: physx/cpu", "physics: physx/cuda")

    with pytest.raises(ConfigurationError, match="PhysX CPU.*Newton CUDA"):
        load_mirror_config("physx_cpu", configs_root=root)


def test_canonical_kaleidoscope_profiles_build_strict_gpu_backend_union() -> None:
    config = load_kaleidoscope_config()
    newton = load_kaleidoscope_config("newton_cuda")

    assert isinstance(config, KaleidoscopeConfig)
    assert isinstance(config.physics, PhysxCudaSettings)
    assert config.physics.engine == "physx"
    assert config.physics.execution == "cuda"
    assert config.compute.cuda_device == 0
    assert not hasattr(config.physics, "cuda_device")
    assert config.cuda_device == 0
    assert config.torch_device == "cuda:0"
    assert isinstance(config.environments, KaleidoscopeEnvironmentSettings)
    assert config.environments.num_envs == 256
    assert config.environments.base_env_path == "/World/envs"
    assert config.environments.env_prefix == "env"
    assert config.environments.origin_xyz == (0.0, 0.0, 0.0)
    assert not hasattr(config.profiles, "control")
    assert not hasattr(config, "control")
    assert config.default_controller_bundle == "physx"
    assert "control" not in config.sources
    assert config.sources["object.Tblock"].name == "TblockV1_default.yaml"
    assert all(item.resolved_profile is not None for item in config.scene.objects)
    assert all(item.resolved_profile is not None for item in config.scene.robots)
    assert tuple(config.controller_bundles) == ("physx",)
    assert not hasattr(config, "replication")
    assert not hasattr(config.profiles, "replication")
    assert "replication" not in config.sources
    assert isinstance(config.task.action, JointControlActionSettings)
    assert config.task.action.physics_ticks_per_action == 2
    assert config.profiles.curobo is None
    assert config.curobo is None
    assert "curobo" not in config.sources

    assert isinstance(newton, KaleidoscopeConfig)
    assert isinstance(newton.physics, NewtonCudaSettings)
    assert newton.physics.engine == "newton"
    assert newton.physics.execution == "cuda"
    assert newton.compute.cuda_device == 0
    assert not hasattr(newton.physics, "cuda_device")
    assert newton.cuda_device == 0
    assert newton.torch_device == "cuda:0"
    assert not hasattr(newton.physics, "world_count")
    assert newton.environments == config.environments
    assert newton.profiles.physics == "newton/cuda"
    assert not hasattr(newton.profiles, "control")
    assert not hasattr(newton, "control")
    assert newton.default_controller_bundle == "newton"
    assert newton.sources["physics"].name == "cuda.yaml"
    assert "control" not in newton.sources
    _assert_controller_projection_reuses_profile_cache(config)
    _assert_controller_projection_reuses_profile_cache(newton)


@pytest.mark.parametrize("num_envs", (0, -1, True))
def test_kaleidoscope_environment_direct_construction_rejects_invalid_count(
    num_envs: object,
) -> None:
    """公开 typed facade 的直接构造不能绕过 YAML parser 的数量约束。"""

    with pytest.raises(ConfigurationError, match="environments.num_envs"):
        KaleidoscopeEnvironmentSettings(
            num_envs=num_envs,  # type: ignore[arg-type]
            base_env_path="/World/envs",
            env_prefix="env",
            origin_xyz=(0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    "origin_xyz",
    (
        (0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, True, 0.0),
    ),
)
def test_kaleidoscope_environment_direct_construction_rejects_invalid_origin(
    origin_xyz: object,
) -> None:
    """环境逻辑原点始终是三个有限数值，不能以 bool 冒充坐标。"""

    with pytest.raises(ConfigurationError, match="environments.origin_xyz"):
        KaleidoscopeEnvironmentSettings(
            num_envs=1,
            base_env_path="/World/envs",
            env_prefix="env",
            origin_xyz=origin_xyz,  # type: ignore[arg-type]
        )


def test_kaleidoscope_environment_direct_construction_canonicalizes_origin() -> None:
    settings = KaleidoscopeEnvironmentSettings(
        num_envs=1,
        base_env_path="/World/envs",
        env_prefix="env",
        origin_xyz=[0, 1, 2],  # type: ignore[arg-type]
    )

    assert settings.origin_xyz == (0.0, 1.0, 2.0)
    assert isinstance(settings.origin_xyz, tuple)


@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
def test_kaleidoscope_root_environment_facts_enter_semantic_fingerprint(
    tmp_path: Path,
    profile: str,
) -> None:
    root = _isolated_configs(tmp_path)
    baseline = load_kaleidoscope_config(profile, configs_root=root)
    mode_path = root / "modes" / "kaleidoscope" / f"{profile}.yaml"
    _replace(
        mode_path,
        "  origin_xyz: [0.0, 0.0, 0.0]",
        "  origin_xyz: [0.25, -0.5, 1.0]",
    )

    changed = load_kaleidoscope_config(profile, configs_root=root)

    assert changed.environments.origin_xyz == (0.25, -0.5, 1.0)
    assert semantic_config_fingerprint(changed) != semantic_config_fingerprint(baseline)


@pytest.mark.parametrize(
    ("relative_path", "old", "new"),
    [
        pytest.param(
            Path("curobo/mirror.yaml"),
            "    seed_count: 32",
            "    seed_count: 24",
            id="curobo-kinematics",
        ),
        pytest.param(
            Path("curobo/mirror.yaml"),
            "    trajectory_seed_count: 4",
            "    trajectory_seed_count: 3",
            id="curobo-motion-planner",
        ),
        pytest.param(
            Path("planning/mirror.yaml"),
            "    timeout_s: 30.0",
            "    timeout_s: 12.5",
            id="planning-request-defaults",
        ),
    ],
)
def test_mirror_curobo_and_planning_facts_enter_semantic_fingerprint(
    tmp_path: Path,
    relative_path: Path,
    old: str,
    new: str,
) -> None:
    root = _isolated_configs(tmp_path)
    baseline = load_mirror_config("physx_cpu", configs_root=root)
    _replace(root / relative_path, old, new)

    changed = load_mirror_config("physx_cpu", configs_root=root)

    assert semantic_config_payload(changed) != semantic_config_payload(baseline)
    assert semantic_config_fingerprint(changed) != semantic_config_fingerprint(baseline)


def test_mirror_scene3_and_kaleidoscope_tblock_push_are_distinct_facts() -> None:
    mirror = load_mirror_config()
    kaleidoscope = load_kaleidoscope_config()

    assert mirror.scene.scene_id != kaleidoscope.scene.scene_id
    assert mirror.scene.physics_frequency_hz == 60.0
    assert kaleidoscope.scene.physics_frequency_hz == 240.0
    assert {item.label for item in mirror.scene.robots} == {"left_arm", "right_arm"}
    assert {item.label for item in kaleidoscope.scene.robots} == {
        "ar5v2_l6v1_0",
        "ar5v2_l6v1_1",
    }
    assert "warehouse" not in {item.name for item in kaleidoscope.scene.objects}
    # KaleidoscopeSceneSettings 从类型上就没有相机或渲染频率，不是解析后丢弃字段。
    assert not hasattr(kaleidoscope.scene, "cameras")
    assert not hasattr(kaleidoscope.scene, "render_frequency_hz")


def test_mirror_scene_rejects_unknown_planning_startup_policy(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    scene_path = root / "scenes" / "mirror" / "scene3.yaml"
    _replace(scene_path, "planning_startup: prewarm", "planning_startup: eager")

    with pytest.raises(ConfigurationError, match=r"planning_startup.*lazy, prewarm"):
        load_mirror_config("physx_cpu", configs_root=root)


@pytest.mark.parametrize(
    ("profile_path", "old", "new", "expected"),
    [
        (
            Path("objects/TblockV1_default.yaml"),
            "    static: false",
            "    static: true",
            "恰好包含一个非静态 rigid object",
        ),
        (
            Path("objects/workstation_armbase.yaml"),
            "    static: true",
            "    static: false",
            "恰好包含一个非静态 rigid object",
        ),
        (
            Path("tasks/kaleidoscope/tblock_push_v1.yaml"),
            "  dynamic_object: Tblock",
            "  dynamic_object: workstation",
            "必须命名 scene 中唯一的非静态 rigid object",
        ),
        (
            Path("scenes/kaleidoscope/tblock_push.yaml"),
            "      object_profile: workstation_armbase",
            "      object_profile: capsule_rope",
            "不支持 dynamic_chain",
        ),
    ],
)
def test_kaleidoscope_object_profiles_close_the_complete_state_schema(
    tmp_path: Path,
    profile_path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = _isolated_configs(tmp_path)
    _replace(root / profile_path, old, new)

    with pytest.raises(ConfigurationError, match=expected):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_mode_files_only_contain_composition_facts() -> None:
    expected_profiles = {
        "mirror": (
            "physx_cpu",
            "physx_cpu_hybrid",
            "newton_cpu",
            "newton_cuda",
        ),
        "kaleidoscope": ("physx_cuda", "newton_cuda"),
    }
    expected_scene_profiles = {
        ("mirror", "physx_cpu"): "mirror/scene3",
        ("mirror", "physx_cpu_hybrid"): "mirror/scene3_hybrid",
        ("mirror", "newton_cpu"): "mirror/scene3",
        ("mirror", "newton_cuda"): "mirror/scene3",
        ("kaleidoscope", "physx_cuda"): "kaleidoscope/tblock_push",
        ("kaleidoscope", "newton_cuda"): "kaleidoscope/tblock_push",
    }
    for mode, profiles in expected_profiles.items():
        mode_root = REPO_ROOT / "configs" / "modes" / mode
        assert {path.stem for path in mode_root.glob("*.yaml")} == set(profiles)
        for profile in profiles:
            document = load_yaml_mapping(mode_root / f"{profile}.yaml")
            expected_keys = {"mode", "profiles", "compute"}
            if mode == "kaleidoscope":
                expected_keys.add("environments")
            assert set(document) == expected_keys
            assert document["compute"] == {"cuda_device": 0}
            profile_references = document["profiles"]
            assert isinstance(profile_references, Mapping)
            assert (
                profile_references["scene"] == expected_scene_profiles[(mode, profile)]
            )
            if mode == "kaleidoscope":
                assert document["environments"] == {
                    "num_envs": 256,
                    "base_env_path": "/World/envs",
                    "env_prefix": "env",
                    "origin_xyz": [0.0, 0.0, 0.0],
                }


def test_kaleidoscope_leaf_profiles_do_not_duplicate_cuda_device() -> None:
    for profile in ("physx_cuda", "newton_cuda"):
        config = load_kaleidoscope_config(profile)
        for source_name, path in config.sources.items():
            document = load_yaml_mapping(path)
            if source_name == "mode":
                assert document["compute"] == {"cuda_device": 0}
            else:
                assert "cuda_device" not in repr(document)


def test_public_physics_profiles_use_only_engine_and_execution_discriminants() -> None:
    profile_paths = {
        REPO_ROOT / "configs" / "physics" / "physx" / "cpu.yaml": (
            "physx",
            "cpu",
        ),
        REPO_ROOT / "configs" / "physics" / "physx" / "cuda.yaml": (
            "physx",
            "cuda",
        ),
        REPO_ROOT / "configs" / "physics" / "newton" / "cpu.yaml": (
            "newton",
            "cpu",
        ),
        REPO_ROOT / "configs" / "physics" / "newton" / "cuda.yaml": (
            "newton",
            "cuda",
        ),
    }

    for path, expected in profile_paths.items():
        physics = load_yaml_mapping(path)["physics"]
        assert isinstance(physics, Mapping)
        assert (physics["engine"], physics["execution"]) == expected
        assert "kind" not in physics
        assert "cuda_device" not in physics
        assert "world_count" not in physics


@pytest.mark.parametrize(
    "legacy_path",
    [
        Path("modes/mirror/default.yaml"),
        Path("modes/mirror/physx.yaml"),
        Path("modes/mirror/newton.yaml"),
        Path("modes/kaleidoscope/default.yaml"),
        Path("modes/kaleidoscope/default_newton_cuda.yaml"),
        Path("modes/kaleidoscope/physx.yaml"),
        Path("modes/kaleidoscope/newton.yaml"),
        Path("physics/newton/direct.yaml"),
        Path("control/mirror_default.yaml"),
        Path("control/kaleidoscope_position.yaml"),
        Path("controllers/default"),
        Path("controllers/newton_cuda"),
        Path("curobo/default.yaml"),
        Path("kinematics"),
        Path("planning/curobo"),
        Path("replication"),
        Path("replication/grid_cuda.yaml"),
        Path("replication/newton_worlds.yaml"),
        Path("scenes/scene3_mirror.yaml"),
        Path("scenes/tblock_push_scene.yaml"),
    ],
)
def test_legacy_public_profile_names_are_removed(legacy_path: Path) -> None:
    assert not (REPO_ROOT / "configs" / legacy_path).exists()


def test_scene_catalog_root_contains_only_product_namespaces() -> None:
    scene_root = REPO_ROOT / "configs" / "scenes"

    assert not tuple(scene_root.glob("*.yaml"))
    assert {path.name for path in scene_root.iterdir() if path.is_dir()} == {
        "kaleidoscope",
        "mirror",
    }


def test_old_physics_kind_discriminant_is_not_accepted(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    physics_path = root / "physics" / "physx" / "cpu.yaml"
    _replace(
        physics_path,
        "  engine: physx",
        "  kind: physx_cpu",
    )

    with pytest.raises(ConfigurationError, match="physics.engine"):
        load_mirror_config("physx_cpu", configs_root=root)


@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
def test_kaleidoscope_requires_root_compute(
    tmp_path: Path,
    profile: str,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / f"{profile}.yaml"
    content = mode_path.read_text(encoding="utf-8")
    start = content.index("compute:\n")
    end = content.index("environments:\n", start)
    mode_path.write_text(content[:start] + content[end:], encoding="utf-8")

    with pytest.raises(ConfigurationError, match="compute"):
        load_kaleidoscope_config(profile, configs_root=root)


@pytest.mark.parametrize(
    ("profile", "path", "old", "new", "expected"),
    [
        (
            "physx_cuda",
            Path("physics/physx/cuda.yaml"),
            "  execution: cuda",
            "  execution: cuda\n  cuda_device: 0",
            "cuda_device",
        ),
        (
            "physx_cuda",
            Path("modes/kaleidoscope/physx_cuda.yaml"),
            "  cuda_device: 0",
            "  cuda_device: 0\n  active_gpu: 0",
            "active_gpu",
        ),
        (
            "newton_cuda",
            Path("physics/newton/cuda.yaml"),
            "  execution: cuda",
            "  execution: cuda\n  cuda_device: 0",
            "cuda_device",
        ),
    ],
)
def test_kaleidoscope_rejects_duplicate_device_facts(
    tmp_path: Path,
    profile: str,
    path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = _isolated_configs(tmp_path)
    _replace(root / path, old, new)

    with pytest.raises(ConfigurationError, match=expected):
        load_kaleidoscope_config(profile, configs_root=root)


def test_physx_cpu_mirror_compute_is_consumed_by_curobo() -> None:
    config = load_mirror_config("physx_cpu")
    robot_profile = config.scene.robots[0].resolved_profile
    assert robot_profile is not None

    backend = curobo_config_from_profiles(
        robot_profile,
        curobo_settings=config.curobo,
        cuda_device=config.cuda_device,
    )

    assert backend.device.device == "cuda:0"
    assert backend.ik.num_seeds == 32
    assert backend.ik.seed_solver_num_seeds == 32
    assert backend.ik.max_batch_size == 8
    assert backend.ik.use_cuda_graph is True
    assert backend.ik.self_collision_check is False
    assert backend.ik.collision_cache == {}
    assert backend.motion_planner.use_cuda_graph is False
    assert backend.motion_planner.collision_cache == {
        "cuboid": 48,
        "mesh": 4,
    }


def test_mirror_rejects_motion_planner_cuda_graph(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    curobo_path = root / "curobo" / "mirror.yaml"
    _replace(
        curobo_path,
        "    use_cuda_graph: false",
        "    use_cuda_graph: true",
    )

    with pytest.raises(
        ConfigurationError, match=r"motion_planner\.use_cuda_graph 必须为 false"
    ):
        load_mirror_config("physx_cpu", configs_root=root)


def test_mirror_rejects_legacy_kinematics_profile_slot(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "mirror" / "physx_cpu.yaml"
    _replace(mode_path, "  curobo: mirror", "  kinematics: curobo/mirror_default")

    with pytest.raises(ConfigurationError, match="curobo|kinematics"):
        load_mirror_config("physx_cpu", configs_root=root)


@pytest.mark.parametrize("profile", ["physx_cpu", "newton_cpu", "newton_cuda"])
def test_mirror_requires_root_compute(tmp_path: Path, profile: str) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "mirror" / f"{profile}.yaml"
    content = mode_path.read_text(encoding="utf-8")
    start = content.index("compute:\n")
    end = content.index("profiles:\n", start)
    mode_path.write_text(content[:start] + content[end:], encoding="utf-8")

    with pytest.raises(ConfigurationError, match="compute"):
        load_mirror_config(profile, configs_root=root)


def test_kaleidoscope_rejects_physx_cpu_backend(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(mode_path, "physics: physx/cuda", "physics: physx/cpu")

    with pytest.raises(ConfigurationError, match="PhysX CUDA.*Newton CUDA"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_newton_physics_leaf_rejects_duplicate_world_count(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    physics_path = root / "physics" / "newton" / "cuda.yaml"
    _replace(
        physics_path,
        "  execution: cuda",
        "  execution: cuda\n  world_count: 255",
    )

    with pytest.raises(ConfigurationError, match="world_count"):
        load_kaleidoscope_config("newton_cuda", configs_root=root)


def test_mirror_accepts_newton_cpu_runtime() -> None:
    config = load_mirror_config("newton_cpu")

    assert isinstance(config.physics, NewtonCpuSettings)
    assert config.physics.execution == "cpu"
    assert not hasattr(config.physics, "use_cuda_graph")


def test_kaleidoscope_rejects_newton_cpu_backend(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "newton_cuda.yaml"
    _replace(mode_path, "physics: newton/cuda", "physics: newton/cpu")

    with pytest.raises(ConfigurationError, match="PhysX CUDA.*Newton CUDA"):
        load_kaleidoscope_config("newton_cuda", configs_root=root)


@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
def test_kaleidoscope_requires_root_environments(
    tmp_path: Path,
    profile: str,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / f"{profile}.yaml"
    content = mode_path.read_text(encoding="utf-8")
    start = content.index("environments:\n")
    end = content.index("profiles:\n", start)
    mode_path.write_text(content[:start] + content[end:], encoding="utf-8")

    with pytest.raises(ConfigurationError, match="environments"):
        load_kaleidoscope_config(profile, configs_root=root)


def test_kaleidoscope_rejects_legacy_replication_profile_selector(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(
        mode_path,
        "  physics: physx/cuda",
        "  physics: physx/cuda\n  replication: grid_envs",
    )

    with pytest.raises(ConfigurationError, match="replication"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize(
    ("addition", "expected"),
    [
        ("  spacing_m: 3.0\n", "spacing_m"),
        (
            "  clone:\n    replicate_physics: true\n    copy_from_source: true\n",
            "clone",
        ),
        ("  collision_isolation:\n    strategy: env_ids\n", "collision_isolation"),
    ],
)
def test_kaleidoscope_environments_rejects_backend_replication_policy(
    tmp_path: Path,
    addition: str,
    expected: str,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(
        mode_path,
        "  origin_xyz: [0.0, 0.0, 0.0]\n",
        f"  origin_xyz: [0.0, 0.0, 0.0]\n{addition}",
    )

    with pytest.raises(ConfigurationError, match=expected):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("  num_envs: 256", "  num_envs: 0", "num_envs.*>= 1"),
        (
            "  base_env_path: /World/envs",
            "  base_env_path: World/envs",
            "base_env_path.*绝对 USD path",
        ),
        (
            "  base_env_path: /World/envs",
            "  base_env_path: /",
            "base_env_path.*非根",
        ),
        (
            "  base_env_path: /World/envs",
            "  base_env_path: /World//envs",
            "base_env_path.*空 USD path component",
        ),
        ("  env_prefix: env", "  env_prefix: env/nested", "env_prefix"),
        ("  env_prefix: env", "  env_prefix: ''", "env_prefix"),
        (
            "  origin_xyz: [0.0, 0.0, 0.0]",
            "  origin_xyz: [0.0, 0.0]",
            "origin_xyz.*3",
        ),
    ],
)
def test_kaleidoscope_environments_are_strictly_validated(
    tmp_path: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(mode_path, old, new)

    with pytest.raises(ConfigurationError, match=expected):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize(
    ("path", "addition", "concept"),
    [
        (
            Path("scenes/kaleidoscope/tblock_push.yaml"),
            "\n  cameras: []\n",
            "camera",
        ),
        (
            Path("tasks/kaleidoscope/tblock_push_v1.yaml"),
            "\n  planner: {}\n",
            "planner",
        ),
        (
            Path("modes/kaleidoscope/physx_cuda.yaml"),
            "\n  telemetry: {}\n",
            "telemetry",
        ),
    ],
)
def test_kaleidoscope_recursively_rejects_forbidden_product_concepts(
    tmp_path: Path,
    path: Path,
    addition: str,
    concept: str,
) -> None:
    root = _isolated_configs(tmp_path)
    target = root / path
    target.write_text(target.read_text(encoding="utf-8") + addition, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=concept):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_joint_delta_action_rejects_ik_or_linear_fields(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    task_path = root / "tasks" / "kaleidoscope" / "tblock_push_v1.yaml"
    _replace(
        task_path,
        "    clip: 1.0",
        "    clip: 1.0\n    kinematics_profile: curobo/kaleidoscope_batch",
    )

    with pytest.raises(ConfigurationError, match="kinematics_profile"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize(
    "action",
    [
        """  action:
    mode: ee_pose_full
    physics_ticks_per_action: 2
    reference_velocity_limit_rad_s: 1.0
    failure_policy: hold_penalty_truncate
""",
        """  action:
    mode: ee_linear_path_full
    waypoint_count: 8
    physics_ticks_per_action: 20
    reference_velocity_limit_rad_s: 1.0
    progress_mode: linear
    failure_policy: hold_from_first_failure
""",
    ],
)
@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
def test_ee_and_linear_actions_load_noncollision_batch_kinematics(
    tmp_path: Path,
    action: str,
    profile: str,
) -> None:
    root = _isolated_configs(tmp_path)
    task_path = root / "tasks" / "kaleidoscope" / "tblock_push_v1.yaml"
    content = task_path.read_text(encoding="utf-8")
    start = content.index("  action:\n")
    end = content.index("\n  observation:\n", start)
    task_path.write_text(content[:start] + action + content[end:], encoding="utf-8")
    mode_path = root / "modes" / "kaleidoscope" / f"{profile}.yaml"
    _replace(
        mode_path,
        "  task: kaleidoscope/tblock_push_v1",
        "  task: kaleidoscope/tblock_push_v1\n  curobo: kaleidoscope_batch_ik",
    )

    config = load_kaleidoscope_config(profile, configs_root=root)

    assert config.profiles.curobo == "kaleidoscope_batch_ik"
    assert config.curobo is not None
    assert config.curobo.motion_planner is None
    assert config.curobo.kinematics.max_batch_size == 1024
    assert config.curobo.kinematics.collision_check is False
    assert config.curobo.kinematics.collision_cache is None
    assert "curobo" in config.sources


def test_ee_action_requires_mode_curobo_profile(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    task_path = root / "tasks" / "kaleidoscope" / "tblock_push_v1.yaml"
    content = task_path.read_text(encoding="utf-8")
    start = content.index("  action:\n")
    end = content.index("\n  observation:\n", start)
    action = """  action:
    mode: ee_pose_full
    physics_ticks_per_action: 2
    reference_velocity_limit_rad_s: 1.0
    failure_policy: hold_penalty_truncate
"""
    task_path.write_text(content[:start] + action + content[end:], encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"profiles\.curobo"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_joint_delta_rejects_mode_curobo_profile(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(
        mode_path,
        "  task: kaleidoscope/tblock_push_v1",
        "  task: kaleidoscope/tblock_push_v1\n  curobo: kaleidoscope_batch_ik",
    )

    with pytest.raises(ConfigurationError, match=r"profiles\.curobo"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize("profile", ["physx_cuda", "newton_cuda"])
def test_kaleidoscope_accepts_dormant_kinematics_collision_cache(
    tmp_path: Path,
    profile: str,
) -> None:
    root = _isolated_configs(tmp_path)
    task_path = root / "tasks" / "kaleidoscope" / "tblock_push_v1.yaml"
    content = task_path.read_text(encoding="utf-8")
    start = content.index("  action:\n")
    end = content.index("\n  observation:\n", start)
    ee_action = """  action:
    mode: ee_pose_full
    physics_ticks_per_action: 2
    reference_velocity_limit_rad_s: 1.0
    failure_policy: hold_penalty_truncate
"""
    task_path.write_text(
        content[:start] + ee_action + content[end:],
        encoding="utf-8",
    )
    mode_path = root / "modes" / "kaleidoscope" / f"{profile}.yaml"
    _replace(
        mode_path,
        "  task: kaleidoscope/tblock_push_v1",
        "  task: kaleidoscope/tblock_push_v1\n  curobo: kaleidoscope_batch_ik",
    )
    curobo_path = root / "curobo" / "kaleidoscope_batch_ik.yaml"
    _replace(
        curobo_path,
        "    use_cuda_graph: true",
        "    use_cuda_graph: true\n"
        "    collision_cache:\n"
        "      cuboid: 0\n"
        "      mesh: 0",
    )

    config = load_kaleidoscope_config(profile, configs_root=root)

    assert config.curobo is not None
    cache = config.curobo.kinematics.collision_cache
    assert cache is not None
    assert cache.as_backend_mapping() == {"cuboid": 0, "mesh": 0}


def test_kaleidoscope_rejects_replication_at_mode_root(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(
        mode_path,
        "profiles:\n",
        "replication:\n  strategy: env_ids\nprofiles:\n",
    )

    with pytest.raises(ConfigurationError, match="replication"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_kaleidoscope_rejects_non_boolean_fabric_flag(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    physics_path = root / "physics" / "physx" / "cuda.yaml"
    _replace(physics_path, "  use_fabric: true", "  use_fabric: 1")

    with pytest.raises(ConfigurationError, match="use_fabric.*boolean"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_kaleidoscope_rejects_scene_query_manager(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    physics_path = root / "physics" / "physx" / "cuda.yaml"
    _replace(
        physics_path,
        "  enable_scene_query_support: false",
        "  enable_scene_query_support: true",
    )

    with pytest.raises(ConfigurationError, match="collision-query"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


def test_mirror_rejects_rl_profile_slot(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "mirror" / "physx_cpu.yaml"
    _replace(
        mode_path,
        "  outputs: mirror_default",
        "  outputs: mirror_default\n  task: kaleidoscope/tblock_push_v1",
    )

    with pytest.raises(ConfigurationError, match="task"):
        load_mirror_config("physx_cpu", configs_root=root)


def test_profile_reference_cannot_escape_group_directory(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    _replace(
        mode_path,
        "task: kaleidoscope/tblock_push_v1",
        "task: ../modes/kaleidoscope/physx",
    )

    with pytest.raises(ConfigurationError, match="非法路径分量"):
        load_kaleidoscope_config("physx_cuda", configs_root=root)


@pytest.mark.parametrize(
    ("mode", "profile", "canonical", "invalid", "expected"),
    [
        ("mirror", "physx_cpu", "mirror/scene3", "scene3", "mirror.*namespace"),
        (
            "mirror",
            "physx_cpu",
            "mirror/scene3",
            "kaleidoscope/tblock_push",
            "mirror.*namespace",
        ),
        (
            "kaleidoscope",
            "physx_cuda",
            "kaleidoscope/tblock_push",
            "mirror/scene3",
            "kaleidoscope.*namespace",
        ),
        (
            "mirror",
            "physx_cpu",
            "mirror/scene3",
            r"mirror\scene3",
            "namespace 分隔符",
        ),
        (
            "mirror",
            "physx_cpu",
            "mirror/scene3",
            "mirror/scene3.yaml",
            "不带扩展名",
        ),
        (
            "mirror",
            "physx_cpu",
            "mirror/scene3",
            "mirror/scene3.v2",
            "不能包含",
        ),
    ],
)
def test_scene_profile_reference_requires_exact_product_namespace(
    tmp_path: Path,
    mode: str,
    profile: str,
    canonical: str,
    invalid: str,
    expected: str,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / mode / f"{profile}.yaml"
    _replace(mode_path, f"scene: {canonical}", f"scene: {invalid}")
    loader = load_mirror_config if mode == "mirror" else load_kaleidoscope_config

    with pytest.raises(ConfigurationError, match=expected):
        loader(profile, configs_root=root)


def test_scene_profile_symlink_cannot_cross_product_namespace(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    mirror_scene = root / "scenes" / "mirror" / "scene3.yaml"
    mirror_scene.unlink()
    mirror_scene.symlink_to(Path("../kaleidoscope/tblock_push.yaml"))

    with pytest.raises(
        ConfigurationError,
        match=r"profiles\.scenes 'mirror' namespace.*逃逸",
    ):
        load_mirror_config("physx_cpu", configs_root=root)


def test_scene_group_root_symlink_cannot_escape_configs_root(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    scene_root = root / "scenes"
    external_scene_root = tmp_path / "external-scenes"
    shutil.copytree(scene_root, external_scene_root)
    shutil.rmtree(scene_root)
    scene_root.symlink_to(external_scene_root, target_is_directory=True)

    with pytest.raises(
        ConfigurationError,
        match=r"profiles\.scenes root.*逃逸配置根目录",
    ):
        load_mirror_config("physx_cpu", configs_root=root)


def test_scene_namespace_directory_cannot_alias_another_product(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    mirror_root = root / "scenes" / "mirror"
    shutil.rmtree(mirror_root)
    mirror_root.symlink_to("kaleidoscope", target_is_directory=True)

    with pytest.raises(
        ConfigurationError,
        match=r"profiles\.scenes 'mirror' namespace 根目录不能是符号链接",
    ):
        load_mirror_config("physx_cpu", configs_root=root)


def test_missing_scene_selector_is_reported_as_configuration_error(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    mode_path = root / "modes" / "mirror" / "physx_cpu.yaml"
    _replace(mode_path, "scene: mirror/scene3", "scene: mirror/missing")

    with pytest.raises(
        ConfigurationError,
        match=r"无法读取 YAML 配置 .*scenes/mirror/missing\.yaml",
    ):
        load_mirror_config("physx_cpu", configs_root=root)


def test_explicit_mode_path_infers_its_isolated_configs_root(tmp_path: Path) -> None:
    root = _isolated_configs(tmp_path)
    config = load_kaleidoscope_config(
        root / "modes" / "kaleidoscope" / "physx_cuda.yaml"
    )

    assert (
        config.sources["mode"]
        == (root / "modes" / "kaleidoscope" / "physx_cuda.yaml").resolve()
    )
    assert (
        config.sources["scene"]
        == (root / "scenes" / "kaleidoscope" / "tblock_push.yaml").resolve()
    )


def test_custom_configs_root_closes_robot_and_controller_resources(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    robot_path = root / "robots" / "ar5v2_l6v1_l.yaml"
    controller_path = root / "controllers" / "physx" / "arm_controller.yaml"
    _replace(robot_path, "  name: ar5v2_l6v1_l", "  name: custom_left_robot")
    _replace(controller_path, "    stiffness: 1000.0", "    stiffness: 321.0")

    config = load_mirror_config("physx_cpu", configs_root=root)
    left_profile = config.scene.robots[0].resolved_profile

    assert left_profile is not None
    assert left_profile.name == "custom_left_robot"
    assert config.controller_bundles["physx"].arm.position_control.stiffness == (321.0,)
    assert config.sources["robot.left_arm"] == robot_path.resolve()
    assert config.sources["controller.physx.arm"] == controller_path.resolve()


def test_mirror_assembly_consumes_custom_root_object_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linkerbot_sim.mirror.scene_assembly as mirror_assembly

    root = _isolated_configs(tmp_path)
    object_path = root / "objects" / "TblockV1_default.yaml"
    _replace(object_path, "      static_friction: 0.8", "      static_friction: 0.37")
    config = load_mirror_config("physx_cpu", configs_root=root)
    captured: dict[str, object] = {}

    class _AssemblyCaptured(RuntimeError):
        pass

    def capture_resources(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _AssemblyCaptured("captured")

    monkeypatch.setattr(mirror_assembly, "prepare_mcap_output", lambda *_a, **_k: None)
    monkeypatch.setattr(
        mirror_assembly,
        "create_mirror_scene_resources",
        capture_resources,
    )

    with pytest.raises(_AssemblyCaptured, match="captured"):
        mirror_assembly.build_mirror_assembly(config)

    scene = captured["scene"]
    assert scene is config.scene
    assert captured["controller_bundles"] is config.controller_bundles
    tblock = next(item for item in config.scene.objects if item.name == "Tblock")
    assert isinstance(tblock.resolved_profile, RigidObjectProfileConfig)
    assert tblock.resolved_profile.physics.material is not None
    assert tblock.resolved_profile.physics.material.static_friction == 0.37


def test_instance_controller_override_is_resolved_from_custom_root(
    tmp_path: Path,
) -> None:
    root = _isolated_configs(tmp_path)
    shutil.copytree(
        root / "controllers" / "physx",
        root / "controllers" / "lab",
    )
    scene_path = root / "scenes" / "mirror" / "scene3.yaml"
    _replace(
        scene_path,
        "      robot_profile: ar5v2_l6v1_l",
        "      robot_profile: ar5v2_l6v1_l\n      controller_profile: lab",
    )

    config = load_mirror_config("physx_cpu", configs_root=root)

    assert config.scene.robots[0].controller_profile == "lab"
    assert tuple(config.controller_bundles) == ("lab", "physx")
    assert (
        config.sources["controller.lab.arm"]
        == (root / "controllers" / "lab" / "arm_controller.yaml").resolve()
    )


def test_newton_mode_fingerprint_is_root_independent_but_tracks_controller_data(
    tmp_path: Path,
) -> None:
    from scripts.validate_mode_config import _json_value

    first_root = _isolated_configs(tmp_path / "first")
    second_root = _isolated_configs(tmp_path / "second")
    first = load_kaleidoscope_config("newton_cuda", configs_root=first_root)
    second = load_kaleidoscope_config("newton_cuda", configs_root=second_root)

    first_payload = _json_value(first)
    second_payload = semantic_config_payload(second)
    assert first_payload == second_payload
    assert "sources" not in first_payload  # type: ignore[operator]
    assert semantic_config_fingerprint(first) == semantic_config_fingerprint(second)

    controller_path = second_root / "controllers" / "newton" / "arm_controller.yaml"
    _replace(controller_path, "    stiffness: 200.0", "    stiffness: 199.0")
    changed = load_kaleidoscope_config("newton_cuda", configs_root=second_root)

    assert _json_value(changed) != first_payload
    assert semantic_config_fingerprint(changed) != semantic_config_fingerprint(first)


def test_deep_freeze_preserves_semantic_payload_and_fingerprint() -> None:
    mutable = {
        "nested": {
            "sequence": [1, {"enabled": True}],
            "tuple": ("left", "right"),
        }
    }
    payload = semantic_config_payload(mutable)
    fingerprint = semantic_config_fingerprint(mutable)

    frozen = deep_freeze_configuration(mutable)

    assert semantic_config_payload(frozen) == payload
    assert semantic_config_fingerprint(frozen) == fingerprint
    mutable["nested"]["sequence"][1]["enabled"] = False  # type: ignore[index]
    assert semantic_config_payload(frozen) == payload
    assert semantic_config_fingerprint(frozen) == fingerprint


@pytest.mark.parametrize(
    ("loader", "profile"),
    [
        (load_mirror_config, "physx_cpu"),
        (load_mirror_config, "newton_cpu"),
        (load_mirror_config, "newton_cuda"),
        (load_kaleidoscope_config, "physx_cuda"),
        (load_kaleidoscope_config, "newton_cuda"),
    ],
)
def test_resolved_configuration_graph_is_deeply_immutable(
    loader: Callable[[str], MirrorConfig | KaleidoscopeConfig],
    profile: str,
) -> None:
    config = loader(profile)
    payload = semantic_config_payload(config)
    fingerprint = semantic_config_fingerprint(config)

    robot_profile = config.scene.robots[0].resolved_profile
    assert robot_profile is not None
    gravity = robot_profile.gravity_policy
    with pytest.raises((AttributeError, TypeError)):
        gravity.arm = True  # type: ignore[misc]

    arm_joints = robot_profile.joint_groups.arm
    assert isinstance(arm_joints, tuple)
    with pytest.raises(TypeError):
        arm_joints[0] = "mutated_joint"  # type: ignore[index]

    object_profile = next(
        item.resolved_profile for item in config.scene.objects if item.name == "Tblock"
    )
    assert isinstance(object_profile, RigidObjectProfileConfig)
    assert object_profile.physics.material is not None
    with pytest.raises((AttributeError, TypeError)):
        object_profile.physics.material.static_friction = 0.0  # type: ignore[misc]

    controller_name = config.default_controller_bundle
    position = config.controller_bundles[controller_name].arm.position_control
    stiffness = position.stiffness
    assert isinstance(stiffness, tuple)
    with pytest.raises(TypeError):
        stiffness[0] = 0.0  # type: ignore[index]

    assert semantic_config_payload(config) == payload
    assert semantic_config_fingerprint(config) == fingerprint


def test_validator_reports_the_public_semantic_fingerprint(capsys) -> None:
    from scripts.validate_mode_config import main

    assert main(["--mode", "kaleidoscope", "--profile", "newton_cuda"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["fingerprint"] == semantic_config_fingerprint(
        load_kaleidoscope_config("newton_cuda")
    )


def test_new_loader_rejects_duplicate_keys_at_any_depth(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("outer:\n  value: 1\n  value: 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate mapping key 'value'"):
        load_yaml_mapping(path)


def test_new_loader_rejects_non_string_keys_at_any_depth(tmp_path: Path) -> None:
    path = tmp_path / "non-string-key.yaml"
    path.write_text(
        "object:\n  physics:\n    material:\n      1: 0.5\n      typo: 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match=r"object\.physics\.material 的键必须是非空字符串，得到 1",
    ):
        load_yaml_mapping(path)


def test_new_loader_rejects_recursive_yaml_aliases(tmp_path: Path) -> None:
    path = tmp_path / "recursive-alias.yaml"
    path.write_text("root: &root\n  self: *root\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"root\.self 不得包含递归 YAML alias"):
        load_yaml_mapping(path)


def test_new_loader_retains_safe_yaml_tag_restrictions(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "value: !!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="could not determine a constructor"):
        load_yaml_mapping(path)


def test_configuration_facade_has_no_runtime_dependencies() -> None:
    # 这是 source-level import closure contract：配置包不能因可选仿真依赖未安装而失效。
    configuration_root = REPO_ROOT / "src" / "linkerbot_sim" / "configuration"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in configuration_root.glob("*.py")
    )
    for forbidden_import in (
        "import omni",
        "from omni",
        "import isaacsim",
        "from isaacsim",
        "import torch",
        "from torch",
        "import curobo",
        "from curobo",
        "import gymnasium",
        "from gymnasium",
    ):
        assert forbidden_import not in sources
