from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.configs.cli import (
    resolve_runtime_profile as resolve_runtime_profile_graph,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
import linkerbot_sim.configs.profiles as profile_loader
from linkerbot_sim.configs.runtime import load_runtime_profile, resolve_runtime_config
from linkerbot_sim.configs.validator import validate_profile_graph
import linkerbot_sim.configs.validator as graph_validator
from linkerbot_sim.envs.config import validate_env_profile, validate_per_env_fragment
from linkerbot_sim.sensors import SceneSensorSettings
from linkerbot_sim.tiled.config import TiledEnvConfig


_BUNDLED_RUNTIME_PROFILE_NAMES = tuple(
    path.stem for path in sorted(Path("configs/runtime").glob("*.yaml"))
)
_BUNDLED_ENV_PROFILE_NAMES = tuple(
    sorted(
        {
            *(path.stem for path in Path("configs/envs").glob("*.yaml")),
            *(path.parent.name for path in Path("configs/envs").glob("*/base.yaml")),
        }
    )
)


def _minimal_env(**env_updates: object) -> dict[str, object]:
    env: dict[str, object] = {"name": "unit", **env_updates}
    return {
        "env": env,
        "robots": [
            {
                "label": "robot_0",
                "robot_profile": "unit_robot",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            }
        ],
    }


@pytest.mark.parametrize("profile_name", _BUNDLED_ENV_PROFILE_NAMES)
def test_all_bundled_env_profiles_pass_the_strict_loader(profile_name: str) -> None:
    load_profile_yaml("env", profile_name)


@pytest.mark.parametrize("runtime_name", _BUNDLED_RUNTIME_PROFILE_NAMES)
def test_all_bundled_runtime_profiles_resolve_complete_graphs(
    runtime_name: str,
) -> None:
    profile, resolved = resolve_runtime_profile_graph(runtime_name)

    assert profile.profiles.env == resolved.profiles.env


def test_configs_package_and_validator_import_without_cycle() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import linkerbot_sim.configs; "
                "import linkerbot_sim.objects.config; "
                "import linkerbot_sim.configs.validator"
            ),
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_generic_robot_profile_loader_uses_strict_domain_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "broken.yaml").write_text(
        """
robot:
  kind: arm
  asset_type: mjcf
  asset_path: robot.xml
typo_section: {}
curobo:
  enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setitem(profile_loader.PROFILE_GROUP_DIRS, "robot", tmp_path)

    with pytest.raises(ValueError, match=r"typo_section"):
        load_profile_yaml("robot", "broken")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("env.add_ground", "false"),
        ("env.add_ground", 0),
        ("env.physics_frequency", "240"),
        ("env.gravity_z", True),
    ),
)
def test_env_scalar_types_are_strict(path: str, value: object) -> None:
    key = path.rsplit(".", maxsplit=1)[1]
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        validate_env_profile(_minimal_env(**{key: value}))


def test_env_runtime_settings_never_treats_false_string_as_true() -> None:
    with pytest.raises(ValueError, match=r"env\.add_ground must be a boolean"):
        EnvRuntimeSettings.from_env_config(
            {"env": {"name": "unit", "add_ground": "false"}}
        )


@pytest.mark.parametrize("field", ("planner", "telemetry", "playback", "transport"))
def test_env_rejects_unknown_top_level_fields(field: str) -> None:
    data = _minimal_env()
    data[field] = {}
    with pytest.raises(ValueError, match=rf"env profile\.{field} is not supported"):
        validate_env_profile(data)


def test_tiled_rejects_unknown_runtime_section_with_full_path() -> None:
    with pytest.raises(ValueError, match=r"unsupported keys: runtime.*tiled\.runtime"):
        TiledEnvConfig.from_env_config(
            {
                "tiled": {
                    "enabled": True,
                    "runtime": {"planner": {"backend": "curobo"}},
                },
            }
        )


def test_tiled_unknown_nested_key_reports_complete_path() -> None:
    with pytest.raises(ValueError, match=r"tiled\.clone\.copy_form_source"):
        TiledEnvConfig.from_env_config(
            {
                "tiled": {"clone": {"copy_form_source": True}},
            }
        )


@pytest.mark.parametrize(
    ("mutation", "path"),
    (
        ({"visuals": None}, "visuals"),
        ({"sensors": None}, "sensors"),
        ({"tiled": {"layout": None}}, "tiled.layout"),
        ({"tiled": {"clone": None}}, "tiled.clone"),
        ({"tiled": {"diagnostics": None}}, "tiled.diagnostics"),
    ),
)
def test_env_rejects_null_non_nullable_mapping_sections(
    mutation: dict[str, object], path: str
) -> None:
    data = _minimal_env()
    data.update(mutation)

    with pytest.raises(
        ValueError,
        match=path.replace(".", r"\.") + " must be a mapping",
    ):
        validate_env_profile(data)


def test_env_rejects_unknown_visual_camera_field() -> None:
    data = _minimal_env()
    data["visuals"] = {
        "camera": {
            "enabled": True,
            "eye": [1, 2, 3],
            "target": [0, 0, 0],
            "prim_path": "/Viewport",
        }
    }
    with pytest.raises(ValueError, match=r"visuals\.camera"):
        validate_env_profile(data)


def test_unknown_nested_visual_and_camera_fields_report_full_paths() -> None:
    visual = _minimal_env()
    visual["visuals"] = {"viewport": {"targte": [0, 0, 0]}}
    with pytest.raises(ValueError, match=r"visuals\.viewport\.targte"):
        validate_env_profile(visual)

    camera = _minimal_env()
    camera["sensors"] = {
        "cameras": {"world": {"prim_path": "/World/Camera", "frequncy": 30}}
    }
    with pytest.raises(ValueError, match=r"sensors\.cameras\.world\.frequncy"):
        validate_env_profile(camera)


def test_camera_clipping_range_parses_after_length_validation() -> None:
    settings = SceneSensorSettings.from_env_config(
        {
            "sensors": {
                "cameras": {
                    "world": {
                        "prim_path": "/World/Camera",
                        "clipping_range": [0.05, 8.0],
                    }
                }
            }
        }
    )
    assert settings.cameras[0].clipping_range == (0.05, 8.0)


def test_per_env_fragment_rejects_unknown_nested_pose_field() -> None:
    with pytest.raises(
        ValueError,
        match=r"tiled\.per_env\[0\]\.objects\.rope\.root_pose.*rrpy",
    ):
        validate_per_env_fragment(
            {
                "env_id": 0,
                "objects": {
                    "rope": {
                        "root_pose": {
                            "xyz": [0, 0, 0],
                            "rrpy": [0, 0, 0],
                        }
                    }
                },
            }
        )


def test_per_env_fragment_requires_explicit_env_id() -> None:
    source = "/tmp/env_003.yaml"
    with pytest.raises(
        ValueError, match=r"per-env profile\.env_id is required"
    ) as exc_info:
        validate_per_env_fragment({"objects": {}}, source_path=source)

    assert source in str(exc_info.value)


@pytest.mark.parametrize(
    "runtime_name", ("default_single_scene", "default_tiled_scene")
)
def test_bundled_runtime_profile_validates_complete_dependency_graph(
    runtime_name: str,
) -> None:
    profile = load_runtime_profile(runtime_name)
    env_config = load_profile_yaml("env", profile.profiles.env)
    resolved = resolve_runtime_config(
        profile,
        cli_overrides={},
        env_config=env_config,
    )

    graph = validate_profile_graph(
        runtime_profile=runtime_name,
        profile=profile,
        resolved=resolved,
        env_config=env_config,
    )

    assert graph.dependencies["runtime"] == (runtime_name,)
    assert graph.dependencies["env"] == (profile.profiles.env,)
    assert graph.dependencies["robot"]
    assert graph.dependencies["planning_robot"]
    assert graph.dependencies["object"]
    assert graph.dependencies["controller"]
    assert graph.dependencies["curobo"] == (profile.profiles.curobo,)
    assert graph.dependencies["logging"] == (profile.profiles.logging,)


def _default_graph_inputs():
    profile = load_runtime_profile("default_single_scene")
    env_config = load_profile_yaml("env", profile.profiles.env)
    resolved = resolve_runtime_config(
        profile,
        cli_overrides={},
        env_config=env_config,
    )
    return profile, env_config, resolved


def test_profile_graph_rejects_robot_curobo_resource_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, env_config, resolved = _default_graph_inputs()
    original = graph_validator.load_robot_profile

    def load_with_typo(path):
        data = deepcopy(original(path))
        data["curobo"]["robot"]["urdf_pth"] = "typo.urdf"
        return data

    monkeypatch.setattr(graph_validator, "load_robot_profile", load_with_typo)
    with pytest.raises(ValueError, match=r"curobo\.robot\.urdf_pth"):
        validate_profile_graph(
            runtime_profile="default",
            profile=profile,
            resolved=resolved,
            env_config=env_config,
        )


def test_profile_graph_rejects_merged_robot_without_curobo_model_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, env_config, resolved = _default_graph_inputs()
    original = graph_validator.load_robot_profile

    def load_without_resource(path):
        data = deepcopy(original(path))
        robot = data["curobo"]["robot"]
        robot.pop("urdf_path", None)
        robot.pop("robot_config_path", None)
        return data

    monkeypatch.setattr(
        graph_validator,
        "load_robot_profile",
        load_without_resource,
    )
    with pytest.raises(ValueError, match=r"requires robot_config_path or urdf_path"):
        validate_profile_graph(
            runtime_profile="default",
            profile=profile,
            resolved=resolved,
            env_config=env_config,
        )


def test_profile_graph_rejects_robot_object_prim_tree_overlap() -> None:
    profile, env_config, resolved = _default_graph_inputs()
    conflicting = deepcopy(env_config)
    conflicting["robots"][0]["prim_path"] = "/World/WorkstationArmBase/Robot"

    with pytest.raises(
        ValueError, match=r"Robot and object instance prim paths overlap"
    ):
        validate_profile_graph(
            runtime_profile="default",
            profile=profile,
            resolved=resolved,
            env_config=conflicting,
        )


def test_profile_graph_rejects_nested_robot_prim_trees() -> None:
    profile, env_config, resolved = _default_graph_inputs()
    conflicting = deepcopy(env_config)
    first = conflicting["robots"][0]
    first["prim_path"] = "/World/SharedRobot"
    second = deepcopy(first)
    second["label"] = "nested_robot"
    second["prim_path"] = "/World/SharedRobot/child"
    conflicting["robots"].append(second)

    with pytest.raises(ValueError, match=r"Robot instance prim paths overlap"):
        validate_profile_graph(
            runtime_profile="default",
            profile=profile,
            resolved=resolved,
            env_config=conflicting,
        )
