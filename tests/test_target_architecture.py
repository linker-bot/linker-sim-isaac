from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from scripts import update_architecture_inventory as inventory_generator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "linkerbot_sim"
ARCHITECTURE_MANIFEST = ROOT / "architecture" / "module_disposition.yaml"
ARCHITECTURE_GENERATOR = ROOT / "scripts" / "update_architecture_inventory.py"
EXPECTED_PRODUCT_PHYSICS = {
    "mirror": {("physx", "cpu"), ("newton", "cpu"), ("newton", "cuda")},
    "kaleidoscope": {("physx", "cuda"), ("newton", "cuda")},
}
EXPECTED_KIT_EXPERIENCES = {
    "apps/linkerbot_sim.mirror.physx.python.kit",
    "apps/linkerbot_sim.mirror.newton.python.kit",
    "apps/linkerbot_sim.mirror.newton_render.python.kit",
    "apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
    "apps/linkerbot_sim.kaleidoscope.newton.python.kit",
    "apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit",
    "apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit",
}
EXPECTED_MODE_PROFILES = {
    "mirror": {
        "configs/modes/mirror/physx_cpu.yaml",
        "configs/modes/mirror/physx_cpu_hybrid.yaml",
        "configs/modes/mirror/newton_cpu.yaml",
        "configs/modes/mirror/newton_cuda.yaml",
    },
    "kaleidoscope": {
        "configs/modes/kaleidoscope/physx_cuda.yaml",
        "configs/modes/kaleidoscope/newton_cuda.yaml",
    },
}
EXPECTED_CANONICAL_CONFIGURATION_PROFILES = {
    "configs/scenes": set(),
    "configs/scenes/mirror": {
        "configs/scenes/mirror/scene3.yaml",
        "configs/scenes/mirror/scene3_hybrid.yaml",
    },
    "configs/scenes/kaleidoscope": {"configs/scenes/kaleidoscope/tblock_push.yaml"},
    "configs/physics/physx": {
        "configs/physics/physx/cpu.yaml",
        "configs/physics/physx/cuda.yaml",
    },
    "configs/physics/newton": {
        "configs/physics/newton/cpu.yaml",
        "configs/physics/newton/cuda.yaml",
    },
    "configs/control": {
        "configs/control/hybrid_force_position.yaml",
        "configs/control/mirror.yaml",
    },
    "configs/curobo": {
        "configs/curobo/kaleidoscope_batch_ik.yaml",
        "configs/curobo/mirror.yaml",
    },
    "configs/planning": {"configs/planning/mirror.yaml"},
    "configs/visualization": {"configs/visualization/kaleidoscope.yaml"},
}
EXPECTED_CANONICAL_CONFIGURATION_DIRECTORIES = {
    "configs/controllers": {"newton", "physx"},
    "configs/physics": {"newton", "physx"},
    "configs/scenes": {"kaleidoscope", "mirror"},
}
EXPECTED_FORBIDDEN_PUBLIC_CONFIGURATION_PATHS = {
    "configs/controllers/example.yaml",
    "configs/curobo/example.yaml",
    "configs/kinematics",
    "configs/logging",
    "configs/objects/example.yaml",
    "configs/planning/curobo",
    "configs/replication",
    "configs/robots/example.yaml",
    "configs/visualization/kaleidoscope",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _python_files(relative: str) -> tuple[Path, ...]:
    return tuple(sorted((SOURCE / relative).rglob("*.py")))


def test_manifest_freezes_dual_backend_seven_kit_profile_contract() -> None:
    """产品矩阵与 canonical profile 集合是手写合同，不能靠刷新 inventory 绕过。"""

    manifest = yaml.safe_load(ARCHITECTURE_MANIFEST.read_text(encoding="utf-8"))
    products = manifest["breaking_migration"]["products"]
    for product_name, expected in EXPECTED_PRODUCT_PHYSICS.items():
        actual = {
            (selection["engine"], selection["execution"])
            for selection in products[product_name]["physics"]
        }
        assert actual == expected

    required = manifest["required_targets"]
    declared_kits = required["kit_experiences"]["exact"]
    assert len(declared_kits) == len(EXPECTED_KIT_EXPERIENCES)
    assert set(declared_kits) == EXPECTED_KIT_EXPERIENCES
    assert {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "apps").glob("*.kit")
    } == EXPECTED_KIT_EXPERIENCES

    for product_name, expected in EXPECTED_MODE_PROFILES.items():
        parent = Path("configs/modes") / product_name
        declared_profiles = {
            path
            for path in required["canonical_configuration"]["paths"]
            if Path(path).parent == parent
        }
        assert declared_profiles == expected
        assert {
            path.relative_to(ROOT).as_posix() for path in (ROOT / parent).glob("*.yaml")
        } == expected

    for parent_name, expected in EXPECTED_CANONICAL_CONFIGURATION_PROFILES.items():
        parent = Path(parent_name)
        declared_profiles = {
            path
            for path in required["canonical_configuration"]["paths"]
            if Path(path).parent == parent
        }
        assert declared_profiles == expected
        assert {
            path.relative_to(ROOT).as_posix() for path in (ROOT / parent).glob("*.yaml")
        } == expected
        assert (
            set(inventory_generator.CANONICAL_CONFIGURATION_FILE_SETS[parent_name])
            == expected
        )
    assert {
        parent: set(children)
        for parent, children in inventory_generator.CANONICAL_CONFIGURATION_DIRECTORY_SETS.items()
    } == EXPECTED_CANONICAL_CONFIGURATION_DIRECTORIES
    assert (
        set(inventory_generator.FORBIDDEN_PUBLIC_CONFIGURATION_PATHS)
        == EXPECTED_FORBIDDEN_PUBLIC_CONFIGURATION_PATHS
    )
    for relative in EXPECTED_FORBIDDEN_PUBLIC_CONFIGURATION_PATHS:
        assert not (ROOT / relative).exists()
    assert not (SOURCE / "configuration" / "visualization.py").exists()


def test_inventory_write_runs_architecture_violations_after_writes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """更新机械清单不能绕过 check 模式已有的架构规则。"""

    manifest = {"sentinel": "final manifest"}
    events: list[str] = []

    monkeypatch.setattr(inventory_generator, "_manifest", lambda: manifest)
    monkeypatch.setattr(inventory_generator, "_desired_files", lambda _value: {})
    monkeypatch.setattr(
        inventory_generator,
        "_manifest_with_generated_inventory",
        lambda _value: "generated manifest",
    )

    def fake_write(_desired: object) -> tuple[Path, ...]:
        events.append("write")
        return ()

    def fake_violations(value: object, *, require_final: bool) -> tuple[str, ...]:
        assert value is manifest
        assert require_final is True
        events.append("violations")
        return ("forced violation",)

    monkeypatch.setattr(inventory_generator, "_write", fake_write)
    monkeypatch.setattr(
        inventory_generator, "_architecture_violations", fake_violations
    )

    assert inventory_generator.main(["--write", "--require-final"]) == 1
    assert events == ["write", "write", "violations"]
    assert "architecture violation: forced violation" in capsys.readouterr().out


def test_configuration_is_a_pure_leaf_and_has_one_yaml_catalog_owner() -> None:
    files = _python_files("configuration")
    forbidden = (
        "torch",
        "gymnasium",
        "skrl",
        "newton",
        "mujoco_warp",
        "linkerbot_sim.isaac",
        "linkerbot_sim.mirror",
        "linkerbot_sim.kaleidoscope",
        "linkerbot_sim.training",
        "linkerbot_sim.app",
        "linkerbot_sim.tiled",
    )
    for path in files:
        imports = _imports(path)
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in forbidden
        ), path
        assert not any(name == "yaml" or name.startswith("yaml.") for name in imports)

    yaml_importers = [
        path.relative_to(ROOT).as_posix()
        for path in SOURCE.rglob("*.py")
        if any(name == "yaml" or name.startswith("yaml.") for name in _imports(path))
    ]
    assert yaml_importers == ["src/linkerbot_sim/utils/config.py"]

    profile_io_owners = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if "load_yaml" in path.read_text(encoding="utf-8")
    ]
    assert profile_io_owners == ["src/linkerbot_sim/configuration/catalog.py"]


def test_configuration_schema_names_match_configs_top_level_groups() -> None:
    """每个公开 profile 目录恰有一个同名 schema owner，基础设施模块除外。"""

    profile_groups = {
        path.name for path in (ROOT / "configs").iterdir() if path.is_dir()
    }
    configuration_root = SOURCE / "configuration"
    schema_modules = {
        path.stem
        for path in configuration_root.glob("*.py")
        if path.stem not in {"__init__", "catalog", "common", "fingerprint"}
    }
    schema_packages = {
        path.name
        for path in configuration_root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert schema_modules | schema_packages == profile_groups
    assert not (configuration_root / "visualization" / "common.py").exists()


def test_mode_roots_do_not_import_each_other_and_training_does_not_own_isaac() -> None:
    mirror_imports = set().union(*(_imports(path) for path in _python_files("mirror")))
    kaleidoscope_imports = set().union(
        *(_imports(path) for path in _python_files("kaleidoscope"))
    )
    training_imports = set().union(
        *(_imports(path) for path in _python_files("training"))
    )
    assert not any(
        name.startswith("linkerbot_sim.kaleidoscope") for name in mirror_imports
    )
    assert not any(
        name.startswith("linkerbot_sim.mirror") for name in kaleidoscope_imports
    )
    assert not any(name.startswith("linkerbot_sim.isaac") for name in training_imports)


def test_numerical_backends_do_not_depend_on_product_packages() -> None:
    imports = set().union(*(_imports(path) for path in _python_files("backends")))
    forbidden = ("linkerbot_sim.mirror", "linkerbot_sim.kaleidoscope")
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )


def test_kaleidoscope_product_closure_has_no_planner_or_external_engine_import() -> (
    None
):
    imports = set().union(*(_imports(path) for path in _python_files("kaleidoscope")))
    forbidden_fragments = (
        ".planning",
        ".collision",
        "mujoco_warp",
        ".telemetry",
        ".camera",
        ".transport",
        ".playback",
    )
    violations = sorted(
        name
        for name in imports
        if any(fragment in name for fragment in forbidden_fragments)
        or name == "newton"
        or name.startswith("newton.")
    )
    assert violations == []


def test_kaleidoscope_facade_import_is_lazy_in_a_clean_interpreter() -> None:
    code = """
import sys
import linkerbot_sim.kaleidoscope
from linkerbot_sim.kaleidoscope import KaleidoscopeTrainingPort
assert KaleidoscopeTrainingPort.__name__ == "KaleidoscopeTrainingPort"
forbidden = ('torch', 'gymnasium', 'skrl', 'newton', 'mujoco_warp', 'omni', 'isaacsim')
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in forbidden)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_curobo_kinematics_facade_defers_torch_device_solver_import() -> None:
    """纯类型消费者经过 package 初始化时不能被迫安装或导入 Torch。"""

    code = """
import sys
import linkerbot_sim.backends.curobo.kinematics as kinematics
assert kinematics.__all__ == [
    'BatchIKTensorResult',
    'BatchIKWaypointTensorResult',
    'CuroboDeviceBatchIKSolver',
    'CuroboKinematicsContext',
    'create_kinematics_context',
    'kinematics_config_from_robot_profile',
]
from linkerbot_sim.backends.curobo.kinematics import BatchIKTensorResult
assert BatchIKTensorResult.__name__ == 'BatchIKTensorResult'
assert 'CuroboDeviceBatchIKSolver' in dir(kinematics)
assert 'torch' not in sys.modules
assert 'linkerbot_sim.backends.curobo.kinematics.device_batch_ik' not in sys.modules
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_kaleidoscope_native_bootstrap_does_not_import_gymnasium() -> None:
    code = """
import sys
from linkerbot_sim.kaleidoscope.bootstrap import make_torch_env
assert callable(make_torch_env)
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] == 'gymnasium')
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_asset_configuration_import_does_not_expand_catalog_cycle() -> None:
    """资产叶模块必须能在干净解释器中独立导入，且不触发根配置图。"""

    code = """
import sys
from linkerbot_sim.configuration.robots import AssetImportConfig
assert AssetImportConfig.__name__ == "AssetImportConfig"
assert "linkerbot_sim.configuration.resource_paths" not in sys.modules
assert "linkerbot_sim.configuration.catalog" not in sys.modules
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_object_configuration_load_stays_out_of_runtime_layers() -> None:
    """完整 object 判别联合解析不能反向导入 object/Isaac runtime。"""

    code = """
import sys
from linkerbot_sim.configuration import load_kaleidoscope_config, load_mirror_config
from linkerbot_sim.configuration.objects import object_profile_from_mapping

mirror = load_mirror_config()
kaleidoscope = load_kaleidoscope_config()
chain = object_profile_from_mapping(
    {
        'object': {
            'kind': 'dynamic_chain',
            'source': 'usd',
            'asset_path': 'rope.usda',
            'root_path': '/CapsuleRope',
            'state_summary': {'reference_body': 'left_box'},
        }
    },
    profile_name='rope',
)
profiles = (
    mirror.scene.objects[0].resolved_profile,
    kaleidoscope.scene.objects[0].resolved_profile,
    chain,
)
assert all(profile is not None for profile in profiles)
assert all(profile.__class__.__module__ == 'linkerbot_sim.configuration.objects' for profile in profiles)
assert all(profile.physics.__class__.__module__ == 'linkerbot_sim.configuration.objects' for profile in profiles)
blocked = ('linkerbot_sim.objects', 'linkerbot_sim.isaac')
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
)
assert loaded == [], loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_target_source_has_no_product_level_rl_package_or_alias() -> None:
    assert not (SOURCE / "rl").exists()
    for path in (
        SOURCE / "configuration",
        SOURCE / "mirror",
        SOURCE / "kaleidoscope",
        SOURCE / "training",
        SOURCE / "isaac",
    ):
        for source in path.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            assert "linkerbot_sim.rl" not in text
            assert "tiled_rl" not in text


def test_generated_target_architecture_is_current_and_final() -> None:
    """最终门禁同时校验 inventory、facade、依赖方向和破坏性迁移清理。"""

    subprocess.run(
        [sys.executable, str(ARCHITECTURE_GENERATOR), "--check", "--require-final"],
        cwd=ROOT,
        check=True,
    )
    manifest = yaml.safe_load(ARCHITECTURE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    inventory = manifest["generated_inventory"]
    assert inventory["status"] == "final"
    assert inventory["pending_facades"] == []
    for name in (
        "production_python",
        "test_python",
        "configuration_yaml",
        "kit_experiences",
        "entry_scripts",
        "automation_scripts",
        "maintained_docs",
    ):
        group = inventory[name]
        assert group["count"] == len(group["files"])
        assert len(group["path_sha256"]) == 64
        assert len(group["path_content_sha256"]) == 64
    allowed_layers = set(manifest["layers"]["order"])
    for entry in inventory["production_python"]["files"]:
        assert entry["layer"] in allowed_layers
        assert entry["owner"].startswith("linkerbot_sim")
        assert entry["module"].startswith("linkerbot_sim")


def test_breaking_migration_leaves_no_legacy_product_or_physics_shim() -> None:
    """不兼容迁移完成后，正式源码、配置和脚本不能再回退到旧产品入口。"""

    forbidden_names = re.compile(
        r"single_scene|SingleScene|tiled_scene|TiledScene|"
        r"linkerbot_sim\.(?:tiled|physics)(?:\.|\b)"
    )
    roots_and_suffixes = (
        (SOURCE, {".py"}),
        (ROOT / "configs", {".yaml", ".yml", ".md"}),
        (ROOT / "scripts", {".py"}),
    )
    violations: list[str] = []
    for root, suffixes in roots_and_suffixes:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in suffixes:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if forbidden_names.search(relative) or forbidden_names.search(
                path.read_text(encoding="utf-8")
            ):
                violations.append(relative)
    assert violations == []

    shim_imports = []
    for path in SOURCE.rglob("*.py"):
        for imported in _imports(path):
            if imported == "linkerbot_sim.physics" or imported.startswith(
                "linkerbot_sim.physics."
            ):
                shim_imports.append(
                    f"{path.relative_to(SOURCE).as_posix()} -> {imported}"
                )
    assert shim_imports == []
    assert not (SOURCE / "physics").exists()
    assert not (ROOT / "configs" / "envs").exists()
