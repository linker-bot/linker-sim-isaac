from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

from linkerbot_sim.backends.curobo.config import CuroboTaskBundle
from linkerbot_sim.utils.config import load_yaml


ALLOWLIST_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "config_audit"
    / "hardcoded_allowlist.yaml"
)
REQUIRED_ENTRY_FIELDS = {
    "id",
    "path",
    "symbol",
    "category",
    "decision",
    "reason",
    "owner",
}
ALLOWED_CATEGORIES = {
    "protocol_contract",
    "mathematical_invariant",
    "third_party_contract",
    "safety_invariant",
    "internal_scheduling",
    "repository_layout_contract",
    "visual_preference",
}
ALLOWED_DECISIONS = {
    "retain_in_code",
    "retain_pinned",
}


def _allowlist() -> dict[str, object]:
    return load_yaml(ALLOWLIST_PATH)


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for item in target.elts for name in _assigned_names(item)}
    return set()


def _defined_symbol_paths(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{member.name}")
                elif isinstance(member, ast.Assign):
                    for target in member.targets:
                        symbols.update(
                            f"{node.name}.{name}" for name in _assigned_names(target)
                        )
                elif isinstance(member, ast.AnnAssign):
                    symbols.update(
                        f"{node.name}.{name}" for name in _assigned_names(member.target)
                    )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                symbols.update(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            symbols.update(_assigned_names(node.target))
    return symbols


def test_hardcoded_allowlist_entries_have_reviewable_decisions() -> None:
    data = _allowlist()
    source_root = Path(data["inventory"]["source_root"])
    assert source_root.is_dir()
    entries = data["entries"]
    assert isinstance(entries, list) and entries

    identifiers: set[str] = set()
    for entry in entries:
        assert isinstance(entry, dict)
        assert REQUIRED_ENTRY_FIELDS <= set(entry)
        for field in REQUIRED_ENTRY_FIELDS:
            assert isinstance(entry[field], str) and entry[field].strip()
        identifier = entry["id"]
        assert identifier not in identifiers
        identifiers.add(identifier)
        assert entry["category"] in ALLOWED_CATEGORIES
        assert entry["decision"] in ALLOWED_DECISIONS
        assert len(entry["reason"].strip()) >= 80
        assert Path(entry["path"]).exists()

        for site in entry.get("sites", ()):
            assert isinstance(site, str)
            relative_path, symbol = site.rsplit(":", 1)
            source_path = source_root / relative_path
            assert source_path.is_file(), site
            assert symbol in _defined_symbol_paths(source_path), site


def test_source_inventory_and_candidate_scan_rule_are_current() -> None:
    data = _allowlist()
    inventory = data["inventory"]
    assert isinstance(inventory, dict)
    source_root = Path(inventory["source_root"])
    source_files = sorted(Path().glob(str(inventory["source_python_glob"])))

    assert all(path.is_file() and path.suffix == ".py" for path in source_files)
    # 数量由架构 inventory 生成器冻结；测试只验证工作树与清单完全一致，避免在测试
    # 源码里再维护一份会随模块移动失效的“魔法数字”。
    assert len(source_files) == inventory["source_python_count"]
    assert all(path.is_relative_to(source_root) for path in source_files)

    scan = inventory["candidate_scan"]
    assert isinstance(scan, dict)
    assert scan["engine"] == "python_ast"
    assert scan["assignment_scope"] == "module"
    pattern = re.compile(str(scan["name_regex"]))
    assert scan["review_dispositions"] == [
        "yaml_or_request_owned",
        "allowlisted_retained",
    ]

    candidates = sorted(
        {
            f"{path.relative_to(source_root).as_posix()}:{symbol}"
            for path in source_files
            for symbol in _defined_symbol_paths(path)
            if "." not in symbol and pattern.fullmatch(symbol)
        }
    )
    candidate_manifest = "\n".join(candidates).encode("utf-8")
    assert len(candidates) == scan["reviewed_candidate_count"]
    assert (
        hashlib.sha256(candidate_manifest).hexdigest()
        == scan["reviewed_candidate_sha256"]
    )
    required_protocol_candidates = {
        "mirror/controller.py:_OUT_OF_BAND_OPERATIONS",
        "mirror/interface/protocol.py:MIRROR_V1_OPERATIONS",
        "mirror/motion/owner.py:MOTION_OPERATIONS",
    }
    assert required_protocol_candidates <= set(candidates)

    entries = data["entries"]
    assert isinstance(entries, list)
    allowlisted_sites = {site for entry in entries for site in entry.get("sites", ())}
    assert required_protocol_candidates <= allowlisted_sites


def test_configuration_inventory_keeps_project_and_third_party_yaml_separate() -> None:
    inventory = _allowlist()["inventory"]
    assert isinstance(inventory, dict)
    project_files = sorted(Path().glob(str(inventory["project_yaml_glob"])))
    task_files = sorted(Path().glob(str(inventory["third_party_task_glob"])))

    assert len(project_files) == inventory["project_yaml_count"]
    assert len(task_files) == inventory["third_party_task_count"]
    assert not set(project_files) & set(task_files)
    assert all(path.suffix == ".yaml" for path in project_files)
    assert all(path.suffix == ".yml" for path in task_files)


def test_curobo_task_bundle_matches_allowlist_source_and_pinned_hashes() -> None:
    entries = _allowlist()["entries"]
    assert isinstance(entries, list)
    entry = next(item for item in entries if item["id"] == "curobo_v0_8_task_bundle")
    expected_files = entry["files"]
    assert isinstance(expected_files, dict)

    task_root = Path(entry["path"])
    actual_files = {
        path.relative_to(task_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in task_root.rglob("*.yml")
    }
    assert actual_files == expected_files

    bundle = CuroboTaskBundle.named("curobo_v0_8_default")
    referenced_files = {
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
    assert referenced_files == set(expected_files)
    assert entry["bundle_version"] == "0.8.0"
    assert bundle.compatible_versions == frozenset({entry["bundle_version"]})


def test_mirror_wire_operation_sets_are_allowlisted() -> None:
    entries = _allowlist()["entries"]
    assert isinstance(entries, list)
    entry = next(item for item in entries if item["id"] == "mirror_v1_wire_vocabulary")
    assert set(entry["sites"]) == {
        "mirror/controller.py:_OUT_OF_BAND_OPERATIONS",
        "mirror/interface/protocol.py:MIRROR_V1_OPERATIONS",
        "mirror/motion/owner.py:MOTION_OPERATIONS",
    }
