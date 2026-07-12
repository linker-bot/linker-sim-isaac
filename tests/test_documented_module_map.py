from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "linkerbot_sim"
DOCUMENT_PATHS = (
    REPO_ROOT / "docs" / "en" / "development" / "module-map.md",
    REPO_ROOT / "docs" / "zh-CN" / "development" / "module-map.md",
)
GROUP_ORDER = (
    "root",
    "app",
    "assets",
    "backends",
    "configs",
    "controllers",
    "envs",
    "execution",
    "logging",
    "objects",
    "planning",
    "robots",
    "sensors",
    "snapshots",
    "telemetry",
    "tiled",
    "trajectories",
    "utils",
    "visualization",
)
RUNTIME_LABELS = {"pure", "Isaac main thread", "cuRobo/CUDA"}
CLASSIFICATIONS = {"documented facade", "owner path", "internal"}
DOCUMENTED_FACADES = {
    "linkerbot_sim",
    "linkerbot_sim.app.interactive.single_scene",
    "linkerbot_sim.app.interactive.tiled_scene",
    "linkerbot_sim.backends.curobo",
    "linkerbot_sim.controllers",
    "linkerbot_sim.execution",
    "linkerbot_sim.objects",
    "linkerbot_sim.planning",
    "linkerbot_sim.robots",
    "linkerbot_sim.sensors",
    "linkerbot_sim.snapshots",
}
LINK_RE = re.compile(r"^\[[^]]+\]\((?P<target>[^)]+)\)$")


@dataclass(frozen=True)
class ModuleEntry:
    group: str
    module: str
    responsibility: str
    runtime: str
    classification: str
    documentation_target: str


def _marked_section(text: str, name: str) -> str:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _module_name(cell: str) -> str | None:
    if len(cell) < 2 or not cell.startswith("`linkerbot_sim") or not cell.endswith("`"):
        return None
    return cell[1:-1]


def _parse_inventory(path: Path) -> list[ModuleEntry]:
    text = path.read_text(encoding="utf-8")
    section = _marked_section(text, "module-inventory")
    entries: list[ModuleEntry] = []
    for line in section.splitlines():
        cells = _table_cells(line)
        if len(cells) != 6:
            continue
        module = _module_name(cells[1])
        if module is None:
            continue
        link = LINK_RE.fullmatch(cells[5])
        assert link is not None, (path, module, cells[5])
        entries.append(
            ModuleEntry(
                group=cells[0],
                module=module,
                responsibility=cells[2],
                runtime=cells[3],
                classification=cells[4],
                documentation_target=link.group("target"),
            )
        )
    return entries


def _parse_registry(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    section = _marked_section(text, "module-interface-registry")
    entries: list[tuple[str, str, str]] = []
    for line in section.splitlines():
        cells = _table_cells(line)
        if len(cells) != 3:
            continue
        module = _module_name(cells[0])
        if module is not None:
            entries.append((module, cells[1], cells[2]))
    return entries


def _source_modules() -> set[str]:
    modules: set[str] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        suffix = ".".join(parts)
        modules.add("linkerbot_sim" + (f".{suffix}" if suffix else ""))
    return modules


def _module_group(module: str) -> str:
    parts = module.split(".")
    return "root" if len(parts) == 1 else parts[1]


@pytest.mark.parametrize("document_path", DOCUMENT_PATHS)
def test_module_inventory_exactly_covers_source(document_path: Path) -> None:
    entries = _parse_inventory(document_path)
    documented_modules = [entry.module for entry in entries]
    source_modules = _source_modules()
    source_group_counts = Counter(_module_group(module) for module in source_modules)

    assert len(documented_modules) == len(set(documented_modules))
    assert set(documented_modules) == source_modules

    assert Counter(entry.group for entry in entries) == source_group_counts
    assert list(dict.fromkeys(entry.group for entry in entries)) == list(GROUP_ORDER)
    assert all(entry.group == _module_group(entry.module) for entry in entries)

    text = document_path.read_text(encoding="utf-8")
    section = _marked_section(text, "module-inventory")
    headings = re.findall(r"^### ([a-z]+) \(([0-9]+)\)$", section, re.MULTILINE)
    assert headings == [
        (group, str(source_group_counts[group])) for group in GROUP_ORDER
    ]

    for entry in entries:
        assert entry.responsibility
        assert entry.runtime in RUNTIME_LABELS
        assert entry.classification in CLASSIFICATIONS
        target = (document_path.parent / entry.documentation_target).resolve()
        assert target.is_file(), (document_path, entry.module, target)


@pytest.mark.parametrize("document_path", DOCUMENT_PATHS)
def test_registry_matches_inventory_classification(document_path: Path) -> None:
    entries = _parse_inventory(document_path)
    registry_rows = _parse_registry(document_path)
    assert len(registry_rows) == len({row[0] for row in registry_rows})

    registry = {
        module: (classification, runtime)
        for module, classification, runtime in registry_rows
    }
    non_internal = {
        entry.module: (entry.classification, entry.runtime)
        for entry in entries
        if entry.classification != "internal"
    }
    assert registry == non_internal

    facades = {
        entry.module for entry in entries if entry.classification == "documented facade"
    }
    assert facades == DOCUMENTED_FACADES
    assert not {
        module
        for module in facades
        if module == "linkerbot_sim.tiled" or module.startswith("linkerbot_sim.tiled.")
    }


def test_bilingual_module_maps_have_identical_facts() -> None:
    english = _parse_inventory(DOCUMENT_PATHS[0])
    chinese = _parse_inventory(DOCUMENT_PATHS[1])

    def facts(entry: ModuleEntry) -> tuple[str, str, str, str, str]:
        return (
            entry.group,
            entry.module,
            entry.runtime,
            entry.classification,
            entry.documentation_target,
        )

    assert [facts(entry) for entry in english] == [facts(entry) for entry in chinese]
    assert _parse_registry(DOCUMENT_PATHS[0]) == _parse_registry(DOCUMENT_PATHS[1])


def test_tiled_package_declares_no_top_level_exports() -> None:
    path = SOURCE_ROOT / "tiled" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "__all__"
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.List) and value.elts == []
