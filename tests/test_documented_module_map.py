from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "linkerbot_sim"
DOCUMENT_PATHS = (
    REPO_ROOT / "docs" / "en" / "development" / "module-map.md",
    REPO_ROOT / "docs" / "zh-CN" / "development" / "module-map.md",
)
MANIFEST_PATH = REPO_ROOT / "architecture" / "module_disposition.yaml"
MANIFEST = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
DECLARED_GROUP_ORDER = tuple(MANIFEST["module_map"]["group_order"])
GROUP_LAYERS = dict(MANIFEST["module_map"]["group_layers"])
RUNTIME_LABELS = {"pure", "Isaac main thread", "cuRobo/CUDA"}
CLASSIFICATIONS = {"documented facade", "owner path", "internal"}
DOCUMENTED_FACADES = set(MANIFEST["public_facades"])
LINK_RE = re.compile(r"^\[[^]]+\]\((?P<target>[^)]+)\)$")


@dataclass(frozen=True)
class ModuleEntry:
    group: str
    module: str
    layer: str
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
        if len(cells) != 7:
            continue
        module = _module_name(cells[1])
        if module is None:
            continue
        link = LINK_RE.fullmatch(cells[6])
        assert link is not None, (path, module, cells[6])
        entries.append(
            ModuleEntry(
                group=cells[0],
                module=module,
                layer=cells[2],
                responsibility=cells[3],
                runtime=cells[4],
                classification=cells[5],
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
    expected_group_order = tuple(
        group for group in DECLARED_GROUP_ORDER if source_group_counts[group]
    )

    assert len(documented_modules) == len(set(documented_modules))
    assert set(documented_modules) == source_modules

    assert Counter(entry.group for entry in entries) == source_group_counts
    assert (
        tuple(dict.fromkeys(entry.group for entry in entries)) == expected_group_order
    )
    assert all(entry.group == _module_group(entry.module) for entry in entries)
    assert all(entry.layer == GROUP_LAYERS[entry.group] for entry in entries)

    text = document_path.read_text(encoding="utf-8")
    section = _marked_section(text, "module-inventory")
    headings = re.findall(r"^### ([a-z]+) \(([0-9]+)\)$", section, re.MULTILINE)
    assert headings == [
        (group, str(source_group_counts[group])) for group in expected_group_order
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


def test_bilingual_module_maps_have_identical_facts() -> None:
    english = _parse_inventory(DOCUMENT_PATHS[0])
    chinese = _parse_inventory(DOCUMENT_PATHS[1])

    def facts(entry: ModuleEntry) -> tuple[str, str, str, str, str, str]:
        return (
            entry.group,
            entry.module,
            entry.layer,
            entry.runtime,
            entry.classification,
            entry.documentation_target,
        )

    assert [facts(entry) for entry in english] == [facts(entry) for entry in chinese]
    assert _parse_registry(DOCUMENT_PATHS[0]) == _parse_registry(DOCUMENT_PATHS[1])
