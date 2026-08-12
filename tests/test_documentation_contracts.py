from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror.interface.protocol import decode_request
from linkerbot_sim.mirror.motion.hybrid_executor import parse_hybrid_motion_request
from linkerbot_sim.mirror.motion.request_parser import parse_mirror_motion_request


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
ENGLISH_ROOT = DOCS_ROOT / "en"
CHINESE_ROOT = DOCS_ROOT / "zh-CN"
SOURCE_ROOT = REPO_ROOT / "src" / "linkerbot_sim"
MANIFEST_PATH = REPO_ROOT / "architecture" / "module_disposition.yaml"

MIRROR_CLI = SOURCE_ROOT / "mirror" / "cli.py"
MODE_CONFIG_CLI = REPO_ROOT / "scripts" / "validate_mode_config.py"
MIRROR_PROTOCOL = SOURCE_ROOT / "mirror" / "interface" / "protocol.py"
MIRROR_MOTION_OWNER = SOURCE_ROOT / "mirror" / "motion" / "owner.py"
CLI_CONTRACTS = (
    (
        MIRROR_CLI,
        "build_parser",
        ENGLISH_ROOT / "reference" / "mirror-cli.md",
        CHINESE_ROOT / "reference" / "mirror-cli.md",
        "Complete Option Table",
        "参数",
    ),
)

_FENCE_OPEN_RE = re.compile(r"^(?: {0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)$")
_LINK_RE = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")
_LONG_OPTION_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
_OPERATION_RE = re.compile(
    r"`((?:control|motion|queue|runtime|snapshot|state)\.[a-z][a-z0-9_]*)`"
)


@dataclass(frozen=True)
class FencedBlock:
    path: Path
    language: str
    opening_line: int
    content: str


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _markdown_paths(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*.md")}


def _maintained_markdown_paths() -> tuple[Path, ...]:
    paths = set(REPO_ROOT.glob("README*.md"))
    paths.update(DOCS_ROOT.rglob("*.md"))
    return tuple(sorted(paths))


def _iter_fenced_blocks(path: Path) -> Iterator[FencedBlock]:
    active_fence: str | None = None
    language = ""
    opening_line = 0
    content: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if active_fence is None:
            match = _FENCE_OPEN_RE.fullmatch(line)
            if match is None:
                continue
            active_fence = match.group("fence")
            info = match.group("info").strip().split(maxsplit=1)
            language = info[0].lower() if info else ""
            opening_line = line_number
            content = []
            continue

        closing_re = (
            rf"^(?: {{0,3}}){re.escape(active_fence[0])}"
            rf"{{{len(active_fence)},}}[ \t]*$"
        )
        if re.fullmatch(closing_re, line):
            if language in {"json", "jsonl", "yaml"}:
                yield FencedBlock(path, language, opening_line, "\n".join(content))
            active_fence = None
            language = ""
            content = []
            continue
        content.append(line)
    assert active_fence is None, f"unclosed fence: {path}:{opening_line}"


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_strict_json(content: str) -> object:
    return json.loads(
        content,
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value!r}")
        ),
    )


def _validate_fenced_block(block: FencedBlock) -> None:
    if block.language == "json":
        _load_strict_json(block.content)
        return
    if block.language == "yaml":
        yaml.load(block.content, Loader=_UniqueKeyLoader)
        return
    lines = block.content.splitlines()
    assert any(line.strip() for line in lines), "JSONL block is empty"
    for line in lines:
        if line.strip():
            _load_strict_json(line)


def _named_section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section {heading!r}: {path}"
    return match.group("body")


def _function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name} in {path}"
    return matches[0]


def _parser_long_options(path: Path, function_name: str) -> set[str]:
    options: set[str] = set()
    for call in (
        node
        for node in ast.walk(_function(path, function_name))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ):
        declared = {
            argument.value
            for argument in call.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }
        assert declared, f"non-static long option in {path}:{call.lineno}"
        options.update(declared)
        action = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "action"),
            None,
        )
        if isinstance(action, ast.Attribute) and action.attr == "BooleanOptionalAction":
            options.update(f"--no-{option[2:]}" for option in declared)
    return options


def _documented_table_options(path: Path, heading: str) -> set[str]:
    options: set[str] = set()
    for line in _named_section(path, heading).splitlines():
        if line.startswith("|"):
            first_cell = line.strip().strip("|").split("|", maxsplit=1)[0]
            options.update(_LONG_OPTION_RE.findall(first_cell))
    return options


def _manifest() -> Mapping[str, object]:
    value = yaml.load(
        MANIFEST_PATH.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
    )
    assert isinstance(value, Mapping)
    return value


def _facade_path(module: str) -> Path:
    return SOURCE_ROOT.joinpath(*module.split(".")[1:], "__init__.py")


def _static_exports(module: str) -> tuple[str, ...]:
    path = _facade_path(module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dictionaries: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        keys = tuple(
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
        if len(keys) != len(value.keys):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                dictionaries[target.id] = keys

    assignments: list[tuple[str, ...]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        value = node.value
        try:
            literal = ast.literal_eval(value)
        except (TypeError, ValueError):
            literal = None
        if isinstance(literal, (tuple, list)) and all(
            isinstance(item, str) for item in literal
        ):
            assignments.append(tuple(literal))
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "sorted"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in dictionaries
        ):
            assignments.append(tuple(sorted(dictionaries[value.args[0].id])))
    assert len(assignments) == 1, f"cannot resolve {module}.__all__ statically"
    return assignments[0]


def _documented_facade_symbols(path: Path, module: str) -> set[str]:
    section = _named_section(path, f"`{module}`")
    section = section.split("\n### ", maxsplit=1)[0]
    symbols: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first_cell = line.strip().strip("|").split("|", maxsplit=1)[0]
        for value in re.findall(r"`([A-Za-z_]\w*)", first_cell):
            if value not in {"Symbol", "导出"}:
                symbols.add(value)
    return symbols


def _static_assignment_strings(path: Path, name: str) -> set[str]:
    """Collect literal string leaves from one module-level assignment."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[set[str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        values.append(
            {
                child.value
                for child in ast.walk(node.value)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
        )
    assert len(values) == 1, f"expected one assignment to {name} in {path}"
    return values[0]


def test_bilingual_markdown_trees_and_language_switches_match() -> None:
    english = _markdown_paths(ENGLISH_ROOT)
    chinese = _markdown_paths(CHINESE_ROOT)
    assert english == chinese, (
        f"English-only={sorted(map(str, english - chinese))}; "
        f"Chinese-only={sorted(map(str, chinese - english))}"
    )
    for relative in sorted(english):
        left = ENGLISH_ROOT / relative
        right = CHINESE_ROOT / relative
        for source, counterpart in ((left, right), (right, left)):
            header = "\n".join(source.read_text(encoding="utf-8").splitlines()[:10])
            targets = []
            for match in _LINK_RE.finditer(header):
                target = unquote(match.group("target").split("#", maxsplit=1)[0])
                if target and "://" not in target:
                    targets.append((source.parent / target).resolve())
            assert counterpart.resolve() in targets, source.relative_to(REPO_ROOT)


def test_structured_markdown_fences_parse_strictly() -> None:
    issues: list[str] = []
    checked = 0
    for path in _maintained_markdown_paths():
        for block in _iter_fenced_blocks(path):
            checked += 1
            try:
                _validate_fenced_block(block)
            except (AssertionError, TypeError, ValueError, yaml.YAMLError) as exc:
                issues.append(
                    f"{block.path.relative_to(REPO_ROOT)}:{block.opening_line}: {exc}"
                )
    assert checked > 0
    assert not issues, "invalid structured documentation fences:\n" + "\n".join(issues)


def test_cli_option_contracts_match_static_parsers() -> None:
    for source, function, english, chinese, en_heading, zh_heading in CLI_CONTRACTS:
        expected = _parser_long_options(source, function)
        assert _documented_table_options(english, en_heading) == expected
        assert _documented_table_options(chinese, zh_heading) == expected

    validator_options = _parser_long_options(MODE_CONFIG_CLI, "main")
    assert validator_options == {"--mode", "--profile"}
    for root in (ENGLISH_ROOT, CHINESE_ROOT):
        text = (root / "reference" / "configuration.md").read_text(encoding="utf-8")
        assert validator_options <= set(_LONG_OPTION_RE.findall(text))


def test_mirror_protocol_inventory_and_examples_match_implementation() -> None:
    expected = (
        _static_assignment_strings(MIRROR_MOTION_OWNER, "MOTION_OPERATIONS")
        | _static_assignment_strings(
            MIRROR_PROTOCOL,
            "MIRROR_V1_OPERATIONS",
        )
        | _static_assignment_strings(
            MIRROR_PROTOCOL,
            "MIRROR_V2_OPERATIONS",
        )
        | _static_assignment_strings(
            MIRROR_PROTOCOL,
            "MIRROR_V3_OPERATIONS",
        )
    )
    for root in (ENGLISH_ROOT, CHINESE_ROOT):
        path = root / "reference" / "mirror-json.md"
        text = path.read_text(encoding="utf-8")
        assert set(_OPERATION_RE.findall(text)) == expected
        for block in _iter_fenced_blocks(path):
            if block.language != "json":
                continue
            payload = _load_strict_json(block.content)
            if isinstance(payload, Mapping) and set(payload) == {
                "protocol",
                "request_id",
                "operation",
                "arguments",
            }:
                request = decode_request(json.dumps(payload, ensure_ascii=True))
                assert isinstance(payload["request_id"], str) and payload["request_id"]
                assert request.operation in expected
                assert isinstance(payload["arguments"], Mapping)


def test_bilingual_motion_examples_cover_and_parse_with_current_wire_schema() -> None:
    expected = _static_assignment_strings(
        MIRROR_MOTION_OWNER, "MOTION_OPERATIONS"
    ) | _static_assignment_strings(MIRROR_MOTION_OWNER, "MIRROR_V3_MOTION_OPERATIONS")
    config = load_mirror_config("newton_cpu")
    bilingual_payloads: list[tuple[Mapping[str, object], ...]] = []

    for root in (ENGLISH_ROOT, CHINESE_ROOT):
        path = root / "reference" / "mirror-json.md"
        payloads: list[Mapping[str, object]] = []
        for block in _iter_fenced_blocks(path):
            if block.language != "json":
                continue
            payload = _load_strict_json(block.content)
            if not isinstance(payload, Mapping):
                continue
            operation = payload.get("operation")
            if operation not in expected:
                continue
            request = decode_request(json.dumps(payload, ensure_ascii=True))
            if request.operation == "motion.hybrid_force_position":
                parse_hybrid_motion_request(request.arguments_dict())
            else:
                parse_mirror_motion_request(
                    request.operation,
                    request.arguments_dict(),
                    request_id=request.request_id,
                    config=config,
                    allow_effort=request.protocol
                    in {"linkerbot.mirror.v2", "linkerbot.mirror.v3"},
                )
            payloads.append(payload)

        assert {payload["operation"] for payload in payloads} == expected, path
        # Every operation has an example; plan_timeline additionally has a second
        # multi-robot example.
        assert len(payloads) == len(expected) + 1, path
        request_ids = [payload["request_id"] for payload in payloads]
        assert len(request_ids) == len(set(request_ids)), path
        bilingual_payloads.append(tuple(payloads))

    assert bilingual_payloads[0] == bilingual_payloads[1]


def test_python_reference_matches_frozen_facades_exactly() -> None:
    facades = _manifest()["public_facades"]
    assert isinstance(facades, Mapping)
    documents = (
        ENGLISH_ROOT / "reference" / "python-api.md",
        CHINESE_ROOT / "reference" / "python-api.md",
    )
    for module, raw_settings in facades.items():
        assert isinstance(module, str)
        assert isinstance(raw_settings, Mapping)
        assert raw_settings.get("freeze_status") == "frozen", module
        expected = set(raw_settings.get("exports", ()))
        assert set(_static_exports(module)) == expected, module
        for path in documents:
            documented = _documented_facade_symbols(path, module)
            assert documented == expected, (
                f"{path.relative_to(REPO_ROOT)} {module}: "
                f"missing={sorted(expected - documented)}, "
                f"invented={sorted(documented - expected)}"
            )


def test_pure_facades_import_without_heavy_runtime_modules() -> None:
    facades = _manifest()["public_facades"]
    assert isinstance(facades, Mapping)
    pure = tuple(
        module
        for module, settings in facades.items()
        if isinstance(module, str)
        and isinstance(settings, Mapping)
        and settings.get("runtime") == "pure"
    )
    script = f"""
import importlib
import sys
for module in {pure!r}:
    importlib.import_module(module)
blocked = {{'curobo', 'gymnasium', 'isaacsim', 'mujoco_warp', 'newton', 'omni', 'skrl', 'torch'}}
loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in blocked)
assert loaded == [], loaded
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
