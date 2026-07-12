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

from linkerbot_sim.app.interactive.single_scene.protocol import (
    parse_interactive_motion_message,
)
from linkerbot_sim.app.interactive.single_scene.queue import ResetRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
ENGLISH_ROOT = DOCS_ROOT / "en"
CHINESE_ROOT = DOCS_ROOT / "zh-CN"
SOURCE_ROOT = REPO_ROOT / "src" / "linkerbot_sim"
SINGLE_SCENE_PROTOCOL_SOURCE = (
    SOURCE_ROOT / "app" / "interactive" / "single_scene" / "protocol.py"
)
TILED_SCENE_PROTOCOL_SOURCE = (
    SOURCE_ROOT / "app" / "interactive" / "tiled_scene" / "protocol.py"
)
TILED_SCENE_ACTION_SOURCE = (
    SOURCE_ROOT / "app" / "interactive" / "tiled_scene" / "action_messages.py"
)
TILED_CONTROL_TYPES_SOURCE = SOURCE_ROOT / "tiled" / "control" / "types.py"
RUNTIME_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "runtime" / "example.yaml"

FACADE_SECTIONS: Mapping[str, int] = {
    "linkerbot_sim": 3,
    "linkerbot_sim.planning": 4,
    "linkerbot_sim.backends.curobo": 5,
    "linkerbot_sim.controllers": 6,
    "linkerbot_sim.execution": 7,
    "linkerbot_sim.objects": 8,
    "linkerbot_sim.robots": 9,
    "linkerbot_sim.sensors": 10,
    "linkerbot_sim.snapshots": 11,
    "linkerbot_sim.app.interactive.single_scene": 12,
    "linkerbot_sim.app.interactive.tiled_scene": 13,
}
CLI_CONTRACTS = (
    (
        SOURCE_ROOT / "app" / "interactive" / "single_scene" / "cli.py",
        "parse_args",
        ENGLISH_ROOT / "reference" / "single-scene-cli.md",
        CHINESE_ROOT / "reference" / "single-scene-cli.md",
        "Complete Option Table",
        "完整参数表",
    ),
    (
        SOURCE_ROOT / "app" / "interactive" / "tiled_scene" / "cli.py",
        "parse_args",
        ENGLISH_ROOT / "reference" / "tiled-scene-cli.md",
        CHINESE_ROOT / "reference" / "tiled-scene-cli.md",
        "Complete Option Table",
        "完整参数表",
    ),
    (
        SOURCE_ROOT / "configs" / "cli.py",
        "build_parser",
        ENGLISH_ROOT / "reference" / "configuration.md",
        CHINESE_ROOT / "reference" / "configuration.md",
        "Complete Validator Option Table",
        "校验器完整参数表",
    ),
    (
        REPO_ROOT / "tools" / "object_assets" / "flexible" / "rope" / "build_asset.py",
        "parse_args",
        ENGLISH_ROOT / "development" / "object-assets.md",
        CHINESE_ROOT / "development" / "object-assets.md",
        "Complete Builder Option Table",
        "构建器完整参数表",
    ),
    (
        REPO_ROOT / "tools" / "object_assets" / "rigid" / "tblock" / "build_asset.py",
        "parse_args",
        ENGLISH_ROOT / "development" / "object-assets.md",
        CHINESE_ROOT / "development" / "object-assets.md",
        "Complete Builder Option Table",
        "构建器完整参数表",
    ),
)

_FENCE_OPEN_RE = re.compile(r"^(?: {0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)$")
_LINK_RE = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_LONG_OPTION_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_FACADE_PATH_RE = re.compile(r"linkerbot_sim(?:\.[A-Za-z_]\w*)*")
_OWNER_SYMBOL_RE = re.compile(r"linkerbot_sim(?:\.[A-Za-z_]\w*)+")


@dataclass(frozen=True)
class FencedBlock:
    path: Path
    language: str
    opening_line: int
    content: str


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys, including merged keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


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
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
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
            rf"^(?: {{0,3}}){re.escape(active_fence[0])}{{{len(active_fence)},}}[ \t]*$"
        )
        if re.fullmatch(closing_re, line):
            if language in {"json", "jsonl", "yaml"}:
                yield FencedBlock(
                    path=path,
                    language=language,
                    opening_line=opening_line,
                    content="\n".join(content),
                )
            active_fence = None
            language = ""
            content = []
            continue
        content.append(line)

    assert active_fence is None, f"unclosed fenced block: {path}:{opening_line}"


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_strict_json(content: str) -> object:
    return json.loads(
        content,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
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
    for offset, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            _load_strict_json(line)
        except (TypeError, ValueError) as exc:
            source_line = block.opening_line + offset
            raise ValueError(
                f"invalid JSONL record at source line {source_line}"
            ) from exc


def _markdown_section(text: str, section_number: int) -> str:
    match = re.search(
        rf"^## {section_number}\.(?: |$)(?P<body>.*?)(?=^## \d+\.(?: |$)|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing numbered section {section_number}"
    return match.group(0)


def _named_markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section {heading!r}"
    return match.group("body")


def _marked_section(text: str, name: str) -> str:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _marked_json_object(path: Path, name: str) -> dict[str, object]:
    section = _marked_section(path.read_text(encoding="utf-8"), name)
    match = re.fullmatch(
        r"\s*```json\s*\n(?P<payload>.*?)\n```\s*",
        section,
        re.DOTALL,
    )
    assert match is not None, f"{name} must contain exactly one JSON fence"
    payload = _load_strict_json(match.group("payload"))
    assert isinstance(payload, dict), f"{name} must contain one JSON object"
    return payload


def _table_first_cell_tokens(
    section: str,
    token_pattern: re.Pattern[str],
) -> list[str]:
    tokens: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first_cell = line.strip().strip("|").split("|", maxsplit=1)[0]
        for code_span in _CODE_SPAN_RE.findall(first_cell):
            tokens.extend(token_pattern.findall(code_span))
    return tokens


def _function_node(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(functions) == 1, f"expected one {name} function in {path}"
    return functions[0]


def _literal_strings(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values = {
            element.value for element in node.elts if isinstance(element, ast.Constant)
        }
        assert all(isinstance(value, str) for value in values)
        return values
    return set()


def _strings_compared_with_variable(
    path: Path,
    function_name: str,
    variable_name: str,
) -> set[str]:
    function = _function_node(path, function_name)
    values: set[str] = set()
    for comparison in (
        node for node in ast.walk(function) if isinstance(node, ast.Compare)
    ):
        operands = [comparison.left, *comparison.comparators]
        for left, right in zip(operands, operands[1:]):
            if isinstance(left, ast.Name) and left.id == variable_name:
                values.update(_literal_strings(right))
            if isinstance(right, ast.Name) and right.id == variable_name:
                values.update(_literal_strings(left))
    return values


def _static_string_collection(path: Path, name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    ]
    assert len(assignments) == 1, f"expected one {name} assignment in {path}"
    value = assignments[0].value
    assert value is not None
    if isinstance(value, ast.Call):
        assert len(value.args) == 1
        value = value.args[0]
    result = _literal_strings(value)
    assert result, f"expected a nonempty static string collection for {name}"
    return result


def _documented_marker_tokens(path: Path, marker: str, header: str) -> set[str]:
    section = _marked_section(path.read_text(encoding="utf-8"), marker)
    tokens = set(_table_first_cell_tokens(section, _IDENTIFIER_RE))
    tokens.discard(header)
    return tokens


def _mapping_leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_mapping_leaf_paths(item, path))
        return result
    return {prefix}


def _all_exports(module: str) -> tuple[str, ...]:
    relative_parts = module.split(".")[1:]
    path = SOURCE_ROOT.joinpath(*relative_parts, "__init__.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[object] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and node.value is not None
        ):
            values.append(ast.literal_eval(node.value))

    assert len(values) == 1, f"expected one static __all__ assignment in {path}"
    exports = values[0]
    assert isinstance(exports, (list, tuple))
    assert all(isinstance(name, str) for name in exports)
    assert len(exports) == len(set(exports))
    return tuple(exports)


def _documented_facade_exports(
    path: Path,
    section_number: int,
    expected: set[str],
) -> set[str]:
    section = _markdown_section(path.read_text(encoding="utf-8"), section_number)
    if section_number == 3:
        return {
            token
            for token in _CODE_SPAN_RE.findall(section)
            if _IDENTIFIER_RE.fullmatch(token) and token != "linkerbot_sim"
        }
    documented = set(_table_first_cell_tokens(section, _IDENTIFIER_RE))
    documented.update(expected.intersection(_CODE_SPAN_RE.findall(section)))
    return documented


def _advanced_owner_symbols(path: Path) -> list[str]:
    section = _markdown_section(path.read_text(encoding="utf-8"), 14)
    return _table_first_cell_tokens(section, _OWNER_SYMBOL_RE)


def _module_bound_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                names.update(
                    child.id
                    for child in ast.walk(target)
                    if isinstance(child, ast.Name)
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
    return names


def _assert_owner_symbol_exists(qualified_name: str) -> None:
    module, symbol = qualified_name.rsplit(".", maxsplit=1)
    relative_parts = module.split(".")[1:]
    module_path = SOURCE_ROOT.joinpath(*relative_parts).with_suffix(".py")
    if not module_path.is_file():
        module_path = SOURCE_ROOT.joinpath(*relative_parts, "__init__.py")
    assert module_path.is_file(), f"owner module does not exist: {module}"
    assert symbol in _module_bound_names(module_path), (
        f"owner symbol does not exist: {qualified_name}"
    )


def _parser_long_options(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(functions) == 1, f"expected one {function_name} function in {path}"

    options = {"--help"}
    for call in (
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ):
        declared = {
            value
            for argument in call.args
            if isinstance(argument, ast.Constant)
            and isinstance((value := argument.value), str)
            and value.startswith("--")
        }
        assert declared, f"non-static long option declaration in {path}:{call.lineno}"
        options.update(declared)

        action = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "action"),
            None,
        )
        if isinstance(action, ast.Attribute) and action.attr == "BooleanOptionalAction":
            options.update(f"--no-{option[2:]}" for option in declared)
    return options


def _documented_long_options(path: Path, heading: str) -> set[str]:
    section = _named_markdown_section(path.read_text(encoding="utf-8"), heading)
    return set(_table_first_cell_tokens(section, _LONG_OPTION_RE))


def test_bilingual_markdown_trees_have_identical_relative_paths() -> None:
    english = _markdown_paths(ENGLISH_ROOT)
    chinese = _markdown_paths(CHINESE_ROOT)

    assert english == chinese, (
        f"English-only pages: {sorted(map(str, english - chinese))}\n"
        f"Chinese-only pages: {sorted(map(str, chinese - english))}"
    )


def test_each_bilingual_page_header_links_to_its_counterpart() -> None:
    paired_paths = _markdown_paths(ENGLISH_ROOT).intersection(
        _markdown_paths(CHINESE_ROOT)
    )
    for relative_path in sorted(paired_paths):
        english = ENGLISH_ROOT / relative_path
        chinese = CHINESE_ROOT / relative_path
        for source, counterpart in ((english, chinese), (chinese, english)):
            header = "\n".join(source.read_text(encoding="utf-8").splitlines()[:10])
            targets = []
            for match in _LINK_RE.finditer(header):
                target = unquote(match.group("target").split("#", maxsplit=1)[0])
                if target and "://" not in target:
                    targets.append((source.parent / target).resolve())
            assert counterpart.resolve() in targets, (
                f"header language switch in {source.relative_to(REPO_ROOT)} "
                f"does not target {counterpart.relative_to(REPO_ROOT)}"
            )


def test_structured_markdown_fences_parse_strictly() -> None:
    issues: list[str] = []
    checked = 0
    for path in _maintained_markdown_paths():
        for block in _iter_fenced_blocks(path):
            checked += 1
            try:
                _validate_fenced_block(block)
            except (AssertionError, TypeError, ValueError, yaml.YAMLError) as exc:
                relative = block.path.relative_to(REPO_ROOT)
                issues.append(
                    f"{relative}:{block.opening_line} ({block.language}): {exc}"
                )

    assert checked > 0, "no JSON, JSONL, or YAML documentation fences found"
    assert not issues, "invalid structured documentation fences:\n" + "\n".join(issues)


def test_cli_complete_option_tables_match_static_parsers() -> None:
    for (
        source,
        function_name,
        english,
        chinese,
        english_heading,
        chinese_heading,
    ) in CLI_CONTRACTS:
        actual = _parser_long_options(source, function_name)
        documented_english = _documented_long_options(english, english_heading)
        documented_chinese = _documented_long_options(chinese, chinese_heading)

        assert documented_english == actual, (
            f"{english.relative_to(REPO_ROOT)} option drift: "
            f"missing={sorted(actual - documented_english)}, "
            f"invented={sorted(documented_english - actual)}"
        )
        assert documented_chinese == actual, (
            f"{chinese.relative_to(REPO_ROOT)} option drift: "
            f"missing={sorted(actual - documented_chinese)}, "
            f"invented={sorted(documented_chinese - actual)}"
        )
        assert documented_english == documented_chinese


def test_runtime_reference_covers_complete_example_leaves() -> None:
    example = yaml.load(
        RUNTIME_EXAMPLE_CONFIG.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    expected = _mapping_leaf_paths(example)
    assert expected

    for path in (
        ENGLISH_ROOT / "reference" / "configuration.md",
        CHINESE_ROOT / "reference" / "configuration.md",
    ):
        text = path.read_text(encoding="utf-8")
        missing = {field for field in expected if f"`{field}`" not in text}
        assert not missing, (
            f"{path.relative_to(REPO_ROOT)} omits runtime example leaves: "
            f"{sorted(missing)}"
        )


def test_protocol_indexes_match_static_dispatchers() -> None:
    expected_single_scene = _strings_compared_with_variable(
        SINGLE_SCENE_PROTOCOL_SOURCE,
        "parse_interactive_motion_message",
        "command_type",
    )
    expected_tiled_scene = _strings_compared_with_variable(
        TILED_SCENE_PROTOCOL_SOURCE,
        "handle_tiled_interactive_message",
        "message_type",
    ) | _strings_compared_with_variable(
        TILED_SCENE_ACTION_SOURCE,
        "_action_message",
        "message_type",
    )
    expected_actions = _static_string_collection(
        TILED_CONTROL_TYPES_SOURCE,
        "SUPPORTED_COMMAND_KINDS",
    )
    expected_env_ids_required = _static_string_collection(
        TILED_SCENE_PROTOCOL_SOURCE,
        "_ENV_IDS_REQUIRED_MESSAGE_TYPES",
    )
    for language_root in (ENGLISH_ROOT, CHINESE_ROOT):
        reference_root = language_root / "reference"
        assert (
            _documented_marker_tokens(
                reference_root / "single-scene-json.md",
                "scene-message-index",
                "type",
            )
            == expected_single_scene
        )
        assert (
            _documented_marker_tokens(
                reference_root / "tiled-scene-json.md",
                "tiled-message-index",
                "type",
            )
            == expected_tiled_scene
        )
        assert (
            _documented_marker_tokens(
                reference_root / "tiled-scene-json.md",
                "tiled-action-index",
                "kind",
            )
            == expected_actions
        )
        assert (
            _documented_marker_tokens(
                reference_root / "tiled-scene-json.md",
                "tiled-env-ids-required-index",
                "type",
            )
            == expected_env_ids_required
        )


def test_single_scene_reset_examples_match_current_request_and_response_contract() -> (
    None
):
    for language_root in (ENGLISH_ROOT, CHINESE_ROOT):
        path = language_root / "reference" / "single-scene-json.md"
        request = _marked_json_object(path, "scene-reset-request")
        command = parse_interactive_motion_message(request)

        assert command.kind == "reset"
        assert command.reset_id is not None
        expected_response = {
            "event": "reset",
            "accepted": True,
            **ResetRequest(
                reset_id=command.reset_id,
                clear_queue=command.reset_clear_queue,
                hold_after_reset=command.reset_hold_after_reset,
            ).snapshot(),
        }
        assert _marked_json_object(path, "scene-reset-response") == expected_response


def test_python_reference_names_every_facade_export() -> None:
    document_paths = (
        ENGLISH_ROOT / "reference" / "python-api.md",
        CHINESE_ROOT / "reference" / "python-api.md",
    )
    expected_exports = {facade: set(_all_exports(facade)) for facade in FACADE_SECTIONS}
    for path in document_paths:
        summary = _markdown_section(path.read_text(encoding="utf-8"), 2)
        documented_facades = set(_table_first_cell_tokens(summary, _FACADE_PATH_RE))
        assert documented_facades == set(FACADE_SECTIONS)

        for facade, section_number in FACADE_SECTIONS.items():
            documented = _documented_facade_exports(
                path,
                section_number,
                expected_exports[facade],
            )
            assert documented == expected_exports[facade], (
                f"{path.relative_to(REPO_ROOT)} section {section_number} export drift "
                f"for {facade}: missing={sorted(expected_exports[facade] - documented)}, "
                f"invented={sorted(documented - expected_exports[facade])}"
            )


def test_documented_facades_import_without_heavy_runtime_modules() -> None:
    script = f"""
import importlib
import sys

for module_name in {tuple(FACADE_SECTIONS)!r}:
    importlib.import_module(module_name)

blocked_roots = {{"omni", "isaacsim", "pxr", "torch", "curobo"}}
loaded = sorted(
    module_name
    for module_name in sys.modules
    if module_name.split(".", maxsplit=1)[0] in blocked_roots
)
if loaded:
    raise RuntimeError(f"facade import loaded runtime modules: {{loaded}}")
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


def test_bilingual_advanced_owner_symbols_match_and_exist() -> None:
    english = _advanced_owner_symbols(ENGLISH_ROOT / "reference" / "python-api.md")
    chinese = _advanced_owner_symbols(CHINESE_ROOT / "reference" / "python-api.md")

    assert english, "advanced owner symbol inventory is empty"
    assert len(english) == len(set(english)), "duplicate English advanced owner symbol"
    assert len(chinese) == len(set(chinese)), "duplicate Chinese advanced owner symbol"
    assert set(english) == set(chinese)
    for qualified_name in english:
        _assert_owner_symbol_exists(qualified_name)
