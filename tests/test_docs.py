"""The API reference is the public API's contract (CPU).

- ``docs/api.md`` names each public module in a ``## `linker_lab.<module>``` heading (the
  package itself: ``## `linker_lab```) and each public name in a ``### `<name>...``` heading
  below it; the module's ``__all__`` must be the same names.
- A class section's top-level bullets (``- `member`, `other`: ...``) list its members: every
  member the class defines itself (methods, properties, configuration fields; not the ones
  it inherits from Isaac Lab) must be listed, and every listed member must exist.
- Every ``python`` code block of the user docs (README, ``docs/``, the task template's
  README) is valid Python. The ones that need no simulation run here, each with
  ``workstation`` and an ``LWv0EnvCfg`` named ``cfg`` at hand (the fragments change one); the others run in
  ``tests/gpu/test_docs_gpu.py``, one process each.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api.md"
NEEDS_SIMULATION = "launch_simulation"


def sections() -> dict[str, dict[str, list[str]]]:
    """Module -> public name -> the members its section's top-level bullets name."""
    modules: dict[str, dict[str, list[str]]] = {}
    module = name = None
    # A bullet's continuation lines (indented, not a nested bullet) join its first line.
    text = re.sub(r"\n  (?![ -])", " ", API.read_text())
    for line in text.splitlines():
        if m := re.match(r"## `(linker_lab(?:\.\w+)?)`", line):
            module = modules.setdefault(m[1], {})
            name = None
        elif line.startswith("## "):
            module = name = None
        elif module is not None and (m := re.match(r"### `(\w+)", line)):
            name = m[1]
            module[name] = []
        elif name and line.startswith("- `"):
            head = line[2:].split("`: ", 1)[0] + "`"  # the backticked names before the colon
            module[name] += [re.match(r"\w+", group)[0] for group in re.findall(r"`([^`]+)`", head)]
    return modules


DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), ROOT / "templates" / "task" / "README.md"]


def code_blocks() -> list[tuple[str, str]]:
    """(where, code) of every python block of the user docs."""
    blocks = []
    for doc in DOCS:
        for i, code in enumerate(re.findall(r"```python\n(.*?)```", doc.read_text(), re.S)):
            blocks.append((f"{doc.relative_to(ROOT)}#{i + 1}", code))
    return blocks


def own_members(cls) -> set[str]:
    """Public names assigned or defined in the class body itself (fields, methods, properties)."""
    import ast
    import textwrap

    body = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0].body
    names = set()
    for node in body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    # The term class of a configuration and the configuration of a term are Isaac Lab's.
    return {n for n in names if not n.startswith("_")} - {"class_type", "cfg"}


def exists(cls, member: str) -> bool:
    if hasattr(cls, member):
        return True
    if dataclasses.is_dataclass(cls) and member in {f.name for f in dataclasses.fields(cls)}:
        return True
    # Instance attributes set in a method (e.g. ``self.limit_hits = ...``).
    return re.search(rf"self\.{member}\s*[:=]", inspect.getsource(cls)) is not None


@pytest.mark.parametrize("module", sorted(sections()))
def test_public_names_are_the_documented_ones(module):
    documented = set(sections()[module])
    exported = set(importlib.import_module(module).__all__)
    assert exported == documented, (
        f"{module}: in __all__ only {sorted(exported - documented)}, "
        f"in docs/api.md only {sorted(documented - exported)}"
    )


def test_every_public_module_is_documented():
    documented = set(sections())
    public = {"linker_lab"} | {
        f"linker_lab.{p.stem}"
        for p in (ROOT / "src" / "linker_lab").glob("*.py")
        if not p.stem.startswith("_") and p.stem not in {"engine", "playground", "cli"}
    }
    assert documented == public


@pytest.mark.parametrize(
    "module, name",
    [(m, n) for m, names in sections().items() for n in names],
)
def test_documented_members(module, name):
    obj = getattr(importlib.import_module(module), name)
    listed = sections()[module][name]
    if not inspect.isclass(obj) or issubclass(obj, Exception):
        assert not listed, f"{module}.{name} is not a class but its section lists members"
        return
    missing = [m for m in listed if not exists(obj, m)]
    assert not missing, f"{module}.{name} has no {missing}"
    undocumented = own_members(obj) - set(listed)
    assert not undocumented, f"{module}.{name}: members not in docs/api.md: {sorted(undocumented)}"


def test_code_blocks_compile():
    for where, code in code_blocks():
        compile(code, where, "exec")


CPU_BLOCKS = [b for b in code_blocks() if NEEDS_SIMULATION not in b[1]]


@pytest.mark.parametrize("where, code", CPU_BLOCKS, ids=[w for w, _ in CPU_BLOCKS])
def test_code_blocks_run(where, code):
    from linker_lab import workstation

    namespace = {"__name__": "docs", "workstation": workstation, "cfg": workstation.LWv0EnvCfg()}
    exec(compile(code, where, "exec"), namespace)
