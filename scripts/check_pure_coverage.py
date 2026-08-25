#!/usr/bin/env python3
"""Report coverage for production modules that do not require Isaac or CUDA."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
import sys
import tomllib

from coverage import Coverage
from coverage.exceptions import NoDataError
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "architecture" / "module_disposition.yaml"
DEFAULT_PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PURE_RUNTIME = "pure"
SOURCE_PREFIX = PurePosixPath("src/linkerbot_sim")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def pure_source_paths(manifest_path: Path) -> tuple[str, ...]:
    """Return the final inventory's checked-in pure production Python paths."""

    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(value, label=str(manifest_path))
    inventory = _mapping(
        manifest.get("generated_inventory"), label="generated_inventory"
    )
    if inventory.get("status") != "final":
        raise ValueError("generated_inventory.status must be 'final'")
    production = _mapping(
        inventory.get("production_python"),
        label="generated_inventory.production_python",
    )
    files = production.get("files")
    if not isinstance(files, list):
        raise ValueError("generated_inventory.production_python.files must be a list")

    paths: list[str] = []
    for index, raw_entry in enumerate(files):
        entry = _mapping(raw_entry, label=f"production_python.files[{index}]")
        if entry.get("runtime") != PURE_RUNTIME:
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"production_python.files[{index}].path must be a string")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
            or not path.is_relative_to(SOURCE_PREFIX)
        ):
            raise ValueError(f"pure production path is outside {SOURCE_PREFIX}: {path}")
        paths.append(path.as_posix())

    if not paths:
        raise ValueError("architecture inventory contains no pure production modules")
    if len(paths) != len(set(paths)):
        raise ValueError("architecture inventory contains duplicate pure source paths")
    return tuple(sorted(paths))


def pure_fail_under(pyproject_path: Path) -> float:
    """Load the CPU coverage floor from the repository quality policy."""

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool = _mapping(pyproject.get("tool"), label="tool")
    linkerbot_sim = _mapping(tool.get("linkerbot_sim"), label="tool.linkerbot_sim")
    coverage = _mapping(
        linkerbot_sim.get("coverage"), label="tool.linkerbot_sim.coverage"
    )
    value = coverage.get("pure_fail_under")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("tool.linkerbot_sim.coverage.pure_fail_under must be numeric")
    threshold = float(value)
    if not 0 <= threshold <= 100:
        raise ValueError("pure_fail_under must be between 0 and 100")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT_PATH)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=REPO_ROOT / ".coverage",
        help="coverage.py data file produced by the CPU test run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = pure_source_paths(args.manifest)
        threshold = pure_fail_under(args.pyproject)
        measured = Coverage(data_file=str(args.data_file))
        measured.load()
        percent = measured.report(include=list(paths))
    except (NoDataError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"pure coverage gate: {exc}", file=sys.stderr)
        return 2

    print(
        f"Pure production coverage: {percent:.2f}% "
        f"(required: {threshold:.2f}%; modules: {len(paths)})"
    )
    if percent < threshold:
        print(
            f"pure coverage gate failed: {percent:.2f}% is below {threshold:.2f}%",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
