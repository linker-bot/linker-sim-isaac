#!/usr/bin/env python3
"""Record a Simulation run's runtime and check that every reported acceptance test passed.

Uses only the standard library. Metadata is an explicit allow-list: no environment dump,
hostname, GPU UUID, installed-package location or authenticated package-source URL.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = 1
RUNTIME_PACKAGES = ("isaaclab", "newton", "mujoco-warp", "warp-lang", "torch", "pytest")


class EvidenceError(RuntimeError):
    """Acceptance evidence is missing, incomplete or does not describe this checkout."""


def _sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise EvidenceError("expected SHA must be a full Git commit ID")
    return value.lower()


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError(f"could not inspect the Git checkout ({type(error).__name__})") from error
    return result.stdout.strip()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(root: Path) -> dict:
    return {
        "sha": _sha(_git(root, "rev-parse", "HEAD")),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        "file_sha256": {name: _digest(root / name) for name in ("uv.lock", "pyproject.toml")},
    }


def gpu_metadata() -> tuple[list[dict], str | None]:
    """Query only the GPU model, driver and memory size, without device identifiers."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"nvidia-smi query failed ({type(error).__name__})"
    devices = []
    try:
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 3:
                raise ValueError("expected model, driver and memory")
            name, driver, memory = (value.strip() for value in row)
            if not name or not driver or int(memory) <= 0:
                raise ValueError("incomplete GPU information")
            devices.append({"name": name, "driver_version": driver, "memory_mib": int(memory)})
    except ValueError:
        return [], "nvidia-smi returned incomplete GPU information"
    return (devices, None) if devices else ([], "nvidia-smi reported no GPUs")


def collect_metadata(root: Path, expected_sha: str, sync_outcome: str, run: dict) -> dict:
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    gpus, gpu_error = gpu_metadata()
    packages = sorted(
        ({"name": dist.metadata["Name"], "version": dist.version}
         for dist in importlib.metadata.distributions() if dist.metadata.get("Name")),
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    return {
        "schema": SCHEMA,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "expected_sha": _sha(expected_sha),
        "source": source_metadata(root),
        "workflow": run,
        "dependency_sync_outcome": sync_outcome,
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(),
               "machine": platform.machine(), "libc": list(platform.libc_ver()),
               "distribution": {key: release[key] for key in ("ID", "VERSION_ID", "PRETTY_NAME")
                                if key in release}},
        "packages": packages,
        "gpus": gpus,
        "gpu_error": gpu_error,
    }


def check_junit(path: Path) -> int:
    """Accept a nonempty report only when every testcase passed, with consistent counts."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise EvidenceError(f"cannot read the GPU JUnit report ({type(error).__name__})") from error
    if root.tag not in ("testsuite", "testsuites"):
        raise EvidenceError("GPU JUnit report must contain a testsuite or testsuites root")
    cases = list(root.iter("testcase"))
    if not cases:
        raise EvidenceError("GPU acceptance ran no tests")
    if any(not case.get("name") for case in cases):
        raise EvidenceError("GPU JUnit report contains an unnamed testcase")
    for result in ("skipped", "failure", "error"):
        count = len(list(root.iter(result)))
        if count:
            raise EvidenceError(f"GPU acceptance reported {count} {result} result(s)")
    for suite in (node for node in root.iter() if node.tag in ("testsuite", "testsuites")):
        for name in ("tests", "skipped", "failures", "errors"):
            if name not in suite.attrib:
                continue
            try:
                count = int(suite.attrib[name])
            except ValueError as error:
                raise EvidenceError(f"GPU JUnit {name} count is invalid") from error
            if count < 0:
                raise EvidenceError(f"GPU JUnit {name} count is invalid")
            if name == "tests":
                if count != len(list(suite.iter("testcase"))):
                    raise EvidenceError("GPU JUnit test count does not match its testcases")
            elif count:
                raise EvidenceError(f"GPU acceptance summary reported {count} {name}")
    return len(cases)


def check_evidence(root: Path, metadata_path: Path, junit_path: Path, expected_sha: str) -> dict:
    expected_sha = _sha(expected_sha)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError(f"cannot read runtime metadata ({type(error).__name__})") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise EvidenceError("runtime metadata has an unsupported schema")
    source = metadata.get("source")
    if (not isinstance(source, dict) or source.get("sha") != expected_sha
            or metadata.get("expected_sha") != expected_sha):
        raise EvidenceError("runtime metadata does not match the expected commit")
    if source.get("tracked_dirty") is not False or source != source_metadata(root):
        raise EvidenceError("the checkout changed or contains tracked changes")
    if metadata.get("dependency_sync_outcome") != "success":
        raise EvidenceError("the locked simulation environment was not synced successfully")
    gpus = metadata.get("gpus")
    if (not isinstance(gpus, list) or not gpus or metadata.get("gpu_error") or any(
        not isinstance(gpu, dict) or not isinstance(gpu.get("name"), str) or not gpu["name"]
        or not isinstance(gpu.get("driver_version"), str) or not gpu["driver_version"]
        or type(gpu.get("memory_mib")) is not int or gpu["memory_mib"] <= 0
        for gpu in gpus
    )):
        raise EvidenceError("runtime metadata has no verified NVIDIA GPU information")
    for section, fields in (("python", ("version", "implementation")),
                            ("os", ("system", "release", "machine"))):
        data = metadata.get(section)
        if not isinstance(data, dict) or any(
            not isinstance(data.get(field), str) or not data[field] for field in fields
        ):
            raise EvidenceError(f"runtime metadata has incomplete {section} information")
    packages = metadata.get("packages")
    if not isinstance(packages, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        or not isinstance(item.get("version"), str) or not item["version"]
        for item in packages
    ):
        raise EvidenceError("runtime metadata has invalid package versions")
    installed = {item["name"].lower().replace("_", "-") for item in packages}
    if missing := set(RUNTIME_PACKAGES) - installed:
        raise EvidenceError(f"runtime metadata is missing packages: {', '.join(sorted(missing))}")
    tests = check_junit(junit_path)
    return {"schema": SCHEMA, "accepted": True, "sha": expected_sha, "tests_passed": tests}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="record source and runtime metadata")
    check = commands.add_parser("check", help="require complete passing GPU acceptance evidence")
    for command in (record, check):
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--expected-sha", required=True)
        command.add_argument("--output", type=Path, required=True)
    record.add_argument("--sync-outcome", choices=("success", "failure", "skipped", "cancelled"),
                        required=True)
    record.add_argument("--repository", default="")
    record.add_argument("--run-id", default="")
    record.add_argument("--run-attempt", default="")
    check.add_argument("--metadata", type=Path, required=True)
    check.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            result = collect_metadata(
                args.root, args.expected_sha, args.sync_outcome,
                {"repository": args.repository, "run_id": args.run_id,
                 "run_attempt": args.run_attempt},
            )
        else:
            result = check_evidence(args.root, args.metadata, args.junit, args.expected_sha)
        _write_json(args.output, result)
    except (EvidenceError, OSError) as error:
        _write_json(args.output, {"schema": SCHEMA, "accepted": False, "error": str(error)})
        print(f"Simulation evidence rejected: {error}", file=sys.stderr)
        return 1
    print(f"Simulation {args.command}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
