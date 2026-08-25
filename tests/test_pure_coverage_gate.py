from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_pure_coverage as coverage_gate
from scripts.check_pure_coverage import pure_fail_under, pure_source_paths


def _write_manifest(path: Path, files: list[dict[str, object]]) -> None:
    path.write_text(
        "generated_inventory:\n"
        "  status: final\n"
        "  production_python:\n"
        "    files:\n"
        + "".join(
            "    - path: "
            + str(entry["path"])
            + "\n      runtime: "
            + str(entry["runtime"])
            + "\n"
            for entry in files
        ),
        encoding="utf-8",
    )


def test_pure_source_paths_are_selected_from_architecture_runtime(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest,
        [
            {
                "path": "src/linkerbot_sim/zeta.py",
                "runtime": "Isaac main thread",
            },
            {"path": "src/linkerbot_sim/beta.py", "runtime": "pure"},
            {"path": "src/linkerbot_sim/alpha.py", "runtime": "pure"},
        ],
    )

    assert pure_source_paths(manifest) == (
        "src/linkerbot_sim/alpha.py",
        "src/linkerbot_sim/beta.py",
    )


@pytest.mark.parametrize(
    "bad_path",
    [
        "tests/test_example.py",
        "src/another_package/module.py",
        "src/linkerbot_sim/../../outside.py",
        "README.md",
    ],
)
def test_pure_source_paths_reject_invalid_production_paths(
    tmp_path: Path, bad_path: str
) -> None:
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, [{"path": bad_path, "runtime": "pure"}])

    with pytest.raises(ValueError, match="outside"):
        pure_source_paths(manifest)


def test_pure_source_paths_require_a_final_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "generated_inventory:\n"
        "  status: provisional\n"
        "  production_python:\n"
        "    files: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status must be 'final'"):
        pure_source_paths(manifest)


@pytest.mark.parametrize("threshold", [-1, 101, "84", True])
def test_pure_fail_under_rejects_invalid_thresholds(
    tmp_path: Path, threshold: object
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    rendered = (
        f'"{threshold}"' if isinstance(threshold, str) else str(threshold).lower()
    )
    pyproject.write_text(
        f"[tool.linkerbot_sim.coverage]\npure_fail_under = {rendered}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pure_fail_under"):
        pure_fail_under(pyproject)


def test_repository_pure_coverage_policy_matches_final_inventory() -> None:
    paths = pure_source_paths(Path("architecture/module_disposition.yaml"))

    assert paths
    assert pure_fail_under(Path("pyproject.toml")) == 84


@pytest.mark.parametrize(("measured", "expected_status"), [(84.0, 0), (83.99, 2)])
def test_main_enforces_the_configured_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measured: float,
    expected_status: int,
) -> None:
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest,
        [{"path": "src/linkerbot_sim/example.py", "runtime": "pure"}],
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.linkerbot_sim.coverage]\npure_fail_under = 84\n",
        encoding="utf-8",
    )

    class FakeCoverage:
        def __init__(self, *, data_file: str) -> None:
            assert data_file == str(tmp_path / ".coverage")

        def load(self) -> None:
            pass

        def report(self, *, include: list[str]) -> float:
            assert include == ["src/linkerbot_sim/example.py"]
            return measured

    monkeypatch.setattr(coverage_gate, "Coverage", FakeCoverage)

    assert (
        coverage_gate.main(
            [
                "--manifest",
                str(manifest),
                "--pyproject",
                str(pyproject),
                "--data-file",
                str(tmp_path / ".coverage"),
            ]
        )
        == expected_status
    )
