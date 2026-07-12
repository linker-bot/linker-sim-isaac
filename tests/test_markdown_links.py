from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.check_markdown_links import (
    check_markdown_links,
    discover_markdown_files,
    main,
)


def test_markdown_link_check_accepts_existing_targets_and_ignores_anchors(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    guide = docs / "guide with spaces.md"
    guide.parent.mkdir()
    guide.write_text("# Guide\n", encoding="utf-8")
    index = docs / "index.md"
    index.write_text(
        "[encoded](guide%20with%20spaces.md#section)\n"
        "[literal](guide with spaces.md)\n"
        "[same page](#heading)\n"
        "[external](https://example.com/missing.md)\n",
        encoding="utf-8",
    )

    assert check_markdown_links((index,), repo_root=tmp_path) == ()


def test_markdown_link_check_reports_missing_and_escaping_targets(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text(
        "# Links\n\n[missing](missing.md)\n[escape](../../outside.md)\n",
        encoding="utf-8",
    )

    issues = check_markdown_links((index,), repo_root=tmp_path)

    assert [(issue.line, issue.target, issue.reason) for issue in issues] == [
        (3, "missing.md", "target does not exist"),
        (4, "../../outside.md", "target escapes the repository"),
    ]


def test_markdown_link_check_reports_existing_untracked_target(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    tracked = docs / "tracked.md"
    tracked.write_text("# Tracked\n", encoding="utf-8")
    untracked = docs / "untracked.md"
    untracked.write_text("# Untracked\n", encoding="utf-8")
    index = docs / "index.md"
    index.write_text(
        "[tracked](tracked.md)\n[untracked](untracked.md)\n",
        encoding="utf-8",
    )

    issues = check_markdown_links(
        (index,),
        repo_root=tmp_path,
        tracked_paths=(index, tracked),
    )

    assert [(issue.line, issue.target, issue.reason) for issue in issues] == [
        (2, "untracked.md", "target is not tracked by Git")
    ]


def test_markdown_link_check_ignores_code_and_checks_reference_definitions(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text(
        "`[inline code](missing-inline.md)`\n"
        "```markdown\n```python\n[fenced](missing-fenced.md)\n```\n"
        '[guide]: missing-reference.md "Guide"\n',
        encoding="utf-8",
    )

    issues = check_markdown_links((index,), repo_root=tmp_path)

    assert len(issues) == 1
    assert issues[0].line == 6
    assert issues[0].target == "missing-reference.md"


def test_markdown_discovery_uses_maintained_roots(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    ignored = tmp_path / "runtime-data"
    ignored.mkdir()
    (ignored / "report.md").write_text("# Runtime\n", encoding="utf-8")

    discovered = discover_markdown_files(tmp_path)

    assert {path.relative_to(tmp_path).as_posix() for path in discovered} == {
        "README.md",
        "docs/guide.md",
    }


def test_markdown_link_cli_returns_failure_for_broken_link(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("[missing](missing.md)\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "README.md"), cwd=tmp_path, check=True)

    assert main(["--repo-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "README.md:1: target does not exist: missing.md" in output


def test_markdown_link_cli_returns_failure_for_untracked_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("[draft](draft.md)\n", encoding="utf-8")
    (tmp_path / "draft.md").write_text("# Draft\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "README.md"), cwd=tmp_path, check=True)

    assert main(["--repo-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "README.md:1: target is not tracked by Git: draft.md" in output
