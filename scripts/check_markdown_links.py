#!/usr/bin/env python3
"""Check that repository-local Markdown links resolve to tracked paths.

The checker deliberately ignores URL fragments.  Markdown renderers use different
heading-slug rules, while a missing or untracked target file is unambiguous on every
renderer.  HTTP, mail, absolute, and protocol-relative links are outside this
repository-local gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_SEARCH_ROOTS = ("configs", "docs", "scripts", "src", "tests", "tools")
_FENCE_RE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})")
_REFERENCE_DEFINITION_RE = re.compile(r"^[ ]{0,3}\[[^]]+\]:\s*(?P<target>.+?)\s*$")
_INLINE_CODE_RE = re.compile(r"(?P<ticks>`+).*?(?P=ticks)")


@dataclass(frozen=True)
class LinkIssue:
    """One repository-local link whose resolved target cannot be used."""

    source: Path
    line: int
    target: str
    reason: str


def discover_markdown_files(repo_root: Path = REPO_ROOT) -> tuple[Path, ...]:
    """Return maintained Markdown files without traversing runtime-data directories."""

    root = repo_root.resolve()
    files = {path for path in root.glob("*.md") if path.is_file()}
    for relative_root in MARKDOWN_SEARCH_ROOTS:
        search_root = root / relative_root
        if search_root.is_dir():
            files.update(path for path in search_root.rglob("*.md") if path.is_file())
    return tuple(sorted(files))


def markdown_files_from_paths(
    paths: Sequence[Path], *, repo_root: Path = REPO_ROOT
) -> tuple[Path, ...]:
    """Expand explicit files/directories, or use the maintained default search roots."""

    root = repo_root.resolve()
    if not paths:
        return discover_markdown_files(root)

    files: set[Path] = set()
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        path = path.resolve()
        if path.is_dir():
            files.update(item for item in path.rglob("*.md") if item.is_file())
        elif path.is_file() and path.suffix.lower() == ".md":
            files.add(path)
        else:
            raise ValueError(
                f"Markdown input does not exist or is not a .md file: {raw_path}"
            )
    return tuple(sorted(files))


def _git_paths(
    repo_root: Path,
    *,
    include_untracked: bool,
) -> frozenset[Path]:
    """Return index paths and, when requested, non-ignored worktree additions."""

    root = repo_root.resolve()
    command = ["git", "-C", str(root), "ls-files", "-z"]
    if include_untracked:
        command.extend(("--cached", "--others", "--exclude-standard"))
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ValueError("Git is required to verify Markdown link ownership") from exc
    except subprocess.CalledProcessError as exc:
        detail = os.fsdecode(exc.stderr).strip()
        message = "cannot list Git-tracked repository paths"
        if detail:
            message = f"{message}: {detail}"
        raise ValueError(message) from exc

    return frozenset(
        (root / os.fsdecode(relative_path)).resolve()
        for relative_path in completed.stdout.split(b"\0")
        if relative_path
    )


def git_tracked_paths(repo_root: Path = REPO_ROOT) -> frozenset[Path]:
    """Return absolute paths tracked by the repository index."""

    return _git_paths(repo_root, include_untracked=False)


def git_worktree_paths(repo_root: Path = REPO_ROOT) -> frozenset[Path]:
    """Return tracked plus untracked, non-ignored paths without changing the index."""

    return _git_paths(repo_root, include_untracked=True)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _inline_targets(line: str) -> Iterator[str]:
    """Yield inline link/image destinations, including balanced parentheses."""

    cursor = 0
    while True:
        marker = line.find("](", cursor)
        if marker < 0:
            return
        cursor = marker + 2
        if _is_escaped(line, marker):
            continue

        if cursor < len(line) and line[cursor] == "<":
            end = line.find(">", cursor + 1)
            if end >= 0:
                yield line[cursor + 1 : end].strip()
                cursor = end + 1
            continue

        start = cursor
        depth = 1
        while cursor < len(line):
            character = line[cursor]
            if character == "\\":
                cursor += 2
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    yield line[start:cursor].strip()
                    cursor += 1
                    break
            cursor += 1


def _reference_target(line: str) -> str | None:
    match = _REFERENCE_DEFINITION_RE.match(line)
    if match is None:
        return None
    payload = match.group("target")
    if payload.startswith("<"):
        end = payload.find(">", 1)
        return None if end < 0 else payload[1:end].strip()
    return payload.split(maxsplit=1)[0]


def _targets_with_lines(markdown: str) -> Iterator[tuple[int, str]]:
    fence_character: str | None = None
    fence_length = 0
    for line_number, original_line in enumerate(markdown.splitlines(), start=1):
        fence = _FENCE_RE.match(original_line)
        if fence is not None:
            marker = fence.group("fence")
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not original_line[fence.end() :].strip()
            ):
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue

        line = _INLINE_CODE_RE.sub("", original_line)
        yield from ((line_number, target) for target in _inline_targets(line))
        reference_target = _reference_target(line)
        if reference_target is not None:
            yield line_number, reference_target


def _resolved_local_target(
    source: Path, raw_target: str, *, repo_root: Path
) -> tuple[Path | None, str | None]:
    target = raw_target.strip()
    if not target or target.startswith("#"):
        return None, None

    split = urlsplit(target)
    if split.scheme or split.netloc or target.startswith(("/", "\\")):
        return None, None
    decoded_path = unquote(split.path)
    if not decoded_path:
        return None, None

    root = repo_root.resolve()
    resolved = (source.parent / decoded_path).resolve()
    if not resolved.is_relative_to(root):
        return resolved, "target escapes the repository"
    return resolved, None


def _target_is_git_tracked(target: Path, tracked_paths: frozenset[Path]) -> bool:
    if target in tracked_paths:
        return True
    return target.is_dir() and any(
        tracked.is_relative_to(target) for tracked in tracked_paths
    )


def check_markdown_links(
    markdown_files: Iterable[Path],
    *,
    repo_root: Path = REPO_ROOT,
    tracked_paths: Iterable[Path] | None = None,
) -> tuple[LinkIssue, ...]:
    """Check local targets and optional Git ownership in deterministic order."""

    root = repo_root.resolve()
    tracked = (
        None
        if tracked_paths is None
        else frozenset(path.resolve() for path in tracked_paths)
    )
    issues: list[LinkIssue] = []
    for source in sorted(path.resolve() for path in markdown_files):
        markdown = source.read_text(encoding="utf-8")
        for line, raw_target in _targets_with_lines(markdown):
            try:
                target, reason = _resolved_local_target(
                    source, raw_target, repo_root=root
                )
                if target is None:
                    continue
                if reason is None and not target.exists():
                    reason = "target does not exist"
                elif (
                    reason is None
                    and tracked is not None
                    and not _target_is_git_tracked(target, tracked)
                ):
                    reason = "target is not tracked by Git"
            except (OSError, ValueError) as exc:
                reason = f"invalid local target: {exc}"
            if reason is not None:
                issues.append(
                    LinkIssue(
                        source=source,
                        line=line,
                        target=raw_target,
                        reason=reason,
                    )
                )
    return tuple(issues)


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files or directories (default: maintained repository roots)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-untracked-existing",
        action="store_true",
        help=(
            "also accept existing Git-unignored worktree targets; useful while a "
            "multi-file documentation change has not been staged"
        ),
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        files = markdown_files_from_paths(args.paths, repo_root=repo_root)
        tracked = (
            git_worktree_paths(repo_root)
            if args.allow_untracked_existing
            else git_tracked_paths(repo_root)
        )
    except ValueError as exc:
        parser.error(str(exc))

    issues = check_markdown_links(
        files,
        repo_root=repo_root,
        tracked_paths=tracked,
    )
    for issue in issues:
        source = _display_path(issue.source, repo_root=repo_root)
        print(f"{source}:{issue.line}: {issue.reason}: {issue.target}")
    if issues:
        print(f"Found {len(issues)} broken repository-local Markdown link(s).")
        return 1
    ownership = (
        "Git-tracked or visible non-ignored worktree files"
        if args.allow_untracked_existing
        else "Git-tracked files"
    )
    print(f"Checked {len(files)} Markdown file(s); local targets are {ownership}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
