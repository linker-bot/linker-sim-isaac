from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_pull_request_template_covers_public_change_evidence() -> None:
    content = (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "Public contract and compatibility" in content
    assert "just quality" in content
    assert "Simulation" in content
    assert "CHANGELOG.md" in content
    assert "English and Chinese documentation" in content
    assert "credential" in content


def test_issue_chooser_disables_unstructured_reports_and_routes_support() -> None:
    path = ROOT / ".github/ISSUE_TEMPLATE/config.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["blank_issues_enabled"] is False
    links = config["contact_links"]
    assert {link["name"] for link in links} == {
        "Documentation",
        "Support guide",
        "Private security reports",
    }
    assert all(
        link["url"].startswith("https://github.com/linker-bot/") for link in links
    )


def test_support_routes_are_bilingual_and_request_reproducible_identity() -> None:
    english = (ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    chinese = (ROOT / "SUPPORT_zh.md").read_text(encoding="utf-8")

    for content in (english, chinese):
        assert "python scripts/mirror.py --version" in content
        assert "git rev-parse HEAD" in content
        assert "SECURITY.md" in content
        assert "credential" in content
    assert "SUPPORT_zh.md" in english
    assert "SUPPORT.md" in chinese


def test_changelogs_have_matching_release_sections() -> None:
    english = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    chinese = (ROOT / "CHANGELOG_zh.md").read_text(encoding="utf-8")

    for content in (english, chinese):
        assert "## [Unreleased]" in content
        assert "## [0.3.0] - 2026-08-26" in content
