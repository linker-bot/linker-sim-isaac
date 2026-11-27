"""Feedback templates must link to current upstream documentation and commands."""

from pathlib import Path
from urllib.parse import urlparse

import yaml


def test_issue_templates_link_to_current_documentation():
    root = Path(__file__).resolve().parents[1]
    templates = root / ".github" / "ISSUE_TEMPLATE"
    config = yaml.safe_load((templates / "config.yml").read_text())
    docs = []
    for link in config["contact_links"]:
        url = urlparse(link["url"])
        assert url.scheme == "https" and url.netloc == "github.com"
        assert url.path.startswith("/linker-bot/linker-sim-isaac/")
        if "/blob/main/" in url.path:
            target = root / url.path.split("/blob/main/", 1)[1]
            assert target.is_file(), target
            docs.append(target)
        else:
            assert url.path.endswith("/security")
    assert docs
    for name in ("bug_report.md", "question.md"):
        text = (templates / name).read_text()
        assert "uv run linker-lab doctor" in text
        assert "scripts/mirror.py" not in text
        assert "Mirror" not in text and "Kaleidoscope" not in text
