"""Everything that points at this repository points at it, at the right release (CPU).

- The clone commands in the user docs clone this repository.
- The task template installs linker-lab from a fixed source snapshot or this repository's
  tag ``v<version>``. Release preparation additionally requires the tag.
- Each gallery artifact is a file of this repository's release ``<entry>-<entry version>``.
"""

import re
import tomllib
from pathlib import Path

import pytest

from linker_lab import __version__, gallery

REPOSITORY = "https://github.com/linker-bot/linker-sim-isaac"
ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), ROOT / "templates" / "task" / "README.md"]


def test_clone_commands_clone_this_repository():
    clones = [(doc, url) for doc in DOCS for url in re.findall(r"git clone (\S+)", doc.read_text())]
    assert clones
    for doc, url in clones:
        assert url.removesuffix(".git") == REPOSITORY, f"{doc.relative_to(ROOT)} clones {url}"


def test_task_template_uses_a_fixed_source():
    project = tomllib.loads((ROOT / "templates" / "task" / "pyproject.toml").read_text())
    source = project["tool"]["uv"]["sources"]["linker-lab"]
    assert source["git"].removesuffix(".git") == REPOSITORY
    if "rev" in source:
        assert set(source) == {"git", "rev"}
        assert re.fullmatch(r"[0-9a-f]{40}", source["rev"]), "use a complete commit ID"
    else:
        assert set(source) == {"git", "tag"}
        assert source.get("tag") == f"v{__version__}", f"the template installs {source}, not tag v{__version__}"


@pytest.mark.parametrize("name", gallery.entries())
def test_gallery_artifacts_are_files_of_the_entrys_release(name):
    manifest = gallery.manifest(name)
    release = f"{REPOSITORY}/releases/download/{name}-{manifest['version']}"
    for artifact, spec in manifest["artifacts"].items():
        assert spec["url"] == f"{release}/{artifact}", f"{name}: {artifact} is at {spec['url']}"
