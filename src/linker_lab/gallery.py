"""The policy gallery: pretrained policies that run on LW-v0 (``linker-lab gallery``).

An entry is a folder in ``policy-gallery/`` of the repository:

    manifest.yaml   name, version, title, description, license, the linker-lab versions it
                    runs on (``linker_lab``, a version specifier) and its ``artifacts`` (file
                    name -> url and sha256)
    play.py         run(artifacts): opens the entry's scene in the viewer

``run`` checks the version range, downloads the artifacts into the user cache (checked
against their sha256), then calls the entry's ``play.run`` with the artifact paths. Each
entry owns its scene, observation, action and episode code; the runner interprets nothing
else. An entry is for trying a policy interactively; measuring it belongs to the repository
it was trained in.

To try an entry before its artifacts are published, put each file at its cache path
(:func:`artifact_path`); it is used if its sha256 matches.

Importing this module imports nothing from Isaac Lab.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import shutil
import sys
import urllib.request
from pathlib import Path

from linker_lab import __version__, assets

__all__ = [
    "entries",
    "info",
    "manifest",
    "supported",
    "run",
    "load",
    "fetch",
    "artifact_path",
    "GalleryError",
]

# Wheels carry the entries inside the package; editable checkouts use the source
# gallery so edits to a policy are visible without rebuilding the distribution.
_PACKAGED_ROOT = Path(str(importlib.resources.files("linker_lab") / "_gallery"))
ROOT = (
    _PACKAGED_ROOT
    if _PACKAGED_ROOT.is_dir()
    else Path(__file__).resolve().parents[2] / "policy-gallery"
)
FIELDS = (
    "name",
    "version",
    "title",
    "description",
    "license",
    "linker_lab",
    "artifacts",
)


class GalleryError(RuntimeError):
    """An entry is missing, malformed, not for this linker-lab, or its artifacts do not match."""


def entries() -> list[str]:
    """Names of the gallery's entries."""
    return sorted(p.parent.name for p in ROOT.glob("*/manifest.yaml"))


def manifest(name: str) -> dict:
    """An entry's manifest, checked: every field present, every artifact with a url and a
    sha256, a valid version specifier."""
    import yaml
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    file = ROOT / name / "manifest.yaml"
    if not file.is_file():
        raise GalleryError(f"no gallery entry {name!r}; entries: {', '.join(entries())}")
    data = yaml.safe_load(file.read_text())
    missing = [f for f in FIELDS if f not in data]
    if missing:
        raise GalleryError(f"{file}: missing {', '.join(missing)}")
    if data["name"] != name:
        raise GalleryError(f"{file}: name {data['name']!r} is not the folder name {name!r}")
    for artifact, spec in data["artifacts"].items():
        if not (
            isinstance(spec, dict) and spec.get("url") and len(str(spec.get("sha256", ""))) == 64
        ):
            raise GalleryError(f"{file}: artifact {artifact!r} needs a url and a sha256")
    try:
        SpecifierSet(data["linker_lab"])
    except InvalidSpecifier as error:
        raise GalleryError(f"{file}: linker_lab: {error}") from error
    if not (ROOT / name / "play.py").is_file():
        raise GalleryError(f"gallery entry {name!r} has no play.py")
    return data


def supported(data: dict) -> bool:
    """Whether this linker-lab's version is in the entry's range."""
    from packaging.specifiers import SpecifierSet

    return SpecifierSet(data["linker_lab"]).contains(__version__, prereleases=True)


def info(name: str) -> str:
    """A readable summary of an entry."""
    data = manifest(name)
    lines = [
        f"{data['name']} {data['version']}: {data['title']}",
        "",
        data["description"].strip(),
        "",
        f"License: {data['license'].strip()}",
        f"Runs on linker-lab {data['linker_lab']} (this is {__version__}"
        + ("" if supported(data) else ", NOT supported")
        + ")",
        "Artifacts:",
        *(f"  {k}: {v['url']}" for k, v in data["artifacts"].items()),
    ]
    return "\n".join(lines)


def cache_dir() -> Path:
    return assets.cache_dir().parent / "gallery"


def artifact_path(name: str, artifact: str) -> Path:
    return cache_dir() / name / artifact


def fetch(name: str) -> dict[str, Path]:
    """The entry's artifacts in the user cache, downloaded if missing, checked against their
    sha256."""
    data = manifest(name)
    paths = {}
    for artifact, spec in data["artifacts"].items():
        path = artifact_path(name, artifact)
        if not (path.is_file() and assets.sha256(path) == spec["sha256"]):
            _download(spec["url"], path)
            if assets.sha256(path) != spec["sha256"]:
                path.unlink()
                raise GalleryError(f"{spec['url']} does not match the sha256 in {name}'s manifest")
        paths[artifact] = path
    return paths


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"linker-lab: downloading {url}", file=sys.stderr)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise GalleryError(
            f"could not download {url} ({error}); download it by hand to {destination}"
        ) from error
    partial.replace(destination)


def run(name: str) -> None:
    """Open an entry's scene (see the module docstring); returns when its window is closed."""
    data = manifest(name)
    if not supported(data):
        raise GalleryError(f"{name} runs on linker-lab {data['linker_lab']}, this is {__version__}")
    load(name).run(fetch(name))


def load(name: str):
    """An entry's ``play.py`` as a module (importing it prepares the engine)."""
    spec = importlib.util.spec_from_file_location(
        f"linker_lab_gallery_{name}", ROOT / name / "play.py"
    )
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    return play
