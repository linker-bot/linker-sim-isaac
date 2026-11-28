"""The glibc that install.sh and the guides require is the one the locked wheels need (CPU)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ["README.md", "docs/getting-started.md", "docs/requirements.md", "docs/zh-CN/README.md"]


def locked_glibc() -> str:
    """The newest glibc any locked wheel-only package needs, on either architecture."""
    needed = []
    for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]:
        if "sdist" in package:
            continue  # buildable anywhere
        for arch in ("x86_64", "aarch64"):
            tags = [
                tuple(map(int, tag))
                for wheel in package.get("wheels", [])
                for tag in re.findall(rf"manylinux_(\d+)_(\d+)_{arch}", wheel["url"])
            ]
            if tags:
                needed.append(min(tags))  # uv picks the wheel with the oldest glibc
    major, minor = max(needed)
    return f"{major}.{minor}"


def test_install_script_and_guides_require_the_locked_glibc():
    glibc = locked_glibc()
    script = (ROOT / "install.sh").read_text()
    assert f'MIN_GLIBC="{glibc}"' in script
    for guide in GUIDES:
        stated = set(re.findall(r"glibc (\d+\.\d+)", (ROOT / guide).read_text()))
        assert stated == {glibc}, f"{guide} states glibc {stated}, the locked wheels need {glibc}"
