"""The installed wheel must discover its gallery without a source checkout or GPU stack."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_wheel_gallery_outside_checkout(tmp_path):
    pytest.importorskip("hatchling")
    root = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(wheel_dir)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as wheel:
        for entry in (root / "policy-gallery").glob("*/manifest.yaml"):
            prefix = f"linker_lab/_gallery/{entry.parent.name}"
            assert f"{prefix}/manifest.yaml" in wheel.namelist()
            assert f"{prefix}/play.py" in wheel.namelist()
        wheel.extractall(installed)
    env = dict(os.environ, PYTHONPATH=str(installed))
    for args, expected in ((["list"], "ycb-grasp"), (["info", "ycb-grasp"], "License:")):
        result = subprocess.run(
            [sys.executable, "-m", "linker_lab.cli", "gallery", *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert expected in result.stdout
    result = subprocess.run(
        [sys.executable, "-c", "from linker_lab import gallery; "
         "assert gallery.ROOT.is_relative_to(__import__('pathlib').Path(__import__('sys').argv[1])); "
         "assert (gallery.ROOT / 'ycb-grasp' / 'play.py').is_file()", str(installed)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
