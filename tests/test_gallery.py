"""The gallery (CPU): every entry's manifest, artifact download and verification, the CLI."""

import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from linker_lab import __version__, cli, gallery


def test_every_entry_has_a_valid_manifest_for_this_version():
    names = gallery.entries()
    assert "ycb-grasp" in names
    for name in names:
        assert gallery.supported(gallery.manifest(name)), f"{name}: not for {__version__}"


def make_entry(root, name="demo", content=b"weights", **changes):
    (root / name).mkdir(parents=True)
    artifact = root / f"{name}.bin"
    artifact.write_bytes(content)
    sha = hashlib.sha256(b"weights").hexdigest()
    data = dict(name=name, version="0.1.0", title="A demo", description="d", license="l",
                linker_lab=">=0.1.0a1", artifacts={"weights.bin": {"url": artifact.as_uri(), "sha256": sha}})
    data.update(changes)
    data = {k: v for k, v in data.items() if v is not None}
    (root / name / "manifest.yaml").write_text(yaml.safe_dump(data))
    (root / name / "play.py").write_text("RAN = []\ndef run(artifacts):\n    RAN.append(sorted(artifacts))\n")


@pytest.fixture
def fake_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "ROOT", tmp_path / "gallery")
    monkeypatch.setattr(gallery, "cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "gallery"


def test_run_downloads_verifies_and_calls_play(fake_gallery, monkeypatch):
    make_entry(fake_gallery)
    loaded = []
    load = gallery.load
    monkeypatch.setattr(gallery, "load", lambda name: loaded.append(load(name)) or loaded[-1])
    gallery.run("demo")
    assert loaded[0].RAN == [["weights.bin"]]
    assert gallery.artifact_path("demo", "weights.bin").read_bytes() == b"weights"


def test_wrong_checksum_is_rejected_and_not_kept(fake_gallery):
    make_entry(fake_gallery, content=b"tampered")
    with pytest.raises(gallery.GalleryError, match="sha256"):
        gallery.fetch("demo")
    assert not gallery.artifact_path("demo", "weights.bin").exists()


def test_unsupported_version_and_bad_manifests(fake_gallery):
    make_entry(fake_gallery, "old", linker_lab="<0.1")
    with pytest.raises(gallery.GalleryError, match="runs on linker-lab <0.1"):
        gallery.run("old")
    make_entry(fake_gallery, "nolicense", license=None)
    with pytest.raises(gallery.GalleryError, match="missing license"):
        gallery.manifest("nolicense")
    with pytest.raises(gallery.GalleryError, match="no gallery entry"):
        gallery.manifest("absent")


def test_cli_list_and_info(capsys):
    assert cli.main(["gallery", "list"]) == 0
    assert "ycb-grasp" in capsys.readouterr().out
    assert cli.main(["gallery", "info", "ycb-grasp"]) == 0
    assert "License:" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["gallery", "run", "ycb-grasp", "--headless"])  # no headless option


def test_load_registers_annotations_and_reloads_a_fresh_module(fake_gallery, monkeypatch):
    make_entry(fake_gallery, "typed")
    module_name = "linker_lab_gallery_typed"
    monkeypatch.setitem(sys.modules, module_name, None)
    (fake_gallery / "typed" / "play.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Settings:\n    title: str = 'demo'\n"
        "def run(artifacts): pass\n"
    )
    first = gallery.load("typed")
    assert sys.modules[module_name] is first
    assert first.Settings().title == "demo"
    second = gallery.load("typed")
    assert second is not first
    assert sys.modules[module_name] is second
    assert second.Settings().title == "demo"


def test_failed_load_removes_partial_module_and_preserves_previous(fake_gallery, monkeypatch):
    make_entry(fake_gallery, "broken")
    module_name = "linker_lab_gallery_broken"
    monkeypatch.setitem(sys.modules, module_name, None)
    source = fake_gallery / "broken" / "play.py"
    good_source = source.read_text()
    source.write_text("raise RuntimeError('failed import')\n")
    with pytest.raises(RuntimeError, match="failed import"):
        gallery.load("broken")
    assert module_name not in sys.modules
    source.write_text(good_source)
    previous = gallery.load("broken")
    source.write_text("raise RuntimeError('failed reload')\n")
    with pytest.raises(RuntimeError, match="failed reload"):
        gallery.load("broken")
    assert sys.modules[module_name] is previous


def test_concurrent_fetches_use_independent_temporary_files(fake_gallery, monkeypatch):
    make_entry(fake_gallery)
    barrier = threading.Barrier(2)
    copy = gallery.shutil.copyfileobj
    temporary_files = []

    def simultaneous_copy(response, destination):
        temporary_files.append(destination.name)
        copy(response, destination)
        destination.flush()
        barrier.wait(timeout=5)

    monkeypatch.setattr(gallery.shutil, "copyfileobj", simultaneous_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gallery.fetch, "demo") for _ in range(2)]
        paths = [future.result(timeout=10)["weights.bin"] for future in futures]
    assert len(set(temporary_files)) == 2
    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == b"weights"
    assert list(paths[0].parent.iterdir()) == [paths[0]]


@pytest.mark.parametrize("failure", ["checksum", "network"])
def test_failed_competing_download_does_not_remove_valid_cache(tmp_path, monkeypatch, failure):
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.write_bytes(b"bad")
    good.write_bytes(b"weights")
    target = tmp_path / "cache" / "weights.bin"
    checksum = hashlib.sha256(b"weights").hexdigest()
    bad_started, good_published = threading.Event(), threading.Event()

    def competing_copy(response, destination):
        data = response.read()
        destination.write(data)
        if data == b"bad":
            bad_started.set()
            assert good_published.wait(timeout=5)
            if failure == "network":
                raise OSError("connection interrupted")

    def publish_good():
        assert bad_started.wait(timeout=5)
        try:
            gallery._download(good.as_uri(), target, checksum)
        finally:
            good_published.set()

    monkeypatch.setattr(gallery.shutil, "copyfileobj", competing_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(gallery._download, bad.as_uri(), target, checksum)
        passed = pool.submit(publish_good)
        passed.result(timeout=10)
        expected = "sha256" if failure == "checksum" else "connection interrupted"
        with pytest.raises(gallery.GalleryError, match=expected):
            failed.result(timeout=10)
    assert target.read_bytes() == b"weights"
    assert list(target.parent.iterdir()) == [target]
