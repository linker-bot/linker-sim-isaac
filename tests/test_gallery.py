"""The gallery (CPU): every entry's manifest, artifact download and verification, the CLI."""

import hashlib

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
