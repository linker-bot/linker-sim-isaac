"""Concurrent object conversion publishes complete USD files and cleans failed writes."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import trimesh
from pxr import Usd, UsdPhysics

from linker_lab import objects


@pytest.fixture
def parts(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "cache_dir", lambda: tmp_path)
    mesh = trimesh.creation.box((0.05, 0.05, 0.05))
    return [(mesh.vertices, mesh.faces)]


def assert_object(path):
    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    assert UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetMassAttr().Get() == pytest.approx(0.1)


def test_threads_convert_the_same_object_with_distinct_usd_layers(parts, monkeypatch):
    barrier = threading.Barrier(2)
    write_stage = objects._write_stage
    temporary_paths = []

    def simultaneous_write(path, *args):
        temporary_paths.append(path)
        barrier.wait(timeout=10)
        write_stage(path, *args)

    monkeypatch.setattr(objects, "_write_stage", simultaneous_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(objects.write_usd, parts, 0.1) for _ in range(2)]
        paths = [future.result(timeout=30) for future in futures]
    assert len(set(temporary_paths)) == 2
    assert paths[0] == paths[1]
    assert_object(paths[0])
    assert list(paths[0].parent.iterdir()) == [paths[0]]


@pytest.mark.parametrize("complete", [False, True])
def test_failed_conversion_leaves_no_partial_file(parts, monkeypatch, tmp_path, complete):
    write_stage = objects._write_stage

    def fail(path, *args):
        if complete:
            write_stage(path, *args)
        else:
            path.write_text("unfinished stage")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(objects, "_write_stage", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        objects.write_usd(parts, 0.1)
    assert list(tmp_path.iterdir()) == []


def test_failed_competing_conversion_preserves_completed_object(parts, monkeypatch):
    write_stage = objects._write_stage
    failed_started = threading.Event()
    good_published = threading.Event()
    calls = iter(("fail", "succeed"))

    def competing_write(path, *args):
        if next(calls) == "fail":
            path.write_text("unfinished stage")
            failed_started.set()
            assert good_published.wait(timeout=10)
            raise RuntimeError("conversion failed")
        write_stage(path, *args)

    def publish_good():
        assert failed_started.wait(timeout=10)
        try:
            return objects.write_usd(parts, 0.1)
        finally:
            good_published.set()

    monkeypatch.setattr(objects, "_write_stage", competing_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(objects.write_usd, parts, 0.1)
        succeeded = pool.submit(publish_good)
        path = succeeded.result(timeout=30)
        with pytest.raises(RuntimeError, match="conversion failed"):
            failed.result(timeout=30)
    assert_object(path)
    assert list(path.parent.iterdir()) == [path]
