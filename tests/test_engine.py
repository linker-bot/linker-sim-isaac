"""Engine pins and compatibility patches (CPU)."""

import subprocess
import sys
import types

import pytest

from linker_lab import engine


def run(code: str) -> str:
    """Run code in a fresh interpreter and return its last line of output."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_import_starts_nothing():
    out = run(
        "import sys, linker_lab, linker_lab.engine, linker_lab.cli;"
        "print(sorted(m for m in ('isaaclab', 'newton', 'warp', 'torch', 'pxr') if m in sys.modules))"
    )
    assert out == "[]"


def test_installed_versions_match_pins():
    checks = engine.check_versions()
    names = {c.name for c in checks}
    assert {"isaaclab", "newton", "mujoco-warp", "mujoco", "warp-lang", "torch"} <= names
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_apply_refuses_after_isaaclab_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.sim", types.ModuleType("isaaclab.sim"))
    monkeypatch.setattr(engine, "_applied", [])
    with pytest.raises(RuntimeError, match="before Isaac Lab is imported"):
        engine.apply()


def test_apply_after_isaac_labs_command_line():
    """``isaaclab train`` imports its command modules, then the task packages, which bootstrap."""
    out = run(
        "import isaaclab.cli, linker_lab; linker_lab.bootstrap();"
        "import isaaclab.actuators.newton.adapter; print('ok')"
    )
    assert out == "ok"


def test_patches_restore_the_names_isaac_lab_reads():
    out = run(
        "import linker_lab; linker_lab.bootstrap();"
        "import newton, newton.actuators as a;"
        "b = newton.ModelBuilder();"
        "print(a.Clamping is a.ClampingBase,"
        " b._cloth_label is b.surface_label, b._soft_particle_end is b._volume_particle_end)"
    )
    assert out == "True True True"


def test_bootstrap_is_idempotent():
    out = run(
        "import linker_lab; from linker_lab import engine;"
        "linker_lab.bootstrap(); linker_lab.bootstrap(); print(engine.applied())"
    )
    assert out == (
        "['actuator_clamping_alias', 'deformable_label_aliases', 'viewer_image_logger',"
        " 'viewer_unchanged_colors']"
    )


def test_cli_help():
    out = subprocess.run(
        [sys.executable, "-m", "linker_lab.cli", "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "doctor" in out and "play" in out


def test_viewer_keeps_newtons_image_logger_draw():
    """Isaac Lab's viewer no longer replaces Newton 1.8's ``ImageLogger.draw``."""
    out = run(
        "import linker_lab, types; linker_lab.bootstrap();"
        "from isaaclab_visualizers.newton.newton_visualizer import _NewtonViewerUIMixin as M;"
        "from newton._src.viewer.image_logger import ImageLogger;"
        "v = types.SimpleNamespace(_image_logger=ImageLogger.__new__(ImageLogger));"
        "M._patch_image_logger(v);"
        "print('draw' not in vars(v._image_logger), v._image_logger.draw_controls(None) is None)"
    )
    assert out == "True True"


def test_viewer_syncs_colors_only_when_they_change():
    """Unchanged colours are not synced again; changed colours and a rebuilt viewer are."""
    out = run(
        "import linker_lab, types; linker_lab.bootstrap();"
        "import numpy as np, warp as wp;"
        "from newton._src.viewer.viewer import ViewerBase;"
        "sync = ViewerBase._sync_shape_colors_from_model; calls = [];"
        "sync.newton_sync = lambda self: calls.append(1);"
        "color = lambda c: wp.array(np.full((4, 3), c), dtype=wp.vec3, device='cpu');"
        "v = types.SimpleNamespace(model=types.SimpleNamespace(shape_color=color(1.0)),"
        " model_changed=False);"
        "[sync(v) for _ in range(3)];"  # first sync, then unchanged twice
        "v.model.shape_color = color(0.5); sync(v);"
        "v.model_changed = True; sync(v);"
        "print(len(calls))"
    )
    assert out == "3"


def test_bootstrap_failure_requires_a_fresh_process(monkeypatch):
    calls = []

    def first():
        calls.append("first")

    def failing():
        calls.append("failing")
        raise ValueError("patch failed")

    monkeypatch.setattr(engine, "_applied", [])
    monkeypatch.setattr(engine, "_apply_failed", False)
    monkeypatch.setattr(engine, "_SIMULATION_MODULES", ())
    monkeypatch.setattr(engine, "_PATCHES", (first, failing))
    with pytest.raises(ValueError, match="patch failed"):
        engine.apply()
    with pytest.raises(RuntimeError, match="restart the process"):
        engine.apply()
    assert calls == ["first", "failing"]


def test_bootstrap_first_patch_failure_also_prevents_retry(monkeypatch):
    def failing():
        raise ValueError("patch failed")

    monkeypatch.setattr(engine, "_applied", [])
    monkeypatch.setattr(engine, "_apply_failed", False)
    monkeypatch.setattr(engine, "_SIMULATION_MODULES", ())
    monkeypatch.setattr(engine, "_PATCHES", (failing,))
    with pytest.raises(ValueError):
        engine.apply()
    with pytest.raises(RuntimeError, match="restart the process"):
        engine.apply()
