"""LW-v0's files from the package, and their conversion to USD (CPU)."""

from pathlib import Path

import pytest

from linker_lab import assets


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A fresh user cache, so every test converts."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assets.usd.cache_clear()
    yield tmp_path
    assets.usd.cache_clear()


def test_files_come_from_the_package():
    assert assets.root().name == "lw_v0" and assets.path("lw_v0.xml").is_file()
    assert (assets.root() / "lw_v0.urdf").is_file()
    with pytest.raises(FileNotFoundError):
        assets.path("layout.yaml")


@pytest.mark.parametrize("part", assets.PARTS)
def test_conversion_writes_the_newton_settings(cache, part):
    from pxr import Usd, UsdPhysics

    usd = assets.usd(part)
    assert usd.is_relative_to(cache / "linker-lab" / "usd")
    stage = Usd.Stage.Open(str(usd))
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MeshCollisionAPI)]
    assert all(p.GetAttribute("newton:maxHullVertices").Get() == -1 for p in colliders)
    if part != "support":  # the support collides with boxes, not meshes
        assert colliders
    bodies = [p for p in stage.Traverse() if p.GetAttribute("mjc:body:gravcomp").HasAuthoredValue()]
    for p in bodies:
        assert p.GetAttribute("mjc:gravcomp").Get() == p.GetAttribute("mjc:body:gravcomp").Get()
    if part in ("left", "right"):  # only the arm units are gravity-compensated
        assert bodies


def test_every_palette_reaches_the_usd(cache):
    """An arm part is coloured by its parts' palettes (arm, flange, hand, bracket, camera); each
    reaches the USD under its own name, none replaces another."""
    import xml.etree.ElementTree as ET

    from pxr import Usd, UsdShade

    usd = assets.usd("left")
    palettes = {Path(t.get("file")).name for t in ET.parse(assets.part_file("left")).iter("texture")}
    assert len(palettes) == 5
    stage = Usd.Stage.Open(str(usd))
    textures = {
        p.GetAttribute("inputs:ColorTexture").Get().path
        for p in stage.Traverse()
        if p.IsA(UsdShade.Material) and p.GetAttribute("inputs:ColorTexture").Get() is not None
    }
    assert textures == {f"./Textures/{name}" for name in palettes}
    for texture in textures:
        assert (usd.parent / texture).is_file()


def test_the_work_surface_is_green(cache):
    """The support's palette has two colours: the work surface's green (an anti-static mat) and
    the grey of the rest of the table and the column. In the USD, every upward face of the
    support's visual at the table top's height shows the green, every other face the grey."""
    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdShade

    from linker_lab import objects

    green, grey = (0x2E, 0x7D, 0x5B), (0xCB, 0xCB, 0xCB)
    usd = assets.usd("support")
    stage = Usd.Stage.Open(str(usd))
    (texture,) = {
        p.GetAttribute("inputs:ColorTexture").Get().path
        for p in stage.Traverse()
        if p.IsA(UsdShade.Material) and p.GetAttribute("inputs:ColorTexture").Get() is not None
    }
    palette = np.asarray(Image.open(usd.parent / texture).convert("RGB"))
    assert {tuple(int(c) for c in rgb) for rgb in palette.reshape(-1, 3)} == {green, grey}

    (prim,) = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    mesh = UsdGeom.Mesh(prim)
    to_world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float) @ to_world[:3, :3] + to_world[3, :3]
    assert set(mesh.GetFaceVertexCountsAttr().Get()) == {3}
    corners = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    up = (normals[:, 2] > 0.99) & (np.abs(corners[:, :, 2] - objects.table()["top"]).max(axis=1) < 1e-3)

    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    uv = np.asarray(st.ComputeFlattened(), dtype=float).reshape(-1, 3, 2).mean(axis=1)  # face centres
    height, width = palette.shape[:2]
    rows = ((1 - uv[:, 1]) * height).astype(int).clip(0, height - 1)  # st's origin is bottom left
    columns = (uv[:, 0] * width).astype(int).clip(0, width - 1)
    colours = palette[rows, columns]
    assert up.any() and (colours[up] == green).all()
    assert (colours[~up] == grey).all()


def test_conversion_is_cached(cache, monkeypatch):
    first = assets.usd("support")
    assets.usd.cache_clear()
    monkeypatch.setattr(assets, "convert", lambda *_: pytest.fail("converted again"))
    assert assets.usd("support") == first


def test_without_the_converter_the_error_says_what_to_do(cache, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_converter(name, *args, **kwargs):
        if name == "mujoco_usd_converter":
            raise ImportError("no converter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_converter)
    with pytest.raises(RuntimeError, match="copy .* to the same path"):
        assets.usd("support")


def test_the_installed_files_are_the_pinned_release():
    """pyproject.toml installs LW-v0's files from a 1.x release tag of linker-sim-assets, and the
    installed package is that release (a stale environment fails here). Within 1.x LW-v0 stays
    the same robot (linker-sim-assets' promise), so moving to another 1.x release changes only
    pyproject.toml and uv.lock; 2.0.0 would be a product decision."""
    import importlib.metadata
    import re
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    (pin,) = [d for d in project["dependencies"] if d.startswith("linker-lw-v0 ")]
    tag = re.fullmatch(r"linker-lw-v0 @ git\+https://github\.com/linker-bot/linker-sim-assets@v(1\.\d+\.\d+)", pin)
    assert tag, f"pyproject.toml pins {pin!r}, not a 1.x release tag"
    assert importlib.metadata.version("linker-lw-v0") == tag[1]


# ---------------------------------------------------------------- controller data


@pytest.mark.parametrize("side", ["left", "right"])
def test_motor_constants_and_derived_passive_damping(side):
    """motor.yaml read for one hand; the passive damping derived from the robot files gives
    the values linker-lab used before they were data (thumb flexion 4 + 0.4 × 1.86²,
    thumb yaw 24 − 20, fingers 4 + 0.4 × 0.89²), exactly."""
    motor = assets.motor(side)
    s = side[0]
    assert motor["shafts"] == tuple(
        f"hand_{s}h_{j}"
        for j in ("thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")
    )  # linker_lab.hand.SHAFTS order
    assert motor["motor_stiffness"] == 600.0 and motor["speed_limit"] == 1.0
    assert motor["advancing_cap"] == (5.0, 3.0, 5.0, 5.0, 5.0, 5.0)
    assert motor["output_damping"] == (40.0, 20.0, 40.0, 40.0, 40.0, 40.0)
    assert motor["viscosity"] == (20.0, 20.0, 4.2, 4.2, 4.2, 4.2)
    assert motor["passive_damping"] == (5.38384, 4.0, 4.31684, 4.31684, 4.31684, 4.31684)


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_controller_limits(side):
    import math

    limits = assets.controller(side)
    assert limits["joints"] == tuple(f"arm_AR5_5_08{side[0].upper()}_W4C4A6_joint_{i}" for i in range(1, 8))
    assert limits["max_velocity"] == (0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49)
    assert limits["max_acceleration"] == (10.0,) * 7 and limits["max_jerk"] == (40.0,) * 7
    corners = ((-23, 90), (23, 90), (60, 49), (60, -49), (23, -90), (-23, -90), (-60, -49), (-60, 49))
    assert limits["wrist_region"] == tuple((math.radians(a), math.radians(b)) for a, b in corners)
