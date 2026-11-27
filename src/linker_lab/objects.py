"""Rigid objects for LW-v0: import from meshes, contact material, placement on the table.

An object is one rigid body made of one or more mesh *parts* (for example the pieces of a
convex decomposition, or a single mesh). Each part collides as its exact convex hull. The
object's mass, centre of mass and inertia are written into its USD file explicitly; if
you give only the mass, they are computed for a uniform density over the parts' hulls
(what collides). Every collider gets the object contact material below.

    from linker_lab import objects
    parts = objects.load_mesh("004_sugar_box.stl")
    cfg.scene.object = objects.object_cfg(parts, mass=0.5, pos=objects.on_table(parts, (0.35, 0.0)))

``object_cfg`` returns a standard Isaac Lab ``RigidObjectCfg``; ``write_usd`` returns the
USD file itself (e.g. for Isaac Lab's ``MultiUsdFileCfg`` with a different object per
environment). ``fixed_cfg`` places a fixed object (no body, it does not move: a shelf, a
bin, the playground's floor) with the same material. Files are written once into the user
cache, named by their content.

Importing this module imports nothing from Isaac Lab.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np

from linker_lab import assets

__all__ = [
    "object_cfg",
    "fixed_cfg",
    "write_usd",
    "load_mesh",
    "mass_properties",
    "on_table",
    "table",
    "Part",
    "MATERIAL",
]

Part = tuple[np.ndarray, np.ndarray]
"""One mesh part: vertices (N, 3) in metres in the object frame, triangle indices (M, 3)."""

MATERIAL = {
    "friction": 1.0,
    "torsional_friction": 0.005,
    "rolling_friction": 0.0,
    "condim": 4,
    "solref": (0.004, 1.0),
    "solimp": (0.9, 0.95, 0.001, 0.5, 2.0),
    "margin": 0.0,
    "gap": 0.0001,
}
"""Contact settings of every object surface (MuJoCo meanings; solref in seconds).

Where an object touches the robot, the engine uses the larger friction and condim of the
two surfaces: hand-object contacts run with friction 4 (the hand's), condim 4 and this
torsional friction.
"""


def load_mesh(path: str | Path, scale: float = 1.0) -> list[Part]:
    """Read a mesh file (STL, OBJ, PLY, ...) as a single part. Units: metres after ``scale``."""
    import trimesh

    mesh = trimesh.load_mesh(Path(path), force="mesh")
    return [(np.asarray(mesh.vertices, dtype=np.float64) * scale, np.asarray(mesh.faces))]


def mass_properties(parts: Sequence[Part], mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and 3x3 inertia about it [kg m^2], uniform density over the parts' hulls."""
    import trimesh

    hulls = [trimesh.convex.convex_hull(np.asarray(v, dtype=np.float64)) for v, _ in parts]
    volumes = np.array([h.volume for h in hulls])
    if np.any(volumes <= 0):
        raise ValueError("every part needs a convex hull with positive volume")
    density = mass / volumes.sum()
    com = sum(h.volume * h.center_mass for h in hulls) / volumes.sum()
    inertia = np.zeros((3, 3))
    for hull in hulls:
        m, d = density * hull.volume, hull.center_mass - com
        # moment_inertia is about the hull's own centre of mass for unit density
        inertia += density * hull.moment_inertia + m * (d @ d * np.eye(3) - np.outer(d, d))
    return com, inertia


def write_usd(
    parts: Sequence[Part],
    mass: float | None,
    com: Sequence[float] | None = None,
    inertia: np.ndarray | None = None,
    visual: Part | None = None,
) -> Path:
    """Write the object as a USD rigid body into the user cache and return its path.

    ``mass=None`` writes a fixed object instead: the same colliders and material, but no
    body, so the engine attaches them to the world. ``visual``: a mesh that viewers and
    cameras show instead of the parts (it never collides); without it they show the parts.
    """
    if (com is None) != (inertia is None):
        raise ValueError("give both com and inertia, or neither")
    if mass is None:
        if com is not None:
            raise ValueError("a fixed object (mass None) has no centre of mass or inertia")
        body = None
    else:
        if com is None:
            com, inertia = mass_properties(parts, mass)
        com, inertia = np.asarray(com, dtype=np.float64), np.asarray(inertia, dtype=np.float64)
        principal, axes = np.linalg.eigh(inertia)
        a, b, c = principal
        if a <= 0 or a + b < c:  # eigh sorts ascending, so this covers every triangle inequality
            raise ValueError(f"not a physical inertia (principal moments {principal})")
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        body = (mass, com, principal, axes)

    key = ["fixed"] if body is None else [mass, com.tolist(), inertia.tolist()]
    digest = hashlib.sha256(json.dumps([*key, MATERIAL, visual is not None]).encode())
    for vertices, faces in [*parts, *([visual] if visual is not None else [])]:
        digest.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    path = cache_dir() / f"{digest.hexdigest()[:16]}.usda"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.usda")
        _write_stage(partial, parts, body, visual)
        partial.replace(path)
    return path


def object_cfg(
    parts: Sequence[Part],
    mass: float,
    *,
    com: Sequence[float] | None = None,
    inertia: np.ndarray | None = None,
    pos: Sequence[float] = (0.35, 0.0, 0.0),
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    prim_path: str = "{ENV_REGEX_NS}/object",
    visual: Part | None = None,
):
    """An Isaac Lab ``RigidObjectCfg`` for the object at ``pos`` / ``rot`` (quaternion x, y, z, w).

    Poses are in the environment frame, which is the arm-base frame of LW-v0.
    """
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim import UsdFileCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(write_usd(parts, mass, com, inertia, visual))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot)),
    )


def fixed_cfg(
    parts: Sequence[Part],
    *,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    prim_path: str = "{ENV_REGEX_NS}/fixed",
):
    """An Isaac Lab ``AssetBaseCfg`` for a fixed object at ``pos`` / ``rot`` (x, y, z, w): its
    parts collide with the object material but never move."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import UsdFileCfg

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(write_usd(parts, None))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot)),
    )


@cache
def table() -> dict:
    """The table's collision box in the environment frame: ``top`` height, ``center`` and
    ``half_size`` (x, y) [m], read from LW-v0's support model."""
    root = ET.parse(assets.part_file("support")).getroot()
    geom = root.find(".//geom[@name='table_body']")
    pos = [float(v) for v in geom.get("pos").split()]
    size = [float(v) for v in geom.get("size").split()]
    support = assets.part_pose("support")
    if any(support["pos"]) or support["rot"] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("the support is expected at the origin of the environment")
    return {"top": pos[2] + size[2], "center": tuple(pos[:2]), "half_size": tuple(size[:2])}


def on_table(
    parts: Sequence[Part],
    xy: Sequence[float],
    rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    clearance: float = 0.001,
) -> tuple[float, float, float]:
    """Position that puts the object, turned by ``rot`` (x, y, z, w), ``clearance`` above the
    table top at ``xy``. Released there it drops onto the table."""
    lowest = min(float((np.asarray(v) @ _rotation(rot).T)[:, 2].min()) for v, _ in parts)
    return (float(xy[0]), float(xy[1]), table()["top"] + clearance - lowest)


def cache_dir() -> Path:
    return assets.cache_dir().parent / "objects"


def _rotation(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64) / np.linalg.norm(quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _write_stage(path, parts, body, visual=None) -> None:
    """The USD layout Isaac Lab and Newton read: a rigid body with MassAPI (``body`` = mass,
    com, principal moments, principal axes; None for a fixed object) and one collision mesh
    per part, bound to one physics material."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from scipy.spatial.transform import Rotation

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/object").GetPrim()
    stage.SetDefaultPrim(root)

    if body is not None:
        mass, com, principal, axes = body
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr(float(mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*principal))
        x, y, z, w = Rotation.from_matrix(axes).as_quat()
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(w), float(x), float(y), float(z)))

    m = MATERIAL
    material = UsdShade.Material.Define(stage, "/object/material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(m["friction"])
    physics_material.CreateDynamicFrictionAttr(m["friction"])
    physics_material.CreateRestitutionAttr(0.0)
    for name, value in (
        ("newton:torsionalFriction", m["torsional_friction"]),
        ("newton:rollingFriction", m["rolling_friction"]),
    ):
        material.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)

    for i, (vertices, faces) in enumerate(parts):
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        mesh = UsdGeom.Mesh.Define(stage, f"/object/part_{i}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
        for name, kind, value in (
            ("newton:maxHullVertices", Sdf.ValueTypeNames.Int, -1),  # exact hull, no vertex cap
            ("newton:contactMargin", Sdf.ValueTypeNames.Float, m["margin"]),
            ("newton:contactGap", Sdf.ValueTypeNames.Float, m["gap"]),
            ("mjc:condim", Sdf.ValueTypeNames.Int, m["condim"]),
            ("mjc:solref", Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray(m["solref"])),
            ("mjc:solimp", Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray(m["solimp"])),
        ):
            prim.CreateAttribute(name, kind).Set(value)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    if visual is not None:  # no CollisionAPI: shown, never collides; Isaac Lab then hides the parts
        vertices, faces = visual
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        mesh = UsdGeom.Mesh.Define(stage, "/object/visual")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
