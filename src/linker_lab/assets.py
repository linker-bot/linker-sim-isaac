"""LW-v0's robot files, and their USD form for Isaac Lab.

The files are linker-sim-assets' ``assemblies/lw_v0/`` and the parts it is built from (MJCF
and URDF with their meshes), installed as the data package ``linker-lw-v0`` from the release
tag named in ``pyproject.toml`` (``uv.lock`` locks its commit). The scene ``lw_v0.xml`` attaches four parts (``left``, ``right``,
``top_camera``, ``support``) at their poses and holds the solver options and the ``home``
keyframe; the parts' files hold the cameras' intrinsics. This module reads them from there.

The package also holds what the robot files cannot describe, the constants of the real
controllers: the O6 hand motor (``hands/o6/motor.yaml``, :func:`motor`) and the AR5_08 arm's
motion limits (``arms/ar5_08/controller.yaml``, :func:`controller`). linker-sim-assets' README,
"Robot controllers", defines both models; ``linker_lab.hand`` and ``linker_lab.arm`` implement them.

Isaac Lab spawns USD. On first use each part's MJCF is converted with mujoco-usd-converter
into the user cache (``~/.cache/linker-lab/usd/<key>/<part>.usd``, about 1 s for all four);
the key is a hash of the package's files and the converter's version, so new files convert
again and unchanged ones are reused. Two settings the converter does not write are added:

- ``newton:maxHullVertices = -1`` on every collision mesh: the meshes already are exact
  convex hulls, and Newton would otherwise reduce them to 64 vertices.
- ``mjc:gravcomp`` copied from the converter's ``mjc:body:gravcomp``, the name Newton reads:
  without it the robot's bodies are not gravity-compensated in Isaac Lab (they are in MJCF).

Importing this module imports nothing from Isaac Lab.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

__all__ = ["NAME", "root", "path"]

NAME = "lw_v0"
PARTS = ("left", "right", "top_camera", "support")
CONVERTER = "mujoco-usd-converter"


def root() -> Path:
    """The installed ``assemblies/lw_v0/`` folder."""
    return Path(str(importlib.resources.files("linker_lw_v0") / "assemblies" / NAME))


def path(name: str) -> Path:
    """Path of one of LW-v0's files, e.g. ``path("lw_v0.xml")`` or ``path("left.xml")``."""
    file = root() / name
    if not file.is_file():
        raise FileNotFoundError(f"LW-v0 has no file {name!r} (looked in {root()})")
    return file


@cache
def part_file(part: str) -> Path:
    """The MJCF of one of :data:`PARTS`, as ``lw_v0.xml`` loads it (``<model file=...>``):
    the arm units and the top camera are next to the scene, the support is a part of its own."""
    for model in scene().find("asset").iter("model"):
        if model.get("name") == part:
            return (root() / model.get("file")).resolve()
    raise ValueError(f"lw_v0.xml loads no part {part!r}; one of {PARTS}")


def cache_dir() -> Path:
    """Where the USD files are converted to."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "linker-lab" / "usd"


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@cache
def _key() -> str:
    """The package's file hashes (its wheel RECORD) and the converter's version."""
    record = importlib.metadata.distribution("linker-lw-v0").read_text("RECORD") or ""
    text = record + importlib.metadata.version(CONVERTER)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@cache
def usd(part: str) -> Path:
    """The part's USD file, converted from its MJCF on first use."""
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; one of {PARTS}")
    target = cache_dir() / _key() / f"{part}.usd"
    if not target.is_file():
        convert(part_file(part), target)
    return target


def convert(mjcf: Path, target: Path) -> None:
    """Convert one part's MJCF to ``target`` and add the two Newton settings."""
    try:
        import mujoco_usd_converter
        from pxr import Sdf, Usd, UsdPhysics
    except ImportError as error:  # e.g. a machine whose glibc is too old for the USD libraries
        raise RuntimeError(
            f"cannot convert {mjcf.name} to USD here ({error}). Run linker-lab once on a machine "
            f"that can, then copy {target.parent} to the same path on this one."
        ) from error
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.stem}-") as tmp:
        converted = mujoco_usd_converter.Converter(layer_structure=False, scene=False).convert(
            str(mjcf), tmp
        )
        layer = Sdf.Layer.FindOrOpen(converted.path)
        stage = Usd.Stage.Open(layer)
        for prim in stage.Traverse():
            gravcomp = prim.GetAttribute("mjc:body:gravcomp")
            if gravcomp and gravcomp.HasAuthoredValue():
                prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(gravcomp.Get())
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.CreateAttribute(
                    "newton:maxHullVertices",
                    Sdf.ValueTypeNames.Int,
                    variability=Sdf.VariabilityUniform,
                ).Set(-1)
        layer.documentation = f"LW-v0 {mjcf.stem}, converted from {mjcf.name} by linker_lab.assets"
        out = Path(tmp) / f"{mjcf.stem}.usd"
        layer.Export(str(out))
        textures = Path(converted.path).parent / "Textures"  # the palette PNG, next to the USD
        if textures.is_dir():
            shutil.copytree(textures, target.parent / "Textures", dirs_exist_ok=True)
        out.replace(target)


# ------------------------------------------------------------------ facts in lw_v0.xml


@cache
def scene() -> ET.Element:
    return ET.parse(path("lw_v0.xml")).getroot()


@cache
def part_pose(part: str) -> dict:
    """The part's pose in the station frame: ``pos`` (x, y, z) [m] and ``rot`` (x, y, z, w)."""
    for frame in scene().find("worldbody").iter("frame"):
        attach = frame.find("attach")
        if attach is not None and attach.get("model") == part:
            pos = tuple(float(v) for v in frame.get("pos", "0 0 0").split())
            w, x, y, z = (float(v) for v in frame.get("quat", "1 0 0 0").split())
            return {"pos": pos, "rot": (x, y, z, w)}
    raise ValueError(f"lw_v0.xml attaches no part {part!r}")


@cache
def _model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(path("lw_v0.xml")))


@cache
def home(side: str) -> dict[str, float]:
    """The ``home`` keyframe of one arm unit: joint name (without the part prefix) -> angle."""
    model = _model()
    qpos = model.key("home").qpos
    prefix = f"{side}/"
    return {
        model.joint(j).name.removeprefix(prefix): float(qpos[model.jnt_qposadr[j]])
        for j in range(model.njnt)
        if model.joint(j).name.startswith(prefix)
    }


@cache
def camera(part: str, body: str) -> dict:
    """The ``<camera>`` on a body of a part: ``resolution`` (width, height) [px] and the
    pinhole ``fx, fy, cx, cy`` [px] (OpenCV convention)."""
    for element in ET.parse(part_file(part)).getroot().iter("body"):
        if element.get("name") == body and element.find("camera") is not None:
            cam = element.find("camera")
            width, height = (int(v) for v in cam.get("resolution").split())
            fx, fy = (float(v) for v in cam.get("focalpixel").split())
            if any(float(v) for v in cam.get("principalpixel", "0 0").split()):
                raise ValueError(f"{part}/{body}: principal point off the image centre")
            return {"resolution": (width, height), "fx": fx, "fy": fy, "cx": width / 2, "cy": height / 2}
    raise ValueError(f"{part_file(part).name} has no camera on body {body!r}")


# ------------------------------------------------------------------ controller data

MOTOR_MODEL = "geared-shaft-v1"
"""The hand motor model ``linker_lab.hand`` implements (``motor.yaml``'s ``model``)."""


@cache
def _data(name: str) -> dict:
    """One of the package's YAML files (schema version 1), e.g. ``hands/o6/motor.yaml``."""
    import yaml

    file = Path(str(importlib.resources.files("linker_lw_v0") / name))
    data = yaml.safe_load(file.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{name}: schema_version {data.get('schema_version')!r}, linker-lab reads 1")
    return data


def _unit_joints(side: str, part: str, names: list[str], side_letter: str) -> tuple[str, ...]:
    """The joint names of a file ("{S}" templates) as they are in the arm unit ``side``,
    each checked to be one of the unit's joints."""
    joints = tuple(f"{part}_{name.replace('{S}', side_letter)}" for name in names)
    model = _model()
    unit = {model.joint(j).name.removeprefix(f"{side}/") for j in range(model.njnt)}
    if missing := [j for j in joints if j not in unit]:
        raise ValueError(f"LW-v0's {side} unit has no joints {missing}")
    return joints


@cache
def motor(side: str) -> dict:
    """The O6 motor of one hand, from ``hands/o6/motor.yaml``.

    ``shafts``: the six driven joints' names in the arm unit ``side``, the order of every
    list; ``motor_stiffness`` [N·m/rad] and ``speed_limit`` [rad/s]; per shaft
    ``advancing_cap`` [N·m], ``output_damping`` and ``viscosity`` [N·m·s/rad]; and
    ``passive_damping`` [N·m·s/rad], the passive damping the shaft sees, derived from LW-v0's
    files (it is not stored): the joint's damping, plus the part of the drive's ``kv`` that is
    not the motor's ``output_damping``, plus each coupled joint's damping times its coupling
    ratio squared.
    """
    import mujoco

    data = _data("hands/o6/motor.yaml")
    if data["model"] != MOTOR_MODEL:
        raise ValueError(f"motor.yaml: model {data['model']!r}, linker-lab implements {MOTOR_MODEL!r}")
    shafts = _unit_joints(side, "hand", data["shafts"], side[0])
    model = _model()
    passive = []
    for i, shaft in enumerate(shafts):
        joint = model.joint(f"{side}/{shaft}").id
        (drive,) = [u for u in range(model.nu) if model.actuator_trnid[u, 0] == joint]
        damping = model.dof_damping[model.jnt_dofadr[joint]] - model.actuator_biasprm[drive][2]
        damping -= data["output_damping"][i]
        for e in range(model.neq):
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] == joint:
                coefficients = model.eq_data[e][:5]
                if coefficients[0] or any(coefficients[2:]):
                    raise ValueError(f"{shaft}: coupling not linear ({coefficients})")
                damping += model.dof_damping[model.jnt_dofadr[model.eq_obj1id[e]]] * coefficients[1] ** 2
        passive.append(float(damping))
    constants = {"motor_stiffness": float(data["motor_stiffness"]), "speed_limit": float(data["speed_limit"])}
    for key in ("advancing_cap", "output_damping", "viscosity"):
        constants[key] = tuple(float(v) for v in data[key])
    return {"shafts": shafts, **constants, "passive_damping": tuple(passive)}


@cache
def controller(side: str) -> dict:
    """The AR5_08 arm controller's limits, from ``arms/ar5_08/controller.yaml`` (the xCore
    SDK's motion profile; the same for both arms).

    ``joints``: J1..J7's names in the arm unit ``side``, the order of every list;
    ``max_velocity`` [rad/s], ``max_acceleration`` [rad/s²] and ``max_jerk`` [rad/s³] per
    joint; ``wrist_region``: the convex octagon (J6, J7) must stay in, vertices in order [rad].
    """
    data = _data("arms/ar5_08/controller.yaml")
    joints = _unit_joints(side, "arm", data["joints"], side[0].upper())
    region = data["wrist_region"]
    if _unit_joints(side, "arm", region["joints"], side[0].upper()) != joints[5:7]:
        raise ValueError("controller.yaml: the wrist region is not on J6 and J7")
    limits = {k: tuple(float(v) for v in data[k]) for k in ("max_velocity", "max_acceleration", "max_jerk")}
    vertices = tuple((math.radians(j6), math.radians(j7)) for j6, j7 in region["vertices_deg"])
    return {"joints": joints, **limits, "wrist_region": vertices}
