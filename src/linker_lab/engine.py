"""The physics engine linker-lab runs on, and the glue Isaac Lab 3.0 needs to run on it.

Isaac Lab 3.0 was released against Newton 1.6.1. linker-lab pins Newton 1.8.0 with
MuJoCo-Warp 3.17 (exact versions: this package's requirements in ``pyproject.toml``),
because LW-v0 depends on physics added after 1.6, such as the implicit gyroscopic
term for free objects.

Newton 1.8 removed or renamed a few things Isaac Lab 3.0 still uses, and changed one
viewer interface Isaac Lab's Newton viewer wraps; its viewer also spends most of each frame
re-reading unchanged shape colours. Each is patched by one function below, applied by
:func:`apply` (called by ``linker_lab.bootstrap()``) before the rest of Isaac Lab is
imported. Nothing here changes numerical behaviour. Only what
rigid-body scenes such as LW-v0 need is patched: Isaac Lab's deformable bodies
(cloth, soft bodies) and its native Newton actuators do not work with this engine.

This module also holds the checks ``linker-lab doctor`` runs.

Delete the patches when an Isaac Lab release depends on Newton >= 1.8.
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from dataclasses import dataclass

_applied: list[str] = []


def _actuator_clamping_alias() -> None:
    """``isaaclab.actuators.newton.adapter`` imports ``newton.actuators.Clamping``.

    Newton 1.8 removed that name (deprecated in 1.6) in favour of ``ClampingBase``.
    The import runs whenever an Isaac Lab articulation is created. Only the alias
    is restored: Isaac Lab's opt-in native Newton actuator path (``use_newton_actuators``
    or ``NewtonActuator`` USD prims) also relies on other removed names and is not
    supported with this engine; LW-v0 does not use it.
    """
    import newton.actuators

    if hasattr(newton.actuators, "Clamping"):
        raise RuntimeError("newton.actuators.Clamping exists; this patch is no longer needed")
    newton.actuators.Clamping = newton.actuators.ClampingBase


def _deformable_label_aliases() -> None:
    """``isaaclab_newton.cloner.replicate`` reads ``ModelBuilder._cloth_*`` and ``_soft_*``.

    It reads them every time environments are cloned, also in scenes without any
    deformable body. Newton 1.8 renamed cloth/soft deformables to surface/volume
    (``surface_label``, ``_surface_particle_start``, ...). The old names are restored
    as read-only properties of the new ones; the data is the same.
    """
    import newton

    for old, new in (("cloth", "surface"), ("soft", "volume")):
        for old_attr, new_attr in (
            (f"_{old}_label", f"{new}_label"),
            (f"_{old}_particle_start", f"_{new}_particle_start"),
            (f"_{old}_particle_end", f"_{new}_particle_end"),
        ):
            if hasattr(newton.ModelBuilder, old_attr):
                raise RuntimeError(f"ModelBuilder.{old_attr} exists; this patch is no longer needed")
            setattr(newton.ModelBuilder, old_attr, property(lambda b, a=new_attr: getattr(b, a)))


def _viewer_image_logger() -> None:
    """Isaac Lab's Newton viewer wraps the image logger's ``draw()`` with Newton 1.6's signature.

    ``_NewtonViewerUIMixin._patch_image_logger`` replaces ``draw`` by a function taking no
    arguments, which calls the original without any; Newton 1.8 calls
    ``draw(ui, sidebar_width_px=...)``, so every frame of a viewer window raises a
    ``TypeError``. The replacement keeps the wrapper's other change (Newton's own "Logged
    Images" sidebar section stays hidden) and leaves ``draw`` to Newton 1.8, which places
    image windows clear of the sidebar itself. Only the first size of Isaac Lab's
    streaming-camera window is lost.

    Unlike the other patches this one changes Isaac Lab code, so it imports Isaac Lab's
    viewer module; it runs after the Newton aliases it needs.
    """
    import inspect

    from isaaclab_visualizers.newton import newton_visualizer
    from newton._src.viewer.image_logger import ImageLogger

    if "ui" not in inspect.signature(ImageLogger.draw).parameters:
        raise RuntimeError("Newton's ImageLogger.draw takes no ui; this patch is no longer needed")

    def _patch_image_logger(self) -> None:
        image_logger = getattr(self, "_image_logger", None)
        if image_logger is not None:
            image_logger.draw_controls = lambda *args, **kwargs: None

    newton_visualizer._NewtonViewerUIMixin._patch_image_logger = _patch_image_logger


def _viewer_unchanged_colors() -> None:
    """Newton 1.8's viewer re-reads every shape colour from the GPU every frame.

    ``ViewerBase._sync_shape_colors_from_model`` runs each frame and marks every shape batch
    "colours changed", so ``ViewerGL.log_state`` copies each batch's colours from the GPU
    and uploads them again: one blocking copy per batch, also when nothing changed. A scene
    with many separate meshes (the grasp gallery's objects: ~1,000 batches) spends ~50 ms
    per frame on it. The replacement compares the model's colours with the last ones synced
    (one copy per frame) and syncs only when they changed or the viewer rebuilt its batches
    (``model_changed``). Rendering only; remove once Newton syncs colours on change.
    """
    import numpy as np
    from newton._src.viewer.viewer import ViewerBase

    if hasattr(ViewerBase._sync_shape_colors_from_model, "newton_sync"):
        return

    def _sync_shape_colors_from_model(self):
        colors = getattr(self.model, "shape_color", None)
        if colors is None:
            return _sync_shape_colors_from_model.newton_sync(self)
        current = colors.numpy()
        last = getattr(self, "_linker_lab_synced_colors", None)
        if self.model_changed or last is None or not np.array_equal(current, last):
            self._linker_lab_synced_colors = current
            _sync_shape_colors_from_model.newton_sync(self)

    _sync_shape_colors_from_model.newton_sync = ViewerBase._sync_shape_colors_from_model
    ViewerBase._sync_shape_colors_from_model = _sync_shape_colors_from_model


_PATCHES = (
    _actuator_clamping_alias,
    _deformable_label_aliases,
    _viewer_image_logger,
    _viewer_unchanged_colors,
)


_SIMULATION_MODULES = ("isaaclab.actuators", "isaaclab.sim", "isaaclab_newton", "isaaclab_visualizers")
"""Isaac Lab modules that must be imported only after the patches.

``isaaclab.actuators.newton.adapter`` imports ``newton.actuators.Clamping`` by name when it is
imported; every other patched name is looked up when it is used. Isaac Lab's command line
(``isaaclab train``, ``isaaclab play``) imports only its own command modules before it loads
the task packages, so a task package can call ``linker_lab.bootstrap()`` when imported."""


def apply() -> list[str]:
    """Apply every patch once. Must run before Isaac Lab's simulation modules are imported."""
    if _applied:
        return list(_applied)
    early = sorted(
        name
        for name in sys.modules
        if any(name == m or name.startswith(m + ".") for m in _SIMULATION_MODULES)
    )
    if early:
        raise RuntimeError(
            "linker_lab.bootstrap() must be called before Isaac Lab is imported "
            f"(already imported: {', '.join(early[:3])}). Import linker_lab first in your script."
        )
    for patch in _PATCHES:
        patch()
        _applied.append(patch.__name__.lstrip("_"))
    return list(_applied)


def applied() -> list[str]:
    """Names of the patches applied in this process."""
    return list(_applied)


# ---------------------------------------------------------------------------- doctor


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_versions() -> list[Check]:
    """Every package linker-lab pins is installed at the pinned version."""
    from packaging.requirements import Requirement

    checks = []
    for line in metadata.requires("linker-lab") or []:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            checks.append(Check(req.name, False, f"not installed (need {req.specifier})"))
            continue
        ok = req.specifier.contains(installed, prereleases=True)
        checks.append(Check(req.name, ok, f"{installed}" + ("" if ok else f" (need {req.specifier})")))
    return checks


def check_gpu() -> Check:
    """A CUDA GPU is visible to torch."""
    import torch

    if not torch.cuda.is_available():
        return Check("gpu", False, "no CUDA device visible to torch (is the NVIDIA driver installed?)")
    name = torch.cuda.get_device_name(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    return Check("gpu", True, f"{name}, driver {driver}, CUDA {torch.version.cuda}")


def check_patches() -> Check:
    """The compatibility patches are active in this process."""
    names = applied()
    expected = [p.__name__.lstrip("_") for p in _PATCHES]
    return Check("engine patches", names == expected, ", ".join(names) or "none applied")


def check_simulation(seconds: float = 1.0, num_envs: int = 4) -> Check:
    """Simulate LW-v0 with a box dropped onto the table; the robot holds the home pose.

    Runs Isaac Lab headless on the GPU (480 Hz physics, MuJoCo-Warp). Needs the LW-v0
    files (converted to USD on first use). Must be called after :func:`apply`.
    """
    import time

    try:
        from linker_lab import objects, workstation  # converts LW-v0 to USD on first use
    except (FileNotFoundError, RuntimeError) as error:
        return Check("simulation", False, str(error))

    import gymnasium as gym
    import torch
    import trimesh
    from isaaclab.app import launch_simulation

    edge = 0.06
    box = trimesh.creation.box((edge, edge, edge))
    parts = [(box.vertices, box.faces)]
    cfg = workstation.LWv0EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.object = objects.object_cfg(
        parts, 0.1, pos=objects.on_table(parts, (0.35, 0.0), clearance=0.02)
    )
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("Linker-LW-v0", cfg=cfg).unwrapped
        env.reset()
        action = workstation.home_action(env)
        robots = [env.scene[side] for side in workstation.SIDES]
        start = time.perf_counter()
        for _ in range(round(seconds / env.step_dt)):
            env.step(action)
        wall = time.perf_counter() - start
        drift = max(
            float((r.data.joint_pos.torch - r.data.default_joint_pos.torch).abs().max())
            for r in robots
        )
        obj = env.scene["object"]
        height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        error = float((height - edge / 2 - objects.table()["top"]).abs().max())
        speed = float(obj.data.root_lin_vel_w.torch.norm(dim=-1).max())
        env.close()
    ok = drift < 1e-4 and error < 1e-3 and speed < 1e-2 and bool(torch.isfinite(height).all())
    detail = (
        f"LW-v0, {num_envs} environments: home pose held to {drift:.1e} rad; a box dropped "
        f"onto the table rests {error * 1000:.2f} mm from its top, speed {speed:.1e} m/s; "
        f"{seconds:g} s simulated in {wall:.1f} s (the first run also compiles GPU kernels)"
    )
    return Check("simulation", ok, detail)
