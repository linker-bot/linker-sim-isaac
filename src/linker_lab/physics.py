"""The physics settings LW-v0 is simulated with.

The robot's own properties (masses, joint limits, drives, collision shapes and the
contact settings of its surfaces) are in LW-v0's files (``linker_lab.assets``); the contact
settings of objects are in ``linker_lab.objects``. The time step and the solver settings are
read from the ``<option>`` and ``<size>`` of ``lw_v0.xml``, so Isaac Lab runs what every MuJoCo
tool loading that file runs: 480 Hz, implicit-fast, Newton solver with 30 iterations and 50
line-search iterations, elliptic cone, impratio 10, multi-CCD, 256 contacts and 512
constraint rows per environment. Only the decision rate is linker-lab's own.

Importing this module imports nothing from Isaac Lab.
"""

from __future__ import annotations

from linker_lab import assets

__all__ = ["PHYSICS_HZ", "DECISION_HZ", "DECIMATION", "SOLVER", "simulation_cfg"]

_option = assets.scene().find("option")
_size = assets.scene().find("size")
_flag = _option.find("flag")

PHYSICS_HZ = round(1 / float(_option.get("timestep")))
"""Physics steps per second (480). Collision detection runs every step."""
if 1 / PHYSICS_HZ != float(_option.get("timestep")):
    raise ValueError("lw_v0.xml's timestep is not 1/N s")

DECISION_HZ = 60
"""Actions per second; each action is held for PHYSICS_HZ // DECISION_HZ physics steps."""

DECIMATION = PHYSICS_HZ // DECISION_HZ
"""Physics steps per action (8)."""

SOLVER = {
    "integrator": _option.get("integrator"),
    "solver": _option.get("solver"),
    "iterations": int(_option.get("iterations")),
    "ls_iterations": int(_option.get("ls_iterations")),
    "cone": _option.get("cone"),
    "impratio": float(_option.get("impratio")),
    # MuJoCo runs multi-CCD unless <flag multiccd="disable"/>.
    "multiccd": _flag is None or _flag.get("multiccd", "enable") == "enable",
}
"""MuJoCo-Warp solver settings (names as in MJCF ``<option>``)."""

CAPACITY = {"nconmax": int(_size.get("nconmax")), "njmax": int(_size.get("njmax"))}
"""Contacts and constraint rows reserved per environment."""


def simulation_cfg(device: str = "cuda:0"):
    """Isaac Lab's ``SimulationCfg`` with the settings above.

    MuJoCo-Warp solves the dynamics; Newton's collision pipeline finds the contacts
    (instead of MuJoCo-Warp's own collision code) every physics step. The robot's drives
    are Isaac Lab's implicit actuators with the gains and caps from LW-v0's files, so
    Newton's own actuator models stay off.
    """
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

    return SimulationCfg(
        device=device,
        dt=1 / PHYSICS_HZ,
        render_interval=DECIMATION,
        use_newton_actuators=False,
        physics=NewtonCfg(
            num_substeps=1,
            solver_cfg=MJWarpSolverCfg(
                integrator=SOLVER["integrator"],
                solver=SOLVER["solver"].lower(),
                iterations=SOLVER["iterations"],
                ls_iterations=SOLVER["ls_iterations"],
                cone=SOLVER["cone"],
                impratio=SOLVER["impratio"],
                enable_multiccd=SOLVER["multiccd"],
                use_mujoco_contacts=False,
                **CAPACITY,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            bvh_constructor_geometry="lbvh",
        ),
    )
