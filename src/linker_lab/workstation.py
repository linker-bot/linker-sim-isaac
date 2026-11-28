"""LW-v0 in Isaac Lab: the scene, the palm point, the cameras and the default environment.

Scene (``LWv0SceneCfg``): the four parts of LW-v0 (``linker_lab.assets``), spawned from their
USD files with nothing changed, at their poses in ``lw_v0.xml``:

    left, right   arm units (AR5 arm + O6 hand + wrist camera), one articulation each,
                  starting in the home pose. The drives' gains and torque caps are read from
                  the files (``ImplicitActuatorCfg`` with every value left unset).
    top_camera    the ZED 2i on its mount (welded to the world)
    support       table and arm-base column (welded to the world)

The environment frame is the arm-base frame: x forward over the table, z up; the table
top is 0.5 m below the origin.

Default environment (``LWv0EnvCfg``, registered as ``Linker-LW-v0``): the scene, the
physics of ``linker_lab.physics`` (480 Hz, actions at 60 Hz), an ``ArmAction`` per arm
(joint goals through the trajectory generator, real limits; ``linker_lab.arm``) and a
``HandAction`` per hand (shaft targets; ``linker_lab.hand``), joint-state observations,
reset to the home pose. User tasks subclass it and change the action options with
:func:`arm_action_cfg` and :func:`hand_action_cfg`.

Importing this module applies ``linker_lab.bootstrap()`` (so import it before Isaac Lab)
and converts LW-v0's parts to USD on first use.
"""

from __future__ import annotations

import linker_lab

linker_lab.bootstrap()

from functools import cache
from math import isfinite

import gymnasium as gym
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass

from linker_lab import arm, assets, hand, physics

__all__ = [
    "LWv0EnvCfg",
    "LWv0SceneCfg",
    "ActionsCfg",
    "ObservationsCfg",
    "EventsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "arm_action_cfg",
    "hand_action_cfg",
    "home_action",
    "palm_pose_w",
    "camera_cfg",
    "body_path",
    "SIDES",
    "ARM_JOINTS",
    "HAND_JOINTS",
    "TCP_BODY",
    "PALM_BODY",
    "PALM_OFFSET",
    "CAMERAS",
]

SIDES = ("left", "right")

ARM_JOINTS = {side: assets.controller(side)["joints"] for side in SIDES}
"""The seven arm joints J1..J7 of each arm unit (``arms/ar5_08/controller.yaml``)."""

HAND_JOINTS = {side: assets.motor(side)["shafts"] for side in SIDES}
"""The six driven hand joints (thumb pitch, thumb yaw, index, middle, ring, pinky;
``hands/o6/motor.yaml``).

The other five hand joints (thumb ip and the four dip joints) follow them through fixed
couplings in LW-v0's files. Isaac Lab orders an articulation's joints its own way, so always
select joints by name (``preserve_order=True``)."""

TCP_BODY = {side: f"arm_AR5_5_08{side[0].upper()}_W4C4A6_tcp" for side in SIDES}
"""The arm's tool point (flange face), the frame the real controller reports."""

PALM_BODY = {side: f"hand_{side[0]}h_hand_base_link" for side in SIDES}
PALM_OFFSET = (0.055, 0.0, 0.055)
"""The palm point in the hand's base-link frame [m], the same on both hands.

The base frame's x axis is the palm normal (fingers flex towards +x), z points from the
wrist to the knuckles. The palm face is at x = 0.02, so the point lies 3.5 cm in front of
the palm, halfway between wrist and knuckles."""


# ---------------------------------------------------------------------------- scene


def _pose(part: str) -> dict:
    return assets.part_pose(part)


def arm_unit_cfg(side: str) -> ArticulationCfg:
    """One arm unit, at its pose in ``lw_v0.xml``, starting in the ``home`` keyframe."""
    home = assets.home(side)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/" + side,
        spawn=UsdFileCfg(usd_path=str(assets.usd(side))),
        init_state=ArticulationCfg.InitialStateCfg(
            **_pose(side), joint_pos=dict(home), joint_vel={".*": 0.0}
        ),
        actuators={
            "drives": ImplicitActuatorCfg(
                joint_names_expr=[*ARM_JOINTS[side], *HAND_JOINTS[side]],
                stiffness=None,
                damping=None,
            )
        },
    )


@configclass
class LWv0SceneCfg(InteractiveSceneCfg):
    """The LW-v0 workstation. Cameras are off; add them with :func:`camera_cfg`."""

    left: ArticulationCfg = arm_unit_cfg("left")
    right: ArticulationCfg = arm_unit_cfg("right")
    top_camera: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        spawn=UsdFileCfg(usd_path=str(assets.usd("top_camera"))),
        init_state=ArticulationCfg.InitialStateCfg(
            **_pose("top_camera"), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    support: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/support",
        spawn=UsdFileCfg(usd_path=str(assets.usd("support"))),
        init_state=AssetBaseCfg.InitialStateCfg(**_pose("support")),
    )


# ---------------------------------------------------------------------------- cameras

CAMERAS = {
    "left_rgb": ("left", "camera_rgb_optical", 60, ("rgb",)),
    "left_depth": ("left", "camera_depth_optical", 30, ("depth",)),
    "right_rgb": ("right", "camera_rgb_optical", 60, ("rgb",)),
    "right_depth": ("right", "camera_depth_optical", 30, ("depth",)),
    "top": ("top_camera", "sensor_left_optical", 60, ("rgb", "depth")),
}
"""The camera streams: name -> (part, optical body, rate [Hz], image types). The rates are the
real cameras' (Gemini 335L colour 60 Hz, depth 30 Hz; ZED 2i 60 Hz). Resolution and intrinsics
are the ``<camera>`` on the optical body in LW-v0's files."""

CLIPPING_RANGE = (0.01, 10.0)
"""Nearest and farthest distance [m] each camera renders."""

CAMERA_FORWARD = 0.006
"""Each camera sensor sits this far [m] in front of its optical frame, along the view axis.

The optical frames lie inside the camera housings' visual meshes (the wrist cameras' front
face is 2-4 mm ahead, the top camera's under 1 mm), and Newton's ray-traced renderer has
no near clipping plane, so a camera exactly on its optical frame sees only the inside of
its own housing. The default 6 mm clears the housings but changes the simulated optical
centre. Use the sensor's actual pose for projection, or set ``camera_cfg(optical_offset=0)``
when the renderer or scene can exclude the housing. This offset is not a calibration
accuracy estimate."""


@cache
def body_path(part: str, body: str) -> str:
    """Path of a rigid body below the part's root prim, as written in its USD file, e.g. for
    a contact sensor on the left index fingertip:
    ``f"{{ENV_REGEX_NS}}/left/{body_path('left', 'hand_lh_index_distal')}"``."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(assets.usd(part)))
    root = stage.GetDefaultPrim()
    found = [
        p
        for p in Usd.PrimRange(root)
        if p.GetName() == body and p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(found) != 1:
        raise ValueError(f"expected one body {body!r} in {part}'s USD, found {len(found)}")
    return str(found[0].GetPath().MakeRelativePath(root.GetPath()))


def camera_cfg(
    name: str,
    width: int | None = None,
    height: int | None = None,
    *,
    optical_offset: float = CAMERA_FORWARD,
):
    """An Isaac Lab camera sensor for one of :data:`CAMERAS`, on its optical body.

    Resolution, intrinsics (the ``<camera>`` in LW-v0's files), rate and image types are the
    real camera's. ``width`` and ``height`` [px], given together, set another image size of
    the same aspect ratio (e.g. 640 x 400 for a 1280 x 800 camera): the intrinsics are scaled
    to it, so the image shows the same view with fewer or more pixels. Another aspect ratio
    is another mode of the real camera, with its own calibration, so it raises
    ``ValueError``. ``optical_offset`` is the signed distance [m] along the optical frame's
    forward axis. It defaults to :data:`CAMERA_FORWARD` to clear the camera housing;
    zero keeps the calibrated optical centre. It must be finite. At zero, a renderer
    without near clipping may see the housing unless it is excluded from rendering.
    Images are rendered with Newton's Warp renderer. Add it to a scene with e.g.
    ``cfg.scene.top = camera_cfg("top")``.
    """
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import OpenCvPinholeDistortionCfg, PinholeCameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    if not isfinite(optical_offset):
        raise ValueError("camera optical_offset must be finite")
    part, body, rate_hz, data = CAMERAS[name]
    k = assets.camera(part, body)
    real_width, real_height = k["resolution"]
    if width is None and height is None:
        width, height = real_width, real_height
    elif width is None or height is None:
        raise ValueError(f"camera {name!r}: give both width and height, or neither")
    elif min(width, height) < 1 or (
        height != round(width * real_height / real_width)
        and width != round(height * real_width / real_height)
    ):
        raise ValueError(
            f"camera {name!r}: {width} x {height} is not the aspect ratio of the real camera's "
            f"{real_width} x {real_height}"
        )
    sx, sy = width / real_width, height / real_height
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + part + "/" + body_path(part, body) + "/" + name,
        update_period=1.0 / rate_hz,
        update_latest_camera_pose=True,
        width=width,
        height=height,
        data_types=list(data),
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=PinholeCameraCfg(
            clipping_range=CLIPPING_RANGE,
            # Calibrated pinhole intrinsics (fx and fy differ); no lens distortion.
            distortion=OpenCvPinholeDistortionCfg(
                fx=k["fx"] * sx,
                fy=k["fy"] * sy,
                cx=k["cx"] * sx,
                cy=k["cy"] * sy,
                image_size=(width, height),
                apply_lens_distortion=False,
            ),
        ),
        # The mount body's axes are the optical axes (OpenCV: z forward, y down), which
        # Isaac Lab calls the "ros" convention.
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, float(optical_offset)), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"
        ),
    )


# ---------------------------------------------------------------------------- palm point


def palm_pose_w(robot):
    """World pose of the palm point of an arm unit: position (N, 3), quaternion (N, 4) x, y, z, w.

    ``robot`` is the unit's articulation (``env.scene["left"]``). The orientation is the
    hand base frame's (see :data:`PALM_OFFSET`).
    """
    import torch
    from isaaclab.utils.math import quat_apply

    side = next(s for s in SIDES if PALM_BODY[s] in robot.body_names)
    body = robot.body_names.index(PALM_BODY[side])
    pose = robot.data.body_link_pose_w.torch[:, body]
    offset = torch.tensor(PALM_OFFSET, device=pose.device, dtype=pose.dtype).expand(len(pose), 3)
    return pose[:, :3] + quat_apply(pose[:, 3:7], offset), pose[:, 3:7]


# ---------------------------------------------------------------------------- environment


def arm_action_cfg(side: str, *, frame: str = "tcp", **options) -> arm.ArmActionCfg:
    """An :class:`~linker_lab.arm.ArmActionCfg` for one arm of LW-v0.

    ``options`` are the action's options (``command``, ``interpolation``, ``stream``,
    ``limits``, ``ik_damping``). Pose commands (``command="pose"``) move ``frame``: ``"tcp"``,
    the arm's tool point (the frame the real controller reports), or ``"palm"``, the palm
    point of the hand (:data:`PALM_OFFSET`). E.g.
    ``cfg.actions.left_arm = arm_action_cfg("left", command="pose", frame="palm")``.
    """
    bodies = {"tcp": (TCP_BODY[side], (0.0, 0.0, 0.0)), "palm": (PALM_BODY[side], PALM_OFFSET)}
    if frame not in bodies:
        raise ValueError(f"frame must be 'tcp' or 'palm', not {frame!r}")
    body, offset = bodies[frame]
    return arm.ArmActionCfg(
        asset_name=side, joint_names=list(ARM_JOINTS[side]), body_name=body, body_offset=offset, **options
    )


def hand_action_cfg(side: str, **options) -> hand.HandActionCfg:
    """A :class:`~linker_lab.hand.HandActionCfg` for one hand of LW-v0 (``options``: e.g.
    ``normalized=True``)."""
    motor = assets.motor(side)
    return hand.HandActionCfg(asset_name=side, joint_names=list(motor["shafts"]), motor=motor, **options)


@configclass
class ActionsCfg:
    """Per arm 7 joint goals [rad] (J1..J7), per hand 6 shaft targets [rad] (``HAND_JOINTS``
    order), in this order: left arm, right arm, left hand, right hand."""

    left_arm = arm_action_cfg("left")
    right_arm = arm_action_cfg("right")
    left_hand = hand_action_cfg("left")
    right_hand = hand_action_cfg("right")


def _joints(side: str) -> SceneEntityCfg:
    return SceneEntityCfg(
        side, joint_names=[*ARM_JOINTS[side], *HAND_JOINTS[side]], preserve_order=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Positions [rad] and velocities [rad/s] of the 13 driven joints of each unit."""

        left_joint_pos = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": _joints("left")})
        right_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos, params={"asset_cfg": _joints("right")}
        )
        left_joint_vel = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": _joints("left")})
        right_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel, params={"asset_cfg": _joints("right")}
        )
        concatenate_terms = True

    policy = PolicyCfg()


@configclass
class EventsCfg:
    reset_to_home = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )


@configclass
class RewardsCfg:
    """None: LW-v0 is a workstation, not a task. Tasks add their own."""


@configclass
class TerminationsCfg:
    """None: episodes never end by themselves, so a session runs as long as you step it. Tasks
    add their own, e.g. a time limit:
    ``time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)`` (with ``episode_length_s``)."""


@configclass
class LWv0EnvCfg(ManagerBasedRLEnvCfg):
    scene: LWv0SceneCfg = LWv0SceneCfg(num_envs=1, env_spacing=3.0)
    sim = physics.simulation_cfg()
    decimation = physics.DECIMATION
    episode_length_s = 10.0  # used only by a task's time-limit term
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


def home_action(env):
    """The action that holds every arm and hand at the home pose.

    For arms with joint commands (any stream but ``"samples"``) and every hand; build the
    action yourself for other arm commands.
    """
    import torch

    parts = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        robot = env.scene[term.cfg.asset_name]
        ids, _ = robot.find_joints(term.cfg.joint_names, preserve_order=True)
        home = robot.data.default_joint_pos.torch[:, ids]
        if isinstance(term.cfg, arm.ArmActionCfg):
            if term.cfg.command != "joint" or term.cfg.stream == "samples":
                raise ValueError(f"home_action: {name} does not take joint points")
            if term.cfg.stream == "points_velocity":
                home = torch.cat((home, torch.zeros_like(home)), -1)
        elif isinstance(term.cfg, hand.HandActionCfg) and term.cfg.normalized:
            limits = robot.data.joint_pos_limits.torch[:, ids]
            lower, upper = limits[..., 0], limits[..., 1]
            home = 2 * (home - lower) / (upper - lower) - 1
        parts.append(home)
    return torch.cat(parts, dim=-1)


gym.register(
    id="Linker-LW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:LWv0EnvCfg"},
)
