"""The O6 hand: a self-locking motor model and the ``HandAction`` term.

The real O6 drives each of its six actuated joints (thumb pitch, thumb yaw, index,
middle, ring, pinky) with a motor behind a self-locking gear: the motor shaft moves
toward its target, slowly and only as far as its torque allows, and an external load
can stop it but never push it back. The finger is connected to the shaft through the
compliance of the transmission, so a pushed finger deflects a little and springs back
when the load goes; a grasp holds without any command.

``HandMotor`` models this once per decision (60 Hz):

* the motor's drive torque is ``motor_stiffness · (target − shaft)``, limited to
  ``advancing_cap`` (the motor's torque budget);
* the shaft advances by the step that torque can make against the measured finger load
  (the compliance ``Kp · (shaft − q)`` and damping, with the finger position predicted
  one decision ahead), at most ``speed_limit`` and at most to the target;
* the step is never negative in the drive's direction: **the shaft never moves away
  from its target**, whatever the finger load.

Each physics step (480 Hz) the joint is pulled toward the shaft by the native implicit
PD drive in LW-v0's files (Kp 2500 thumb flexion, 1250 finger flexion, 600 thumb yaw; output
caps 60 / 10 N·m); the position target is the shaft, linearly interpolated over the
decision, plus a velocity feed-forward ``output_damping / Kp · shaft speed``. The five
coupled finger joints follow their actuated joints through equality constraints in
LW-v0's files.

:class:`HandAction` passes each physics step's target through the arm's torque guard
(:func:`linker_lab.arm.torque_guard`): the drive's torque ``Kp·(target − q) − kd·q̇`` stays
1 N·m (``TORQUE_GUARD_MARGIN``) within the joint's force limit, so a drive is never asked for
more than it can give. On MuJoCo 4 a drive held at its limit loses its implicit damping and
oscillates there, and the motor model, reading that as a load, would stop the shaft for good;
the guard keeps every MuJoCo version on the same physics (motor model
``geared-shaft-v1``).

The stiffness Kp and the shaft range are read from LW-v0's files (the joints' drive gains and
position limits); the motor's own constants from ``hands/o6/motor.yaml`` (``linker_lab.assets.motor``,
which also derives the passive damping each shaft sees from LW-v0's files).

Presets (:data:`PRESETS`, shaft targets in rad) are the hand poses used by the playground.

Importing this module applies ``linker_lab.bootstrap()``, so import it before Isaac Lab.
"""

from __future__ import annotations

import linker_lab

linker_lab.bootstrap()

from dataclasses import MISSING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from linker_lab import arm

__all__ = ["HandAction", "HandActionCfg", "SHAFTS", "PRESETS", "THUMB_OPPOSE_YAW", "thumb_oppose"]

SHAFTS = ("thumb pitch", "thumb yaw", "index", "middle", "ring", "pinky")
"""The six actuated joints, in action order."""

PRESETS = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "close": (0.45, 1.2, 1.45, 1.45, 1.45, 1.45),
    "pinch": (0.37, 1.04, 0.85, 0.0, 0.0, 0.0),
}
"""Shaft targets [rad]. ``thumb_oppose`` (thumb yaw 1.04, the rest unchanged) is not a full
target; see :func:`thumb_oppose`."""

THUMB_OPPOSE_YAW = 1.04
"""Thumb yaw [rad] that turns the thumb opposite the index finger (as in ``pinch``)."""


def thumb_oppose(current: torch.Tensor) -> torch.Tensor:
    """``current`` shaft targets (N, 6) with the thumb rotated opposite the index finger."""
    result = current.clone()
    result[:, 1] = THUMB_OPPOSE_YAW
    return result


class HandMotor:
    """The motor model of :mod:`linker_lab.hand` for N hands, on any torch device.

    ``kp``, ``lower`` and ``upper`` are the joints' drive stiffness and position limits
    (6 values each); ``motor`` holds the motor's constants (``linker_lab.assets.motor``).
    Shaft state is kept in float64; the per-step position targets are float32, as the
    engine's.
    """

    def __init__(self, q, kp, lower, upper, motor: dict, steps_per_decision: int, decision_dt: float):
        self.dt = decision_dt
        self.steps = steps_per_decision
        f64 = dict(device=q.device, dtype=torch.float64)
        self.kp = torch.as_tensor(kp, **f64)
        self.lower = torch.as_tensor(lower, **f64)
        self.upper = torch.as_tensor(upper, **f64)
        self.output_damping = torch.tensor(motor["output_damping"], **f64)
        self.advancing_cap = torch.tensor(motor["advancing_cap"], **f64)
        self.viscosity = torch.tensor(motor["viscosity"], **f64)
        self.passive_damping = torch.tensor(motor["passive_damping"], **f64)
        self.motor_stiffness = motor["motor_stiffness"]
        self.speed_limit = motor["speed_limit"]
        self.shaft = torch.zeros(q.shape, **f64)
        self.previous_q = torch.zeros(q.shape, **f64)
        # fractions of the decision at the end of each physics step: 1/steps, .., 1
        self._fraction = (torch.arange(1, steps_per_decision + 1, device=q.device) / steps_per_decision)
        self._fraction = self._fraction.float()[:, None, None]
        self.targets = torch.zeros((steps_per_decision, *q.shape), device=q.device)  # this decision's
        self.step_index = 0
        self.reset(q)

    def reset(self, q, rows=slice(None)):
        """Start the selected hands with the shaft at the measured joint position."""
        q = q.double()[rows]
        shaft = torch.minimum(torch.maximum(q, self.lower), self.upper)
        self.shaft[rows] = shaft
        self.previous_q[rows] = q
        self.targets[:, rows] = shaft.float()

    def decide(self, q, target):
        """Advance every shaft once toward ``target`` (N, 6) [rad], given the measured
        positions ``q`` (N, 6); return the shaft positions after the decision."""
        q, old = q.double(), self.shaft
        target = torch.minimum(torch.maximum(target.double(), self.lower), self.upper)
        q_speed = (q - self.previous_q) / self.dt
        error = target - old
        drive = (self.motor_stiffness * error).clamp(-self.advancing_cap, self.advancing_cap)
        direction = torch.sign(drive)
        # Step the drive torque can make against the finger load (backward Euler of the
        # shaft against the compliance kp and the dampings), the finger predicted ahead.
        kp, damping = self.kp, self.output_damping
        predicted_q = q + self.dt * q_speed
        possible = (drive - kp * (old - predicted_q) + damping * q_speed) / (
            kp + (damping + self.viscosity) / self.dt
        )
        # Step of the unloaded motor: speed limit, or the viscous losses at this torque.
        free = torch.minimum(
            torch.full_like(drive, self.dt * self.speed_limit),
            self.dt * drive.abs() / (self.viscosity + self.passive_damping),
        )
        # Never against the drive direction (self-locking), never past the target.
        step = direction * torch.minimum(
            (direction * possible).clamp(min=0), torch.minimum(error.abs(), free)
        )
        shaft = old + step
        speed = step / self.dt
        native = shaft + damping / kp * speed  # target with velocity feed-forward
        # per physics step: the shaft interpolated over the decision plus the feed-forward,
        # ending exactly on the feed-forward target
        start, end, offset = old.float(), shaft.float(), (native - shaft).float()
        self.targets = start + self._fraction * (end - start) + offset
        self.targets[-1] = native.float()
        self.shaft = shaft
        self.previous_q = q
        self.step_index = 0
        return shaft

    def step_target(self):
        """The position target for the next physics step (float32, N × 6)."""
        self.step_index += 1
        return self.targets[min(self.step_index, self.steps) - 1]


class HandAction(ActionTerm):
    """Six shaft targets per decision for one O6 hand, through :class:`HandMotor`.

    Actions are absolute radians in :data:`SHAFTS` order, or normalized: -1 is the lower
    joint limit, +1 the upper (``cfg.normalized``). Targets outside the joint range are
    clamped into it. A non-finite target keeps that shaft's previous target.
    """

    cfg: HandActionCfg

    def __init__(self, cfg: HandActionCfg, env):
        super().__init__(cfg, env)
        if tuple(cfg.joint_names) != tuple(cfg.motor["shafts"]):
            raise ValueError(f"HandAction: joint_names {cfg.joint_names} are not the motor's shafts {cfg.motor['shafts']}")
        ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(ids) != 6:
            raise ValueError(f"HandAction needs the 6 actuated hand joints, got {names}")
        self._joint_ids = torch.tensor(ids, device=self.device)
        data = self._asset.data
        limits = data.joint_pos_limits.torch[0, self._joint_ids]
        self._lower, self._upper = limits[:, 0].clone(), limits[:, 1].clone()
        self.motor = HandMotor(
            data.joint_pos.torch[:, self._joint_ids],
            data.joint_stiffness.torch[0, self._joint_ids],
            self._lower,
            self._upper,
            cfg.motor,
            steps_per_decision=env.cfg.decimation,
            decision_dt=env.step_dt,
        )
        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self._target = self.motor.shaft.float().clone()
        self._guard_joints = self._joint_ids.to(torch.int32)
        self._guarded = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        """The shaft targets [rad] in effect."""
        return self._target

    @property
    def shaft(self) -> torch.Tensor:
        """Motor shaft positions [rad] (N, 6)."""
        return self.motor.shaft

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        target = actions
        if self.cfg.normalized:
            target = self._lower + (actions.clamp(-1, 1) + 1) * 0.5 * (self._upper - self._lower)
        self._target = torch.where(torch.isfinite(target), target, self._target)
        self.motor.decide(self._asset.data.joint_pos.torch[:, self._joint_ids], self._target)

    def apply_actions(self):
        data = self._asset.data
        arm.torque_guard(
            self.motor.step_target(), data.joint_pos.warp, data.joint_vel.warp, data.joint_stiffness.warp,
            data.joint_damping.warp, data.joint_effort_limits.warp, self._guard_joints, out=self._guarded,
        )
        self._asset.set_joint_position_target_index(target=self._guarded, joint_ids=self._joint_ids)

    def get_state(self) -> dict:
        """Everything this term carries from one decision to the next (copies): the last
        command, the shaft targets in effect, the guarded drive targets and the motor's state."""
        tensors = {
            "raw": self._raw, "target": self._target, "guarded": self._guarded,
            "shaft": self.motor.shaft, "previous_q": self.motor.previous_q, "targets": self.motor.targets,
        }
        return {**{k: v.clone() for k, v in tensors.items()}, "step_index": self.motor.step_index}

    def set_state(self, state: dict) -> None:
        """Restore what :meth:`get_state` returned, copied into this term's own tensors. Raises
        ``ValueError`` before changing anything if the keys, shapes or dtypes differ."""
        arm.check_state(self.get_state(), state, type(self).__name__)
        for key, tensor in (
            ("raw", self._raw), ("target", self._target), ("guarded", self._guarded),
            ("shaft", self.motor.shaft), ("previous_q", self.motor.previous_q), ("targets", self.motor.targets),
        ):
            tensor.copy_(state[key])
        self.motor.step_index = int(state["step_index"])

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self.motor.reset(q, rows)
        self._raw[rows] = 0
        self._target[rows] = self.motor.shaft[rows].float()


@configclass
class HandActionCfg(ActionTermCfg):
    """:class:`HandAction`. ``linker_lab.workstation.hand_action_cfg`` fills in the joints and the motor."""

    class_type: type = HandAction
    joint_names: list[str] = MISSING
    """The 6 actuated joints in :data:`SHAFTS` order: the motor's ``shafts``."""
    motor: dict = MISSING
    """The hand's motor (``linker_lab.assets.motor(side)``): its shafts and constants."""
    normalized: bool = False
    """Actions in [-1, 1] over the joint range instead of radians."""
