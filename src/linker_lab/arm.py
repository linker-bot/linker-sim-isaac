"""The AR5 arm: how it is commanded (``ArmAction``), its servo model and its motion limits.

How the real arm is commanded. The xCore SDK sends one position per 1 ms cycle to the
servo; smoothing happens before the servo, and the caller chooses who does it:

* **SDK interpolation on (PP tracking).** The caller sends goals; the SDK plans a
  jerk-limited trajectory (Ruckig, all joints arriving together) to the latest goal
  and streams it.
* **SDK interpolation off (CSP servo).** The caller streams the path itself and is
  responsible for its continuity.

``ArmAction`` models both. Each decision (60 Hz) it receives a command and turns it into
one reference (q_ref, v_ref) per physics step (480 Hz):

``interpolation=True`` (default)
    The command is a **goal**. :func:`plan` (a GPU trajectory generator) moves the reference
    toward the latest goal: time-optimal under the velocity, acceleration and jerk
    limits, all seven joints arriving together, replanned every decision from the
    reference's own position, velocity and acceleration.
``interpolation=False``
    The command is the **next point of the caller's own path** (``stream``):
    ``"points"`` (7 joint positions; the point's velocity is the finite difference to the
    previous point, so the path never stops at a point), ``"points_velocity"`` (positions
    and velocities, 14 values), or ``"samples"`` (the 8 per-step positions, then the 8
    per-step velocities: 112 values, subject to ``limits``). Consecutive points are joined by a
    cubic Hermite curve.

Commands are joint positions [rad] (``command="joint"``) or a pose of a frame on the arm
(``command="pose"``: position [m] and quaternion x, y, z, w in the arm's base frame),
converted to joint positions once per decision by Isaac Lab's damped-least-squares
``DifferentialIKController`` from the measured joint positions.

Servo model (:func:`feed_forward` per decision, :func:`torque_guard` per physics step). LW-v0's native implicit PD
(Kp 16000 N·m/rad, Kd 600-614 N·m·s/rad) tracks ``q_ref + Kd/Kp · v_ref`` (velocity
feed-forward). The real arm's torque caps (120 N·m J1-J4, 100 N·m J5-J7) are the drives'
effort limits in LW-v0's files; when the PD torque would exceed ``cap - 1 N·m``, the target
is moved so the torque equals that guard. Gains and caps are read from LW-v0's files.

Limits (``limits="real"``, the default; the xCore SDK motion profile of the AR5_08, read from
``arms/ar5_08/controller.yaml``): velocity :data:`VELOCITY`, acceleration :data:`ACCELERATION`,
jerk :data:`JERK`, the joint ranges in LW-v0's files, and the wrist octagon
:data:`WRIST_OCTAGON` for (J6, J7):

=====================  =================================  ==========================
                       interpolation=True                 interpolation=False
=====================  =================================  ==========================
position               goal and reference projected       point and reference
                       into the limits                    projected into the limits
velocity, acceleration respected by the generator         per-step reference rate-limited*
jerk                   respected by the generator         counted, not enforced
=====================  =================================  ==========================

At a position boundary, position safety takes priority over acceleration; the
reference stops on the boundary and its actual velocity/acceleration state is retained.
Supplied stream velocities are also clamped to the speed limits.

Interventions are counted per environment in ``env.extras["arm_limit_hits"][asset]``
(decisions in which a limit changed the command or one of its physics steps' references,
or the jerk was exceeded), so a policy relying on infeasible motion is visible. ``limits=None`` turns
all of this off; it needs ``interpolation=False``, as the generator *is* the limits.

Importing this module applies ``linker_lab.bootstrap()``, so import it before Isaac Lab.
"""

from __future__ import annotations

import linker_lab

linker_lab.bootstrap()

from dataclasses import MISSING

import torch
import warp as wp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from linker_lab import assets

__all__ = [
    "ArmAction",
    "ArmActionCfg",
    "VELOCITY",
    "ACCELERATION",
    "JERK",
    "WRIST_OCTAGON",
    "TORQUE_GUARD_MARGIN",
]

_CONTROLLER = assets.controller("left")  # controller.yaml has one set of limits for both arms

VELOCITY = _CONTROLLER["max_velocity"]
"""Maximum joint velocity [rad/s], J1..J7 (0.39, 0.32, 0.39, 0.39, 0.49, 0.49, 0.49)."""
ACCELERATION = _CONTROLLER["max_acceleration"]
"""Maximum joint acceleration [rad/s²], J1..J7 (10 each)."""
JERK = _CONTROLLER["max_jerk"]
"""Maximum joint jerk [rad/s³], J1..J7 (40 each)."""
WRIST_OCTAGON = _CONTROLLER["wrist_region"]
"""The region the real controller allows for (J6, J7) [rad]: a convex octagon, vertices in
order. The J6 and J7 ranges in LW-v0's files (±60°, ±90°) are its bounding box."""
TORQUE_GUARD_MARGIN = 1.0
"""The servo model keeps the PD torque this far [N·m] below each joint's cap."""


# ------------------------------------------------------------- trajectory generator
#
# One joint, state (p, v, a), goal g at rest. The time-optimal jerk-limited motion to g
# has 7 phases of constant jerk: change the velocity to a peak vp (jerk +-J, 0, -+J;
# acceleration limited to A), cruise at vp, change the velocity back to 0. Its travel
# grows with vp, so vp is found by bisection: vp = +-V with a cruise if even the fastest
# peak does not overshoot, else the vp whose travel without cruise is exactly g - p.
# The joints are then synchronized: every joint is slowed to the slowest one's
# duration T by lowering its peak velocity (and cruising longer) until it takes T.

F64 = wp.float64
_vec7 = wp.types.vector(length=7, dtype=wp.float64)
_BISECTIONS = 50  # vp to 2^-50 of its range


@wp.struct
class _Profile:
    duration: _vec7  # [s] of the 7 phases
    jerk: _vec7  # [rad/s³] in each phase


@wp.func
def _advance(s: wp.vec3d, jerk: F64, t: F64) -> wp.vec3d:
    """State (p, v, a) after ``t`` seconds at constant ``jerk``."""
    return wp.vec3d(
        s[0] + t * (s[1] + t * (s[2] * F64(0.5) + t * jerk / F64(6.0))),
        s[1] + t * (s[2] + t * jerk * F64(0.5)),
        s[2] + t * jerk,
    )


@wp.func
def _velocity_change(v0: F64, a0: F64, vt: F64, amax: F64, jmax: F64) -> wp.vec4d:
    """Fastest change from (v0, a0) to velocity vt with zero acceleration.

    Jerk ``s·jmax`` for t1, 0 for t2, ``-s·jmax`` for t3; returns (t1, t2, t3, s).
    """
    zero = F64(0.0)
    s = F64(1.0)
    if vt < v0 + a0 * wp.abs(a0) / (F64(2.0) * jmax):  # velocity once a0 is ramped to 0
        s = F64(-1.0)
    a0s = s * a0
    dv = s * (vt - v0)
    peak = wp.sqrt(wp.max(jmax * dv + F64(0.5) * a0s * a0s, zero))
    t2 = zero
    if peak > amax:
        peak = amax
        t2 = wp.max((dv - (F64(2.0) * amax * amax - a0s * a0s) / (F64(2.0) * jmax)) / amax, zero)
    return wp.vec4d(wp.max((peak - a0s) / jmax, zero), t2, peak / jmax, s)


@wp.func
def _profile(v0: F64, a0: F64, vp: F64, cruise: F64, amax: F64, jmax: F64) -> _Profile:
    up = _velocity_change(v0, a0, vp, amax, jmax)
    down = _velocity_change(vp, F64(0.0), F64(0.0), amax, jmax)
    out = _Profile()
    out.duration = _vec7(up[0], up[1], up[2], cruise, down[0], down[1], down[2])
    zero = F64(0.0)
    out.jerk = _vec7(up[3] * jmax, zero, -up[3] * jmax, zero, down[3] * jmax, zero, -down[3] * jmax)
    return out


@wp.func
def _state_at(s0: wp.vec3d, profile: _Profile, t: F64) -> wp.vec3d:
    s = s0
    remaining = t
    for i in range(7):
        s = _advance(s, profile.jerk[i], wp.min(profile.duration[i], wp.max(remaining, F64(0.0))))
        remaining = remaining - profile.duration[i]
    return s


@wp.func
def _duration(profile: _Profile) -> F64:
    total = F64(0.0)
    for i in range(7):
        total = total + profile.duration[i]
    return total


@wp.func
def _travel(v0: F64, a0: F64, vp: F64, amax: F64, jmax: F64) -> wp.vec2d:
    """Distance and time to change the velocity to vp and back to rest, without cruise."""
    profile = _profile(v0, a0, vp, F64(0.0), amax, jmax)
    took = _duration(profile)
    return wp.vec2d(_state_at(wp.vec3d(F64(0.0), v0, a0), profile, took)[0], took)


@wp.func
def _fastest(e: F64, v0: F64, a0: F64, vmax: F64, amax: F64, jmax: F64) -> wp.vec3d:
    """Peak velocity, cruise time and duration of the time-optimal motion over distance e."""
    zero = F64(0.0)
    result = wp.vec3d(zero, zero, zero)  # at rest at the goal: nothing to do
    if e != zero or v0 != zero or a0 != zero:
        up = _travel(v0, a0, vmax, amax, jmax)
        down = _travel(v0, a0, -vmax, amax, jmax)
        if e >= up[0]:
            result = wp.vec3d(vmax, (e - up[0]) / vmax, up[1] + (e - up[0]) / vmax)
        elif e <= down[0]:
            result = wp.vec3d(-vmax, (e - down[0]) / (-vmax), down[1] + (e - down[0]) / (-vmax))
        else:
            lo = -vmax
            hi = vmax
            for _ in range(_BISECTIONS):
                mid = F64(0.5) * (lo + hi)
                if _travel(v0, a0, mid, amax, jmax)[0] < e:
                    lo = mid
                else:
                    hi = mid
            vp = F64(0.5) * (lo + hi)
            result = wp.vec3d(vp, zero, _travel(v0, a0, vp, amax, jmax)[1])
    return result


@wp.func
def _within(T: F64, e: F64, v0: F64, a0: F64, fastest_peak: F64, amax: F64, jmax: F64) -> wp.vec2d:
    """Peak velocity and cruise time of a motion over distance e that takes T (at least as
    long as the fastest one): the same shape with a lower peak velocity."""
    zero = F64(0.0)
    stop = _travel(v0, a0, zero, amax, jmax)
    rest = e - stop[0]  # distance left after stopping
    result = wp.vec2d(zero, wp.max(T - stop[1], zero))  # stopping arrives: stop, then wait
    if wp.abs(rest) > F64(1.0e-12):
        sign = wp.sign(rest)
        lo = zero
        hi = wp.abs(fastest_peak)
        for _ in range(_BISECTIONS):
            mid = F64(0.5) * (lo + hi)
            travel = _travel(v0, a0, sign * mid, amax, jmax)
            if travel[1] + (e - travel[0]) / (sign * mid) > T:  # too slow
                lo = mid
            else:
                hi = mid
        vp = sign * hi
        result = wp.vec2d(vp, wp.max((e - _travel(v0, a0, vp, amax, jmax)[0]) / vp, zero))
    return result


@wp.kernel
def _fastest_kernel(
    q: wp.array2d(dtype=F64),
    v: wp.array2d(dtype=F64),
    a: wp.array2d(dtype=F64),
    goal: wp.array2d(dtype=F64),
    vmax: wp.array(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    out: wp.array3d(dtype=F64),
):
    n, j = wp.tid()
    f = _fastest(goal[n, j] - q[n, j], v[n, j], a[n, j], vmax[j], amax[j], jmax[j])
    out[n, j, 0] = f[0]
    out[n, j, 1] = f[1]
    out[n, j, 2] = f[2]


@wp.kernel
def _sample_kernel(
    q: wp.array2d(dtype=F64),
    v: wp.array2d(dtype=F64),
    a: wp.array2d(dtype=F64),
    goal: wp.array2d(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    fastest: wp.array3d(dtype=F64),
    slowest: wp.array(dtype=F64),
    dt: F64,
    out_q: wp.array3d(dtype=F64),
    out_v: wp.array3d(dtype=F64),
    out_a: wp.array3d(dtype=F64),
):
    n, j = wp.tid()
    f = wp.vec2d(fastest[n, j, 0], fastest[n, j, 1])
    if fastest[n, j, 2] < slowest[n] - F64(1.0e-9):  # slow down to arrive with the slowest joint
        f = _within(slowest[n], goal[n, j] - q[n, j], v[n, j], a[n, j], f[0], amax[j], jmax[j])
    profile = _profile(v[n, j], a[n, j], f[0], f[1], amax[j], jmax[j])
    total = _duration(profile)
    s0 = wp.vec3d(q[n, j], v[n, j], a[n, j])
    for k in range(out_q.shape[1]):
        t = F64(k + 1) * dt
        s = wp.vec3d(goal[n, j], F64(0.0), F64(0.0))  # arrived
        if t < total:
            s = _state_at(s0, profile, t)
        out_q[n, k, j] = s[0]
        out_v[n, k, j] = s[1]
        out_a[n, k, j] = s[2]


def plan(q, v, a, goal, vmax, amax, jmax, dt: float, steps: int):
    """Sample the trajectory from state (q, v, a) to ``goal`` at t = dt, 2·dt, .., steps·dt.

    All inputs are float64 torch tensors on one device: states and goal (N, 7), limits (7,).
    The state must be within the limits and able to stop within them (any state on a
    previous plan is). Returns positions, velocities and accelerations (N, steps, 7):
    time-optimal under the limits with every joint arriving at its goal together, at
    rest from then on.
    """
    wp.init()  # no-op once Warp is initialized
    n = q.shape[0]
    device = wp.device_from_torch(q.device)
    stream = wp.stream_from_torch(q.device) if device.is_cuda else None
    state = [wp.from_torch(x.contiguous(), dtype=wp.float64) for x in (q, v, a, goal)]
    limits = [wp.from_torch(x.contiguous(), dtype=wp.float64) for x in (vmax, amax, jmax)]
    fastest = torch.empty((n, 7, 3), device=q.device, dtype=torch.float64)  # peak, cruise, duration
    wp.launch(_fastest_kernel, dim=(n, 7), inputs=[*state, *limits], outputs=[wp.from_torch(fastest)],
              device=device, stream=stream)
    slowest = fastest[:, :, 2].amax(1).contiguous()
    out = [torch.empty((n, steps, 7), device=q.device, dtype=torch.float64) for _ in range(3)]
    wp.launch(
        _sample_kernel,
        dim=(n, 7),
        inputs=[*state, *limits[1:], wp.from_torch(fastest), wp.from_torch(slowest), float(dt)],
        outputs=[wp.from_torch(x) for x in out],
        device=device,
        stream=stream,
    )
    return tuple(out)


# ------------------------------------------------------------- servo, stream, limits


def feed_forward(q_ref, v_ref, kp, kd):
    """The servo's position target before the torque guard: ``q_ref + kd/kp · v_ref``.

    Computed in float64 and rounded to the engine's float32; inputs broadcast.
    """
    return (q_ref.double() + kd.double() * v_ref.double() / kp.double()).float()


@wp.kernel
def _guard_kernel(
    requested: wp.array2d(dtype=wp.float32),
    q: wp.array2d(dtype=wp.float32),
    qd: wp.array2d(dtype=wp.float32),
    kp: wp.array2d(dtype=wp.float32),
    kd: wp.array2d(dtype=wp.float32),
    cap: wp.array2d(dtype=wp.float32),
    joints: wp.array(dtype=wp.int32),
    margin: wp.float32,
    out: wp.array2d(dtype=wp.float32),
):
    n, i = wp.tid()
    j = joints[i]
    guard = cap[n, j] - margin
    target = requested[n, i]
    torque = kp[n, j] * (target - q[n, j]) - kd[n, j] * qd[n, j]
    if wp.abs(torque) > guard:  # the target that gives exactly the guard torque
        target = q[n, j] + (kd[n, j] * qd[n, j] + wp.clamp(torque, -guard, guard)) / kp[n, j]
    out[n, i] = target


def torque_guard(requested, q, qd, kp, kd, cap, joints, out=None, margin: float = TORQUE_GUARD_MARGIN):
    """Position targets (N, K) for one physics step: ``requested`` (N, K), unless the PD torque
    ``kp·(requested − q) − kd·qd`` would exceed ``cap − margin``; then the target that gives
    exactly that torque.

    ``q``, ``qd``, ``kp``, ``kd``, ``cap`` are the articulation's float32 joint arrays (N, J)
    (Warp arrays or torch tensors); ``joints`` (K, int32) selects the driven columns: the
    arm's seven, or a hand's six (``linker_lab.hand.HandAction``).
    """
    wp.init()  # no-op once Warp is initialized
    out = torch.empty_like(requested) if out is None else out
    device = wp.device_from_torch(requested.device)
    arrays = [x if isinstance(x, wp.array) else wp.from_torch(x) for x in (requested, q, qd, kp, kd, cap, joints)]
    wp.launch(_guard_kernel, dim=requested.shape, inputs=[*arrays, float(margin)], outputs=[wp.from_torch(out)],
              device=device, stream=wp.stream_from_torch(requested.device) if device.is_cuda else None)
    return out


def hermite(q0, q1, v0, v1, u, duration: float):
    """Cubic Hermite curve from (q0, v0) to (q1, v1) over ``duration``, at fractions ``u``.

    ``q0``..``v1`` are (N, 7), ``u`` is (S, 1); returns positions and velocities (N, S, 7).
    """
    q0, q1, v0, v1 = (x[:, None, :] for x in (q0, q1, v0, v1))
    q = (
        (2 * u**3 - 3 * u**2 + 1) * q0
        + (u**3 - 2 * u**2 + u) * duration * v0
        + (-2 * u**3 + 3 * u**2) * q1
        + (u**3 - u**2) * duration * v1
    )
    v = (
        (6 * u**2 - 6 * u) * q0
        + (3 * u**2 - 4 * u + 1) * duration * v0
        + (-6 * u**2 + 6 * u) * q1
        + (3 * u**2 - 2 * u) * duration * v1
    ) / duration
    return q, v


@wp.kernel
def _project_kernel(
    q: wp.array2d(dtype=F64),
    lower: wp.array(dtype=F64),
    upper: wp.array(dtype=F64),
    corners: wp.array2d(dtype=F64),
    hit: wp.array(dtype=wp.int32),
):
    m = wp.tid()
    changed = wp.int32(0)
    for j in range(7):
        clamped = wp.clamp(q[m, j], lower[j], upper[j])
        if clamped != q[m, j]:
            changed = 1
            q[m, j] = clamped
    # wrist (J6, J7): inside the octagon, or to the nearest point on its boundary
    p = wp.vec2d(q[m, 5], q[m, 6])
    inside = True
    nearest = p
    best = F64(1.0e30)
    for i in range(8):
        a = wp.vec2d(corners[i, 0], corners[i, 1])
        edge = wp.vec2d(corners[(i + 1) % 8, 0], corners[(i + 1) % 8, 1]) - a
        r = p - a
        if edge[0] * r[1] - edge[1] * r[0] > F64(0.0):  # left of a clockwise edge: outside
            inside = False
        c = a + wp.clamp(wp.dot(r, edge) / wp.dot(edge, edge), F64(0.0), F64(1.0)) * edge
        if wp.length_sq(p - c) < best:
            best = wp.length_sq(p - c)
            nearest = c
    if not inside:
        changed = 1
        q[m, 5] = nearest[0]
        q[m, 6] = nearest[1]
    hit[m] = changed


def project(q, lower, upper):
    """Joint positions (..., 7) moved into the joint ranges ``lower``, ``upper`` (7,) and the
    wrist octagon. Returns the positions (float64) and where they changed (bool, q.shape[:-1])."""
    wp.init()  # no-op once Warp is initialized
    out = q.double().reshape(-1, 7).clone()
    hit = torch.empty(out.shape[0], device=q.device, dtype=torch.int32)
    corners = torch.tensor(WRIST_OCTAGON, device=q.device, dtype=torch.float64)
    device = wp.device_from_torch(q.device)
    wp.launch(_project_kernel, dim=out.shape[0],
              inputs=[wp.from_torch(out)] + [wp.from_torch(x.double().contiguous()) for x in (lower, upper, corners)],
              outputs=[wp.from_torch(hit)], device=device,
              stream=wp.stream_from_torch(q.device) if device.is_cuda else None)
    return out.reshape(q.shape), hit.reshape(q.shape[:-1]).bool()


@wp.kernel
def _rate_limit_kernel(
    q: wp.array3d(dtype=F64),
    v: wp.array3d(dtype=F64),
    last: wp.array3d(dtype=F64),
    vmax: wp.array(dtype=F64),
    amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64),
    dt: F64,
    lower: wp.array(dtype=F64),
    upper: wp.array(dtype=F64),
    corners: wp.array2d(dtype=F64),
    position_limits: wp.int32,
    hit: wp.array(dtype=wp.int32),
):
    n = wp.tid()
    previous = _vec7()
    previous_v = _vec7()
    previous_a = _vec7()
    for j in range(7):
        previous[j] = last[n, j, 0]
        previous_v[j] = last[n, j, 1]
        previous_a[j] = last[n, j, 2]
    changed = wp.int32(0)
    for k in range(q.shape[1]):
        candidate = _vec7()
        velocity = _vec7()
        modified = _vec7()
        for j in range(7):
            wanted = (q[n, k, j] - previous[j]) / dt
            speed = wp.clamp(
                wp.clamp(wanted, -vmax[j], vmax[j]),
                previous_v[j] - amax[j] * dt, previous_v[j] + amax[j] * dt,
            )
            candidate[j] = q[n, k, j]
            velocity[j] = wp.clamp(v[n, k, j], -vmax[j], vmax[j])
            if velocity[j] != v[n, k, j]:
                changed = 1
            if speed != wanted:
                candidate[j] = previous[j] + speed * dt
                modified[j] = F64(1.0)
                changed = 1
        # Restrict the whole step to the convex feasible region. Unlike projecting a
        # completed batch, this keeps the next sample's velocity state consistent.
        fraction = F64(1.0)
        if position_limits != 0:
            for j in range(7):
                delta = candidate[j] - previous[j]
                if candidate[j] > upper[j] and delta > F64(0.0):
                    fraction = wp.min(fraction, (upper[j] - previous[j]) / delta)
                if candidate[j] < lower[j] and delta < F64(0.0):
                    fraction = wp.min(fraction, (lower[j] - previous[j]) / delta)
            for edge in range(8):
                ax = corners[edge, 0]
                ay = corners[edge, 1]
                ex = corners[(edge + 1) % 8, 0] - ax
                ey = corners[(edge + 1) % 8, 1] - ay
                before = ex * (previous[6] - ay) - ey * (previous[5] - ax)
                after = ex * (candidate[6] - ay) - ey * (candidate[5] - ax)
                if after > F64(0.0) and after > before:
                    fraction = wp.min(fraction, -before / (after - before))
            fraction = wp.clamp(fraction, F64(0.0), F64(1.0))
        if fraction < F64(1.0):
            changed = 1
        for j in range(7):
            final = candidate[j]
            if fraction < F64(1.0):
                final = previous[j] + fraction * (candidate[j] - previous[j])
            speed = (final - previous[j]) / dt
            acceleration = (speed - previous_v[j]) / dt
            if modified[j] != F64(0.0) or fraction < F64(1.0):
                velocity[j] = speed
            if wp.abs(acceleration) > amax[j] * F64(1.000000001):
                changed = 1  # a boundary stop takes priority over acceleration
            if wp.abs(acceleration - previous_a[j]) > jmax[j] * dt * F64(1.000000001):
                changed = 1
            q[n, k, j] = final
            v[n, k, j] = velocity[j]
            previous[j] = final
            previous_v[j] = speed
            previous_a[j] = acceleration
    for j in range(7):
        last[n, j, 0] = previous[j]
        last[n, j, 1] = previous_v[j]
        last[n, j, 2] = previous_a[j]
    hit[n] = changed


def rate_limit(q, v, last, vmax, amax, jmax, dt: float, lower=None, upper=None):
    """Limit stream references in place, including supplied feed-forward speeds.

    Per-step position differences obey velocity and acceleration limits. Supplied
    velocities are bounded by ``vmax``; when a position is changed, its velocity is
    replaced by the actual finite difference. Optional ``lower``/``upper`` also enforce
    joint ranges and the wrist octagon (the previous position must be feasible).
    At a boundary, position safety takes priority over acceleration and jerk: the step
    is shortened, the intervention counted, and ``last`` records the resulting position,
    finite-difference velocity and acceleration, not the rejected candidate.
    Returns where a limit acted (bool, N). Jerk violations are counted, not enforced.
    """
    wp.init()
    if (lower is None) != (upper is None):
        raise ValueError("give both lower and upper, or neither")
    hit = torch.empty(q.shape[0], device=q.device, dtype=torch.int32)
    device = wp.device_from_torch(q.device)
    bounded = lower is not None
    if not bounded:
        lower, upper = torch.zeros_like(vmax), torch.zeros_like(vmax)
    corners = torch.tensor(WRIST_OCTAGON, device=q.device, dtype=torch.float64)
    wp.launch(
        _rate_limit_kernel, dim=q.shape[0],
        inputs=[wp.from_torch(x) for x in (q, v, last, vmax, amax, jmax)]
        + [float(dt), wp.from_torch(lower), wp.from_torch(upper), wp.from_torch(corners), int(bounded)],
        outputs=[wp.from_torch(hit)], device=device,
        stream=wp.stream_from_torch(q.device) if device.is_cuda else None,
    )
    return hit.bool()


# ------------------------------------------------------------- action term


class ArmAction(ActionTerm):
    """Joint or pose commands for one AR5 arm, through the trajectory generator or the
    Hermite stream, into the servo model. See :mod:`linker_lab.arm`."""

    cfg: ArmActionCfg

    def __init__(self, cfg: ArmActionCfg, env):
        super().__init__(cfg, env)
        _check(cfg)
        ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(ids) != 7:
            raise ValueError(f"ArmAction needs the 7 arm joints, got {names}")
        self._joint_ids = torch.tensor(ids, device=self.device)
        self._steps = env.cfg.decimation
        self._dt = env.physics_dt
        self._decision_dt = env.step_dt
        f64 = dict(device=self.device, dtype=torch.float64)
        limits = self._asset.data.joint_pos_limits.torch[0, self._joint_ids].double()
        self._lower, self._upper = limits[:, 0].clone(), limits[:, 1].clone()
        self._vmax = torch.tensor(VELOCITY, **f64)
        self._amax = torch.tensor(ACCELERATION, **f64)
        self._jmax = torch.tensor(JERK, **f64)
        self._u = torch.arange(1, self._steps + 1, **f64)[:, None] / self._steps

        n, s = self.num_envs, self._steps
        self._ref_q = torch.zeros((n, s, 7), **f64)  # per-step references of this decision
        self._ref_v = torch.zeros((n, s, 7), **f64)
        self._requested = torch.zeros((s, n, 7), device=self.device)  # their servo targets
        self._target = torch.zeros((n, 7), device=self.device)
        self._joints = self._joint_ids.int()
        self._step = 0
        self._plan = [torch.zeros((n, 7), **f64) for _ in range(3)]  # generator state q, v, a
        self._point = [torch.zeros((n, 7), **f64) for _ in range(2)]  # last stream point q, v
        self._last = torch.zeros((n, 7, 3), **f64)  # last reference: q, velocity, acceleration
        self.limit_hits = torch.zeros(n, device=self.device, dtype=torch.long)
        self._raw = torch.zeros((n, self.action_dim), device=self.device)

        if cfg.command == "pose":
            from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

            body_ids, _ = self._asset.find_bodies(cfg.body_name)
            if len(body_ids) != 1:
                raise ValueError(f"body {cfg.body_name!r} not found exactly once")
            self._body = body_ids[0]
            # Isaac Lab leaves the fixed root out of the Jacobian
            self._jacobian_body = self._body - 1 if self._asset.is_fixed_base else self._body
            self._offset = torch.tensor(cfg.body_offset, device=self.device).expand(n, 3)
            self._ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose", ik_method="dls", ik_params={"lambda_val": cfg.ik_damping}
                ),
                num_envs=n,
                device=self.device,
            )
        self.reset()

    # -- Isaac Lab interface

    @property
    def action_dim(self) -> int:
        if self.cfg.stream == "points_velocity":
            return 14
        if self.cfg.stream == "samples":
            return 2 * self._steps * 7
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        """The last command; non-finite values are replaced by the previous command's."""
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The reference (q_ref, v_ref) of the current physics step, (N, 7) each."""
        k = min(max(self._step - 1, 0), self._steps - 1)
        return self._ref_q[:, k], self._ref_v[:, k]

    def process_actions(self, actions: torch.Tensor):
        self._raw = torch.where(torch.isfinite(actions), actions, self._raw)
        limits = self.cfg.limits is not None
        hits = torch.zeros_like(self.limit_hits, dtype=torch.bool)

        if self.cfg.stream == "samples":
            s = self._steps
            ref_q = self._raw[:, : s * 7].view(-1, s, 7).double()
            ref_v = self._raw[:, s * 7 :].view(-1, s, 7).double()
        else:
            point = self._joint_command(self._raw)
            if limits:
                point, changed = project(point, self._lower, self._upper)
                hits |= changed
            if self.cfg.interpolation:
                ref_q, ref_v, ref_a = plan(
                    *self._plan, point, self._vmax, self._amax, self._jmax, self._dt, self._steps
                )
                self._plan = [x[:, -1].clone() for x in (ref_q, ref_v, ref_a)]
            else:
                q0, v0 = self._point
                if self.cfg.stream == "points_velocity":
                    v1 = self._raw[:, 7:].double()
                else:  # the finite difference, so the path does not stop at the point
                    v1 = (point - q0) / self._decision_dt
                ref_q, ref_v = hermite(q0, point, v0, v1, self._u, self._decision_dt)
                self._point = [point, v1]

        if limits:
            ref_q, changed = project(ref_q, self._lower, self._upper)
            hits |= changed.any(-1)
            if not self.cfg.interpolation:
                ref_v = ref_v.contiguous()
                hits |= rate_limit(
                    ref_q, ref_v, self._last, self._vmax, self._amax, self._jmax, self._dt,
                    self._lower, self._upper,
                )
        self._ref_q, self._ref_v = ref_q, ref_v
        data, ids = self._asset.data, self._joint_ids
        kp, kd = data.joint_stiffness.torch[:, ids], data.joint_damping.torch[:, ids]
        self._requested = feed_forward(ref_q, ref_v, kp[:, None], kd[:, None]).transpose(0, 1).contiguous()
        self._step = 0
        self.limit_hits += hits
        self._env.extras.setdefault("arm_limit_hits", {})[self.cfg.asset_name] = self.limit_hits

    def apply_actions(self):
        k = min(self._step, self._steps - 1)
        self._step += 1
        data = self._asset.data
        torque_guard(
            self._requested[k], data.joint_pos.warp, data.joint_vel.warp, data.joint_stiffness.warp,
            data.joint_damping.warp, data.joint_effort_limits.warp, self._joints, out=self._target,
        )
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos.torch[:, self._joint_ids].double()[rows]
        if self.cfg.limits is not None:
            q, _ = project(q, self._lower, self._upper)
        for state in (self._plan, self._point):
            state[0][rows] = q
            for derivative in state[1:]:
                derivative[rows] = 0
        self._last[rows] = 0
        self._last[rows, :, 0] = q
        self._ref_q[rows] = q[:, None, :]
        self._ref_v[rows] = 0
        self._requested[:, rows] = q.float()
        self.limit_hits[rows] = 0
        if self.cfg.command == "pose":
            self._ik.reset()
            pos, quat = self._frame_pose()
            self._raw[rows] = torch.cat((pos, quat), -1)[rows]
        elif self.cfg.stream == "samples":
            self._raw[rows] = torch.cat(
                (q[:, None, :].expand(-1, self._steps, 7).flatten(1), torch.zeros_like(q).repeat(1, self._steps)), -1
            ).float()
        else:
            self._raw[rows] = 0
            self._raw[rows, :7] = q.float()

    # -- state, for linker_lab.snapshot

    def get_state(self) -> dict:
        """Everything this term carries from one decision to the next (copies): the generator's
        and the stream's state, this decision's references and servo targets, the last command,
        the limit counter and, with pose commands, the IK controller's command."""
        state = {
            "plan_q": self._plan[0], "plan_v": self._plan[1], "plan_a": self._plan[2],
            "point_q": self._point[0], "point_v": self._point[1], "last": self._last,
            "ref_q": self._ref_q, "ref_v": self._ref_v, "requested": self._requested,
            "target": self._target, "raw": self._raw, "limit_hits": self.limit_hits,
        }
        if self.cfg.command == "pose":
            state |= {"ik_command": self._ik._command, "ik_pos": self._ik.ee_pos_des, "ik_quat": self._ik.ee_quat_des}
        return {**{k: v.clone() for k, v in state.items()}, "step": self._step}

    def set_state(self, state: dict) -> None:
        """Restore what :meth:`get_state` returned, copied into this term's own tensors. Raises
        ``ValueError`` before changing anything if the keys, shapes or dtypes differ."""
        check_state(self.get_state(), state, type(self).__name__)
        targets = {
            "plan_q": self._plan[0], "plan_v": self._plan[1], "plan_a": self._plan[2],
            "point_q": self._point[0], "point_v": self._point[1], "last": self._last,
            "ref_q": self._ref_q, "ref_v": self._ref_v, "requested": self._requested,
            "target": self._target, "raw": self._raw, "limit_hits": self.limit_hits,
        }
        if self.cfg.command == "pose":
            targets |= {"ik_command": self._ik._command, "ik_pos": self._ik.ee_pos_des, "ik_quat": self._ik.ee_quat_des}
        for key, tensor in targets.items():
            tensor.copy_(state[key])
        self._step = int(state["step"])

    # -- helpers

    def _joint_command(self, raw):
        """Joint positions (N, 7, float64) the command asks for."""
        if self.cfg.command == "joint":
            return raw[:, :7].double()
        pos, quat = self._frame_pose()
        self._ik.set_command(raw, pos, quat)
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        return self._ik.compute(pos, quat, self._frame_jacobian(), q).double()

    def _frame_pose(self):
        """Pose of the commanded frame in the arm's base (root) frame."""
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        data = self._asset.data
        root = data.root_link_pose_w.torch
        body = data.body_link_pose_w.torch[:, self._body]
        pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], body[:, :3], body[:, 3:7])
        identity = torch.zeros_like(quat)
        identity[:, 3] = 1
        return combine_frame_transforms(pos, quat, self._offset, identity)

    def _frame_jacobian(self):
        """Geometric Jacobian (N, 6, 7) of the commanded frame in the arm's base frame."""
        from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul

        data = self._asset.data
        jacobian = data.body_link_jacobian_w.torch[:, self._jacobian_body][:, :, self._joint_ids]
        root_inv = quat_inv(data.root_link_pose_w.torch[:, 3:7])
        rotation = matrix_from_quat(root_inv)
        linear, angular = rotation @ jacobian[:, :3], rotation @ jacobian[:, 3:]
        body_quat = quat_mul(root_inv, data.body_link_pose_w.torch[:, self._body, 3:7])
        lever = quat_apply(body_quat, self._offset)  # offset in base axes
        linear = linear + torch.linalg.cross(angular, lever[:, :, None].expand_as(angular), dim=1)
        return torch.cat((linear, angular), 1)


def _check(cfg: ArmActionCfg):
    if cfg.command not in ("joint", "pose"):
        raise ValueError(f"command must be 'joint' or 'pose', not {cfg.command!r}")
    if cfg.stream not in ("points", "points_velocity", "samples"):
        raise ValueError(f"stream must be 'points', 'points_velocity' or 'samples', not {cfg.stream!r}")
    if cfg.limits not in ("real", None):
        raise ValueError(f"limits must be 'real' or None, not {cfg.limits!r}")
    if cfg.interpolation and cfg.stream != "points":
        raise ValueError("with interpolation=True the command is a goal: stream must be 'points'")
    if cfg.interpolation and cfg.limits is None:
        raise ValueError("interpolation=True needs limits='real' (the generator moves within them)")
    if cfg.command == "pose" and cfg.stream != "points":
        raise ValueError("pose commands are points; velocities and samples need joint commands")
    if cfg.command == "pose" and cfg.body_name is None:
        raise ValueError("pose commands need body_name")


def check_state(current: dict, state: dict, owner: str) -> None:
    """Raise ``ValueError`` unless ``state`` has exactly ``current``'s keys, and each tensor its
    shape and dtype (each other value its type)."""
    if set(state) != set(current):
        raise ValueError(f"{owner} state: keys {sorted(set(state) ^ set(current))} differ")
    for key, value in current.items():
        new = state[key]
        if isinstance(value, torch.Tensor):
            if not isinstance(new, torch.Tensor) or new.shape != value.shape or new.dtype != value.dtype:
                got = (tuple(new.shape), new.dtype) if isinstance(new, torch.Tensor) else type(new).__name__
                raise ValueError(f"{owner} state {key!r}: expected {(tuple(value.shape), value.dtype)}, got {got}")
        elif type(new) is not type(value):
            raise ValueError(f"{owner} state {key!r}: expected {type(value).__name__}, got {type(new).__name__}")


@configclass
class ArmActionCfg(ActionTermCfg):
    """:class:`ArmAction`. ``linker_lab.workstation.arm_action_cfg`` fills in the names."""

    class_type: type = ArmAction
    joint_names: list[str] = MISSING
    """The arm joints J1..J7."""
    command: str = "joint"
    """``"joint"``: 7 joint positions [rad]. ``"pose"``: position [m] and quaternion (x, y, z,
    w) of the frame ``body_name`` + ``body_offset`` in the arm's base frame."""
    interpolation: bool = True
    """True: the command is a goal for the trajectory generator (SDK PP tracking). False: it
    is the next point of the caller's own path (SDK CSP servo)."""
    stream: str = "points"
    """With ``interpolation=False``: ``"points"``, ``"points_velocity"`` or ``"samples"``."""
    limits: str | None = "real"
    """``"real"``: the real controller's limits. None: no limits (``interpolation=False`` only)."""
    body_name: str | None = None
    """Pose commands: the body whose frame is commanded."""
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Pose commands: the commanded point in that body's frame [m]."""
    ik_damping: float = 0.01
    """Damping of the least-squares IK step."""
