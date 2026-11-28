"""Accepted arm trajectories, with position bounds checked over the complete path.

Every phase is cubic in time. Joint extrema are its stationary points; wrist
half-planes are cubics on the union of the two joints' phase boundaries. Checking
those extrema certifies the whole path, rather than just the sampled positions.

An infeasible proposal from rest uses a shared scalar progress along the straight
line to the goal. The joint box and wrist octagon are convex, so that fallback
stays feasible. While moving, an infeasible retarget leaves the previously checked
path in effect. The latest goal is tried again on the next decision.
"""

from __future__ import annotations

import torch
import warp as wp

from linker_lab.arm import (
    F64,
    _advance,
    _duration,
    _fastest,
    _Profile,
    _profile,
    _state_at,
    _within,
)

# These controllers are not differentiated. Keep the bounded phase searches as
# loops: unrolling their nested checks makes compilation needlessly large.
wp.set_module_options({"enable_backward": False, "max_unroll": 1})

_Mat7 = wp.types.matrix(shape=(7, 7), dtype=F64)
_Mat73 = wp.types.matrix(shape=(7, 3), dtype=F64)
_Vec7 = wp.types.vector(length=7, dtype=F64)
_Vec15 = wp.types.vector(length=15, dtype=F64)
_BOUND_TOLERANCE = 1.0e-12


@wp.struct
class _Path:
    duration: _Mat7
    jerk: _Mat7


@wp.func
def _joint(path: _Path, j: int) -> _Profile:
    result = _Profile()
    result.duration = path.duration[j]
    result.jerk = path.jerk[j]
    return result


@wp.func
def _range(s: wp.vec3d, jerk: F64, duration: F64) -> wp.vec2d:
    """Exact position range of one constant-jerk phase, up to floating-point error."""
    end = _advance(s, jerk, duration)[0]
    low, high = wp.min(s[0], end), wp.max(s[0], end)
    if jerk == F64(0.0):
        if s[2] != F64(0.0):
            t = -s[1] / s[2]
            if t > F64(0.0) and t < duration:
                value = _advance(s, jerk, t)[0]
                low, high = wp.min(low, value), wp.max(high, value)
    else:
        discriminant = s[2] * s[2] - F64(2.0) * jerk * s[1]
        if discriminant >= F64(0.0):
            root = wp.sqrt(discriminant)
            for sign in range(2):
                numerator = -s[2] - root
                if sign == 1:
                    numerator = -s[2] + root
                t = numerator / jerk
                if t > F64(0.0) and t < duration:
                    value = _advance(s, jerk, t)[0]
                    low, high = wp.min(low, value), wp.max(high, value)
    return wp.vec2d(low, high)


@wp.func
def _jerk_at(profile: _Profile, t: F64) -> F64:
    end = F64(0.0)
    result = F64(0.0)
    for phase in range(7):
        begin = end
        end = end + profile.duration[phase]
        if t >= begin and t < end:
            result = profile.jerk[phase]
    return result


@wp.func
def _feasible(
    start: _Mat73, goal: _Vec7, path: _Path, vmax: wp.array(dtype=F64),
    amax: wp.array(dtype=F64), jmax: wp.array(dtype=F64), lower: wp.array(dtype=F64),
    upper: wp.array(dtype=F64), corners: wp.array2d(dtype=F64),
) -> bool:
    feasible = bool(True)
    tolerance = F64(_BOUND_TOLERANCE)
    for j in range(7):
        s = wp.vec3d(start[j, 0], start[j, 1], start[j, 2])
        for phase in range(7):
            jerk, duration = path.jerk[j, phase], path.duration[j, phase]
            bounds = _range(s, jerk, duration)
            if bounds[0] < lower[j] - tolerance or bounds[1] > upper[j] + tolerance:
                feasible = False
            speeds = _range(wp.vec3d(s[1], s[2], jerk), F64(0.0), duration)
            if speeds[0] < -vmax[j] - tolerance or speeds[1] > vmax[j] + tolerance:
                feasible = False
            phase_end = _advance(s, jerk, duration)
            if wp.max(wp.abs(s[2]), wp.abs(phase_end[2])) > amax[j] + tolerance:
                feasible = False
            if duration > F64(0.0) and wp.abs(jerk) > jmax[j] + tolerance:
                feasible = False
            s = phase_end
        # Sampling after a path ends holds the goal at rest. Reject a profile
        # that would need a discontinuous correction at that transition.
        if wp.abs(s[0] - goal[j]) > tolerance or wp.abs(s[1]) > tolerance or wp.abs(s[2]) > tolerance:
            feasible = False
    # There are at most fourteen phase ends across J6 and J7. Zero-length phases
    # simply give empty intervals after sorting.
    times = _Vec15()
    for j in range(2):
        total = F64(0.0)
        for phase in range(7):
            total = total + path.duration[j + 5, phase]
            times[1 + j * 7 + phase] = total
    for i in range(15):
        for j in range(14):
            if times[j] > times[j + 1]:
                saved = times[j]
                times[j] = times[j + 1]
                times[j + 1] = saved
    p6, p7 = _joint(path, 5), _joint(path, 6)
    initial6 = wp.vec3d(start[5, 0], start[5, 1], start[5, 2])
    initial7 = wp.vec3d(start[6, 0], start[6, 1], start[6, 2])
    for interval in range(14):
        begin, end = times[interval], times[interval + 1]
        if end > begin:
            s6, s7 = _state_at(initial6, p6, begin), _state_at(initial7, p7, begin)
            midpoint = F64(0.5) * (begin + end)
            j6, j7 = _jerk_at(p6, midpoint), _jerk_at(p7, midpoint)
            for edge in range(8):
                ax = F64(corners[edge, 0])
                ay = F64(corners[edge, 1])
                ex = corners[(edge + 1) % 8, 0] - ax
                ey = corners[(edge + 1) % 8, 1] - ay
                distance = wp.vec3d(
                    ex * (s7[0] - ay) - ey * (s6[0] - ax),
                    ex * s7[1] - ey * s6[1], ex * s7[2] - ey * s6[2],
                )
                if _range(distance, ex * j7 - ey * j6, end - begin)[1] > tolerance:
                    feasible = False
    return feasible


@wp.kernel
def _sample(
    q: wp.array2d(dtype=F64), v: wp.array2d(dtype=F64), a: wp.array2d(dtype=F64),
    goal: wp.array2d(dtype=F64), vmax: wp.array(dtype=F64), amax: wp.array(dtype=F64),
    jmax: wp.array(dtype=F64), lower: wp.array(dtype=F64), upper: wp.array(dtype=F64),
    corners: wp.array2d(dtype=F64),
    starts: wp.array3d(dtype=F64), durations: wp.array3d(dtype=F64), jerks: wp.array3d(dtype=F64),
    elapsed: wp.array(dtype=F64), accepted_goal: wp.array2d(dtype=F64), dt: F64,
    out_q: wp.array3d(dtype=F64), out_v: wp.array3d(dtype=F64), out_a: wp.array3d(dtype=F64),
    hits: wp.array(dtype=wp.int32),
):
    n = wp.tid()
    changed, resting = bool(False), bool(True)
    start = _Mat73()
    target = _Vec7()
    for j in range(7):
        start[j, 0] = q[n, j]
        start[j, 1] = v[n, j]
        start[j, 2] = a[n, j]
        target[j] = goal[n, j]
        changed = changed or goal[n, j] != accepted_goal[n, j]
        resting = resting and v[n, j] == F64(0.0) and a[n, j] == F64(0.0)
    hits[n] = 0
    if changed:
        fastest = _Mat73()
        slowest = F64(0.0)
        for j in range(7):
            fast = _fastest(goal[n, j] - q[n, j], v[n, j], a[n, j], vmax[j], amax[j], jmax[j])
            fastest[j] = fast
            slowest = wp.max(slowest, fast[2])
        candidate = _Path()
        for j in range(7):
            f = wp.vec2d(fastest[j, 0], fastest[j, 1])
            if fastest[j, 2] < slowest - F64(1.0e-9):
                f = _within(slowest, goal[n, j] - q[n, j], v[n, j], a[n, j], f[0], amax[j], jmax[j])
            p = _profile(v[n, j], a[n, j], f[0], f[1], amax[j], jmax[j])
            for phase in range(7):
                candidate.duration[j, phase] = p.duration[phase]
                candidate.jerk[j, phase] = p.jerk[phase]
        feasible = _feasible(start, target, candidate, vmax, amax, jmax, lower, upper, corners)
        if not feasible:
            hits[n] = 1
            if resting:
                # Shared progress s in [0,1]: every joint traverses the same
                # convex line segment, with scalar limits chosen for all joints.
                sv, sa, sj = F64(1.0e30), F64(1.0e30), F64(1.0e30)
                for j in range(7):
                    distance = wp.abs(goal[n, j] - q[n, j])
                    if distance > F64(0.0):
                        sv = wp.min(sv, vmax[j] / distance)
                        sa = wp.min(sa, amax[j] / distance)
                        sj = wp.min(sj, jmax[j] / distance)
                scalar = _fastest(F64(1.0), F64(0.0), F64(0.0), sv, sa, sj)
                p = _profile(F64(0.0), F64(0.0), scalar[0], scalar[1], sa, sj)
                for j in range(7):
                    for phase in range(7):
                        candidate.duration[j, phase] = p.duration[phase]
                        candidate.jerk[j, phase] = p.jerk[phase] * (goal[n, j] - q[n, j])
                feasible = _feasible(start, target, candidate, vmax, amax, jmax, lower, upper, corners)
        if feasible:
            for j in range(7):
                for k in range(3):
                    starts[n, j, k] = start[j, k]
                for phase in range(7):
                    durations[n, j, phase] = candidate.duration[j, phase]
                    jerks[n, j, phase] = candidate.jerk[j, phase]
                accepted_goal[n, j] = goal[n, j]
            elapsed[n] = F64(0.0)
    for j in range(7):
        profile = _Profile()
        for phase in range(7):
            profile.duration[phase] = durations[n, j, phase]
            profile.jerk[phase] = jerks[n, j, phase]
        total = _duration(profile)
        initial = wp.vec3d(starts[n, j, 0], starts[n, j, 1], starts[n, j, 2])
        for k in range(out_q.shape[1]):
            t = elapsed[n] + F64(k + 1) * dt
            s = wp.vec3d(accepted_goal[n, j], F64(0.0), F64(0.0))
            if t < total:
                s = _state_at(initial, profile, t)
            out_q[n, k, j] = s[0]
            out_v[n, k, j] = s[1]
            out_a[n, k, j] = s[2]
    elapsed[n] = elapsed[n] + F64(out_q.shape[1]) * dt


class BoundedPlanner:
    """A per-environment accepted path; all state is included in action snapshots."""

    def __init__(self, q, vmax, amax, jmax, lower, upper, corners):
        self.limits = [x.double().contiguous() for x in (vmax, amax, jmax, lower, upper, corners)]
        n = len(q)
        options = dict(device=q.device, dtype=torch.float64)
        self.state = {
            "path_start": torch.zeros(n, 7, 3, **options),
            "path_duration": torch.zeros(n, 7, 7, **options),
            "path_jerk": torch.zeros(n, 7, 7, **options),
            "path_elapsed": torch.zeros(n, **options),
            "path_goal": torch.zeros(n, 7, **options),
        }
        self.reset(q)

    def reset(self, q, rows=slice(None)):
        for tensor in self.state.values():
            tensor[rows] = 0
        self.state["path_start"][rows, :, 0] = q[rows]
        self.state["path_goal"][rows] = q[rows]

    def sample(self, q, v, a, goal, dt, steps):
        wp.init()
        out = [torch.empty((len(q), steps, 7), device=q.device, dtype=torch.float64) for _ in range(3)]
        hits = torch.empty(len(q), device=q.device, dtype=torch.int32)
        device = wp.device_from_torch(q.device)
        wp.launch(
            _sample, dim=len(q),
            inputs=[wp.from_torch(x.contiguous()) for x in (q, v, a, goal)]
            + [wp.from_torch(x) for x in self.limits]
            + [wp.from_torch(x) for x in self.state.values()] + [float(dt)],
            outputs=[wp.from_torch(x) for x in (*out, hits)], device=device,
            stream=wp.stream_from_torch(q.device) if device.is_cuda else None,
        )
        return (*out, hits.bool())
