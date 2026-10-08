"""Sampled sphere/world validation with explicit link/object contact allowances."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np

from linkerbot_sim.planning.collision_objects import CollisionObject


@dataclass(frozen=True)
class AllowedPlanningContact:
    robot_id: int
    link_name: str
    geometry_name: str


def sphere_obstacle_clearance(
    spheres: np.ndarray, obstacle: CollisionObject
) -> np.ndarray:
    """Match cuRobo's conservative cuboid materialization, returning signed metres."""
    pose = obstacle.pose_matrix()
    local = (spheres[..., :3] - pose[:3, 3]) @ pose[:3, :3]
    size = obstacle.padded_size()
    if obstacle.shape == "cuboid":
        half = np.asarray(size) / 2
    elif obstacle.shape == "sphere":
        half = np.full(3, size[0])
    elif obstacle.shape == "capsule":
        half = np.array((size[0], size[0], size[1] / 2 + size[0]))
    else:
        raise ValueError(f"unsupported planning geometry {obstacle.shape!r}")
    outside = np.abs(local) - half
    return (
        np.linalg.norm(np.maximum(outside, 0), axis=-1)
        + np.minimum(np.max(outside, axis=-1), 0)
        - spheres[..., 3]
    )


def validate_sampled_path(
    path: np.ndarray,
    *,
    sample_spheres: Callable[[np.ndarray], np.ndarray],
    sphere_links: Sequence[str],
    obstacles: Sequence[CollisionObject],
    allowed_pairs: frozenset[tuple[str, str]] = frozenset(),
    max_joint_step: float = 0.02,
) -> dict[str, object]:
    """Check endpoints and bounded joint-space samples, not continuous collision detection."""
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("collision validation needs a finite nonempty joint path")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError("max_joint_step must be finite and positive")
    counts = np.maximum(
        1,
        np.ceil(
            np.max(np.abs(np.diff(values, axis=0)), axis=1) / max_joint_step
        ).astype(int),
    )
    if 1 + int(counts.sum()) > 10000:
        raise ValueError("collision validation exceeds 10000 samples")
    samples = [values[:1]]
    for first, second, count in zip(values[:-1], values[1:], counts, strict=True):
        samples.append(np.linspace(first, second, int(count) + 1)[1:])
    q = np.vstack(samples)
    minimum = float("inf")
    for offset in range(0, len(q), 128):
        spheres = np.asarray(sample_spheres(q[offset : offset + 128]), dtype=float)
        if spheres.shape != (
            len(q[offset : offset + 128]),
            len(sphere_links),
            4,
        ) or not np.all(np.isfinite(spheres)):
            raise ValueError("kinematics returned invalid planning spheres")
        for obstacle in obstacles:
            if not obstacle.enabled:
                continue
            active = np.asarray(
                [(name, obstacle.name) not in allowed_pairs for name in sphere_links]
            )
            if not active.any():
                continue
            gaps = sphere_obstacle_clearance(spheres[:, active], obstacle)
            minimum = min(minimum, float(gaps.min()))
            hit = np.argwhere(gaps < 0)
            if len(hit):
                row, col = hit[0]
                return {
                    "valid": False,
                    "sample_index": int(offset + row),
                    "link_name": np.asarray(sphere_links)[active][col].item(),
                    "geometry_name": obstacle.name,
                    "clearance_m": float(gaps[row, col]),
                    "samples": len(q),
                    "max_joint_step_rad": max_joint_step,
                }
    result = {
        "valid": True,
        "samples": len(q),
        "max_joint_step_rad": max_joint_step,
        "continuous_collision_detection": False,
    }
    if np.isfinite(minimum):
        result["minimum_sampled_clearance_m"] = minimum
    return result
