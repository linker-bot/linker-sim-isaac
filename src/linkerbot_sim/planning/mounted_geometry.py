"""Immutable fixed-hand geometry shared by planning snapshots and backend adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class MountedCollisionModel:
    robot_id: int
    flange_frame: str
    spheres: tuple[tuple[float, float, float, float], ...]
    source_links: tuple[str, ...]
    cell_size_m: float = 0.03
    sphere_links: tuple[str, ...] = ()
    payload_contacts: tuple[tuple[str, str], ...] = ()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                (
                    self.flange_frame,
                    self.spheres,
                    self.source_links,
                    self.cell_size_m,
                    self.sphere_links,
                    self.payload_contacts,
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def diagnostics(self) -> dict[str, object]:
        return {
            "representation": "fixed_hand_collider_box_covering_spheres",
            "sphere_count": len(self.spheres),
            "source_links": list(self.source_links),
            "cell_size_m": self.cell_size_m,
            "fingerprint": self.fingerprint,
            "payload_contacts": list(self.payload_contacts),
            "limitations": [
                "Hand and attachment pose is frozen for this arm plan.",
                "Collision inside the frozen assembly is not checked by the flange envelope.",
                "Collider boxes and covering spheres are conservative approximations.",
            ],
        }

    def payload_collision(self) -> dict[str, object] | None:
        """Check carried geometry against non-contact links in the frozen assembly."""
        if not self.sphere_links:
            return None
        spheres = np.asarray(self.spheres)
        links = np.asarray(self.sphere_links)
        allowed = frozenset(self.payload_contacts)
        for name in sorted(
            {link for link in self.sphere_links if link.startswith("payload:")}
        ):
            payload = spheres[links == name]
            for other in sorted(set(self.sphere_links) - {name}):
                if (name, other) in allowed or (other, name) in allowed:
                    continue
                target = spheres[links == other]
                gap = (
                    np.linalg.norm(payload[:, None, :3] - target[None, :, :3], axis=-1)
                    - payload[:, None, 3]
                    - target[None, :, 3]
                )
                if np.any(gap < 0):
                    return {
                        "payload": name,
                        "link_name": other,
                        "clearance_m": float(gap.min()),
                        "approximation": "covering_spheres",
                    }
        return None


def cover_box_with_spheres(
    size: tuple[float, ...], *, cell_size: float = 0.03
) -> np.ndarray:
    """Tile the volume; each sphere covers an entire cell, not just its surface."""
    dimensions = np.asarray(size, dtype=float)
    if (
        dimensions.shape != (3,)
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0)
    ):
        raise ValueError("box size must contain three finite positive dimensions")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be finite and positive")
    count = np.ceil(dimensions / cell_size).astype(int)
    if int(np.prod(count)) > 100_000:
        raise ValueError(
            "box sphere covering exceeds 100000 cells; use a coarser explicit approximation"
        )
    step = dimensions / count
    axes = [
        (np.arange(n) + 0.5) * width - length / 2
        for n, width, length in zip(count, step, dimensions, strict=True)
    ]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.column_stack((centers, np.full(len(centers), np.linalg.norm(step) / 2)))
