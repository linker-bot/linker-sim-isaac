"""Read imported collision bounds once; never use visual bounds as collision geometry."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class LinkColliderBox:
    name: str
    link_name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def read_link_collider_boxes(
    stage: object, root_path: str, link_names: set[str]
) -> tuple[LinkColliderBox, ...]:
    """Bound each collider in its nearest named URDF link, including instance proxies.

    Each source collider keeps its own bound, preserving spaces between decomposed
    parts. Boxes conservatively fill concavities *inside* an individual collider.
    Geometry and mount changes require rebuilding this provider, not a pose refresh.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"collision root does not exist: {root_path}")
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        link = prim
        while link.IsValid() and link != root and link.GetName() not in link_names:
            link = link.GetParent()
        if link.GetName() not in link_names:
            raise ValueError(f"collider has no named URDF link: {prim.GetPath()}")
        # Collision meshes may be invisible guide-purpose instance proxies.
        # Read their geometry directly; render-purpose bounds may be empty.
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        else:
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
                UsdGeom.Boundable(prim), Usd.TimeCode.Default()
            )
            if extent is None or len(extent) != 2:
                raise ValueError(
                    f"unsupported collision bound: {prim.GetPath()} ({prim.GetTypeName()})"
                )
            points = np.asarray(list(product(*zip(*extent, strict=True))), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(
                f"collider has no finite-volume geometry: {prim.GetPath()}"
            )
        relative = np.asarray(
            cache.GetLocalToWorldTransform(prim)
            * cache.GetLocalToWorldTransform(link).GetInverse()
        ).T
        points = points @ relative[:3, :3].T + relative[:3, 3]
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        size = upper - lower
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError(
                f"collider has invalid finite-volume bounds: {prim.GetPath()}, "
                f"type={prim.GetTypeName()}, lower={lower}, upper={upper}"
            )
        result.append(
            LinkColliderBox(
                str(prim.GetPath()),
                link.GetName(),
                tuple((lower + upper) / 2),
                tuple(size),
            )
        )
    if not result:
        raise ValueError(f"no enabled colliders below {root_path}")
    return tuple(result)
