"""Product-neutral resolution of a configured TCP to one physical rigid body."""

from __future__ import annotations

from dataclasses import dataclass
import math

from linkerbot_sim.configuration.robots import RobotProfileSettings


@dataclass(frozen=True, slots=True)
class PhysicalTcpBinding:
    tcp_frame_name: str
    parent_frame_name: str
    parent_body_path: str
    offset_xyz: tuple[float, float, float]
    offset_rpy: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("tcp_frame_name", "parent_frame_name", "parent_body_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("offset_xyz", "offset_rpy"):
            values = tuple(float(item) for item in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(item) for item in values):
                raise ValueError(f"{name} must contain three finite numbers")
            object.__setattr__(self, name, values)

    def as_legacy_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Preserve the replicated-scene tuple contract during extraction."""

        return (
            self.tcp_frame_name,
            self.parent_frame_name,
            self.parent_body_path,
            self.offset_xyz,
            self.offset_rpy,
        )


def resolve_physical_tcp_binding(
    *,
    stage: object,
    imported_root_path: str,
    profile: RobotProfileSettings,
) -> PhysicalTcpBinding:
    """Resolve the catalog's default TCP to a unique rigid body below a robot."""

    if not isinstance(profile, RobotProfileSettings):
        raise TypeError("profile must be RobotProfileSettings")
    robot = profile.curobo.robot
    if not profile.curobo.binding.enabled or robot is None:
        raise ValueError(
            f"robot profile {profile.name!r} does not enable a physical TCP model"
        )
    default_tcp = robot.default_tcp_frame or robot.resolved_tool_frames[0]
    custom = {frame.frame_name: frame for frame in robot.custom_tcp_frames}
    if default_tcp in custom:
        frame = custom[default_tcp]
        parent = frame.parent_frame
        xyz = tuple(float(value) for value in frame.xyz)
        rpy = tuple(float(value) for value in frame.rpy)
    else:
        parent = default_tcp
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
    return PhysicalTcpBinding(
        tcp_frame_name=default_tcp,
        parent_frame_name=parent,
        parent_body_path=unique_rigid_body_path(
            stage,
            root_path=imported_root_path,
            body_name=parent,
        ),
        offset_xyz=xyz,
        offset_rpy=rpy,
    )


def unique_rigid_body_path(
    stage: object,
    *,
    root_path: str,
    body_name: str,
) -> str:
    """Return the only rigid body with ``body_name`` below ``root_path``."""

    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"robot imported root does not exist: {root_path}")
    matches = tuple(
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"physical TCP parent {body_name!r} must match exactly one rigid body "
            f"below {root_path}; found {len(matches)}"
        )
    return matches[0]


__all__ = [
    "PhysicalTcpBinding",
    "resolve_physical_tcp_binding",
    "unique_rigid_body_path",
]
