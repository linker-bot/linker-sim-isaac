"""Cold AR5/O6 contact-material configuration; no per-step parameter switching."""

from pathlib import Path

from linkerbot_sim.configuration.physics import NewtonContactSettings
from linkerbot_sim.robots.classification import component_for_name


def _robot_roots(robot) -> tuple[str, ...]:
    return tuple(
        str(root).rstrip("/")
        for root in (
            getattr(robot, "imported_root_paths", None) or (robot.imported_root_path,)
        )
    )


def _within(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _set_shape_attribute(builder, name, index, value):
    column = builder.custom_attributes[f"mujoco:{name}"]
    if column.values is None:
        column.values = {}
    if isinstance(column.values, list):
        while len(column.values) <= index:
            column.values.append(None)
    column.values[index] = value


def configure_contacts(
    builder,
    robots,
    object_handles,
    world_roots,
    settings: NewtonContactSettings | None,
) -> dict[str, int]:
    """Apply materials once before finalize; preserve geometry, mass and authored gap.

    Any AR5/O6 enables the scene-wide solver policy. Other robots keep their own
    material columns, even in a mixed scene. Hand/station membership uses the
    imported body name, not a substring in its ancestor arm/hand paths.
    """
    if settings is None:
        return {}
    selected, other = (), ()
    for robot in robots.values():
        roots = _robot_roots(robot)
        if Path(robot.asset_path).parent.name.startswith(("ar5_08_o6_", "ar5_o6_")):
            selected += roots
        else:
            other += roots
    if not selected:
        return {}

    import newton
    import warp as wp
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW

    from linkerbot_sim.objects.runtime import runtime_object_prim_path

    object_roots = ()
    for handle in object_handles:
        physics = getattr(handle.config, "physics", None)
        if handle.kind != "dynamic_chain" and getattr(physics, "static", False):
            continue
        path = runtime_object_prim_path(handle)
        if path is not None:
            if _within(path, (world_roots[0],)):
                suffix = path[len(world_roots[0]) :]
                object_roots += tuple(root + suffix for root in world_roots)
            else:
                object_roots += (path,)

    counts = dict(hand=0, object=0, station=0)
    time_constant, damping_ratio = settings.solref
    for i, path in enumerate(builder.shape_label):
        if not (int(builder.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue
        if _within(path, other):
            continue
        body = builder.shape_body[i]
        body_name = builder.body_label[body].rsplit("/", 1)[-1] if body >= 0 else ""
        if _within(path, selected) and component_for_name(body_name) == "hand":
            role = "hand"
        elif _within(path, object_roots):
            role = "object"
        else:
            role = "station"
        material = getattr(settings, role)
        counts[role] += 1
        builder.shape_material_mu[i] = material.friction[0]
        builder.shape_material_mu_torsional[i] = material.friction[1]
        builder.shape_material_mu_rolling[i] = material.friction[2]
        builder.shape_material_ke[i] = 1.0 / time_constant**2
        builder.shape_material_kd[i] = 2.0 * damping_ratio / time_constant
        builder.shape_material_kf[i] = settings.friction_gain
        builder.shape_material_restitution[i] = settings.restitution
        builder.shape_margin[i] = settings.margin_m
        # Gap was resolved before import. Do not overwrite explicit asset values.
        _set_shape_attribute(builder, "solref", i, wp.vec2(*settings.solref))
        _set_shape_attribute(builder, "solref_mode", i, SOLREF_MODE_RAW)
        _set_shape_attribute(builder, "geom_solimp", i, material.solimp)
        _set_shape_attribute(builder, "condim", i, material.condim)
        _set_shape_attribute(builder, "geom_priority", i, 0)
        _set_shape_attribute(builder, "geom_solmix", i, 1.0)
    return counts
