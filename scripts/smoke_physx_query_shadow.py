#!/usr/bin/env python3
"""Validate query-only shadow geometry against PhysX CUDA scene queries."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.isaac.physics.query_shadow import (  # noqa: E402
    PhysxQueryShadowOwner,
)
from linkerbot_sim.isaac.physics.scene_query import (  # noqa: E402
    PhysxOverlapHit,
    PhysxSceneQueryAdapter,
)
from linkerbot_sim.isaac.session import create_isaac_session_from_spec  # noqa: E402
from linkerbot_sim.isaac.spec import (  # noqa: E402
    IsaacComputeSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)
from linkerbot_sim.kaleidoscope import (  # noqa: E402
    CollisionHitCategory,
    CollisionQueryTarget,
)
from scripts.runtime_worker_supervisor import (  # noqa: E402
    in_runtime_worker,
    run_supervised_worker,
)


PROBE_MARKER = "LINKERBOT_PHYSX_QUERY_SHADOW_VALID"
PRODUCT_PROBE_MARKER = "LINKERBOT_KALEIDOSCOPE_COLLISION_QUERY_VALID"
FAILURE_MARKER = "LINKERBOT_PHYSX_QUERY_SHADOW_FAILED"
SUCCESS_MARKER = "LINKERBOT_PHYSX_QUERY_SHADOW_SMOKE_OK"

ACTIVE_CONCAVE_PATH = "/World/Issue19ActiveConcave"
ACTIVE_CAVITY_PROBE_PATH = "/World/Issue19ActiveCavityProbe"
CANONICAL_SOURCE_PATH = "/World/Issue19CanonicalSource"
SHADOW_CAVITY_PROBE_PATH = "/World/Issue19ShadowCavityProbe"
SHADOW_SCALE_X = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--product-env",
        action="store_true",
        help="query through the real Kaleidoscope Torch environment",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="small environment count used only by --product-env",
    )
    parser.add_argument(
        "--portable-reset-description",
        type=Path,
        help=(
            "restore this portable reset description before querying; requires "
            "--product-env"
        ),
    )
    result = parser.parse_args(argv)
    if result.cuda_device < 0:
        parser.error("--cuda-device must be non-negative")
    if result.num_envs < 1:
        parser.error("--num-envs must be positive")
    if result.portable_reset_description is not None and not result.product_env:
        parser.error("--portable-reset-description requires --product-env")
    return result


def _session_spec(*, cuda_device: int) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=cuda_device),
        physics=IsaacPhysxCudaSpec(enable_scene_query_support=True),
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 120.0,
        gravity_z=0.0,
        add_ground=False,
    )


def _append_box(
    points: list[tuple[float, float, float]],
    face_vertex_counts: list[int],
    face_vertex_indices: list[int],
    *,
    center: tuple[float, float, float],
    half_extent: tuple[float, float, float],
) -> None:
    """Append one closed box into a combined polygon mesh."""

    cx, cy, cz = center
    hx, hy, hz = half_extent
    base = len(points)
    points.extend(
        (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        )
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    face_vertex_counts.extend((4,) * len(faces))
    face_vertex_indices.extend(base + index for face in faces for index in face)


def _define_concave_u_mesh(stage: object, path: str, *, translate_x: float) -> None:
    """Create one concave mesh whose cavity distinguishes decomposition from hull."""

    from pxr import Gf, UsdGeom, UsdPhysics

    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []
    _append_box(
        points,
        counts,
        indices,
        center=(-0.75, 0.0, 0.0),
        half_extent=(0.25, 0.25, 1.0),
    )
    _append_box(
        points,
        counts,
        indices,
        center=(0.75, 0.0, 0.0),
        half_extent=(0.25, 0.25, 1.0),
    )
    _append_box(
        points,
        counts,
        indices,
        center=(0.0, 0.0, -0.75),
        half_extent=(0.5, 0.25, 0.25),
    )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr().Set(counts)
    mesh.CreateFaceVertexIndicesAttr().Set(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.AddTranslateOp().Set(Gf.Vec3d(float(translate_x), 0.0, 0.0))
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set("convexDecomposition")


def _define_probe_cube(stage: object, path: str, *, translate_x: float) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(0.2)
    cube.AddTranslateOp().Set(Gf.Vec3d(float(translate_x), 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)


def _build_probe_scene(stage: object) -> None:
    _define_concave_u_mesh(stage, ACTIVE_CONCAVE_PATH, translate_x=0.0)
    _define_probe_cube(stage, ACTIVE_CAVITY_PROBE_PATH, translate_x=0.0)
    _define_concave_u_mesh(stage, CANONICAL_SOURCE_PATH, translate_x=10.0)
    _define_probe_cube(stage, SHADOW_CAVITY_PROBE_PATH, translate_x=4.0)


def _collision_paths(
    hits: Sequence[PhysxOverlapHit],
    *,
    ignored: Sequence[str] = (),
) -> frozenset[str]:
    ignored_paths = frozenset(ignored)
    return frozenset(
        hit.collision_path for hit in hits if hit.collision_path not in ignored_paths
    )


def _scaled_translation_matrix(*, translate_x: float) -> tuple[float, ...]:
    """Return a Gf row-major transform with visible non-unit X scale."""

    return (
        SHADOW_SCALE_X,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        float(translate_x),
        0.0,
        0.0,
        1.0,
    )


def _assert_geometry_semantics(
    *,
    active_cavity_hits: Sequence[PhysxOverlapHit],
    shadow_cavity_hits: Sequence[PhysxOverlapHit],
    shadow_contact_hits: Sequence[PhysxOverlapHit],
) -> None:
    active_paths = _collision_paths(
        active_cavity_hits,
        ignored=(ACTIVE_CAVITY_PROBE_PATH,),
    )
    if ACTIVE_CONCAVE_PATH in active_paths:
        raise AssertionError(
            "active convexDecomposition collider filled its known empty cavity"
        )

    cavity_paths = _collision_paths(shadow_cavity_hits)
    if SHADOW_CAVITY_PROBE_PATH in cavity_paths:
        raise AssertionError(
            "shadow overlap_shape filled the concave cavity; query cooking does "
            "not match the active convexDecomposition collider"
        )

    contact_paths = _collision_paths(shadow_contact_hits)
    if SHADOW_CAVITY_PROBE_PATH not in contact_paths:
        raise AssertionError(
            "shadow overlap_shape missed a probe intersecting a solid U-mesh bar"
        )


def _world_clock(world: object) -> tuple[float, int]:
    return (
        float(getattr(world, "current_time")),
        int(getattr(world, "current_time_step_index")),
    )


def _active_scene_signature(stage: object) -> tuple[object, ...]:
    """Capture active prim transforms and collision-enabled values only."""

    from pxr import Sdf, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    values: list[object] = []
    for path in (
        ACTIVE_CONCAVE_PATH,
        ACTIVE_CAVITY_PROBE_PATH,
        CANONICAL_SOURCE_PATH,
        SHADOW_CAVITY_PROBE_PATH,
    ):
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        matrix = cache.GetLocalToWorldTransform(prim)
        flattened = tuple(
            float(matrix[row][column]) for row in range(4) for column in range(4)
        )
        enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        values.append((path, flattened, enabled))
    return tuple(values)


def _verify_shadow_usd_contract(stage: object, shadow: object) -> str:
    from pxr import Sdf, UsdGeom, UsdPhysics

    root_path = str(getattr(shadow, "prim_path", ""))
    query_paths = tuple(getattr(shadow, "query_prim_paths", ()))
    source_path = str(getattr(shadow, "source_collision_path", ""))
    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid() or not root.IsA(UsdGeom.Xform):
        raise AssertionError("compound query shadow did not create an Xform root")
    source = stage.GetPrimAtPath(Sdf.Path(source_path))
    approximation = UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get()
    if approximation != "convexDecomposition":
        raise AssertionError(
            f"query shadow source does not use convex decomposition: {approximation!r}"
        )
    if len(query_paths) < 2:
        raise AssertionError(
            "concave query shadow did not expose multiple convex pieces"
        )
    for query_path in query_paths:
        piece = stage.GetPrimAtPath(Sdf.Path(query_path))
        if not piece.IsValid() or not piece.IsA(UsdGeom.Mesh):
            raise AssertionError(
                f"query shadow convex piece is not a Mesh: {query_path}"
            )
        if piece.HasAPI(UsdPhysics.CollisionAPI):
            raise AssertionError(
                f"query shadow convex piece became an active collider: {query_path}"
            )
    return str(approximation)


def _overlap_shadow(
    adapter: PhysxSceneQueryAdapter,
    shadow: object,
) -> tuple[PhysxOverlapHit, ...]:
    return tuple(
        hit
        for query_path in tuple(getattr(shadow, "query_prim_paths", ()))
        for hit in adapter.overlap_shape(query_path)
    )


def _run_probe(session: object) -> dict[str, object]:
    stage = session.stage
    runtime = session.physics_runtime
    if getattr(runtime, "kind", None) != "physx_cuda":
        raise RuntimeError("query-shadow smoke requires physx_cuda")
    world = runtime.world
    context = world.get_physics_context()
    if not bool(context.get_enable_scene_query_support()):
        raise RuntimeError("PhysX scene-query support is not active")

    _build_probe_scene(stage)
    runtime.reset()
    runtime.step(render=False)
    before_clock = _world_clock(world)
    before_signature = _active_scene_signature(stage)

    owner = PhysxQueryShadowOwner(stage)
    adapter = PhysxSceneQueryAdapter()
    try:
        shadow = owner.create_shadow(
            name="concave_u",
            source_collision_path=CANONICAL_SOURCE_PATH,
        )
        approximation = _verify_shadow_usd_contract(stage, shadow)

        active_cavity_hits = adapter.overlap_shape(ACTIVE_CAVITY_PROBE_PATH)
        owner.set_world_transform(
            shadow,
            matrix_row_major=_scaled_translation_matrix(translate_x=4.0),
        )
        shadow_cavity_hits = _overlap_shadow(adapter, shadow)
        owner.set_world_transform(
            shadow,
            # With X scale 0.5, the left solid bar spans local X [-0.5, -0.25].
            # Translating by 4.375 places that bar across the probe at X=4.
            matrix_row_major=_scaled_translation_matrix(translate_x=4.375),
        )
        shadow_contact_hits = _overlap_shadow(adapter, shadow)
        _assert_geometry_semantics(
            active_cavity_hits=active_cavity_hits,
            shadow_cavity_hits=shadow_cavity_hits,
            shadow_contact_hits=shadow_contact_hits,
        )

        after_clock = _world_clock(world)
        after_signature = _active_scene_signature(stage)
        if after_clock != before_clock:
            raise AssertionError(
                f"scene query advanced physics time: {before_clock} -> {after_clock}"
            )
        if after_signature != before_signature:
            raise AssertionError("scene query changed active scene prim state")
        return {
            "active_cavity_hit_count": len(active_cavity_hits),
            "approximation": approximation,
            "clock": list(after_clock),
            "scene_query_support": True,
            "shadow_scale_x": SHADOW_SCALE_X,
            "shadow_piece_count": len(shadow.query_prim_paths),
            "shadow_cavity_hit_count": len(shadow_cavity_hits),
            "shadow_contact_hit_count": len(shadow_contact_hits),
        }
    finally:
        adapter.close()
        owner.close()


def _collision_query_targets() -> tuple[CollisionQueryTarget, ...]:
    """Return every scene category required by the first public query API."""

    return (
        CollisionQueryTarget.TABLE,
        CollisionQueryTarget.DYNAMIC_OBJECT,
        CollisionQueryTarget.OTHER_ROBOT,
    )


def _allowed_hit_category_codes(
    targets: tuple[CollisionQueryTarget, ...],
) -> frozenset[int]:
    by_target = {
        CollisionQueryTarget.TABLE: CollisionHitCategory.TABLE,
        CollisionQueryTarget.DYNAMIC_OBJECT: CollisionHitCategory.DYNAMIC_OBJECT,
        CollisionQueryTarget.OTHER_ROBOT: CollisionHitCategory.OTHER_ROBOT,
    }
    requested_codes = tuple(int(by_target[target]) for target in targets)
    return frozenset((int(CollisionHitCategory.NONE), *requested_codes))


def _snapshot_group_counts(field_names: Sequence[str]) -> dict[str, int]:
    """Require snapshot coverage for every state family a query must preserve."""

    names = frozenset(str(name) for name in field_names)
    groups = {
        "physics": frozenset(
            {
                "robot.q",
                "robot.qd",
                "object.pose_local_wxyz",
                "object.com_velocity",
            }
        ),
        "control": frozenset({"robot.target", "robot.position_reference"}),
        "task": frozenset(name for name in names if name.startswith("task.")),
        "rng": frozenset(name for name in names if name.startswith("rng.")),
    }
    missing_physics = groups["physics"] - names
    missing_control = groups["control"] - names
    if missing_physics or missing_control or not groups["task"] or not groups["rng"]:
        raise RuntimeError(
            "collision query smoke snapshot does not cover all state families: "
            f"missing_physics={sorted(missing_physics)}, "
            f"missing_control={sorted(missing_control)}, "
            f"task_fields={len(groups['task'])}, rng_fields={len(groups['rng'])}"
        )
    return {name: len(values) for name, values in groups.items()}


def _assert_exact_snapshot_match(before: object, after: object) -> dict[str, int]:
    """Prove query and backend refresh preserved every snapshot field exactly."""

    import torch

    metadata_names = (
        "compatibility_fingerprint",
        "control_mode",
        "control_generation",
        "schema_version",
    )
    for name in metadata_names:
        if getattr(before, name, None) != getattr(after, name, None):
            raise RuntimeError(f"collision query changed snapshot metadata {name}")
    before_ids = getattr(before, "env_ids", None)
    after_ids = getattr(after, "env_ids", None)
    if not isinstance(before_ids, torch.Tensor) or not isinstance(
        after_ids, torch.Tensor
    ):
        raise RuntimeError("collision query snapshot env_ids are not Torch tensors")
    if not torch.equal(before_ids, after_ids):
        raise RuntimeError("collision query changed snapshot env_ids")
    before_fields = getattr(before, "fields", None)
    after_fields = getattr(after, "fields", None)
    if not isinstance(before_fields, Mapping) or not isinstance(after_fields, Mapping):
        raise RuntimeError("collision query snapshots do not expose field mappings")
    if set(before_fields) != set(after_fields):
        raise RuntimeError("collision query changed the snapshot field schema")
    for name, previous in before_fields.items():
        current = after_fields[name]
        if not isinstance(previous, torch.Tensor) or not isinstance(
            current, torch.Tensor
        ):
            raise RuntimeError(f"snapshot field {name!r} is not a Torch tensor")
        if not torch.equal(previous, current):
            raise RuntimeError(f"collision query changed snapshot field {name!r}")
    counts = _snapshot_group_counts(tuple(before_fields))
    return {"snapshot_fields": len(before_fields), **counts}


def _runtime_transaction_signature(runtime: object) -> tuple[object, ...]:
    """Capture lifecycle state that is outside the episode snapshot schema."""

    return (
        getattr(runtime, "_phase", None),
        getattr(runtime, "_generation", None),
        getattr(runtime, "_outstanding_token", None),
        getattr(runtime, "_outstanding_phase", None),
        getattr(runtime, "_failed", None),
        getattr(runtime, "_fatal_error", None),
        getattr(runtime, "_closing_started", None),
        getattr(runtime, "_closed", None),
    )


def _validate_product_query_result(
    result: object,
    *,
    device: object,
    env_count: int,
    sample_count: int,
    targets: tuple[CollisionQueryTarget, ...],
) -> dict[str, int]:
    """Validate the public CUDA result without requiring any incidental hit."""

    import torch

    mask = getattr(result, "collision_mask", None)
    first_sample = getattr(result, "first_hit_sample", None)
    first_category = getattr(result, "first_hit_category", None)
    expected = (
        (mask, torch.bool, (env_count, sample_count), "collision_mask"),
        (first_sample, torch.int64, (env_count,), "first_hit_sample"),
        (first_category, torch.int64, (env_count,), "first_hit_category"),
    )
    for tensor, dtype, shape, label in expected:
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"collision query {label} is not a Torch tensor")
        if tensor.device != device or tensor.dtype != dtype or tensor.shape != shape:
            raise RuntimeError(
                f"collision query {label} has an invalid CUDA contract: "
                f"device={tensor.device}, dtype={tensor.dtype}, "
                f"shape={tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"collision query {label} must be contiguous")

    assert isinstance(mask, torch.Tensor)
    assert isinstance(first_sample, torch.Tensor)
    assert isinstance(first_category, torch.Tensor)
    allowed_codes = torch.tensor(
        sorted(_allowed_hit_category_codes(targets)),
        device=device,
        dtype=torch.int64,
    )
    if not bool(torch.all(torch.isin(first_category, allowed_codes)).item()):
        raise RuntimeError("collision query returned an unrequested hit category")
    expected_first = torch.full(
        (env_count,),
        -1,
        device=device,
        dtype=torch.int64,
    )
    for sample_index in range(sample_count):
        expected_first = torch.where(
            (expected_first < 0) & mask[:, sample_index],
            sample_index,
            expected_first,
        )
    if not bool(torch.equal(first_sample, expected_first)):
        raise RuntimeError("first_hit_sample does not match collision_mask")
    has_hit = first_sample >= 0
    category_has_hit = first_category != int(CollisionHitCategory.NONE)
    if not bool(torch.equal(has_hit, category_has_hit)):
        raise RuntimeError("first_hit_category does not match collision_mask")
    return {
        "collision_count": int(torch.count_nonzero(mask).item()),
        "rows_with_hit": int(torch.count_nonzero(has_hit).item()),
    }


def _interior_product_query_candidates(
    current,
    *,
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
):
    """Build two valid samples without changing the active control target."""

    import torch

    if not isinstance(current, torch.Tensor) or current.ndim != 2:
        raise TypeError("current robot target must be a rank-2 Torch tensor")
    lower = torch.as_tensor(
        tuple(joint_lower),
        device=current.device,
        dtype=current.dtype,
    )
    upper = torch.as_tensor(
        tuple(joint_upper),
        device=current.device,
        dtype=current.dtype,
    )
    if lower.shape != current.shape[1:] or upper.shape != current.shape[1:]:
        raise ValueError("public collision-query limits do not match robot target")
    if not bool(torch.all(torch.isfinite(current))):
        raise ValueError("current robot target must contain only finite values")
    span = upper - lower
    if not bool(torch.all(torch.isfinite(span) & (span > 0.0))):
        raise ValueError("public collision-query limits must be finite and ordered")

    # A small interior margin survives float32 -> host NumPy conversion and keeps
    # values that sit exactly on a limit from being rejected by strict validation.
    margin = torch.maximum(span * 1.0e-4, torch.full_like(span, 1.0e-6))
    interior_lower = lower + margin
    interior_upper = upper - margin
    if not bool(torch.all(interior_lower < interior_upper)):
        raise ValueError("public collision-query limits have no stable interior")
    safe_current = torch.minimum(
        torch.maximum(current, interior_lower),
        interior_upper,
    )
    midpoint = ((lower + upper) * 0.5).expand_as(current)
    clamped_entries = int(torch.count_nonzero(safe_current != current).item())
    return torch.stack((safe_current, midpoint), dim=1).contiguous(), clamped_entries


def _run_product_env_probe(
    *,
    cuda_device: int,
    num_envs: int,
    portable_reset_description: Path | None,
) -> dict[str, object]:
    """Query the production PhysX composition through its public Env method."""

    import torch

    from linkerbot_sim.configuration import load_kaleidoscope_config
    from linkerbot_sim.kaleidoscope import (
        load_reset_scene_description,
        make_torch_env,
    )

    config = load_kaleidoscope_config("physx_cuda")
    if int(config.cuda_device) != cuda_device:
        raise RuntimeError(
            "--cuda-device must match the locked physx_cuda profile: "
            f"configured={config.cuda_device}, requested={cuda_device}"
        )
    env = make_torch_env(config=config, num_envs=num_envs)
    failed = False
    try:
        if not bool(env.supports_articulated_collision_queries):
            raise RuntimeError(
                "PhysX Kaleidoscope Env did not expose collision queries"
            )
        env.reset(seed=123)
        reset_evidence: dict[str, object] | None = None
        if portable_reset_description is not None:
            description = load_reset_scene_description(portable_reset_description)
            observations, reset_info = env.reset_from_description(
                description,
                env_id=0,
            )
            if not isinstance(observations, torch.Tensor):
                raise RuntimeError("portable reset observations are not a Torch tensor")
            if observations.device != env.device or not bool(
                torch.all(torch.isfinite(observations)).item()
            ):
                raise RuntimeError(
                    "portable reset observations are not finite on the Env device"
                )
            reset_env_ids = reset_info.get("env_ids")
            if not isinstance(reset_env_ids, torch.Tensor) or not torch.equal(
                reset_env_ids,
                torch.tensor([0], device=env.device, dtype=torch.int64),
            ):
                raise RuntimeError("portable reset returned unexpected environment ids")
            reset_state = env.get_state(
                reset_env_ids,
                fields=(
                    "rng.key",
                    "rng.counter",
                    "task.episode_length",
                    "task.episode_physics_steps",
                ),
            )
            generator = description.generator
            expected_reset_state = {
                "rng.key": generator.rng_key,
                "rng.counter": generator.rng_counter_after,
                "task.episode_length": 0,
                "task.episode_physics_steps": 0,
            }
            for field, expected_value in expected_reset_state.items():
                value = reset_state.get(field)
                if not isinstance(value, torch.Tensor) or value.tolist() != [
                    expected_value
                ]:
                    raise RuntimeError(
                        f"portable reset returned unexpected state for {field}: {value}"
                    )
            reset_evidence = {
                "env_id": 0,
                "rng_key": generator.rng_key,
                "rng_counter_after": generator.rng_counter_after,
                "state_verified": True,
            }
        selected_count = min(2, env.num_envs)
        env_ids = torch.arange(
            selected_count,
            device=env.device,
            dtype=torch.int64,
        )
        target_state = env.get_state(env_ids, fields=("robot.target",))
        command_target = target_state.get("robot.target")
        if not isinstance(command_target, torch.Tensor):
            raise RuntimeError("public robot.target state is not a Torch tensor")

        views = env.runtime.views
        ports = tuple(getattr(views, "robot_ports", ()))
        columns = tuple(getattr(views, "robot_columns", ()))
        if not ports or len(ports) != len(columns):
            raise RuntimeError("Kaleidoscope query smoke cannot resolve robot columns")
        port = ports[0]
        robot_label = str(getattr(port, "label", ""))
        robot_columns = getattr(columns[0], "columns", None)
        if not robot_label or not isinstance(robot_columns, slice):
            raise RuntimeError("Kaleidoscope query smoke found invalid robot metadata")

        current = command_target[:, robot_columns]
        query_specs = env.articulated_collision_query_specs
        query_spec = query_specs.get(robot_label)
        if query_spec is None:
            raise RuntimeError(
                "Kaleidoscope query smoke cannot read public robot query limits"
            )
        candidates, clamped_entries = _interior_product_query_candidates(
            current,
            joint_lower=query_spec.joint_lower,
            joint_upper=query_spec.joint_upper,
        )
        candidates_before = candidates.clone()
        env_ids_before = env_ids.clone()
        targets = _collision_query_targets()
        before_snapshot = env.snapshot()
        physics_world = getattr(env.runtime.physics_runtime, "world", None)
        if physics_world is None:
            raise RuntimeError("PhysX Kaleidoscope runtime did not expose its World")
        before_clock = _world_clock(physics_world)
        before_control = env.get_control_mode()
        before_transaction = _runtime_transaction_signature(env.runtime)
        outstanding_done_before = env.runtime._outstanding_done.clone()
        result = env.query_articulated_collisions(
            robot_label=robot_label,
            joint_positions=candidates,
            env_ids=env_ids,
            targets=targets,
        )
        metrics = _validate_product_query_result(
            result,
            device=env.device,
            env_count=selected_count,
            sample_count=candidates.shape[1],
            targets=targets,
        )
        after_clock = _world_clock(physics_world)
        after_control = env.get_control_mode()
        after_transaction = _runtime_transaction_signature(env.runtime)
        if after_clock != before_clock:
            raise RuntimeError(
                f"collision query advanced physics clock: {before_clock} -> "
                f"{after_clock}"
            )
        if after_control != before_control:
            raise RuntimeError("collision query changed control-mode state")
        if after_transaction != before_transaction:
            raise RuntimeError("collision query changed Runtime transaction state")
        if not torch.equal(env.runtime._outstanding_done, outstanding_done_before):
            raise RuntimeError("collision query changed SAME_STEP done state")
        if not torch.equal(candidates, candidates_before) or not torch.equal(
            env_ids, env_ids_before
        ):
            raise RuntimeError("collision query modified caller-owned input tensors")

        # Read active PhysX/Fabric values again before the second snapshot. This
        # catches writes to the real articulation/object, not only canonical buffers.
        env.runtime.views.refresh()
        after_snapshot = env.snapshot()
        invariant_counts = _assert_exact_snapshot_match(
            before_snapshot,
            after_snapshot,
        )
        report = {
            "backend": str(getattr(env.runtime.physics_runtime, "kind", "")),
            "candidate_clamped_entries": clamped_entries,
            "candidate_source": (
                "interior_clamped_control_target_and_joint_limit_midpoint"
            ),
            "device": str(env.device),
            "num_envs": env.num_envs,
            "queried_envs": selected_count,
            "robot_label": robot_label,
            "samples": candidates.shape[1],
            "state_invariance": {
                **invariant_counts,
                "control_mode": True,
                "physics_clock": list(after_clock),
                "query_inputs": True,
                "runtime_transaction": True,
                "same_step_done": True,
            },
            "targets": [target.value for target in targets],
            **metrics,
        }
        if reset_evidence is not None:
            report["portable_reset"] = reset_evidence
        # fast_shutdown may terminate the worker from env.close(). Emit evidence
        # first; the supervisor still requires a clean worker exit afterwards.
        print(
            PRODUCT_PROBE_MARKER
            + " "
            + json.dumps(report, ensure_ascii=True, sort_keys=True),
            flush=True,
        )
        return report
    except BaseException as exc:
        failed = True
        _report_probe_failure(exc, mode="product_env")
        raise
    finally:
        env.close(exit_code=1 if failed else 0)


def _report_probe_failure(exc: BaseException, *, mode: str) -> None:
    """Flush the original failure before Kit fast shutdown can end the worker."""

    print(
        FAILURE_MARKER
        + " "
        + json.dumps(
            {
                "error_message": str(exc),
                "error_type": type(exc).__name__,
                "mode": mode,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    required_marker = PRODUCT_PROBE_MARKER if args.product_env else PROBE_MARKER
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=arguments,
            required_markers=(required_marker,),
            success_marker=SUCCESS_MARKER,
        )

    os.environ["LINKERBOT_RUNTIME_PROVENANCE"] = "0"
    if args.product_env:
        _run_product_env_probe(
            cuda_device=args.cuda_device,
            num_envs=args.num_envs,
            portable_reset_description=args.portable_reset_description,
        )
        return 0
    session = create_isaac_session_from_spec(
        spec=_session_spec(cuda_device=args.cuda_device)
    )
    failed = False
    try:
        report = _run_probe(session)
        print(
            PROBE_MARKER + " " + json.dumps(report, ensure_ascii=True, sort_keys=True),
            flush=True,
        )
    except BaseException as exc:
        failed = True
        _report_probe_failure(exc, mode="low_level")
        raise
    finally:
        session.close(exit_code=1 if failed else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
