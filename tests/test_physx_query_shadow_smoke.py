from __future__ import annotations

import pytest

from linkerbot_sim.isaac.physics.scene_query import PhysxOverlapHit
from scripts import smoke_physx_query_shadow as smoke


def _hit(path: str) -> PhysxOverlapHit:
    return PhysxOverlapHit(path, path, 0xFFFFFFFF)


def test_parse_args_rejects_negative_cuda_device() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--cuda-device", "-1"])


def test_parse_args_selects_small_product_environment() -> None:
    args = smoke.parse_args(["--product-env", "--num-envs", "3"])

    assert args.product_env is True
    assert args.num_envs == 3


def test_parse_args_rejects_non_positive_product_environment_count() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--product-env", "--num-envs", "0"])


def test_product_query_targets_and_hit_codes_cover_the_public_v1_contract() -> None:
    targets = smoke._collision_query_targets()

    assert [target.value for target in targets] == [
        "table",
        "dynamic_object",
        "other_robot",
    ]
    assert smoke._allowed_hit_category_codes(targets) == frozenset({0, 1, 2, 3})


def test_product_query_candidates_stay_strictly_inside_public_limits() -> None:
    torch = pytest.importorskip("torch")
    current = torch.tensor(
        [[-0.25, 0.5, 2.0], [0.4, 0.5, 0.6]],
        dtype=torch.float32,
    )
    lower = (0.0, 0.0, 0.0)
    upper = (1.0, 1.0, 1.0)

    candidates, clamped_entries = smoke._interior_product_query_candidates(
        current,
        joint_lower=lower,
        joint_upper=upper,
    )

    assert candidates.shape == (2, 2, 3)
    assert candidates.is_contiguous()
    assert clamped_entries == 2
    assert torch.all(candidates > torch.tensor(lower))
    assert torch.all(candidates < torch.tensor(upper))
    assert torch.equal(candidates[:, 1], torch.full((2, 3), 0.5))


def test_snapshot_group_counts_require_physics_control_task_and_rng() -> None:
    counts = smoke._snapshot_group_counts(
        (
            "robot.q",
            "robot.qd",
            "robot.target",
            "robot.position_reference",
            "object.pose_local_wxyz",
            "object.com_velocity",
            "task.episode_length",
            "task.reward",
            "rng.key",
            "rng.counter",
        )
    )

    assert counts == {"physics": 4, "control": 2, "task": 2, "rng": 2}


@pytest.mark.parametrize(
    "missing",
    (
        "robot.q",
        "robot.target",
        "task.episode_length",
        "rng.key",
    ),
)
def test_snapshot_group_counts_reject_incomplete_state_families(
    missing: str,
) -> None:
    fields = {
        "robot.q",
        "robot.qd",
        "robot.target",
        "robot.position_reference",
        "object.pose_local_wxyz",
        "object.com_velocity",
        "task.episode_length",
        "rng.key",
    }
    fields.remove(missing)

    with pytest.raises(RuntimeError, match="does not cover all state families"):
        smoke._snapshot_group_counts(tuple(fields))


def test_runtime_transaction_signature_covers_query_lifecycle_state() -> None:
    marker = object()
    runtime = type(
        "RuntimeState",
        (),
        {
            "_phase": "idle",
            "_generation": 4,
            "_outstanding_token": marker,
            "_outstanding_phase": "issued",
            "_failed": False,
            "_fatal_error": None,
            "_closing_started": False,
            "_closed": False,
        },
    )()

    assert smoke._runtime_transaction_signature(runtime) == (
        "idle",
        4,
        marker,
        "issued",
        False,
        None,
        False,
        False,
    )


def test_session_spec_requires_physx_cuda_scene_query_support() -> None:
    spec = smoke._session_spec(cuda_device=2)

    assert spec.physics_kind == "physx_cuda"
    assert spec.compute.cuda_device == 2
    assert spec.physics.enable_scene_query_support is True
    assert spec.gravity_z == 0.0
    assert spec.add_ground is False


def test_concave_mesh_builder_has_three_closed_boxes() -> None:
    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []
    for center, half_extent in (
        ((-0.75, 0.0, 0.0), (0.25, 0.25, 1.0)),
        ((0.75, 0.0, 0.0), (0.25, 0.25, 1.0)),
        ((0.0, 0.0, -0.75), (0.5, 0.25, 0.25)),
    ):
        smoke._append_box(
            points,
            counts,
            indices,
            center=center,
            half_extent=half_extent,
        )

    assert len(points) == 24
    assert counts == [4] * 18
    assert len(indices) == 72
    assert min(indices) == 0
    assert max(indices) == 23


def test_shadow_probe_transform_keeps_non_unit_scale_and_translation() -> None:
    matrix = smoke._scaled_translation_matrix(translate_x=4.375)

    assert matrix[0] == smoke.SHADOW_SCALE_X == 0.5
    assert matrix[5] == 1.0
    assert matrix[10] == 1.0
    assert matrix[12:16] == (4.375, 0.0, 0.0, 1.0)


def test_geometry_semantics_accepts_empty_cavities_and_solid_bar_hit() -> None:
    smoke._assert_geometry_semantics(
        active_cavity_hits=(_hit(smoke.ACTIVE_CAVITY_PROBE_PATH),),
        shadow_cavity_hits=(),
        shadow_contact_hits=(_hit(smoke.SHADOW_CAVITY_PROBE_PATH),),
    )


def test_geometry_semantics_rejects_active_cavity_fill() -> None:
    with pytest.raises(AssertionError, match="active convexDecomposition"):
        smoke._assert_geometry_semantics(
            active_cavity_hits=(_hit(smoke.ACTIVE_CONCAVE_PATH),),
            shadow_cavity_hits=(),
            shadow_contact_hits=(_hit(smoke.SHADOW_CAVITY_PROBE_PATH),),
        )


def test_geometry_semantics_rejects_shadow_convex_hull_fallback() -> None:
    with pytest.raises(AssertionError, match="does not match"):
        smoke._assert_geometry_semantics(
            active_cavity_hits=(),
            shadow_cavity_hits=(_hit(smoke.SHADOW_CAVITY_PROBE_PATH),),
            shadow_contact_hits=(_hit(smoke.SHADOW_CAVITY_PROBE_PATH),),
        )


def test_geometry_semantics_requires_positive_solid_bar_hit() -> None:
    with pytest.raises(AssertionError, match="missed"):
        smoke._assert_geometry_semantics(
            active_cavity_hits=(),
            shadow_cavity_hits=(),
            shadow_contact_hits=(),
        )


def test_low_level_worker_reports_failure_and_closes_with_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Session:
        def __init__(self) -> None:
            self.close_codes: list[int] = []

        def close(self, *, exit_code: int = 0) -> None:
            self.close_codes.append(exit_code)

    session = _Session()
    monkeypatch.setattr(smoke, "in_runtime_worker", lambda: True)
    monkeypatch.setattr(
        smoke,
        "create_isaac_session_from_spec",
        lambda *, spec: session,
    )

    def fail_probe(_session: object) -> dict[str, object]:
        raise RuntimeError("probe geometry failed")

    monkeypatch.setattr(smoke, "_run_probe", fail_probe)

    with pytest.raises(RuntimeError, match="probe geometry failed"):
        smoke.main([])

    assert session.close_codes == [1]
    captured = capsys.readouterr()
    assert smoke.FAILURE_MARKER in captured.err
    assert '"error_type": "RuntimeError"' in captured.err
    assert '"mode": "low_level"' in captured.err
    assert smoke.PROBE_MARKER not in captured.out
