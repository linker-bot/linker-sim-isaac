from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.configs.runtime import (
    RuntimeProfileConfig,
    load_runtime_profile,
    resolve_runtime_config,
)
from linkerbot_sim.tiled.config import TiledEnvConfig


def _profile(
    runtime: dict[str, object],
    *,
    name: str = "unit",
    add_tiled_scene_telemetry_scope: bool = True,
) -> RuntimeProfileConfig:
    runtime = dict(runtime)
    if add_tiled_scene_telemetry_scope and runtime.get("mode") == "tiled_scene":
        telemetry = runtime.get("telemetry", {})
        if isinstance(telemetry, dict):
            telemetry = dict(telemetry)
            telemetry.setdefault("primary_env_id", 0)
            telemetry.setdefault("selected_env_ids", [0])
            runtime["telemetry"] = telemetry
    return RuntimeProfileConfig.from_mapping(
        {"runtime": runtime},
        profile_name=name,
        source_path=f"/profiles/{name}.yaml",
    )


def _resolve(
    profile: RuntimeProfileConfig,
    *,
    cli: dict[str, object] | None = None,
    env: dict[str, object] | None = None,
    expected_mode: str | None = None,
):
    return resolve_runtime_config(
        profile,
        cli_overrides={} if cli is None else cli,
        env_config={"env": {}} if env is None else env,
        expected_mode=expected_mode,
    )


def test_bundled_single_scene_default_resolves() -> None:
    profile = load_runtime_profile("default_single_scene")
    resolved = resolve_runtime_config(
        profile,
        cli_overrides={},
        env_config=load_profile_yaml("env", profile.profiles.env),
        expected_mode="single_scene",
    )

    assert resolved.mode == "single_scene"
    assert resolved.profiles.env == "scene1"
    assert resolved.profiles.curobo == "default"
    assert resolved.profiles.logging == "default_logger"
    assert resolved.simulation_app.gui is False
    assert resolved.execution.control_mode == "position"
    assert resolved.execution.idle_physics_policy == "hold_step"
    assert resolved.interactive.stdin_eof_policy == "exit"
    assert resolved.planner.backend == "curobo"
    assert resolved.telemetry.rate_hz == 60.0
    assert resolved.telemetry.joint_effort_field == "none"


def test_bundled_tiled_scene_default_resolves() -> None:
    profile = load_runtime_profile("default_tiled_scene")
    resolved = resolve_runtime_config(
        profile,
        cli_overrides={},
        env_config=load_profile_yaml("env", profile.profiles.env),
        expected_mode="tiled_scene",
    )

    assert resolved.profiles.env == "scene3_tiled"
    assert resolved.execution.default_decimation == 2
    assert resolved.execution.idle_physics_policy == "pause"
    assert resolved.planner.resources.max_workers == 2
    assert resolved.planner.resources.max_pending_requests == 64
    assert resolved.planner.resources.max_completed_results == 256
    assert resolved.planner.joint_batch_mode == "auto"
    assert resolved.sources["runtime.planner.backend"] == "runtime:default_tiled_scene"
    assert resolved.telemetry.rate_hz == 10.0
    assert resolved.telemetry.selected_env_ids == (0,)
    assert resolved.telemetry.publish_decimation == 1


def test_entrypoint_mode_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _resolve(
            load_runtime_profile("default_tiled_scene"),
            expected_mode="single_scene",
        )


@pytest.mark.parametrize("unsupported_profile", ("default", "default_tiled"))
def test_unsupported_runtime_profile_names_are_not_resolved(
    unsupported_profile: str,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_runtime_profile(unsupported_profile)


@pytest.mark.parametrize("unsupported_mode", ("scene", "tiled"))
def test_runtime_rejects_unsupported_mode_values(unsupported_mode: str) -> None:
    with pytest.raises(ValueError, match="runtime.mode"):
        _profile({"mode": unsupported_mode})


@pytest.mark.parametrize("control_mode", ("velocity", "effort"))
def test_tiled_scene_runtime_rejects_unimplemented_control_modes(
    control_mode: str,
) -> None:
    with pytest.raises(ValueError, match="must be 'position' in tiled_scene mode"):
        _profile(
            {
                "mode": "tiled_scene",
                "execution": {"control_mode": control_mode},
            }
        )


@pytest.mark.parametrize(
    ("runtime", "error"),
    (
        (
            {
                "mode": "single_scene",
                "planner": {"request_defaults": {"coordination": "coupled"}},
            },
            "no coupled planner backend",
        ),
        (
            {
                "mode": "tiled_scene",
                "planner": {"request_defaults": {"coordination": "static_others"}},
            },
            "must be 'independent' in tiled_scene mode",
        ),
        (
            {
                "mode": "tiled_scene",
                "planner": {"request_defaults": {"force_collision_refresh": True}},
            },
            "isolated context",
        ),
        (
            {
                "mode": "single_scene",
                "planner": {
                    "backend": "linear",
                    "request_defaults": {"avoid_collisions": True},
                },
            },
            "linear.*avoid_collisions",
        ),
    ),
)
def test_runtime_rejects_unimplemented_planner_default_combinations(
    runtime: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _profile(runtime)


def test_cli_explicit_values_override_yaml_and_none_does_not() -> None:
    profile = _profile(
        {
            "mode": "single_scene",
            "simulation_app": {"gui": True},
            "telemetry": {"include_objects": True},
        }
    )

    inherited = _resolve(
        profile,
        cli={"simulation_app.gui": None, "telemetry.include_objects": None},
    )
    overridden = _resolve(
        profile,
        cli={"simulation_app.gui": False, "telemetry.include_objects": False},
    )

    assert inherited.simulation_app.gui is True
    assert inherited.telemetry.include_objects is True
    assert overridden.simulation_app.gui is False
    assert overridden.telemetry.include_objects is False
    assert overridden.sources["runtime.simulation_app.gui"] == "cli"


@pytest.mark.parametrize(
    ("runtime", "path"),
    [
        ({"simulaton_app": {}}, "runtime.simulaton_app"),
        (
            {"simulation_app": {"gpu": {"active_gup": 0}}},
            "runtime.simulation_app.gpu.active_gup",
        ),
        (
            {"planner": {"resources": {"max_worker": 1}}},
            "runtime.planner.resources.max_worker",
        ),
        (
            {"telemetry": {"mcap": {"unknown": False}}},
            "runtime.telemetry.mcap.unknown",
        ),
    ],
)
def test_unknown_key_reports_complete_leaf_path(
    runtime: dict[str, object], path: str
) -> None:
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        _profile(runtime)


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_boolean_fields_are_strict(value: object) -> None:
    with pytest.raises(ValueError, match="runtime.simulation_app.gui"):
        _profile({"simulation_app": {"gui": value}})


@pytest.mark.parametrize(
    ("telemetry", "error"),
    (
        ({"selected_env_ids": "0"}, "non-empty sequence"),
        ({"selected_env_ids": []}, "must be non-empty"),
        ({"selected_env_ids": [0, 0]}, "cannot contain duplicates"),
        ({"selected_env_ids": [True]}, r"selected_env_ids\[0\]"),
        ({"publish_decimation": 0}, "positive integer"),
        ({"publish_decimation": True}, "non-negative integer"),
        ({"joint_effort_field": "estimated"}, "joint_effort_field"),
    ),
)
def test_telemetry_runtime_fields_are_strict(
    telemetry: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _profile({"mode": "tiled_scene", "telemetry": telemetry})


@pytest.mark.parametrize("missing", ("primary_env_id", "selected_env_ids"))
def test_tiled_scene_runtime_requires_explicit_telemetry_selection(
    missing: str,
) -> None:
    telemetry: dict[str, object] = {
        "primary_env_id": 0,
        "selected_env_ids": [0],
    }
    telemetry.pop(missing)

    with pytest.raises(ValueError, match=rf"runtime\.telemetry\.{missing} is required"):
        _profile(
            {"mode": "tiled_scene", "telemetry": telemetry},
            add_tiled_scene_telemetry_scope=False,
        )


@pytest.mark.parametrize(
    ("runtime", "error"),
    (
        (
            {"mode": "single_scene", "telemetry": {"selected_env_ids": [1]}},
            "selected_env_ids.*only supported in tiled_scene mode",
        ),
        (
            {"mode": "single_scene", "telemetry": {"publish_decimation": 2}},
            "publish_decimation.*only supported in tiled_scene mode",
        ),
        (
            {
                "mode": "tiled_scene",
                "telemetry": {
                    "include_efforts": True,
                    "joint_effort_field": "measured",
                },
            },
            "joint_effort_field is only supported in single_scene mode",
        ),
        (
            {
                "mode": "single_scene",
                "telemetry": {"joint_effort_field": "measured"},
            },
            "joint_effort_field requires include_efforts=true",
        ),
    ),
)
def test_runtime_rejects_mode_incompatible_telemetry_fields(
    runtime: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _profile(runtime)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("planner.resources.max_workers", 0, "max_workers"),
        ("planner.resources.max_pending_requests", -1, "max_pending_requests"),
        ("interactive.transport.request_queue_capacity", 0, "request_queue_capacity"),
        ("interactive.command_history_capacity", -1, "command_history_capacity"),
        ("interactive.snapshot_request_capacity", 0, "snapshot_request_capacity"),
        ("simulation_app.render.gui_size", [1280, 0], "gui_size"),
        ("interactive.transport.tcp_jsonl.port", 65536, "port"),
    ],
)
def test_invalid_ranges_report_field_path(
    path: str, value: object, message: str
) -> None:
    keys = path.split(".")
    runtime: dict[str, object] = {}
    current = runtime
    for key in keys[:-1]:
        nested: dict[str, object] = {}
        current[key] = nested
        current = nested
    current[keys[-1]] = value
    with pytest.raises(ValueError, match=message):
        _profile(runtime)


@pytest.mark.parametrize("endpoint", ("tcp_jsonl", "websocket", "foxglove_live"))
@pytest.mark.parametrize("host", ("127.0.0.1", "127.0.0.2", "::1", "localhost"))
def test_runtime_listener_hosts_accept_only_explicit_loopback_values(
    endpoint: str,
    host: str,
) -> None:
    if endpoint == "foxglove_live":
        profile = _profile({"telemetry": {endpoint: {"host": host}}})
        assert profile.telemetry.foxglove_live.host == host
    else:
        profile = _profile({"interactive": {"transport": {endpoint: {"host": host}}}})
        assert getattr(profile.interactive.transport, endpoint).host == host


@pytest.mark.parametrize("endpoint", ("tcp_jsonl", "websocket", "foxglove_live"))
@pytest.mark.parametrize(
    "host",
    ("0.0.0.0", "::", "192.0.2.10", "example.invalid"),
)
def test_runtime_listener_hosts_reject_non_loopback_values(
    endpoint: str,
    host: str,
) -> None:
    runtime = (
        {"telemetry": {endpoint: {"host": host}}}
        if endpoint == "foxglove_live"
        else {"interactive": {"transport": {endpoint: {"host": host}}}}
    )
    with pytest.raises(ValueError, match="loopback"):
        _profile(runtime)


def test_paths_reject_parent_traversal_and_accept_nullable_clear() -> None:
    with pytest.raises(ValueError, match="cache_root"):
        _profile({"paths": {"cache_root": "../private"}})

    assert _profile({"paths": {"cache_root": None}}).paths.cache_root is None
    with pytest.raises(ValueError, match="profiles.env"):
        _profile({"profiles": {"env": None}})

    assert (
        _profile(
            {"interactive": {"command_history_capacity": 0}}
        ).interactive.command_history_capacity
        == 0
    )


@pytest.mark.parametrize(
    "runtime",
    [
        {"simulation_app": None},
        {"simulation_app": {"gpu": None}},
        {"planner": None},
        {"interactive": {"transport": None}},
    ],
)
def test_non_nullable_sections_reject_explicit_null(
    runtime: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _profile(runtime)


def test_unknown_tiled_runtime_section_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"unsupported keys: runtime.*tiled\.runtime"):
        TiledEnvConfig.from_env_config(
            {
                "tiled": {"runtime": {"planner": {"backend": "linear"}}},
            }
        )


def test_mapping_order_and_provenance_do_not_change_fingerprint() -> None:
    first = _resolve(
        _profile(
            {
                "mode": "single_scene",
                "execution": {"default_decimation": 3, "control_mode": "position"},
            },
            name="one",
        )
    )
    second = _resolve(
        _profile(
            {
                "execution": {"control_mode": "position", "default_decimation": 3},
                "mode": "single_scene",
            },
            name="two",
        )
    )
    changed = _resolve(
        _profile({"mode": "single_scene", "execution": {"default_decimation": 4}})
    )

    assert first.fingerprint == second.fingerprint
    assert first.sources != second.sources
    assert changed.fingerprint != first.fingerprint
    json.dumps(first.as_dict(), allow_nan=False)
    assert set(first.sources) == set(second.sources)
    assert first.sources["runtime.camera_output.rgb_format"] == "default"


def test_empty_mapping_does_not_claim_default_leaf_provenance() -> None:
    resolved = _resolve(_profile({"planner": {}}, name="empty"))

    assert resolved.sources["runtime.planner.backend"] == "default"


def test_list_override_replaces_whole_value() -> None:
    resolved = _resolve(
        _profile({"simulation_app": {"render": {"gui_size": [100, 200]}}}),
        cli={"simulation_app.render.gui_size": [300, 400]},
    )
    assert resolved.simulation_app.render.gui_size == (300, 400)


def test_max_batch_problems_cannot_exceed_selected_curobo_capacity() -> None:
    with pytest.raises(ValueError, match="max_batch_problems"):
        _resolve(
            _profile(
                {
                    "planner": {
                        "backend": "curobo",
                        "resources": {"max_batch_problems": 257},
                    }
                }
            )
        )


def test_max_batch_problems_auto_resolves_to_linear_num_envs() -> None:
    profile = _profile(
        {
            "planner": {
                "backend": "linear",
                "resources": {"max_batch_problems": "auto"},
            }
        }
    )
    assert profile.planner.resources.max_batch_problems == "auto"

    resolved = _resolve(profile, env={"tiled": {"num_envs": 7}})
    explicit = _resolve(
        _profile(
            {
                "planner": {
                    "backend": "linear",
                    "resources": {"max_batch_problems": 7},
                }
            }
        ),
        env={"tiled": {"num_envs": 7}},
    )

    assert resolved.planner.resources.max_batch_problems == 7
    assert resolved.sources["runtime.planner.resources.max_batch_problems"] == (
        "runtime:unit"
    )
    assert resolved.fingerprint == explicit.fingerprint


def test_max_batch_problems_auto_uses_smallest_relevant_curobo_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "linkerbot_sim.configs.profiles.load_profile_yaml",
        lambda group, name: {
            "curobo": {
                "kinematics": {"ik": {"max_batch_size": 128}},
                "motion_planner": {"max_batch_size": 32},
            }
        },
    )
    auto = _resolve(
        _profile(
            {
                "planner": {
                    "backend": "curobo",
                    "resources": {"max_batch_problems": "auto"},
                }
            }
        ),
        env={"tiled": {"num_envs": 64}},
    )
    assert auto.planner.resources.max_batch_problems == 32

    with pytest.raises(ValueError, match=r"max_batch_size \(32\)"):
        _resolve(
            _profile(
                {
                    "planner": {
                        "backend": "curobo",
                        "resources": {"max_batch_problems": 33},
                    }
                }
            ),
            env={"tiled": {"num_envs": 64}},
        )


def test_max_batch_problems_auto_applies_defaults_for_omitted_curobo_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "linkerbot_sim.configs.profiles.load_profile_yaml",
        lambda group, name: {
            "curobo": {
                "kinematics": {"ik": {"max_batch_size": 128}},
                "motion_planner": {},
            }
        },
    )

    resolved = _resolve(
        _profile(
            {
                "planner": {
                    "backend": "curobo",
                    "resources": {"max_batch_problems": "auto"},
                }
            }
        ),
        env={"tiled": {"num_envs": 300}},
    )

    assert resolved.planner.resources.max_batch_problems == 128


@pytest.mark.parametrize("value", ("AUTO", "64", 0, 1.5, True))
def test_max_batch_problems_rejects_invalid_auto_or_integer(value: object) -> None:
    with pytest.raises(ValueError, match="max_batch_problems"):
        _profile({"planner": {"resources": {"max_batch_problems": value}}})


def test_runtime_planner_and_playback_policies_are_strictly_validated() -> None:
    with pytest.raises(
        ValueError,
        match="hold_failed_env.*reject_request",
    ):
        _profile({"planner": {"failure_policy": "atomic"}})
    with pytest.raises(ValueError, match="runtime.planner.oversize_request_policy"):
        _profile({"planner": {"oversize_request_policy": "allow"}})
    with pytest.raises(ValueError, match="runtime.playback.max_duration_s_per_env"):
        _profile({"playback": {"max_duration_s_per_env": 0.0}})


def test_runtime_playback_limits_and_sync_ik_failure_policy_resolve() -> None:
    resolved = _resolve(
        _profile(
            {
                "planner": {
                    "oversize_request_policy": "reject",
                    "failure_policy": "reject_request",
                },
                "playback": {
                    "max_queue_depth_per_env": 4,
                    "max_samples_per_env": 123,
                    "max_duration_s_per_env": 4.5,
                    "overflow_policy": "reject",
                },
            }
        )
    )

    assert resolved.planner.oversize_request_policy == "reject"
    assert resolved.planner.failure_policy == "reject_request"
    assert resolved.playback.max_queue_depth_per_env == 4
    assert resolved.playback.max_samples_per_env == 123
    assert resolved.playback.max_duration_s_per_env == 4.5


def test_camera_output_and_file_policies_resolve_as_typed_settings() -> None:
    resolved = _resolve(
        _profile(
            {
                "camera_output": {
                    "queue_size": 7,
                    "overflow_policy": "error",
                    "worker_poll_interval_s": 0.02,
                    "existing_data_policy": "timestamped_dir",
                    "shutdown_policy": "abort",
                    "rgb_format": "png",
                    "depth_format": "npz",
                    "metadata_flush_interval_frames": 8,
                    "max_bytes_per_camera": 123_456,
                },
                "output": {
                    "csv_existing_file_policy": "resume",
                    "mcap_existing_file_policy": "timestamped_dir",
                },
                "shutdown": {"camera_publisher_timeout_s": 0.25},
            }
        )
    )

    assert resolved.camera_output.queue_size == 7
    assert resolved.camera_output.overflow_policy == "error"
    assert resolved.camera_output.worker_poll_interval_s == pytest.approx(0.02)
    assert resolved.camera_output.existing_data_policy == "timestamped_dir"
    assert resolved.camera_output.shutdown_policy == "abort"
    assert resolved.shutdown.camera_publisher_timeout_s == pytest.approx(0.25)
    assert resolved.camera_output.rgb_format == "png"
    assert resolved.camera_output.depth_format == "npz"
    assert resolved.camera_output.metadata_flush_interval_frames == 8
    assert resolved.camera_output.max_bytes_per_camera == 123_456
    assert resolved.output.csv_existing_file_policy == "resume"
    assert resolved.output.mcap_existing_file_policy == "timestamped_dir"


def test_telemetry_settings_resolve_all_runtime_owned_fields() -> None:
    resolved = _resolve(
        _profile(
            {
                "telemetry": {
                    "primary_env_id": 0,
                    "rate_hz": 12.5,
                    "buffer_size": 7,
                    "drop_policy": "drop_newest",
                    "on_error": "continue",
                    "include_joint_states": False,
                    "include_state_json": True,
                    "include_scene_markers": False,
                    "include_efforts": True,
                    "include_objects": True,
                    "joint_effort_field": "measured",
                    "topics": {
                        "joint_states": "/run_2/joints",
                        "scene": "/run_2/markers",
                        "state": "/run_2/state",
                    },
                    "mcap": {"path": "logs/run_2/state.mcap"},
                    "foxglove_live": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": 9876,
                    },
                },
                "output": {"mcap_existing_file_policy": "truncate"},
            }
        )
    )

    telemetry = resolved.telemetry
    assert telemetry.rate_hz == pytest.approx(12.5)
    assert telemetry.buffer_size == 7
    assert telemetry.drop_policy == "drop_newest"
    assert telemetry.on_error == "continue"
    assert telemetry.include_joint_states is False
    assert telemetry.include_state_json is True
    assert telemetry.include_scene_markers is False
    assert telemetry.include_efforts is True
    assert telemetry.include_objects is True
    assert telemetry.joint_effort_field == "measured"
    assert telemetry.topics.joint_states == "/run_2/joints"
    assert telemetry.topics.scene == "/run_2/markers"
    assert telemetry.topics.state == "/run_2/state"
    assert telemetry.mcap.path == "logs/run_2/state.mcap"
    assert telemetry.foxglove_live.host == "127.0.0.1"
    assert telemetry.foxglove_live.port == 9876
    assert resolved.output.mcap_existing_file_policy == "truncate"


def test_tiled_scene_telemetry_selection_resolves() -> None:
    resolved = _resolve(
        _profile(
            {
                "mode": "tiled_scene",
                "telemetry": {
                    "primary_env_id": 3,
                    "selected_env_ids": [1, 3],
                    "publish_decimation": 4,
                },
            }
        ),
        env={"tiled": {"num_envs": 4}},
    )

    assert resolved.telemetry.primary_env_id == 3
    assert resolved.telemetry.selected_env_ids == (1, 3)
    assert resolved.telemetry.publish_decimation == 4


def test_tiled_scene_telemetry_primary_env_must_be_in_runtime_range() -> None:
    profile = _profile(
        {
            "mode": "tiled_scene",
            "profiles": {"env": "scene3_tiled"},
            "telemetry": {"primary_env_id": 2, "selected_env_ids": [2]},
        }
    )

    with pytest.raises(ValueError, match="primary_env_id.*tiled.num_envs"):
        _resolve(profile, env={"tiled": {"num_envs": 2}})


def test_tiled_scene_telemetry_selection_must_include_primary_and_stay_in_range() -> (
    None
):
    missing_primary = _profile(
        {
            "mode": "tiled_scene",
            "telemetry": {"primary_env_id": 0, "selected_env_ids": [1]},
        }
    )
    out_of_range = _profile(
        {
            "mode": "tiled_scene",
            "telemetry": {"primary_env_id": 0, "selected_env_ids": [0, 2]},
        }
    )

    with pytest.raises(ValueError, match="primary_env_id.*included"):
        _resolve(missing_primary, env={"tiled": {"num_envs": 2}})
    with pytest.raises(ValueError, match="selected_env_ids.*tiled.num_envs"):
        _resolve(out_of_range, env={"tiled": {"num_envs": 2}})


def test_telemetry_rejects_duplicate_topics() -> None:
    with pytest.raises(ValueError, match="distinct topic paths"):
        _profile(
            {
                "telemetry": {
                    "topics": {
                        "joint_states": "/same",
                        "scene": "/same",
                        "state": "/state",
                    }
                }
            }
        )


def test_scene_markers_require_object_sampling_when_output_is_enabled() -> None:
    with pytest.raises(ValueError, match="include_scene_markers.*include_objects"):
        _profile(
            {
                "telemetry": {
                    "include_joint_states": False,
                    "include_state_json": False,
                    "include_scene_markers": True,
                    "include_objects": False,
                    "mcap": {"path": "state.mcap"},
                }
            }
        )

    _profile(
        {
            "mode": "tiled_scene",
            "telemetry": {
                "include_joint_states": False,
                "include_state_json": False,
                "include_scene_markers": True,
                "include_objects": False,
                "mcap": {"path": "state.mcap"},
            },
        }
    )
    with pytest.raises(ValueError, match="at least one output modality"):
        _profile(
            {
                "telemetry": {
                    "include_joint_states": False,
                    "include_state_json": False,
                    "include_scene_markers": False,
                    "mcap": {"path": "state.mcap"},
                }
            }
        )


@pytest.mark.parametrize(
    ("runtime", "field"),
    (
        ({"camera_output": {"rgb_format": "jpeg"}}, "rgb_format"),
        ({"camera_output": {"depth_format": "exr"}}, "depth_format"),
        (
            {"camera_output": {"metadata_flush_interval_frames": 0}},
            "metadata_flush_interval_frames",
        ),
        (
            {"camera_output": {"max_bytes_per_camera": 0}},
            "max_bytes_per_camera",
        ),
        (
            {"output": {"csv_existing_file_policy": "timestamped"}},
            "csv_existing_file_policy",
        ),
        (
            {"camera_output": {"shutdown_timeout_s": 2.0}},
            "runtime.shutdown.camera_publisher_timeout_s",
        ),
    ),
)
def test_camera_output_and_file_policies_reject_unsupported_values(
    runtime: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _profile(runtime)


@pytest.mark.parametrize(
    "policy",
    ("error", "truncate", "resume", "timestamped_dir"),
)
@pytest.mark.parametrize(
    "runtime",
    (
        {"camera_output": {"existing_data_policy": None}},
        {"output": {"csv_existing_file_policy": None}},
        {"output": {"mcap_existing_file_policy": None}},
    ),
)
def test_all_file_policy_owners_share_the_same_enum(
    runtime: dict[str, object],
    policy: str,
) -> None:
    section = next(iter(runtime.values()))
    assert isinstance(section, dict)
    key = next(iter(section))
    section[key] = policy

    _profile(runtime)


@pytest.mark.parametrize(
    "runtime",
    (
        {"camera_output": {"existing_data_policy": "overwrite"}},
        {"output": {"csv_existing_file_policy": "overwrite"}},
        {"output": {"mcap_existing_file_policy": "overwrite"}},
    ),
)
def test_all_file_policy_owners_reject_values_outside_the_shared_enum(
    runtime: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="policy"):
        _profile(runtime)


def test_runtime_config_import_does_not_load_heavy_runtime_modules() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import linkerbot_sim.configs.runtime; "
        "print(sorted(name for name in sys.modules "
        "if name == 'torch' or name.startswith('curobo') "
        "or name.startswith('isaacsim') or name.startswith('omni')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"
