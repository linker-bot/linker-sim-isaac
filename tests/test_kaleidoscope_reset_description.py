from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.kaleidoscope.reset_description import (
    KaleidoscopeResetDescription,
    KaleidoscopeResetTemplate,
    make_reset_template,
    reset_json,
    sample_reset_description,
)


@pytest.fixture
def template():
    config = load_kaleidoscope_config()
    width = sum(len(r.resolved_profile.controlled_joints) for r in config.scene.robots)
    return make_reset_template(config, nominal_joint_positions=[0.0] * width)


def test_json_roundtrip_and_seed_stream_selection(template, tmp_path):
    restored = KaleidoscopeResetTemplate.from_mapping(json.loads(reset_json(template)))
    assert restored == template
    sample = sample_reset_description(
        restored, seed=42, source_env_ids=(9, 2), reset_index=3
    )
    path = tmp_path / "reset.json"
    path.write_text(reset_json(sample))
    loaded = KaleidoscopeResetDescription.from_mapping(json.loads(path.read_text()))
    assert loaded == sample
    # Logical RNG streams do not depend on the sampled batch's order or size.
    assert (
        sample.rows[1]
        == sample_reset_description(
            template, seed=42, source_env_ids=(2,), reset_index=3
        ).rows[0]
    )
    assert sample.rows[0].rng_counter == 4 * (len(template.joint_names) + 6)
    assert sample != sample_reset_description(
        template, seed=43, source_env_ids=(9, 2), reset_index=3
    )


def test_sampling_in_fresh_process_needs_no_torch_kit_or_cuda(template, tmp_path):
    path = tmp_path / "template.json"
    path.write_text(reset_json(template))
    script = """
import json, sys
from linkerbot_sim.kaleidoscope import KaleidoscopeResetTemplate, sample_reset_description, reset_json
from pathlib import Path
template = KaleidoscopeResetTemplate.from_mapping(json.loads(Path(sys.argv[1]).read_text()))
result = sample_reset_description(template, seed=-15, source_env_ids=(5,), reset_index=9)
assert not {'torch', 'warp', 'newton', 'omni', 'isaacsim', 'pxr'}.intersection(sys.modules)
print(reset_json(result))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", script, str(path)],
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert json.loads(result) == json.loads(
        reset_json(
            sample_reset_description(
                template, seed=-15, source_env_ids=(5,), reset_index=9
            )
        )
    )


def test_template_compatibility_ignores_only_gpu_and_environment_count(template):
    config = load_kaleidoscope_config()
    changed = replace(
        config,
        compute=replace(config.compute, cuda_device=3),
        environments=replace(config.environments, num_envs=2),
    )
    assert (
        make_reset_template(
            changed, nominal_joint_positions=template.nominal_joint_positions
        )
        == template
    )
    changed = replace(config, scene=replace(config.scene, physics_frequency_hz=960))
    assert (
        make_reset_template(
            changed, nominal_joint_positions=template.nominal_joint_positions
        ).fingerprint
        != template.fingerprint
    )
    other_backend = load_kaleidoscope_config(
        "configs/modes/kaleidoscope/newton_cuda.yaml"
    )
    assert (
        make_reset_template(
            other_backend, nominal_joint_positions=template.nominal_joint_positions
        ).fingerprint
        != template.fingerprint
    )
    assert (
        replace(template, control_mode="velocity").fingerprint != template.fingerprint
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future-v2"),
        ("configuration_fingerprint", "bad"),
        ("nominal_joint_positions", []),
        ("heading_axis", [0, 0, 0]),
        ("nominal_block_orientation_wxyz", [0, 0, 0, 0]),
        ("nominal_block_position", [0, float("nan"), 0]),
        ("control_mode", "invalid"),
        ("joint_names", ["same", "same"]),
        ("joint_names", "joint"),
    ],
)
def test_template_rejects_invalid_payload(template, field, value):
    data = template.as_dict()
    data[field] = value
    with pytest.raises((ValueError, TypeError)):
        KaleidoscopeResetTemplate.from_mapping(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rng_counter", -1),
        ("rng_key", 1 << 63),
        ("source_env_id", True),
        ("block_position", [0, float("inf"), 0]),
        ("goal_yaw", 1e300),
        ("block_orientation_wxyz", [0, 0, 0, 0]),
        ("joint_positions", []),
    ],
)
def test_description_rejects_invalid_rows(template, field, value):
    data = sample_reset_description(template, seed=0).as_dict()
    data["rows"][0][field] = value
    with pytest.raises((ValueError, TypeError)):
        KaleidoscopeResetDescription.from_mapping(data)


def test_description_rejects_unknown_fields_and_version(template):
    data = sample_reset_description(template, seed=0).as_dict()
    with pytest.raises(ValueError):
        KaleidoscopeResetDescription.from_mapping({**data, "extra": 1})
    with pytest.raises(ValueError):
        KaleidoscopeResetDescription.from_mapping({**data, "schema": "future"})
    with pytest.raises(ValueError):
        KaleidoscopeResetDescription.from_mapping({**data, "rows": []})


def test_restore_preflight_rejects_mismatched_width_and_degenerate_heading(template):
    sample = sample_reset_description(template, seed=0)
    sample.validate_template(template)
    with pytest.raises(ValueError, match="width"):
        replace(
            sample, rows=(replace(sample.rows[0], joint_positions=(0.0,)),)
        ).validate_template(template)
    # Rotate the configured Y heading axis vertically: finite/unit quaternion
    # alone is insufficient; reject before launching a GPU device assertion.
    bad_row = replace(
        sample.rows[0], block_orientation_wxyz=(2**-0.5, 2**-0.5, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="heading"):
        replace(sample, rows=(bad_row,)).validate_template(template)


def test_runtime_restore_preflight_and_writer_failure(monkeypatch, template):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from test_kaleidoscope_gpu_contracts import _task, _Views, _Session
    from linkerbot_sim.kaleidoscope.actions import (
        JointDeltaActionTerm,
        JointDeltaRuntimeAction,
    )
    from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime
    from linkerbot_sim.kaleidoscope.state_api import KaleidoscopeStateAPI, StateBinding

    task = _task(num_envs=3)
    views = _Views(3, command_dim=6)
    action = JointDeltaRuntimeAction(
        JointDeltaActionTerm(
            lower=torch.full((6,), -1.0, device="cuda"),
            upper=torch.full((6,), 1.0, device="cuda"),
            scale=torch.full((6,), 0.05, device="cuda"),
            clip=1.0,
            num_envs=3,
            target=views.target,
        ),
        physics_ticks_per_action=2,
    )
    state_api = KaleidoscopeStateAPI(
        {
            name: StateBinding(value, finite=value.is_floating_point())
            for name, value in task.state_fields().items()
        },
        num_envs=3,
    )
    runtime = KaleidoscopeRuntime(
        session=_Session(),
        views=views,
        task=task,
        action_term=action,
        state_api=state_api,
    )
    template = replace(
        template,
        joint_names=tuple(f"joint{i}" for i in range(6)),
        nominal_joint_positions=(0.0,) * 6,
        nominal_block_position=tuple(task._nominal_block_position.cpu().tolist()),
    )
    monkeypatch.setattr(runtime, "get_reset_template", lambda: template)
    runtime.reset()
    sample = sample_reset_description(template, seed=25, source_env_ids=(9,))
    ids = torch.tensor([1], device="cuda")
    before = {n: v.clone() for n, v in task.state_fields().items()}
    writes = len(views.reset_selectors)
    with pytest.raises(ValueError, match="mismatch"):
        runtime.restore_reset_description(
            replace(sample, template_fingerprint="0" * 64), target_env_ids=ids
        )
    assert len(views.reset_selectors) == writes and not runtime._failed
    for name, value in task.state_fields().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    token = runtime.issue_same_step_token()
    with pytest.raises(RuntimeError):
        runtime.restore_reset_description(sample, target_env_ids=ids)
    runtime.tokenized_step(token, torch.zeros((3, 6), device="cuda"))
    runtime.complete_same_step_reset(token)

    def fail_writer(_command):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(views, "write_reset", fail_writer)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.restore_reset_description(sample, target_env_ids=ids)
    assert runtime._failed
    with pytest.raises(RuntimeError):
        runtime.reset()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": 1 << 63},
        {"source_env_ids": ()},
        {"source_env_ids": (0, 0)},
        {"source_env_ids": (-1,)},
        {"reset_index": -1},
        {"reset_index": (1 << 63) - 1},
    ],
)
def test_sampler_rejects_invalid_stream_selection(template, kwargs):
    with pytest.raises(ValueError):
        sample_reset_description(template, **({"seed": 0} | kwargs))


@pytest.mark.parametrize("seed", [0, -15, (1 << 63) - 1])
@pytest.mark.parametrize("reset_index", [0, 1, 7])
def test_cpu_sample_matches_cuda_reset_and_counter(template, seed, reset_index):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from linkerbot_sim.kaleidoscope.tasks.tblock_push_v1 import (
        TBlockPushV1,
        TBlockPushV1Settings,
    )

    config = load_kaleidoscope_config()
    task = TBlockPushV1(
        num_envs=10,
        command_dim=len(template.joint_names),
        action_dim=len(template.joint_names),
        robot_count=2,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        nominal_joint_positions=torch.tensor(
            template.nominal_joint_positions, device="cuda"
        ),
        nominal_block_position=torch.tensor(
            template.nominal_block_position, device="cuda"
        ),
        nominal_block_orientation_wxyz=torch.tensor(
            template.nominal_block_orientation_wxyz, device="cuda"
        ),
        settings=TBlockPushV1Settings.from_configuration(config.task),
    )
    ids = torch.tensor([9, 2, 0], device="cuda", dtype=torch.int64)
    task.reseed(seed)
    for _ in range(reset_index + 1):
        command = task.reset_command(ids)
    sample = sample_reset_description(
        template, seed=seed, source_env_ids=(9, 2, 0), reset_index=reset_index
    )
    for name in (
        "joint_positions",
        "block_position",
        "block_orientation_wxyz",
        "goal_position",
        "goal_yaw",
    ):
        expected = np.asarray(
            [getattr(row, name) for row in sample.rows], dtype=np.float32
        )
        np.testing.assert_allclose(
            getattr(command, name).cpu().numpy(), expected, rtol=0, atol=3e-7
        )
    assert task.buffers.rng_key[ids].tolist() == [row.rng_key for row in sample.rows]
    assert task.buffers.rng_counter[ids].tolist() == [
        row.rng_counter for row in sample.rows
    ]
