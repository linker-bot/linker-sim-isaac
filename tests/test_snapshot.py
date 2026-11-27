"""Snapshot files, headers and the configuration fingerprint (CPU). Restoring is tested on the
GPU (tests/gpu/test_snapshot_gpu.py)."""

import pickle
import subprocess
import sys

import pytest
import torch

from linker_lab import snapshot


def example() -> snapshot.Snapshot:
    header = {"schema": snapshot.SCHEMA, "packages": {"torch": "2.11.0"}, "config": "abc", "linker_lab_commit": None,
              "scene": {"articulations": ["left"], "rigid_objects": []}, "actions": {"left_arm": 7}, "created": "now"}
    state = {"control": {"left_arm": {"raw": torch.arange(7.0)[None], "step": 3}},
             "environment": {"rng_python": [[1, 2, 3], None], "sim_time": 6.0}}
    return snapshot.Snapshot(header, state)


def test_save_and_load_round_trip(tmp_path):
    s = example()
    snapshot.save(s, tmp_path / "s.pt")
    loaded = snapshot.load(tmp_path / "s.pt")
    assert loaded.header == s.header
    assert torch.equal(loaded.state["control"]["left_arm"]["raw"], s.state["control"]["left_arm"]["raw"])
    assert loaded.state["control"]["left_arm"]["step"] == 3
    assert loaded.state["environment"] == s.state["environment"]


def test_load_refuses_other_files_and_schemas(tmp_path):
    torch.save({"weights": torch.zeros(3)}, tmp_path / "other.pt")
    with pytest.raises(snapshot.SnapshotError, match="not a linker-lab snapshot") as error:
        snapshot.load(tmp_path / "other.pt")
    assert error.value.physics_modified is False
    s = example()
    snapshot.save(snapshot.Snapshot({**s.header, "schema": 99}, s.state), tmp_path / "future.pt")
    with pytest.raises(snapshot.SnapshotError, match="schema 99"):
        snapshot.load(tmp_path / "future.pt")


class Payload:
    def __reduce__(self):
        return (print, ("code ran",))


def test_load_runs_no_code(tmp_path, capsys):
    torch.save({"header": Payload(), "state": {}}, tmp_path / "evil.pt")
    with pytest.raises(pickle.UnpicklingError):
        snapshot.load(tmp_path / "evil.pt")
    assert "code ran" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("packages", {"torch": "2.12.0"}, "other package versions"),
        ("config", "def", "configuration differs"),
        ("scene", {"articulations": ["left", "right"], "rigid_objects": []}, "scene differs"),
        ("actions", {"left_arm": 14}, "actions differs"),
        ("linker_lab_commit", "1" * 40, "taken with linker-lab"),
    ],
)
def test_header_mismatch_names_the_field(field, value, message):
    current = {**example().header, "linker_lab_commit": "0" * 40}
    with pytest.raises(snapshot.SnapshotError, match=message) as error:
        snapshot._check_header(current, {**current, field: value})
    assert error.value.physics_modified is False


def test_header_ignores_creation_time_and_an_unknown_commit():
    current = example().header
    snapshot._check_header({**current, "linker_lab_commit": "0" * 40}, {**current, "created": "later", "linker_lab_commit": None})


def test_state_layout_mismatch_names_the_entry():
    s = example().state
    bad = {**s, "control": {"left_arm": {"raw": torch.zeros(1, 14), "step": 3}}}
    with pytest.raises(ValueError, match=r"state.control.left_arm.raw"):
        snapshot._check_tree(s, bad, "state")
    with pytest.raises(ValueError, match="keys differ"):
        snapshot._check_tree(s, {"control": s["control"]}, "state")


@pytest.fixture(scope="module")
def cfg():
    from linker_lab import workstation

    return workstation.LWv0EnvCfg


def test_fingerprint_ignores_exactly_the_viewer_fields(cfg):
    base = snapshot._fingerprint(cfg())
    c = cfg()
    c.viewer.eye = (9.0, 9.0, 9.0)
    assert snapshot._fingerprint(c) == base
    assert snapshot.UNCOMPARED_CONFIG == ("viewer", "sim.visualizer_cfgs")
    for change in (lambda c: setattr(c, "episode_length_s", 20.0), lambda c: setattr(c.scene, "num_envs", 2),
                   lambda c: setattr(c.actions.left_arm, "interpolation", False)):
        c = cfg()
        change(c)
        assert snapshot._fingerprint(c) != base


def test_fingerprint_is_the_same_in_a_new_process(cfg):
    code = "import linker_lab; linker_lab.bootstrap(); from linker_lab import snapshot, workstation; print(snapshot._fingerprint(workstation.LWv0EnvCfg()))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()[-1]
    assert out == snapshot._fingerprint(cfg())
