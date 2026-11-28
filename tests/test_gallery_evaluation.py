"""Grasp outcomes and evaluation accounting, without a simulated GPU scene."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from linker_lab import gallery

SPEC = importlib.util.spec_from_file_location(
    "evaluate_gallery", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_gallery.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


@pytest.fixture(scope="module")
def play():
    return gallery.load("ycb-grasp")


@pytest.mark.parametrize("failure", [None, "plan", "finite", "in range", "risen", "alone"])
def test_physical_checks_produce_a_structured_outcome(play, failure):
    scene = play.GraspScene.__new__(play.GraspScene)
    scene.demo = {"end": 20, "hold_start": 10}
    scene.k = 20
    scene.active, scene.sensor = 0, [0]
    scene.items = [{"name": "test", "label": "Object"}]
    scene.plan_valid = torch.tensor([failure != "plan"])
    scene.ok = dict.fromkeys(("finite", "in range", "risen", "alone"), True)
    # A prior failure during the hold must not be cleared by a valid last sample.
    if failure in scene.ok:
        scene.ok[failure] = False
    scene.lower, scene.upper = torch.full((13,), -1.0), torch.ones(13)
    scene._joints = lambda: torch.zeros(1, 13)
    scene.settled_pos = torch.zeros(1, 3)
    scene._object_pose = lambda: (torch.tensor([[0.0, 0.0, 0.11]]), torch.tensor([[0, 0, 0, 1]]))
    scene.contact = SimpleNamespace(data=SimpleNamespace(
        net_forces_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)),
        force_matrix_w=SimpleNamespace(torch=torch.zeros(1, 1, 1, 3)),
    ))
    scene._grasp_check()
    assert scene.state == "done"
    assert scene.result["success"] is (failure is None)
    assert (scene.result["reason"] is None) is (failure is None)
    assert scene.message.endswith(".")  # Both outcomes did satisfy the old assertion.


class Scene:
    """A scripted scene that processes queued buttons only when stepped."""

    def __init__(self, *, success=True, settle=None, grasp=None):
        self.items = [{"name": "object"}]
        self.state = "done"
        self.result = {"success": True}  # must not reuse a previous trial
        self.attempt = 0
        self.message = "Object: missed."
        self.success = success
        self.settle = ["settling", "ready"] if settle is None else settle
        self.grasp = ["grasping", "done"] if grasp is None else grasp
        self._states = iter(())
        self.draws = []

    def press(self, button, item=None):
        self._states = iter(self.settle if button == "drop" else self.grasp)
        if button == "drop":
            self.result = None
            self.draws.append(self.rng.uniform())

    def step(self):
        self.state = next(self._states, self.state)
        if self.state == "done":
            self.result = {"success": self.success, "plan_valid": True,
                           "checks": {"risen": self.success},
                           "reason": None if self.success else "not lifted"}

    def _object_pose(self):
        return np.zeros((1, 3)), np.array([[0, 0, 0, 1]])


@pytest.mark.parametrize("scene,category", [
    (Scene(success=False), "grasp"),
    (Scene(settle=["settling", "empty"]), "placement"),
    (Scene(settle=["settling"]), "settle_timeout"),
    (Scene(grasp=["grasping"]), "grasp_timeout"),
    (Scene(grasp=["grasping", "empty"]), "interrupted"),
])
def test_misses_timeouts_and_interruptions_are_failed_trials(scene, category):
    trial = evaluation.run_trial(scene, 0, 7, settle_steps=3, grasp_steps=3)
    assert trial["success"] is False
    assert trial["failure"] == category
    assert trial["object"] == "object" and trial["seed"] == 7


def test_each_trial_resets_the_placement_seed_and_processes_the_drop():
    scene = Scene()
    first = evaluation.run_trial(scene, 0, 7)
    evaluation.run_trial(scene, 0, 29)
    last = evaluation.run_trial(scene, 0, 7)
    assert first == last
    assert scene.draws[0] == scene.draws[2] != scene.draws[1]
    assert first["success"] is True and first["settled_position"] == [0, 0, 0]


def test_success_rate_keeps_all_failed_trials_in_the_denominator():
    rows = [{"success": True, "failure": None}]
    rows += [{"success": False, "failure": kind}
             for kind in ("grasp", "placement", "settle_timeout", "grasp_timeout")]
    summary = evaluation.summarize(rows)
    assert summary == {"trials": 5, "successes": 1, "success_rate": 0.2,
                       "failures": dict.fromkeys(
                           ("grasp", "placement", "settle_timeout", "grasp_timeout"), 1)}
    with pytest.raises(ValueError, match="at least one trial"):
        evaluation.summarize([])


def test_a_failed_checkpoint_keeps_the_previous_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.json"
    previous = {"complete": False, "trials": [{"success": False, "failure": "grasp"}]}
    evaluation.write_report(path, previous)
    with pytest.raises(ValueError):
        evaluation.write_report(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == previous

    def failed_replace(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="publication failed"):
        evaluation.write_report(path, {"complete": True, "trials": []})
    assert json.loads(path.read_text()) == previous
    assert list(tmp_path.iterdir()) == [path]


def test_report_identifies_loaded_source_and_rejects_another_installation(play, monkeypatch):
    import linker_lab

    hashes = evaluation.source_hashes(play)
    assert "src/linker_lab/arm.py" in hashes
    assert "policy-gallery/ycb-grasp/play.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())
    foreign_entry = SimpleNamespace(__file__="/another/install/play.py")
    with pytest.raises(RuntimeError, match="belong to this checkout"):
        evaluation.source_hashes(foreign_entry)
    monkeypatch.setattr(linker_lab, "__file__", "/another/install/linker_lab/__init__.py")
    with pytest.raises(RuntimeError, match="belong to this checkout"):
        evaluation.source_hashes(play)
