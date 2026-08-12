from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import linkerbot_sim.backends.curobo.kinematics.device_batch_ik as device_ik
from linkerbot_sim.backends.curobo.kinematics import CuroboDeviceBatchIKSolver


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for device IK"
)


class _GoalToolPose:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _Solver:
    def __init__(self) -> None:
        self.calls = 0

    def solve_pose(self, _goal, *, current_state, seed_config):
        del seed_config
        self.calls += 1
        seed = current_state.position
        success = torch.ones(seed.shape[0], device=seed.device, dtype=torch.bool)
        success[-1] = False
        return SimpleNamespace(
            solution=seed + 0.1,
            success=success,
            position_error=torch.arange(seed.shape[0], device=seed.device).float(),
            rotation_error=torch.zeros(seed.shape[0], device=seed.device),
        )


class _Context:
    def __init__(self) -> None:
        self.ik_solver = _Solver()
        self.default_tcp_frame = "tcp"
        self.device_cfg = SimpleNamespace(
            device=torch.device("cuda:0"), dtype=torch.float32
        )
        self.types = SimpleNamespace(GoalToolPose=_GoalToolPose)

    def frame_names(self):
        return ["tcp"]

    def joint_names(self):
        return ["j1", "j2"]

    def joint_state_from_positions(self, positions):
        return SimpleNamespace(position=positions)


@pytest.fixture(autouse=True)
def _no_tool_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device_ik, "update_active_tool_pose_criteria", lambda *a, **k: None
    )


def test_device_batch_ik_keeps_mapping_results_and_diagnostics_on_cuda() -> None:
    context = _Context()
    solver = CuroboDeviceBatchIKSolver(
        context,
        command_joint_names=("extra", "j2", "j1"),
    )
    seeds = torch.tensor([[9.0, 2.0, 1.0], [8.0, 4.0, 3.0]], device="cuda")
    result = solver.solve(
        target_positions=torch.zeros((2, 3), device="cuda"),
        target_orientations_wxyz=None,
        seeds=seeds,
    )
    assert result.joint_positions.device.type == "cuda"
    torch.testing.assert_close(
        result.joint_positions[0], torch.tensor([9.0, 2.1, 1.1], device="cuda")
    )
    torch.testing.assert_close(result.joint_positions[1], seeds[1])
    assert result.success.tolist() == [True, False]


def test_device_waypoint_ik_warm_starts_and_holds_failed_rows() -> None:
    context = _Context()
    solver = CuroboDeviceBatchIKSolver(context)
    seeds = torch.zeros((2, 2), device="cuda")
    result = solver.solve_waypoints(
        target_positions=torch.zeros((3, 2, 3), device="cuda"),
        target_orientations_wxyz=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]] * 2] * 3, device="cuda"
        ),
        seeds=seeds,
    )
    assert context.ik_solver.calls == 3
    assert result.joint_positions.shape == (3, 2, 2)
    torch.testing.assert_close(
        result.joint_positions[:, 1], torch.zeros((3, 2), device="cuda")
    )
    torch.testing.assert_close(
        result.joint_positions[:, 0],
        torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], device="cuda"),
    )
    assert result.first_failure_step.tolist() == [-1, 0]


@pytest.mark.parametrize("field", ["solution", "success", "position_error"])
def test_device_batch_ik_rejects_host_result_tensors(field: str) -> None:
    context = _Context()
    original = context.ik_solver.solve_pose

    def host_result(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(result, field, getattr(result, field).cpu())
        return result

    context.ik_solver.solve_pose = host_result
    solver = CuroboDeviceBatchIKSolver(context)

    with pytest.raises(ValueError, match="must live on CUDA"):
        solver.solve(
            target_positions=torch.zeros((2, 3), device="cuda"),
            target_orientations_wxyz=None,
            seeds=torch.zeros((2, 2), device="cuda"),
        )
