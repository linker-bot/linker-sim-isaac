from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from linkerbot_sim.app.interactive.single_arm import cli as single_arm_cli
from linkerbot_sim.app.interactive.protocol import parse_interactive_motion_message
from linkerbot_sim.app.interactive import dual_arm as dual_arm_interactive
from linkerbot_sim.app.interactive import single_arm as single_arm_interactive
from linkerbot_sim.app.motion.single_arm import (
    _hand_target_command,
    _single_joint_absolute_goal,
)
from linkerbot_sim.app.motion.specs import CSpaceDeltaPlanMoveSpec, HandMoveSpec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "single_arm_interactive.py"
DUAL_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dual_arm_interactive.py"


def test_single_arm_script_wrapper_imports_entrypoint() -> None:
    spec = importlib.util.spec_from_file_location("single_arm_interactive", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.main is single_arm_interactive.main


def test_single_arm_parse_args_defaults_hold_to_false(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["single_arm_interactive.py", "--gui"])

    args = single_arm_cli.parse_args()

    assert args.gui is True
    assert args.hold is False


def test_single_arm_parse_args_accepts_hold(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["single_arm_interactive.py", "--gui", "--hold"])

    args = single_arm_cli.parse_args()

    assert args.hold is True


def test_single_arm_run_interactive_mode_passes_hold_app(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Runtime:
        def close(self) -> None:
            captured["closed"] = True

    def fake_create_single_robot_runtime(**kwargs):
        captured.update(kwargs)
        return Runtime()

    monkeypatch.setattr(
        single_arm_cli, "create_single_robot_runtime", fake_create_single_robot_runtime
    )
    monkeypatch.setattr(
        single_arm_cli,
        "run_interactive_single_arm_motion",
        lambda runtime, **kwargs: 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["single_arm_interactive.py", "--gui", "--hold"],
    )

    single_arm_cli.run_interactive_mode(single_arm_cli.parse_args())

    assert captured["gui"] is True
    assert captured["hold_app"] is True
    assert captured["closed"] is True


def test_dual_arm_script_wrapper_imports_entrypoint() -> None:
    spec = importlib.util.spec_from_file_location("dual_arm_interactive", DUAL_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.main is dual_arm_interactive.main


def test_single_arm_protocol_uses_default_side_when_omitted() -> None:
    command = parse_interactive_motion_message(
        {
            "type": "cspace_delta",
            "joint_deltas": [0.1],
            "duration_s": 0.2,
        },
        default_tcp_by_side={"left": "single_tcp"},
        default_side="left",
    )

    move = command.moves[0]
    assert isinstance(move, CSpaceDeltaPlanMoveSpec)
    assert move.side == "left"
    assert move.tcp_frame_name == "single_tcp"


def test_single_arm_hand_target_updates_hand_joints_by_order() -> None:
    runtime = SimpleNamespace(
        joint_controller=SimpleNamespace(
            command_joint_names=(
                "AR5V2_L_arm_joint_1",
                "L6V1_L_hand_index_mcp_pitch",
                "L6V1_L_hand_middle_mcp_pitch",
            )
        )
    )

    target = _hand_target_command(
        runtime,
        start_command=np.asarray([1.0, 0.0, 0.0], dtype=float),
        hand=HandMoveSpec(
            side="left",
            joint_positions=[0.2, 0.3],
            duration_s=0.5,
        ),
    )

    np.testing.assert_allclose(target, [1.0, 0.2, 0.3])


def test_single_arm_absolute_cspace_goal_updates_prefix() -> None:
    goal = _single_joint_absolute_goal(
        base_q=np.asarray([1.0, 2.0, 3.0], dtype=float),
        joint_positions=[0.1, 0.2],
    )

    np.testing.assert_allclose(goal, [0.1, 0.2, 3.0])
