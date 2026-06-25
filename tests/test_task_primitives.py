from __future__ import annotations

from manipulation_project.controllers.types import ComponentControlSettings, JointControlSettings
from manipulation_project.tasks.primitives import SwitchControlModeTask, TaskRuntime


class _FakeController:
    def __init__(self, settings: JointControlSettings) -> None:
        self.settings = settings
        self.configure_runtime_calls = 0

    def configure_runtime(self) -> None:
        self.configure_runtime_calls += 1


def test_switch_control_mode_task_reconfigures_controller_without_stepping() -> None:
    initial_settings = JointControlSettings(
        default=ComponentControlSettings(mode="position", method="implicit")
    )
    next_settings = JointControlSettings(
        default=ComponentControlSettings(mode="effort", method="direct")
    )
    controller = _FakeController(initial_settings)
    runtime = TaskRuntime(
        robot=object(),
        world=object(),
        articulation_action_type=object(),
        controller=controller,
        simulation_app=None,
        render=False,
    )

    step = SwitchControlModeTask(settings=next_settings, phase="switch_to_effort").run(runtime, 42)

    assert step == 42
    assert controller.settings is next_settings
    assert controller.configure_runtime_calls == 1
