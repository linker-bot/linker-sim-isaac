from __future__ import annotations

from types import MappingProxyType

import pytest

from linkerbot_sim.kaleidoscope.collision_query import (
    ArticulatedCollisionQueryResult,
    ArticulatedCollisionQuerySpec,
    CollisionQueryTarget,
    CollisionQueryUnavailableError,
)
from linkerbot_sim.kaleidoscope.env import TorchKaleidoscopeEnv


class _Runtime:
    num_envs = 2
    device = object()
    action_dim = 4
    action_low = -1.0
    action_high = 1.0
    observation_dim = 8
    viewport_enabled = False
    render_every_n_steps = 0
    supports_articulated_collision_queries = True

    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []
        self.step_calls = 0
        self.close_calls: list[int] = []
        self.closed = False
        self.query_error: BaseException | None = None
        self.articulated_collision_query_specs = MappingProxyType(
            {
                "left": ArticulatedCollisionQuerySpec(
                    robot_label="left",
                    command_joint_names=("joint",),
                    joint_lower=(-1.0,),
                    joint_upper=(1.0,),
                )
            }
        )
        marker = object()
        self.query_result = ArticulatedCollisionQueryResult(
            collision_mask=marker,  # type: ignore[arg-type]
            first_hit_sample=marker,  # type: ignore[arg-type]
            first_hit_category=marker,  # type: ignore[arg-type]
        )

    def query_articulated_collisions(
        self,
        **kwargs: object,
    ) -> ArticulatedCollisionQueryResult:
        if self.closed:
            raise RuntimeError("Kaleidoscope runtime is closed")
        if self.query_error is not None:
            raise self.query_error
        self.query_calls.append(kwargs)
        return self.query_result

    def step(self, _actions: object) -> object:
        self.step_calls += 1
        raise AssertionError("collision query must not call step")

    def close(self, *, exit_code: int = 0) -> None:
        self.close_calls.append(exit_code)
        self.closed = True


def test_env_collision_query_forwards_exact_objects_without_step() -> None:
    runtime = _Runtime()
    env = TorchKaleidoscopeEnv(runtime)  # type: ignore[arg-type]
    joint_positions = object()
    env_ids = object()
    targets = (
        CollisionQueryTarget.TABLE,
        CollisionQueryTarget.OTHER_ROBOT,
    )

    assert env.supports_articulated_collision_queries is True
    assert (
        env.articulated_collision_query_specs
        is runtime.articulated_collision_query_specs
    )
    assert runtime.query_calls == []
    assert runtime.step_calls == 0
    runtime.supports_articulated_collision_queries = False
    assert env.supports_articulated_collision_queries is False
    runtime.supports_articulated_collision_queries = True

    result = env.query_articulated_collisions(
        robot_label="right",
        joint_positions=joint_positions,  # type: ignore[arg-type]
        env_ids=env_ids,  # type: ignore[arg-type]
        targets=targets,
    )

    assert result is runtime.query_result
    assert runtime.query_calls == [
        {
            "robot_label": "right",
            "joint_positions": joint_positions,
            "env_ids": env_ids,
            "targets": targets,
        }
    ]
    assert runtime.step_calls == 0


def test_env_collision_query_preserves_capability_and_closed_errors() -> None:
    runtime = _Runtime()
    env = TorchKaleidoscopeEnv(runtime)  # type: ignore[arg-type]
    unavailable = CollisionQueryUnavailableError("newton_cuda")
    runtime.query_error = unavailable

    with pytest.raises(CollisionQueryUnavailableError) as captured:
        env.query_articulated_collisions(
            robot_label="left",
            joint_positions=object(),  # type: ignore[arg-type]
            env_ids=object(),  # type: ignore[arg-type]
            targets=(CollisionQueryTarget.TABLE,),
        )

    assert captured.value is unavailable
    runtime.query_error = None
    env.close(exit_code=3)
    assert runtime.close_calls == [3]
    with pytest.raises(RuntimeError, match="closed"):
        env.query_articulated_collisions(
            robot_label="left",
            joint_positions=object(),  # type: ignore[arg-type]
            env_ids=object(),  # type: ignore[arg-type]
            targets=(CollisionQueryTarget.TABLE,),
        )
