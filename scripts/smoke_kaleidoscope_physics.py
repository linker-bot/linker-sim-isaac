#!/usr/bin/env python3
"""验证真实 Kaleidoscope PhysX CUDA/Newton rollout 与 GPU 状态 API。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.configuration import load_kaleidoscope_config  # noqa: E402
from linkerbot_sim.kaleidoscope import make_torch_env  # noqa: E402


SUCCESS_MARKER = "LINKERBOT_KALEIDOSCOPE_PHYSICS_SMOKE_OK"
DEFAULT_PROFILE = "physx_cuda"
DEFAULT_NUM_ENVS = 2
DEFAULT_STEPS = 2
DEFAULT_ACTION_MODE = "joint_control"
ACTION_MODES = (
    "joint_control",
    "ee_delta_position",
    "ee_linear_path_position",
)
# reset randomization 最大为 0.03 rad；额外 0.02 rad 覆盖单拍 drive 跟踪误差，同时仍能
# 把越过机械限位的数值发散判为失败。
ZERO_ACTION_JOINT_LIMIT_MARGIN_RAD = 0.05
# 共址的独立 Newton worlds 在 GPU 并行约束归约中仍可能产生约 9.92e-5 的单元素
# qacc_warmstart 舍入差。该字段是下一拍求解提示而非 generalized state；只在 clone 后继
# 比较中允许 1e-4 绝对误差。TIME/ACT、立即 clone 以及全部物理/任务字段不使用此容差。
SOLVER_WARMSTART_SUCCESSOR_ATOL = 1.0e-4


@dataclass(frozen=True, slots=True)
class ProfileContract:
    """正式 Kaleidoscope profile 对应的唯一 backend 与 Kit experience。"""

    engine: str
    execution: str
    runtime_kind: str
    physics_backend: str
    kit_filename: str


PROFILE_CONTRACTS = {
    "physx_cuda": ProfileContract(
        engine="physx",
        execution="cuda",
        runtime_kind="physx_cuda",
        physics_backend="physx",
        kit_filename="linkerbot_sim.kaleidoscope.physx_cuda.python.kit",
    ),
    "newton_cuda": ProfileContract(
        engine="newton",
        execution="cuda",
        runtime_kind="newton_cuda",
        physics_backend="newton",
        kit_filename="linkerbot_sim.kaleidoscope.newton.python.kit",
    ),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONTRACTS),
        default=DEFAULT_PROFILE,
    )
    parser.add_argument("--num-envs", type=_positive_int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default=DEFAULT_ACTION_MODE,
        help="select the canonical joint smoke or the real batch IK / synchronized straight-line diagnostic path",
    )
    parser.add_argument(
        "--exercise-training-adapters",
        action="store_true",
        help="verify the real Gymnasium/skrl SAME_STEP boundary after the native smoke",
    )
    return parser.parse_args(argv)


def _load_smoke_config(profile: str, *, action_mode: str) -> object:
    """通过正式 loader 构造一个仅用于真实 action 验收的严格配置图。

    canonical mode profile 固定使用 ``joint_control``，避免日常训练无条件创建 cuRobo。
    action smoke 在临时 ``configs_root`` 中结构化替换 task action，再调用同一个完整图
    loader。返回对象已经携带 robot/object/controller resolved graph；删除临时目录后
    runtime 不应回读 YAML，这也顺带验证配置闭包没有隐藏的仓库默认根依赖。
    """

    if action_mode == "joint_control":
        return load_kaleidoscope_config(profile)
    try:
        action = {
            "ee_delta_position": {
                "mode": "ee_delta_position",
                "physics_ticks_per_action": 2,
                "reference_velocity_limit_rad_s": 1.0,
                "orientation_mode": "current",
                "failure_policy": "hold_penalty_truncate",
            },
            "ee_linear_path_position": {
                "mode": "ee_linear_path_position",
                "waypoint_count": 4,
                "physics_ticks_per_action": 4,
                "reference_velocity_limit_rad_s": 1.0,
                "orientation_mode": "current",
                "progress_mode": "linear",
                "failure_policy": "hold_from_first_failure",
            },
        }[action_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported smoke action mode {action_mode!r}") from exc

    import yaml

    from linkerbot_sim.configuration.catalog import load_yaml_mapping

    with tempfile.TemporaryDirectory(prefix="linkerbot-kaleidoscope-config-") as temp:
        configs_root = Path(temp) / "configs"
        shutil.copytree(REPO_ROOT / "configs", configs_root)
        mode_path = configs_root / "modes" / "kaleidoscope" / f"{profile}.yaml"
        mode_document = load_yaml_mapping(mode_path)
        profiles = mode_document.get("profiles")
        if not isinstance(profiles, dict) or not isinstance(profiles.get("task"), str):
            raise RuntimeError("Kaleidoscope mode profile has no task reference")
        # task 只表达动作；临时 mode composition 显式选择 kinematics-only cuRobo
        # profile，证明 joint_control canonical profile 不会无条件加载 solver 配置。
        profiles["curobo"] = "kaleidoscope_batch_ik"
        with mode_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(mode_document, stream, sort_keys=False)
        task_path = (configs_root / "tasks" / str(profiles["task"])).with_suffix(
            ".yaml"
        )
        task_document = load_yaml_mapping(task_path)
        task = task_document.get("task")
        if not isinstance(task, dict):
            raise RuntimeError("Kaleidoscope task profile has no mutable task mapping")
        task["action"] = action
        with task_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(task_document, stream, sort_keys=False)
        return load_kaleidoscope_config(profile, configs_root=configs_root)


def _require_cuda_tensor(value: object, *, label: str) -> None:
    """在 smoke 边界拒绝任何静默 CPU fallback。"""

    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{label} is not a Torch tensor")
    if value.device.type != "cuda":
        raise RuntimeError(f"{label} must be CUDA-resident, got {value.device}")


def _require_cuda_mapping(value: Mapping[str, object], *, label: str) -> None:
    for name, tensor in value.items():
        _require_cuda_tensor(tensor, label=f"{label}.{name}")


def _require_finite_tensor(value: object, *, label: str) -> None:
    """拒绝浮点张量中的 NaN/Inf；整数和布尔状态不参与有限性判断。"""

    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{label} is not a Torch tensor")
    if not value.is_floating_point():
        return
    finite = torch.isfinite(value)
    if not bool(torch.all(finite).item()):
        invalid_count = int(torch.count_nonzero(~finite).item())
        raise RuntimeError(
            f"{label} contains {invalid_count} non-finite floating-point values"
        )


def _require_finite_mapping(value: Mapping[str, object], *, label: str) -> None:
    for name, tensor in value.items():
        _require_finite_tensor(tensor, label=f"{label}.{name}")


def _zero_action_physical_metrics(
    env: object,
    state: Mapping[str, object],
    *,
    label: str,
) -> dict[str, float]:
    """用配置关节限位和 physics dt 拒绝“仍有限但已发散”的零动作状态。"""

    import torch

    runtime = getattr(env, "runtime", None)
    action_term = getattr(runtime, "action_term", None)
    # joint-delta 生产路径用 RuntimeAction 包装 GPU target accumulator；测试替身也允许
    # 直接暴露同一组字段。
    action_controller = getattr(action_term, "controller", action_term)
    views = getattr(runtime, "views", None)
    physics_runtime = getattr(runtime, "physics_runtime", None)
    lower = getattr(action_controller, "lower", None)
    upper = getattr(action_controller, "upper", None)
    controlled_q = getattr(views, "joint_positions", None)
    controlled_qd = getattr(views, "joint_velocities", None)
    robot_q = state.get("robot.q")
    robot_qd = state.get("robot.qd")
    target = state.get("robot.target")
    values = {
        "joint lower": lower,
        "joint upper": upper,
        "controlled q": controlled_q,
        "controlled qd": controlled_qd,
        "robot.q": robot_q,
        "robot.qd": robot_qd,
        "robot.target": target,
    }
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        missing = sorted(
            name
            for name, value in values.items()
            if not isinstance(value, torch.Tensor)
        )
        raise RuntimeError(
            f"{label} cannot audit physical bounds; missing tensors: {missing}"
        )
    assert isinstance(lower, torch.Tensor)
    assert isinstance(upper, torch.Tensor)
    assert isinstance(controlled_q, torch.Tensor)
    assert isinstance(controlled_qd, torch.Tensor)
    assert isinstance(robot_q, torch.Tensor)
    assert isinstance(robot_qd, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    if not (
        lower.ndim == upper.ndim == 1
        and lower.shape == upper.shape
        and controlled_q.shape == controlled_qd.shape == target.shape
        and controlled_q.shape[1:] == lower.shape
    ):
        raise RuntimeError(f"{label} joint-limit tensors have inconsistent shapes")
    margin = ZERO_ACTION_JOINT_LIMIT_MARGIN_RAD
    within_controlled_limits = (controlled_q >= lower - margin) & (
        controlled_q <= upper + margin
    )
    if not bool(torch.all(within_controlled_limits).item()):
        raise RuntimeError(
            f"{label} controlled robot.q exceeded configured joint limits"
        )
    if not bool(torch.all((target >= lower) & (target <= upper)).item()):
        raise RuntimeError(f"{label} robot.target exceeded configured joint limits")

    # 当前组合资产的 full robot state 只含有上述 active revolute joints 及其 equality
    # followers；follower 的角域不大于 active 全局角域。因此该全量门禁也能捕获未进入
    # command vector 的 DIP follower 发散。
    full_position_bound = (
        float(torch.maximum(torch.abs(lower), torch.abs(upper)).max().item()) + margin
    )
    max_abs_q = float(torch.abs(robot_q).max().item())
    if max_abs_q > full_position_bound:
        raise RuntimeError(
            f"{label} full robot.q exceeded the asset angle domain: "
            f"actual={max_abs_q}, bound={full_position_bound}"
        )

    get_physics_dt = getattr(physics_runtime, "get_physics_dt", None)
    physics_dt = float(
        get_physics_dt()
        if callable(get_physics_dt)
        else getattr(physics_runtime, "physics_dt", 0.0)
    )
    if not 0.0 < physics_dt:
        raise RuntimeError(f"{label} requires a positive physics_runtime.physics_dt")
    # 这是宽松的稳定性阈值而非电机速度规格：任何关节若能在一个 physics tick 内跨越
    # 配置的最大完整行程，smoke 就视为积分已失稳。
    velocity_bound = float(torch.max(upper - lower).item()) / physics_dt
    max_abs_qd = float(torch.abs(robot_qd).max().item())
    max_abs_controlled_qd = float(torch.abs(controlled_qd).max().item())
    if max(max_abs_qd, max_abs_controlled_qd) > velocity_bound:
        raise RuntimeError(
            f"{label} robot.qd crossed a full joint range in one physics tick: "
            f"actual={max(max_abs_qd, max_abs_controlled_qd)}, bound={velocity_bound}"
        )
    return {
        "max_abs_robot_q_rad": max_abs_q,
        "max_abs_robot_qd_rad_s": max_abs_qd,
        "position_bound_rad": full_position_bound,
        "velocity_bound_rad_s": velocity_bound,
    }


def _selected_kit_name(config: object, *, num_envs: int) -> str:
    """用生产 selector 解析 Kit，而不是在 smoke 中复制一条启动规则。"""

    from linkerbot_sim.isaac.app import _experience_path
    from linkerbot_sim.kaleidoscope.scene_assembly import session_spec_from_config

    spec = session_spec_from_config(config, num_envs=num_envs)
    return _experience_path(spec).name


def _enabled_extension_names() -> tuple[str, ...]:
    """读取当前 Kit 的完整 enabled closure，供 Newton owner 排他性检查。"""

    from linkerbot_sim.isaac.extensions import enumerate_enabled_kit_extensions

    return tuple(item.name for item in enumerate_enabled_kit_extensions())


def _resolve_profile_contract(
    profile: str,
    *,
    num_envs: int,
    action_mode: str = DEFAULT_ACTION_MODE,
) -> tuple[object, ProfileContract]:
    """解析 strict 配置，并证明 profile、physics 判别项和 Kit 三者一致。"""

    try:
        contract = PROFILE_CONTRACTS[profile]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Kaleidoscope smoke profile {profile!r}; "
            f"expected one of {tuple(PROFILE_CONTRACTS)!r}"
        ) from exc
    config = _load_smoke_config(profile, action_mode=action_mode)
    configured_physics = getattr(config, "physics", None)
    configured_selection = (
        str(getattr(configured_physics, "engine", "")),
        str(getattr(configured_physics, "execution", "")),
    )
    expected_selection = (contract.engine, contract.execution)
    if configured_selection != expected_selection:
        raise RuntimeError(
            "Kaleidoscope smoke profile resolved an unexpected physics backend: "
            f"profile={profile!r}, configured={configured_selection!r}, "
            f"expected={expected_selection!r}"
        )
    selected_kit = _selected_kit_name(config, num_envs=num_envs)
    if selected_kit != contract.kit_filename:
        raise RuntimeError(
            "Kaleidoscope smoke profile selected an unexpected Kit experience: "
            f"profile={profile!r}, selected={selected_kit!r}, "
            f"expected={contract.kit_filename!r}"
        )
    return config, contract


def _runtime_metadata(env: object, contract: ProfileContract) -> dict[str, object]:
    """核对已经启动的具体 physics owner，并审计 Newton 扩展闭包。"""

    runtime = getattr(getattr(env, "runtime", None), "physics_runtime", None)
    if runtime is None:
        raise RuntimeError("Kaleidoscope environment did not expose physics_runtime")
    actual = {
        "physics_engine": contract.engine,
        "physics_execution": contract.execution,
        "runtime_kind": str(getattr(runtime, "kind", "")),
        "physics_backend": str(getattr(runtime, "backend", "")),
        "execution": str(getattr(runtime, "execution", "")),
    }
    expected = {
        "physics_engine": contract.engine,
        "physics_execution": contract.execution,
        "runtime_kind": contract.runtime_kind,
        "physics_backend": contract.physics_backend,
        "execution": contract.execution,
    }
    if actual != expected:
        raise RuntimeError(
            "Kaleidoscope runtime differs from its strict profile contract: "
            f"actual={actual!r}, expected={expected!r}"
        )

    extension_audit: dict[str, object] = {"checked": False, "forbidden": []}
    if contract.runtime_kind == "newton_cuda":
        from linkerbot_sim.isaac.physics.exclusivity import (
            newton_forbidden_extensions,
        )

        forbidden = newton_forbidden_extensions(_enabled_extension_names())
        if forbidden:
            raise RuntimeError(
                "Kaleidoscope Newton smoke found Isaac/PhysX physics-owner "
                f"extensions: {list(forbidden)!r}"
            )
        extension_audit = {"checked": True, "forbidden": []}
    return {**actual, "extension_audit": extension_audit}


def _assert_matching_state(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    label: str,
    solver_warmstart_offset: int | None = None,
) -> None:
    """比较两份完整 GPU state，字段缺失也视为 smoke 失败。"""

    import torch

    _require_cuda_mapping(expected, label=f"{label}.expected")
    _require_cuda_mapping(actual, label=f"{label}.actual")
    if set(expected) != set(actual):
        raise RuntimeError(
            f"{label} state fields differ: "
            f"expected={sorted(expected)!r}, actual={sorted(actual)!r}"
        )
    for name in expected:
        if name == "solver.persistent" and solver_warmstart_offset is not None:
            _assert_solver_persistent_successor(
                expected[name],
                actual[name],
                warmstart_offset=solver_warmstart_offset,
                label=f"{label}.{name}",
            )
            continue
        torch.testing.assert_close(
            actual[name],
            expected[name],
            msg=lambda message, field=name: f"{label}.{field}: {message}",
        )


def _assert_solver_persistent_successor(
    expected: object,
    actual: object,
    *,
    warmstart_offset: int,
    label: str,
) -> None:
    """严格比较 TIME/ACT，仅对后继 WARMSTART 使用 GPU 舍入绝对容差。"""

    import torch

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise RuntimeError(f"{label} values must be Torch tensors")
    if expected.shape != actual.shape or expected.ndim != 2:
        raise RuntimeError(f"{label} values must be equal-shape matrices")
    if not 1 <= warmstart_offset < expected.shape[1]:
        raise RuntimeError(
            f"{label} warmstart offset {warmstart_offset} is outside width "
            f"{expected.shape[1]}"
        )
    torch.testing.assert_close(
        actual[:, :warmstart_offset],
        expected[:, :warmstart_offset],
        rtol=0.0,
        atol=0.0,
        msg=lambda message: f"{label}.time_act: {message}",
    )
    torch.testing.assert_close(
        actual[:, warmstart_offset:],
        expected[:, warmstart_offset:],
        rtol=0.0,
        atol=SOLVER_WARMSTART_SUCCESSOR_ATOL,
        msg=lambda message: f"{label}.qacc_warmstart: {message}",
    )


def _assert_state_field_changed(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> None:
    """证明一次状态写入不是 no-op，防止 restore smoke 产生假阳性。"""

    import torch

    _require_cuda_mapping(before, label=f"{label}.before")
    _require_cuda_mapping(after, label=f"{label}.after")
    if field not in before or field not in after:
        raise RuntimeError(f"{label} requires state field {field!r}")
    if torch.equal(before[field], after[field]):
        raise RuntimeError(f"{label}.{field} did not change")


def _require_step_tensors(result: tuple[object, ...], *, label: str) -> None:
    """统一审计 rollout 返回值，后续验证拍也不能绕过 CUDA 门禁。"""

    if len(result) != 5:
        raise RuntimeError(f"{label} must return the five-field step contract")
    observations, rewards, terminated, truncated, info = result
    for name, tensor in (
        ("observations", observations),
        ("rewards", rewards),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        _require_cuda_tensor(tensor, label=f"{label}.{name}")
        _require_finite_tensor(tensor, label=f"{label}.{name}")
    if not isinstance(info, Mapping):
        raise RuntimeError(f"{label}.info must be a mapping")
    _require_cuda_mapping(info, label=f"{label}.info")
    _require_finite_mapping(info, label=f"{label}.info")


def _runtime_owner_identities(env: object) -> dict[str, int]:
    """Capture owners that a control-mode switch is forbidden to rebuild."""

    runtime = getattr(env, "runtime", None)
    if runtime is None:
        raise RuntimeError("control-mode smoke requires env.runtime")
    owners = {
        "runtime": runtime,
        "session": getattr(runtime, "session", None),
        "physics_runtime": getattr(runtime, "physics_runtime", None),
        "action_term": getattr(runtime, "action_term", None),
        "task": getattr(runtime, "task", None),
    }
    missing = [name for name, value in owners.items() if value is None]
    if missing:
        raise RuntimeError(f"control-mode smoke is missing runtime owners: {missing}")
    return {name: id(value) for name, value in owners.items()}


def _require_control_state(
    env: object,
    *,
    mode: str,
    generation: int,
) -> object:
    state = env.get_control_mode()
    actual = (
        getattr(state, "initial_mode", None),
        getattr(state, "active_mode", None),
        getattr(state, "generation", None),
        getattr(state, "scope", None),
    )
    expected = ("position", mode, generation, "all")
    if actual != expected:
        raise RuntimeError(
            f"unexpected control-mode state: actual={actual!r}, expected={expected!r}"
        )
    return state


def _restore_same_mode_snapshot(env: object, snapshot: object, *, label: str) -> None:
    """Perturb the active target and prove same-mode restore is not a no-op."""

    import torch

    state = env.get_control_mode()
    if getattr(snapshot, "schema_version", None) != 2:
        raise RuntimeError(f"{label} snapshot must use schema 2")
    if getattr(snapshot, "control_mode", None) != getattr(state, "active_mode", None):
        raise RuntimeError(f"{label} snapshot control mode metadata is incorrect")
    if getattr(snapshot, "control_generation", None) != getattr(
        state, "generation", None
    ):
        raise RuntimeError(f"{label} snapshot control generation metadata is incorrect")
    selected = snapshot.env_ids[:1]
    before = env.get_state(selected, fields=("robot.target",))
    perturbed = before["robot.target"].clone()
    perturbed[:, 0].add_(1.0e-4)
    env.set_state({"robot.target": perturbed}, selected)
    _assert_state_field_changed(
        before,
        env.get_state(selected, fields=("robot.target",)),
        field="robot.target",
        label=f"{label}.set_state_probe",
    )
    env.restore_snapshot(snapshot)
    _assert_matching_state(
        snapshot.fields,
        env.get_state(snapshot.env_ids),
        label=f"{label}.restore",
    )
    if env.get_control_mode().generation != state.generation:
        raise RuntimeError(f"{label} restore changed the runtime control generation")
    torch.cuda.synchronize(env.device)


def _require_cross_mode_restore_rejected(env: object, snapshot: object) -> None:
    before = env.get_state()
    try:
        env.restore_snapshot(snapshot)
    except ValueError:
        pass
    else:
        raise RuntimeError("cross-mode snapshot restore unexpectedly succeeded")
    _assert_matching_state(
        before,
        env.get_state(),
        label="cross_mode_restore_preflight",
    )


def _exercise_runtime_control_modes(
    env: object,
    zero_actions: object,
) -> dict[str, object]:
    """Exercise all control channels without replacing any runtime owner."""

    import torch

    owners = _runtime_owner_identities(env)
    _require_control_state(env, mode="position", generation=0)
    position_snapshot = env.snapshot()
    _restore_same_mode_snapshot(env, position_snapshot, label="position_mode")
    _require_step_tensors(env.step(zero_actions), label="position_mode.step")

    change = env.set_control_mode("velocity", expected_generation=0)
    if not change.changed or change.generation != 1:
        raise RuntimeError("position-to-velocity switch returned an invalid change")
    _require_control_state(env, mode="velocity", generation=1)
    _require_cross_mode_restore_rejected(env, position_snapshot)
    velocity_actions = torch.zeros_like(zero_actions)
    velocity_actions[:, 0] = 0.05
    _require_step_tensors(env.step(velocity_actions), label="velocity_mode.step")
    velocity_snapshot = env.snapshot()
    _restore_same_mode_snapshot(env, velocity_snapshot, label="velocity_mode")

    change = env.set_control_mode("effort", expected_generation=1)
    if not change.changed or change.generation != 2:
        raise RuntimeError("velocity-to-effort switch returned an invalid change")
    _require_control_state(env, mode="effort", generation=2)
    effort_actions = torch.zeros_like(zero_actions)
    # joint_control maps 0.05 to five percent of the profile limit.
    effort_actions[:, 0] = 0.05
    _require_step_tensors(env.step(effort_actions), label="effort_mode.step")
    effort_snapshot = env.snapshot()
    _restore_same_mode_snapshot(env, effort_snapshot, label="effort_mode")

    change = env.set_control_mode("position", expected_generation=2)
    if not change.changed or change.generation != 3:
        raise RuntimeError("effort-to-position switch returned an invalid change")
    _require_control_state(env, mode="position", generation=3)
    _require_step_tensors(env.step(zero_actions), label="position_return.step")
    return_snapshot = env.snapshot()
    _restore_same_mode_snapshot(env, return_snapshot, label="position_return")

    reset_observations, reset_info = env.reset(seed=123)
    _require_cuda_tensor(reset_observations, label="control_mode_reset.observations")
    _require_cuda_mapping(reset_info, label="control_mode_reset.info")
    _require_control_state(env, mode="position", generation=3)
    if _runtime_owner_identities(env) != owners:
        raise RuntimeError("control-mode switching replaced a runtime owner")
    return {
        "verified": True,
        "sequence": ["position", "velocity", "effort", "position"],
        "generation": 3,
        "identity_owners": sorted(owners),
        "same_mode_snapshot_restores": 4,
        "cross_mode_restore_rejected": True,
        "reset_preserved_mode": True,
    }


def _newton_contact_probe(
    env: object,
    contract: ProfileContract,
    *,
    baseline_snapshot: object | None = None,
) -> dict[str, object]:
    """读取实时 MuJoCo-Warp 接触计数，证明每个 Newton world 都执行物理接触。"""

    if contract.runtime_kind != "newton_cuda":
        return {"performed": False, "reason": "not_newton"}
    if baseline_snapshot is None:
        return _read_newton_contact_report(env)
    try:
        _induce_newton_contacts(env)
        return _read_newton_contact_report(env)
    finally:
        # 行为探针会故意把 T-block 放入左侧 TCP；无论诊断是否成功，都必须恢复完整
        # engine/controller/task/RNG/solver state，后续 snapshot 测试不能继承探针副作用。
        env.restore_snapshot(baseline_snapshot)


def _induce_newton_contacts(env: object) -> None:
    """把每个 world 的任务物体移到左侧 TCP，并真实推进一次 solver。"""

    pose_state = env.get_state(fields=("object.pose_local_wxyz",))
    pose = pose_state.get("object.pose_local_wxyz")
    _require_cuda_tensor(pose, label="contact_probe.object.pose_local_wxyz")
    live = env.runtime.views.refresh()
    tcp = live.tcp_positions_local
    _require_cuda_tensor(tcp, label="contact_probe.tcp_positions_local")
    if pose.shape != (env.num_envs, 7) or tcp.shape[:2] != (env.num_envs, 2):
        raise RuntimeError("contact probe state has an unexpected shape")
    contact_pose = pose.clone()
    contact_pose[:, :3].copy_(tcp[:, 0, :])
    env.set_state({"object.pose_local_wxyz": contact_pose})
    env.runtime.physics_runtime.step(render=False)
    # refresh 建立 owner stream 到 Torch caller stream 的正常产品级 hand-off；下面的
    # 冷诊断同步只负责读取 contact metadata，不替代运行时同步合同。
    env.runtime.views.refresh()


def _read_newton_contact_report(env: object) -> dict[str, object]:
    """按实际 contact pipeline ABI 读取每个 world 的接触分布。"""

    import numpy as np

    runtime = env.runtime.physics_runtime
    synchronize = getattr(runtime, "_synchronize_owner_stream", None)
    if not callable(synchronize):
        raise RuntimeError("Newton runtime has no owner-stream synchronizer")
    # nacon 是冷诊断字段，不进入训练热路径。先同步 owner stream，再做一次小向量 D2H，
    # 避免把读取旧接触计数误判成 contact pipeline 成功。
    synchronize()
    diagnostics = dict(runtime.diagnostics())
    pipeline = str(diagnostics["contact_pipeline"])
    if pipeline == "newton":
        contacts = getattr(runtime, "_contacts", None)
        model = getattr(runtime, "model", None)
        count_values = _numpy_int_vector(
            getattr(contacts, "rigid_contact_count", None),
            label="Newton rigid contact count",
        )
        if count_values.shape != (1,):
            raise RuntimeError("Newton rigid contact count must be scalar")
        contact_count = int(count_values[0])
        capacity = int(getattr(contacts, "rigid_contact_max", 0))
        if contact_count <= 0:
            raise RuntimeError("Newton collision pipeline produced no contacts")
        if capacity <= 0 or contact_count >= capacity:
            raise RuntimeError(
                "Newton raw contact capacity exhausted: "
                f"count={contact_count}, capacity={capacity}"
            )
        shape0 = _numpy_int_vector(
            getattr(contacts, "rigid_contact_shape0", None),
            label="Newton contact shape0",
            allow_negative_one=True,
        )[:contact_count]
        shape1 = _numpy_int_vector(
            getattr(contacts, "rigid_contact_shape1", None),
            label="Newton contact shape1",
            allow_negative_one=True,
        )[:contact_count]
        shape_world = _numpy_int_vector(
            getattr(model, "shape_world", None),
            label="Newton shape worlds",
            allow_negative_one=True,
        )
        if np.any(shape0 < 0) or np.any(shape1 < 0):
            raise RuntimeError("active Newton contacts contain invalid shape ids")
        if np.any(shape0 >= shape_world.size) or np.any(shape1 >= shape_world.size):
            raise RuntimeError("active Newton contact shape id exceeds model metadata")
        world0 = shape_world[shape0]
        world1 = shape_world[shape1]
        cross_world = (world0 >= 0) & (world1 >= 0) & (world0 != world1)
        if np.any(cross_world):
            raise RuntimeError("Newton produced a cross-world physical contact")
        contact_world = np.where(world0 >= 0, world0, world1)
        if np.any(contact_world < 0) or np.any(contact_world >= env.num_envs):
            raise RuntimeError("Newton contact cannot be assigned to an env world")
        per_world = np.bincount(contact_world, minlength=env.num_envs)[: env.num_envs]
        if not np.all(per_world > 0):
            inactive = int(np.count_nonzero(per_world == 0))
            raise RuntimeError(
                f"Newton found no physical contacts in {inactive} world(s)"
            )
        return {
            "performed": True,
            "pipeline": pipeline,
            "active_worlds": int(np.count_nonzero(per_world)),
            "total_contacts": contact_count,
            "min_contacts": int(per_world.min()),
            "max_contacts": int(per_world.max()),
            "raw_contact_capacity": capacity,
        }
    if pipeline != "mujoco":
        raise RuntimeError(f"unsupported Newton contact pipeline {pipeline!r}")
    solver = getattr(runtime, "solver", None)
    count_values = _numpy_int_vector(
        getattr(getattr(solver, "mjw_data", None), "nacon", None),
        label="MuJoCo-Warp contact count",
    )
    if count_values.shape != (1,):
        raise RuntimeError("MuJoCo-Warp contact count must be scalar")
    contact_count = int(count_values[0])
    data = getattr(solver, "mjw_data", None)
    capacity = int(getattr(data, "naconmax", 0))
    if contact_count <= 0:
        raise RuntimeError("MuJoCo-Warp collision pipeline produced no contacts")
    if capacity <= 0 or contact_count >= capacity:
        raise RuntimeError(
            "MuJoCo-Warp contact capacity exhausted: "
            f"count={contact_count}, capacity={capacity}"
        )
    world_ids = _numpy_int_vector(
        getattr(getattr(data, "contact", None), "worldid", None),
        label="MuJoCo-Warp contact world ids",
        allow_negative_one=True,
    )[:contact_count]
    if np.any(world_ids < 0) or np.any(world_ids >= env.num_envs):
        raise RuntimeError("MuJoCo-Warp active contact has an invalid world id")
    per_world = np.bincount(world_ids, minlength=env.num_envs)[: env.num_envs]
    if not np.all(per_world > 0):
        inactive = int(np.count_nonzero(per_world == 0))
        raise RuntimeError(f"Newton found no physical contacts in {inactive} world(s)")
    return {
        "performed": True,
        "pipeline": pipeline,
        "active_worlds": int(np.count_nonzero(per_world)),
        "total_contacts": contact_count,
        "min_contacts": int(per_world.min()),
        "max_contacts": int(per_world.max()),
        "contact_capacity": capacity,
    }


def _numpy_int_vector(
    value: object,
    *,
    label: str,
    allow_negative_one: bool = False,
):
    """读取已同步的 Newton/Warp 小型诊断数组。"""

    import numpy as np

    numpy_method = getattr(value, "numpy", None)
    if not callable(numpy_method):
        raise RuntimeError(f"{label} is unavailable")
    result = np.asarray(numpy_method(), dtype=np.int64).reshape(-1)
    minimum = -1 if allow_negative_one else 0
    if np.any(result < minimum):
        raise RuntimeError(f"{label} contains a value below {minimum}")
    return result


def _diagnostic_action(env: object, *, action_mode: str):
    """构造当前状态附近的 CUDA action，避免用不可达目标制造伪失败。"""

    import torch

    if action_mode == "ee_delta_position":
        return torch.zeros(
            (env.num_envs, env.action_dim),
            device=env.device,
            dtype=torch.float32,
        )
    if action_mode == "ee_linear_path_position":
        state = env.runtime.views.refresh()
        tcp = _require_action_tcp_positions(
            state.tcp_positions_local,
            num_envs=env.num_envs,
            action_dim=env.action_dim,
        )
        # 终点取当前 TCP，仍会完整执行 waypoint 生成、批量 waypoint IK 和固定 tick
        # 重采样；零长度几何路径消除了资产初态附近任意方向扰动导致的可达性歧义。
        return tcp.reshape(env.num_envs, env.action_dim).contiguous()
    raise ValueError(f"unsupported diagnostic action mode {action_mode!r}")


def _require_action_tcp_positions(
    value: object,
    *,
    num_envs: int,
    action_dim: int,
):
    import torch

    tcp = value
    _require_cuda_tensor(tcp, label="diagnostic_action.tcp_positions_local")
    assert isinstance(tcp, torch.Tensor)
    if tcp.ndim != 3 or tcp.shape[0] != num_envs or tcp.shape[2] != 3:
        raise RuntimeError("diagnostic TCP positions must have shape (N,R,3)")
    if tcp.shape[1] * 3 != action_dim:
        raise RuntimeError("diagnostic TCP positions do not cover the action columns")
    return tcp


def _require_action_success(
    env: object,
    info: Mapping[str, object],
    *,
    action_mode: str,
) -> tuple[str, ...]:
    """要求真实 solver 对每个 robot/env 都返回成功，不只检查输出有限。"""

    import torch

    action_term = env.runtime.action_term
    bindings = tuple(getattr(action_term, "bindings", ()))
    if not bindings:
        raise RuntimeError("diagnostic action runtime has no kinematics bindings")
    prefix = "ik_success" if action_mode == "ee_delta_position" else "linear_success"
    fields: list[str] = []
    for binding in bindings:
        name = f"{prefix}.{binding.label}"
        value = info.get(name)
        _require_cuda_tensor(value, label=f"diagnostic_action.info.{name}")
        assert isinstance(value, torch.Tensor)
        if value.dtype is not torch.bool or value.shape != (env.num_envs,):
            raise RuntimeError(f"{name} must be a CUDA bool tensor with shape (N,)")
        if not bool(torch.all(value).item()):
            failed = int(torch.count_nonzero(~value).item())
            raise RuntimeError(f"{name} failed for {failed} environment(s)")
        fields.append(name)
    return tuple(fields)


def _run_action_smoke(
    *,
    profile: str,
    num_envs: int,
    steps: int,
    action_mode: str,
) -> dict[str, object]:
    """用真实 physics owner 和真实 cuRobo context 验证 batch IK/直线动作。"""

    config, contract = _resolve_profile_contract(
        profile,
        num_envs=num_envs,
        action_mode=action_mode,
    )
    env = make_torch_env(config=config, num_envs=num_envs)
    failed = False
    try:
        runtime_metadata = _runtime_metadata(env, contract)
        observations, reset_info = env.reset(seed=123)
        _require_cuda_tensor(observations, label="action_reset.observations")
        _require_cuda_mapping(reset_info, label="action_reset.info")
        success_fields: tuple[str, ...] = ()
        for step_index in range(steps):
            action = _diagnostic_action(env, action_mode=action_mode)
            result = env.step(action)
            _require_step_tensors(result, label=f"{action_mode}_step[{step_index}]")
            info = result[4]
            assert isinstance(info, Mapping)
            success_fields = _require_action_success(
                env,
                info,
                action_mode=action_mode,
            )
            state = env.get_state()
            _require_cuda_mapping(state, label=f"{action_mode}_state[{step_index}]")
            _require_finite_mapping(state, label=f"{action_mode}_state[{step_index}]")
        result = {
            "profile": profile,
            "kit": contract.kit_filename,
            **runtime_metadata,
            "smoke_kind": "action",
            "action_mode": action_mode,
            "device": str(env.device),
            "num_envs": env.num_envs,
            "steps": steps,
            "action_dim": env.action_dim,
            "observation_dim": env.observation_dim,
            "success_fields": list(success_fields),
            "all_rows_succeeded": True,
        }
        print(SUCCESS_MARKER + " " + json.dumps(result, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        failed = True
        traceback.print_exception(exc)
        sys.stderr.flush()
        print(
            f"LINKERBOT_KALEIDOSCOPE_PHYSICS_SMOKE_FAILED {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        env.close(exit_code=1 if failed else 0)


def _prime_horizon_termination(env: object) -> None:
    """只通过 public state API 让下一拍产生全行 time-limit done。"""

    import torch

    current = env.get_state(fields=("task.episode_length",))
    episode_length = current.get("task.episode_length")
    _require_cuda_tensor(episode_length, label="training.task.episode_length")
    assert isinstance(episode_length, torch.Tensor)
    horizon = int(env.runtime.task.settings.horizon)
    env.set_state({"task.episode_length": torch.full_like(episode_length, horizon - 1)})


def _exercise_training_adapters(env: object) -> dict[str, bool]:
    """在同一个真实环境上验证 Gymnasium 与 skrl 的 SAME_STEP 语义。"""

    import numpy as np
    import torch

    from linkerbot_sim.kaleidoscope.adapters.gymnasium import (
        GymnasiumKaleidoscopeAdapter,
    )
    from linkerbot_sim.training.skrl.env import SkrlTorchAdapter

    # Gymnasium 是明确的 CPU/NumPy 冷边界。用 public state API 把 horizon 推到下一拍，
    # 证明 final_obs/final_info 在 reset 覆写 done 行前已经保存。
    env.reset(seed=456)
    _prime_horizon_termination(env)
    gym = GymnasiumKaleidoscopeAdapter(env, autoreset_mode="same_step")
    gym_result = gym.step(np.zeros((env.num_envs, env.action_dim), dtype=np.float32))
    gym_done = np.logical_or(gym_result[2], gym_result[3])
    gym_info = gym_result[4]
    if not np.all(gym_done) or not np.all(gym_info.get("_final_obs", False)):
        raise RuntimeError("Gymnasium SAME_STEP did not preserve/reset every done row")
    final_obs = gym_info.get("final_obs")
    if not isinstance(final_obs, np.ndarray) or final_obs.shape != (
        env.num_envs,
        env.observation_dim,
    ):
        raise RuntimeError("Gymnasium SAME_STEP final_obs has the wrong shape")

    # skrl 保持 CUDA-native：generation token 覆盖 step/reset 事务，adapter-owned
    # final buffer 必须仍是 CUDA tensor，不能经过 Gymnasium/NumPy。
    skrl = SkrlTorchAdapter(env)
    skrl.reset()
    _prime_horizon_termination(env)
    skrl_result = skrl.step(
        torch.zeros(
            (env.num_envs, env.action_dim),
            device=env.device,
            dtype=torch.float32,
        )
    )
    observations, rewards, terminated, truncated, info = skrl_result
    for name, value in (
        ("observations", observations),
        ("rewards", rewards),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        _require_cuda_tensor(value, label=f"skrl_same_step.{name}")
    _require_cuda_mapping(info, label="skrl_same_step.info")
    final_mask = info.get("_final_obs")
    _require_cuda_tensor(final_mask, label="skrl_same_step.info._final_obs")
    assert isinstance(final_mask, torch.Tensor)
    if final_mask.shape != (env.num_envs,) or not bool(torch.all(final_mask).item()):
        raise RuntimeError("skrl SAME_STEP did not preserve/reset every done row")
    return {"gymnasium_same_step": True, "skrl_same_step": True}


def run_smoke(
    *,
    profile: str,
    num_envs: int,
    steps: int,
    action_mode: str = DEFAULT_ACTION_MODE,
    exercise_training_adapters: bool = False,
) -> dict[str, object]:
    """通过正式 composition root 运行一个最小、可判定的 GPU episode。

    成功标记必须在 ``env.close()`` 前输出，因为 Isaac ``fast_shutdown`` 会在 native App
    close 中直接结束解释器。构造/执行失败则由 Session 使用非零 exit status 关闭 Kit，
    shell 和 CI 不会得到伪成功。
    """

    if action_mode != "joint_control":
        if exercise_training_adapters:
            raise ValueError(
                "training adapter smoke must use the canonical joint_control action"
            )
        return _run_action_smoke(
            profile=profile,
            num_envs=num_envs,
            steps=steps,
            action_mode=action_mode,
        )

    import torch

    config, contract = _resolve_profile_contract(profile, num_envs=num_envs)
    env = make_torch_env(config=config, num_envs=num_envs)
    failed = False
    try:
        runtime_metadata = _runtime_metadata(env, contract)
        observations, reset_info = env.reset(seed=123)
        _require_cuda_tensor(observations, label="reset.observations")
        _require_cuda_mapping(reset_info, label="reset.info")
        actions = torch.zeros(
            (env.num_envs, env.action_dim),
            device=env.device,
            dtype=torch.float32,
        )
        control_mode_switching = _exercise_runtime_control_modes(env, actions)
        reset_state = env.get_state()
        _require_cuda_mapping(reset_state, label="get_state")
        _require_finite_mapping(reset_state, label="get_state")
        zero_action_metrics = {
            "max_abs_robot_q_rad": 0.0,
            "max_abs_robot_qd_rad_s": 0.0,
            "position_bound_rad": 0.0,
            "velocity_bound_rad_s": 0.0,
        }
        for step_index in range(steps):
            _require_step_tensors(
                env.step(actions),
                label=f"zero_action_step[{step_index}]",
            )
            zero_action_state = env.get_state()
            _require_cuda_mapping(
                zero_action_state,
                label=f"zero_action_step[{step_index}].state",
            )
            _require_finite_mapping(
                zero_action_state,
                label=f"zero_action_step[{step_index}].state",
            )
            step_metrics = _zero_action_physical_metrics(
                env,
                zero_action_state,
                label=f"zero_action_step[{step_index}]",
            )
            zero_action_metrics["max_abs_robot_q_rad"] = max(
                zero_action_metrics["max_abs_robot_q_rad"],
                step_metrics["max_abs_robot_q_rad"],
            )
            zero_action_metrics["max_abs_robot_qd_rad_s"] = max(
                zero_action_metrics["max_abs_robot_qd_rad_s"],
                step_metrics["max_abs_robot_qd_rad_s"],
            )
            zero_action_metrics["position_bound_rad"] = step_metrics[
                "position_bound_rad"
            ]
            zero_action_metrics["velocity_bound_rad_s"] = step_metrics[
                "velocity_bound_rad_s"
            ]
        zero_action_finite_verified = True
        snapshot = env.snapshot()
        _require_cuda_tensor(snapshot.env_ids, label="snapshot.env_ids")
        _require_cuda_mapping(snapshot.fields, label="snapshot.fields")

        # restore 前必须真实改写一个 engine-owned 字段。若 restore 是 no-op，下面的完整
        # state 比较必然失败；这比“snapshot 后立刻 restore”更能验证 backend writer。
        selected = snapshot.env_ids[:1]
        target_before = env.get_state(selected, fields=("robot.target",))
        perturbed_target = target_before["robot.target"].clone()
        perturbed_target[:, 0].add_(1.0e-3)
        env.set_state({"robot.target": perturbed_target}, selected)
        target_after = env.get_state(selected, fields=("robot.target",))
        _assert_state_field_changed(
            target_before,
            target_after,
            field="robot.target",
            label="set_state_probe",
        )
        if "solver.persistent" in snapshot.fields:
            persistent_before = env.get_state(
                selected,
                fields=("solver.persistent",),
            )
            perturbed_persistent = persistent_before["solver.persistent"].clone()
            # packed state 的第0列固定为 MuJoCo time；只扰动该无坐标字段即可验证
            # selected writer，不把 qacc_warmstart 人为推入不稳定区间。
            perturbed_persistent[:, 0].add_(0.125)
            env.set_state(
                {"solver.persistent": perturbed_persistent},
                selected,
            )
            persistent_after = env.get_state(
                selected,
                fields=("solver.persistent",),
            )
            _assert_state_field_changed(
                persistent_before,
                persistent_after,
                field="solver.persistent",
                label="solver_persistent_set_state_probe",
            )
        env.restore_snapshot(snapshot)
        restored_state = env.get_state(snapshot.env_ids)
        _assert_matching_state(
            snapshot.fields,
            restored_state,
            label="snapshot_round_trip",
        )

        clone_verified = False
        partial_reset_isolation_verified = False
        action_row_isolation_verified = False
        clone_successor_verified = False
        if env.num_envs >= 2:
            source = torch.tensor([0], device=env.device, dtype=torch.int64)
            target = torch.tensor([1], device=env.device, dtype=torch.int64)

            # partial reset 是强化学习的核心路径。这里比较未选行的完整 canonical state，
            # 包括 engine、controller、task history、RNG 以及 Newton solver 持久字段。
            protected_before = env.get_state(target)
            reset_observations, reset_idx_info = env.reset_idx(source)
            _require_cuda_tensor(
                reset_observations,
                label="partial_reset.observations",
            )
            _require_cuda_mapping(reset_idx_info, label="partial_reset.info")
            protected_after = env.get_state(target)
            _assert_matching_state(
                protected_before,
                protected_after,
                label="partial_reset_isolation",
            )
            partial_reset_isolation_verified = True

            env.clone_state(source, target)
            source_state = env.get_state(source)
            target_state = env.get_state(target)
            # 比较完整 state schema，而不是只抽查机器人位置：engine state、controller
            # target、对象速度、task history 和 logical RNG 任一遗漏都会使 smoke 失败。
            _assert_matching_state(source_state, target_state, label="clone_state")
            clone_verified = True

            # partial reset 的随机扰动可能把某个 nominal-at-limit 关节推到边界外极小距离。
            # joint-delta 的零动作仍会执行 limit clamp，若直接把它当“未选行绝对不变”，就会
            # 把合法的全行 clamp 误报成 row 泄漏。先让两行执行同一零动作完成归一化，再 clone
            # 一次建立严格相同且位于控制域内的基线；随后的差异才只可能来自 env 0 非零动作。
            _require_step_tensors(
                env.step(torch.zeros_like(actions)),
                label="action_isolation_baseline_step",
            )
            env.clone_state(source, target)
            target_state = env.get_state(target)

            # 只给 env 0 一个非零 joint-delta；env 1 的 target 必须保持 clone 后的值。
            isolated_actions = torch.zeros_like(actions)
            isolated_actions[0, 0] = 0.25
            target_before_action = target_state["robot.target"].clone()
            _require_step_tensors(
                env.step(isolated_actions),
                label="isolated_action_step",
            )
            source_after_action = env.get_state(source)
            target_after_action = env.get_state(target)
            torch.testing.assert_close(
                target_after_action["robot.target"],
                target_before_action,
            )
            _assert_state_field_changed(
                target_after_action,
                source_after_action,
                field="robot.target",
                label="action_row_isolation",
            )
            action_row_isolation_verified = True

            # 再次 clone 后让两行执行完全相同的下一拍。立即回读相等还不够，只有下一拍
            # 全状态仍相等，才能证明 controller/RNG/Newton warm-start 等隐状态也被复制。
            env.clone_state(source, target)
            _assert_matching_state(
                env.get_state(source),
                env.get_state(target),
                label="clone_successor_initial",
            )
            equal_actions = torch.zeros_like(actions)
            equal_actions[0, 0] = 0.1
            equal_actions[1, 0] = 0.1
            _require_step_tensors(
                env.step(equal_actions),
                label="clone_successor_step",
            )
            source_successor = env.get_state(source)
            target_successor = env.get_state(target)
            activation_width = int(
                getattr(
                    getattr(env.runtime, "physics_runtime", None),
                    "solver_integration_activation_width",
                    0,
                )
            )
            _assert_matching_state(
                source_successor,
                target_successor,
                label="clone_successor",
                solver_warmstart_offset=1 + activation_width,
            )
            clone_successor_verified = True

        contact_baseline = env.snapshot()
        physical_contacts = _newton_contact_probe(
            env,
            contract,
            baseline_snapshot=contact_baseline,
        )
        training_adapters = (
            _exercise_training_adapters(env)
            if exercise_training_adapters
            else {"gymnasium_same_step": False, "skrl_same_step": False}
        )

        result = {
            "profile": profile,
            "kit": contract.kit_filename,
            **runtime_metadata,
            "device": str(env.device),
            "num_envs": env.num_envs,
            "steps": steps,
            "action_dim": env.action_dim,
            "observation_dim": env.observation_dim,
            "state_fields": sorted(reset_state),
            "snapshot_fields": sorted(snapshot.fields),
            "snapshot_round_trip_verified": True,
            "control_mode_switching": control_mode_switching,
            "zero_action_finite_verified": zero_action_finite_verified,
            "zero_action_physical_bounds_verified": True,
            "zero_action_metrics": zero_action_metrics,
            "physical_contacts": physical_contacts,
            "set_state_verified": True,
            "clone_verified": clone_verified,
            "partial_reset_isolation_verified": partial_reset_isolation_verified,
            "action_row_isolation_verified": action_row_isolation_verified,
            "clone_successor_verified": clone_successor_verified,
            "training_adapters": training_adapters,
        }
        print(
            SUCCESS_MARKER + " " + json.dumps(result, sort_keys=True),
            flush=True,
        )
        return result
    except BaseException as exc:
        failed = True
        # fast-shutdown 会在 env.close 内结束解释器；必须在此之前输出主异常与稳定标记。
        traceback.print_exception(exc)
        sys.stderr.flush()
        print(
            f"LINKERBOT_KALEIDOSCOPE_PHYSICS_SMOKE_FAILED {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        env.close(exit_code=1 if failed else 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_smoke(
        profile=str(args.profile),
        num_envs=int(args.num_envs),
        steps=int(args.steps),
        action_mode=str(args.action_mode),
        exercise_training_adapters=bool(args.exercise_training_adapters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
