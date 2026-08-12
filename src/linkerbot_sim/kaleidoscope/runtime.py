"""Kaleidoscope 的 GPU runtime 与唯一资源所有权边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterator, Literal, Protocol, runtime_checkable

from linkerbot_sim.controllers.control_mode import ControlModeLockedError
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.control_commands import (
    ControlTrajectory,
    EffortControlTrajectory,
    PositionControlTrajectory,
    VelocityControlTrajectory,
)
from linkerbot_sim.kaleidoscope.control_mode import (
    KaleidoscopeControlModeCoordinator,
)
from linkerbot_sim.kaleidoscope.observations import TBlockState
from linkerbot_sim.kaleidoscope.resets import TBlockResetCommand
from linkerbot_sim.kaleidoscope.snapshot import KaleidoscopeEpisodeSnapshot
from linkerbot_sim.kaleidoscope.state_api import KaleidoscopeStateAPI
from linkerbot_sim.kaleidoscope.task import TaskStepResult, VectorTask
from linkerbot_sim.kaleidoscope.tensors import normalize_env_ids, require_cuda_tensor

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class ActionExecution:
    """一个 decision 内需要同步写入的固定 tick joint targets。"""

    control: ControlTrajectory
    position_reference: "torch.Tensor"
    failure_mask: "torch.Tensor"
    info: Mapping[str, "torch.Tensor"]

    @property
    def joint_targets(self) -> "torch.Tensor":
        """Deprecated read-only tensor alias for position-only integrations."""

        if isinstance(self.control, PositionControlTrajectory):
            return self.control.positions
        if isinstance(self.control, VelocityControlTrajectory):
            return self.control.velocities
        return self.control.efforts


@runtime_checkable
class ActionTerm(Protocol):
    """固定 action mode 的设备执行边界。"""

    action_dim: int
    action_low: float
    action_high: float
    physics_ticks_per_action: int
    supported_control_modes: tuple[ControlMode, ...]

    def apply(
        self,
        actions: "torch.Tensor",
        state: TBlockState,
        active_mode: ControlMode,
    ) -> ActionExecution: ...

    def reset(self, env_ids: "torch.Tensor", command: TBlockResetCommand) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class KaleidoscopeViews(Protocol):
    """后端 raw tensor port 的 mode adapter；不拥有物理 runtime 或 App。"""

    num_envs: int
    device: "torch.device"

    def write_reset(self, command: TBlockResetCommand) -> None: ...

    def refresh(self, env_ids: "torch.Tensor | None" = None) -> TBlockState: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SameStepToken:
    """skrl SAME_STEP 私有握手令牌；不是 public RL action API。"""

    generation: int


class KaleidoscopeRuntime:
    """拥有一个 IsaacSession、mode views、action term 与 task。

    ``IsaacSession`` 是 App/stage/concrete PhysicsRuntime 的唯一直接 owner。Runtime 在关闭时先
    关闭 task/views，再关闭 session；它从不单独持有或关闭 Isaac World。
    """

    def __init__(
        self,
        *,
        session: object,
        views: KaleidoscopeViews,
        action_term: ActionTerm,
        task: VectorTask,
        state_api: KaleidoscopeStateAPI,
        control_mode_coordinator: KaleidoscopeControlModeCoordinator | None = None,
        ik_failure_penalty: float = -1.0,
        viewport: object | None = None,
        viewport_reconfigure: Callable[[], None] | None = None,
    ) -> None:
        import torch

        physics_runtime = getattr(session, "physics_runtime", None)
        if physics_runtime is None:
            raise TypeError("Kaleidoscope requires IsaacSession.physics_runtime")
        backend_kind = str(getattr(physics_runtime, "kind", ""))
        if backend_kind not in {"physx_cuda", "newton_cuda"}:
            raise ValueError(
                "Kaleidoscope only accepts physx_cuda or newton_cuda PhysicsRuntime, "
                f"got {backend_kind!r}"
            )
        if views.num_envs != task.num_envs or views.num_envs != state_api.num_envs:
            raise ValueError("views/task/state_api num_envs must match")
        if views.device != task.device or views.device != state_api.device:
            raise ValueError("views/task/state_api must share one CUDA device")
        if int(action_term.action_dim) != int(task.action_dim):
            raise ValueError("action term and task action dimensions must match")
        if int(action_term.physics_ticks_per_action) != int(
            getattr(task, "settings").physics_ticks_per_action
        ):
            raise ValueError("action/task physics_ticks_per_action must have one owner")
        self.session = session
        self.views = views
        self.action_term = action_term
        self.task = task
        self.state_api = state_api
        self.num_envs = views.num_envs
        self.device = views.device
        self.action_dim = action_term.action_dim
        self.action_low = float(action_term.action_low)
        self.action_high = float(action_term.action_high)
        if (
            math.isnan(self.action_low)
            or math.isnan(self.action_high)
            or self.action_low >= self.action_high
        ):
            raise ValueError("action term bounds must be ordered")
        self.observation_dim = task.observation_dim
        self.ik_failure_penalty = float(ik_failure_penalty)
        self.viewport_enabled = viewport is not None
        self.render_every_n_steps = (
            0 if viewport is None else int(getattr(viewport, "render_every_n_steps"))
        )
        if self.viewport_enabled:
            capabilities = getattr(physics_runtime, "capabilities", None)
            if not bool(getattr(capabilities, "rendering", False)):
                raise RuntimeError(
                    "configured Kaleidoscope physics runtime cannot render"
                )
            if self.render_every_n_steps < 1:
                raise ValueError("render_every_n_steps must be positive")
        if viewport_reconfigure is not None and not callable(viewport_reconfigure):
            raise TypeError("viewport_reconfigure must be callable or None")
        if (
            self.viewport_enabled
            and backend_kind == "newton_cuda"
            and viewport_reconfigure is None
        ):
            raise RuntimeError(
                "Newton Kaleidoscope viewport requires viewport_reconfigure"
            )
        self._viewport_reconfigure = viewport_reconfigure
        # Newton 的轻量 Kit 会在首个 app.update 中初始化默认 camera controller，
        # 从而覆盖 assembly 已写入的 world view。只在首次显式 render 内补一次 camera 写入
        # 与 renderer update；训练 step 仍固定 render=False，且 body_q 仍只 D2H 一次。
        self._viewport_reconfigure_pending = (
            self.viewport_enabled and backend_kind == "newton_cuda"
        )
        self._closed = False
        self._closing_started = False
        self._close_completed: set[str] = set()
        self._failed = False
        self._fatal_error: str | None = None
        self._phase: Literal[
            "idle", "step", "reset", "same_step", "mode_switch", "closing"
        ] = "idle"
        self._generation = 0
        self._outstanding_token: SameStepToken | None = None
        self._outstanding_phase: Literal["issued", "stepped"] | None = None
        self._outstanding_done = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # num_envs/device 在 runtime 生命周期内固定；全环境 reset 复用同一 CUDA selector，
        # 避免每次 reset 都重新分配并填充 arange buffer。
        self._all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        supported = tuple(
            getattr(action_term, "supported_control_modes", ("position",))
        )
        self.control_mode = (
            control_mode_coordinator
            or KaleidoscopeControlModeCoordinator(
                views=views,
                bindings=(),
                supported_modes=supported,
            )
        )
        self.control_mode.bind_runtime(self)

    @property
    def physics_runtime(self) -> object:
        return self.session.physics_runtime

    @property
    def fatal_error(self) -> str | None:
        return getattr(self, "_fatal_error", None)

    def get_control_mode(self):
        if self._closed:
            raise RuntimeError("Kaleidoscope runtime is closed")
        return self.control_mode.get_mode()

    def set_control_mode(
        self,
        mode: ControlMode,
        *,
        expected_generation: int | None = None,
    ):
        self._require_usable()
        self._require_no_same_step_transaction()
        with self._phase_scope("mode_switch"):
            return self.control_mode.set_mode(
                mode,
                expected_generation=expected_generation,
            )

    def mark_control_mode_fatal(self, message: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = str(message)
        self._failed = True

    def reset(self) -> tuple["torch.Tensor", Mapping[str, "torch.Tensor"]]:
        return self.reset_idx(self._all_env_ids)

    def reseed(self, seed: int) -> None:
        """只允许在 reset 之前调用的确定性 RNG 冷边界。"""

        self._require_usable()
        self._require_no_same_step_transaction()
        reseed = getattr(self.task, "reseed", None)
        if not callable(reseed):
            raise RuntimeError("configured task does not support deterministic reseed")
        reseed(seed)

    def reset_idx(
        self, env_ids: "torch.Tensor"
    ) -> tuple["torch.Tensor", Mapping[str, "torch.Tensor"]]:
        """显式 reset K 行；构造/写入/refresh 任一步失败都会令 runtime fail-stop。"""

        self._require_usable()
        self._require_no_same_step_transaction()
        ids = normalize_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        with self._phase_scope("reset"):
            try:
                command = self.task.reset_command(ids)
                self.control_mode.write_reset(command)
                self.action_term.reset(ids, command)
                self.physics_runtime.forward()
                state = self.views.refresh(ids)
                self.task.initialize_after_reset(ids, state)
            except BaseException:
                self._failed = True
                raise
        # index_select 已返回独立 storage；无需再做第二次 device-to-device copy。
        observations = self.task.buffers.last_finite_observation.index_select(0, ids)
        return observations, {"env_ids": ids.clone()}

    def step(self, actions: "torch.Tensor") -> TaskStepResult:
        """Native/debug 严格入口：done 行未 reset 时在推进 physics 前拒绝。"""

        import torch

        self._require_usable()
        self._require_no_same_step_transaction()
        # 这是 native/debug API 的可恢复生命周期错误，不能用会破坏 CUDA context 的
        # device-side assertion 表达。训练热路径使用 tokenized_step，由 generation 协议
        # 保证 SAME_STEP reset，因此不会执行这里唯一一次显式 D2H scalar readback。
        if bool(torch.any(self.task.buffers.needs_reset).item()):
            raise RuntimeError("done environments must be reset before the next step")
        with self._phase_scope("step"):
            return self._step_core(actions)

    def issue_same_step_token(self) -> SameStepToken:
        """向唯一训练 adapter 发放下一拍 token。"""

        self._require_usable()
        if self._phase != "idle":
            raise ControlModeLockedError(
                f"cannot issue SAME_STEP token during {self._phase!r} phase"
            )
        if self._outstanding_token is not None:
            raise RuntimeError(
                "previous SAME_STEP generation has not been acknowledged"
            )
        token = SameStepToken(self._generation)
        self._outstanding_token = token
        self._outstanding_phase = "issued"
        return token

    def tokenized_step(
        self, token: SameStepToken, actions: "torch.Tensor"
    ) -> TaskStepResult:
        """无 ``Tensor.any/item/bool`` 的训练入口。"""

        self._require_token(token, expected_phase="issued")
        with self._phase_scope("same_step"):
            try:
                result = self._step_core(actions)
                self._outstanding_done.copy_(result.terminated | result.truncated)
            except BaseException:
                self._failed = True
                raise
        self._outstanding_phase = "stepped"
        return result

    def complete_same_step_reset(self, token: SameStepToken) -> "torch.Tensor":
        """用固定 N 行 GPU mask reset 本 generation，完成后轮换 token。

        返回值是 runtime-owned 的 bool mask，只保证在下一次 tokenized step 前有效。它不是
        变长 env-id 列表，因此不会触发 ``torch.nonzero`` 的 CUDA 主机同步。
        """

        self._require_token(token, expected_phase="stepped")
        with self._phase_scope("same_step"):
            try:
                self._reset_same_step_mask(self._outstanding_done)
            except BaseException:
                # 保留 outstanding token；runtime 同时 fail-stop，绝不推进下一拍物理。
                self._failed = True
                raise
        self._outstanding_token = None
        self._outstanding_phase = None
        self._generation += 1
        return self._outstanding_done

    def get_state(self, *args: object, **kwargs: object) -> dict[str, "torch.Tensor"]:
        """返回 state API 的 owned CUDA 字段；不经过 RPC/NumPy transport。"""

        self._require_usable()
        return self.state_api.get_state(*args, **kwargs)

    def set_state(self, *args: object, **kwargs: object) -> None:
        """事务写入 CUDA state，并在提交后统一 forward/refresh 派生状态。"""

        self._require_usable()
        self._require_no_same_step_transaction()
        with self._phase_scope("reset"):
            self.state_api.set_state(*args, **kwargs)
            self._forward_after_state_write()

    def snapshot(self, *args: object, **kwargs: object) -> KaleidoscopeEpisodeSnapshot:
        """捕获同进程 GPU episode snapshot；持久化属于另一个显式冷 API。"""

        self._require_usable()
        return self.state_api.snapshot(*args, **kwargs)

    def restore_snapshot(self, *args: object, **kwargs: object) -> None:
        """恢复兼容的 GPU snapshot，并禁止与 SAME_STEP transaction 交错。"""

        self._require_usable()
        self._require_no_same_step_transaction()
        with self._phase_scope("reset"):
            self.state_api.restore_snapshot(*args, **kwargs)
            self._forward_after_state_write()

    def clone_state(self, *args: object, **kwargs: object) -> None:
        """设备内 clone 后统一 forward，使后端缓存与 canonical state 重新一致。"""

        self._require_usable()
        self._require_no_same_step_transaction()
        with self._phase_scope("reset"):
            self.state_api.clone_state(*args, **kwargs)
            self._forward_after_state_write()

    def render(self) -> None:
        """显式发布一帧 human viewport，不进入默认训练 step 热路径。

        PhysX 只同步 Fabric transformation；Newton 在该边界同步 owner stream 并将
        选中 world 的 CUDA body state 写入 USD。渲染失败表示显式 viewer 合同已失效，
        runtime 随即 fail-stop，防止调用方继续展示过期画面。
        """

        self._require_usable()
        if not self.viewport_enabled:
            raise RuntimeError(
                "Kaleidoscope viewport is disabled; use make_viewport_env()"
            )
        with self._phase_scope("step"):
            try:
                self.physics_runtime.render()
                if self._viewport_reconfigure_pending:
                    assert self._viewport_reconfigure is not None
                    render_update = getattr(self.physics_runtime, "render_update", None)
                    if not callable(render_update):
                        raise RuntimeError(
                            "Newton viewport cannot perform its stabilization update"
                        )
                    self._viewport_reconfigure()
                    render_update()
                    self._viewport_reconfigure_pending = False
            except BaseException:
                self._failed = True
                raise

    def is_running(self) -> bool:
        """返回 GUI App 是否仍在运行，供 viewer 响应窗口关闭。"""

        if self._closed or self._closing_started:
            return False
        self._require_usable()
        if not self.viewport_enabled:
            raise RuntimeError(
                "Kaleidoscope viewport is disabled; use make_viewport_env()"
            )
        callback = getattr(getattr(self.session, "app", None), "is_running", None)
        if not callable(callback):
            raise RuntimeError("Kaleidoscope viewport App does not expose is_running()")
        return bool(callback())

    def close(self, *, exit_code: int = 0) -> None:
        """按 action/IK → views/task → session 的固定依赖顺序幂等关闭。

        child 关闭失败时仍尝试其它 child，但绝不提前关闭 Session；成功项记录进度，
        调用方重试时只处理失败项。这样 cuRobo CUDA graph 或 raw view 不会在 native App
        已销毁后才被迫清理，也不会把一次部分失败伪装成 runtime 已关闭。
        """

        if type(exit_code) is not int or exit_code < 0:
            raise ValueError("exit_code must be a non-negative int")
        if self._closed:
            return
        self._closing_started = True
        self._phase = "closing"
        first_error: BaseException | None = None
        children = (
            ("action_term", self.action_term.close),
            ("views", self.views.close),
            ("task", self.task.close),
        )
        for label, callback in children:
            if label in self._close_completed:
                continue
            try:
                callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"additional close failure: {type(exc).__name__}: {exc}"
                    )
            else:
                self._close_completed.add(label)
        children_closed = all(
            label in self._close_completed for label, _callback in children
        )
        if children_closed and "session" not in self._close_completed:
            try:
                # 正常训练关闭保持无参数调用，便于窄测试替身实现；smoke/构造失败则把
                # 非零状态传给 fast-shutdown SimulationApp，防止 shell 得到伪成功。
                if exit_code:
                    self.session.close(exit_code=exit_code)
                else:
                    self.session.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"session close also failed: {type(exc).__name__}: {exc}"
                    )
            else:
                self._close_completed.add("session")
        self._closed = self._close_completed == {
            "action_term",
            "views",
            "task",
            "session",
        }
        if first_error is not None:
            raise first_error

    def _step_core(self, actions: "torch.Tensor") -> TaskStepResult:
        self._require_usable()
        state_before = self.views.refresh()
        active_mode = self.control_mode.active_mode
        supported = tuple(
            getattr(self.action_term, "supported_control_modes", ("position",))
        )
        if active_mode not in supported:
            from linkerbot_sim.controllers.control_mode import (
                ControlModeIncompatibleError,
            )

            raise ControlModeIncompatibleError(
                f"configured action does not support active mode {active_mode!r}",
                active_mode=active_mode,
                operation="action.apply",
            )
        execution = self.action_term.apply(actions, state_before, active_mode)
        self._validate_action_execution(execution, active_mode=active_mode)
        try:
            for tick in range(self.action_term.physics_ticks_per_action):
                self.control_mode.dispatch(execution.control, tick)
                self.physics_runtime.step(render=False)
            self.control_mode.commit_position_reference(execution.position_reference)
            state_after = self.views.refresh()
            result = self.task.step(state_after, actions)
            return self._apply_action_failures(result, execution)
        except BaseException:
            self._failed = True
            raise

    def _validate_action_execution(
        self,
        execution: ActionExecution,
        *,
        active_mode: ControlMode,
    ) -> None:
        import torch

        if not isinstance(execution, ActionExecution):
            raise TypeError("action term must return ActionExecution")
        expected_type = {
            "position": PositionControlTrajectory,
            "velocity": VelocityControlTrajectory,
            "effort": EffortControlTrajectory,
        }[active_mode]
        if not isinstance(execution.control, expected_type):
            raise ValueError("action control trajectory does not match active mode")
        tensors = (
            (execution.control.positions, execution.control.velocities)
            if isinstance(execution.control, PositionControlTrajectory)
            else (
                (execution.control.velocities,)
                if isinstance(execution.control, VelocityControlTrajectory)
                else (execution.control.efforts,)
            )
        )
        width = int(getattr(self.views, "command_dim", tensors[0].shape[2]))
        expected_shape = (
            self.action_term.physics_ticks_per_action,
            self.num_envs,
            width,
        )
        for value in tensors:
            tensor = require_cuda_tensor(
                value,
                name="action control trajectory",
                ndim=3,
                dtype=torch.float32,
            )
            if tensor.device != self.device or tensor.shape != expected_shape:
                raise ValueError(
                    f"action control trajectory must have shape {expected_shape} "
                    f"on {self.device}"
                )
            torch._assert_async(
                torch.all(torch.isfinite(tensor)),
                "action control trajectory must be finite",
            )
        reference = require_cuda_tensor(
            execution.position_reference,
            name="action position reference",
            ndim=2,
            dtype=torch.float32,
        )
        if reference.device != self.device or reference.shape != (self.num_envs, width):
            raise ValueError(
                "action position reference has the wrong CUDA device or shape"
            )
        torch._assert_async(
            torch.all(torch.isfinite(reference)),
            "action position reference must be finite",
        )

    def _apply_action_failures(
        self, result: TaskStepResult, execution: ActionExecution
    ) -> TaskStepResult:
        import torch

        failure = require_cuda_tensor(
            execution.failure_mask,
            name="action failure mask",
            ndim=1,
            dtype=torch.bool,
        )
        if failure.shape != (self.num_envs,):
            raise ValueError("action failure mask must have shape (N,)")
        reward = (
            result.rewards + failure.to(result.rewards.dtype) * self.ik_failure_penalty
        )
        truncated = result.truncated | failure
        self.task.buffers.reward.copy_(reward)
        self.task.buffers.episode_return.add_(
            failure.to(result.rewards.dtype) * self.ik_failure_penalty
        )
        self.task.buffers.truncated.copy_(truncated)
        self.task.buffers.needs_reset.copy_(result.terminated | truncated)
        info = dict(result.info)
        info.update(execution.info)
        info["action_failure"] = failure
        return TaskStepResult(
            observations=result.observations,
            rewards=reward,
            terminated=result.terminated,
            truncated=truncated,
            info=info,
        )

    def _reset_same_step_mask(self, reset_mask: "torch.Tensor") -> None:
        """固定 N 行写回；task 负责严格保留非 done 行和对应 RNG counter。"""

        state_before = self.views.refresh()
        command = self.task.masked_reset_command(reset_mask, state_before)
        self.control_mode.write_reset(command)
        self.action_term.reset(command.env_ids, command)
        self.physics_runtime.forward()
        state = self.views.refresh()
        self.task.initialize_after_masked_reset(reset_mask, state)

    def _require_token(
        self,
        token: SameStepToken,
        *,
        expected_phase: Literal["issued", "stepped"],
    ) -> None:
        self._require_usable()
        if not isinstance(token, SameStepToken):
            raise TypeError("invalid SAME_STEP token")
        if self._outstanding_token is not token:
            raise RuntimeError(
                "forged, stale, missing, or already completed SAME_STEP token"
            )
        if self._outstanding_phase != expected_phase:
            if expected_phase == "issued":
                raise RuntimeError("SAME_STEP token has already been stepped")
            raise RuntimeError("SAME_STEP token has not been stepped")

    def _require_no_same_step_transaction(self) -> None:
        if self._outstanding_token is not None:
            raise RuntimeError(
                "cannot mutate runtime during an active SAME_STEP transaction"
            )

    @contextmanager
    def _phase_scope(
        self,
        phase: Literal["step", "reset", "same_step", "mode_switch"],
    ) -> Iterator[None]:
        current = getattr(self, "_phase", "idle")
        if current != "idle":
            raise ControlModeLockedError(
                f"Kaleidoscope runtime phase is {current!r}; cannot enter {phase!r}"
            )
        self._phase = phase
        try:
            yield
        finally:
            if not self._closing_started:
                self._phase = "idle"

    def _forward_after_state_write(self) -> None:
        """提交引擎派生状态并回读 canonical buffer；任一失败都进入 fail-stop。"""

        try:
            self.physics_runtime.forward()
            # native equality/follower 投影可能在 forward 中修正 full DOF。立即回读，保证
            # get_state/snapshot 与下一拍 observation 看见同一个引擎稳定状态。
            self.views.refresh()
        except BaseException:
            self._failed = True
            raise

    def _require_usable(self) -> None:
        if self._closed:
            raise RuntimeError("Kaleidoscope runtime is closed")
        if self._closing_started:
            raise RuntimeError("Kaleidoscope runtime teardown has started")
        if (
            self._failed
            or getattr(self, "_fatal_error", None) is not None
            or self.state_api.poisoned
        ):
            raise RuntimeError(
                "Kaleidoscope runtime is fail-stop and must be recreated"
            )


__all__ = [
    "ActionExecution",
    "ActionTerm",
    "KaleidoscopeRuntime",
    "KaleidoscopeViews",
    "SameStepToken",
]
