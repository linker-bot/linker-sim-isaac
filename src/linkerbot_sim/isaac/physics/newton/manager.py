"""多 world Newton/MuJoCo-Warp 物理 runtime 核心。

本模块故意不 import ``isaacsim.physics.newton``：USD 只负责提供已配置的拓扑，manager
自行解析一次 prototype、复制为相互独立的 Newton world，并独占 model/state/control 与
solver；CUDA execution 另外独占 physics stream，CPU execution 则在 Warp CPU device 上
同步执行。这样 Kit 只负责应用和渲染循环，不会再有 extension-owned physics scene 与
second solver 同时更新同一批对象。

状态所有权也在这里统一：``joint_q/joint_qd`` 是 articulated system 的 generalized 权威
状态，``body_q/body_qd`` 是 FK 或求解器导出的 maximal 表示。普通 FREE rigid body setter
可以同步写两者；dynamic-chain 则必须整条链恢复 generalized state，再用 FK 派生 body state，
不能把每个 body 当作互不相关的六自由度对象写入。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Any
import weakref

import numpy as np

from linkerbot_sim.isaac.physics.exclusivity import (
    validate_newton_exclusivity,
)
from linkerbot_sim.isaac.physics.newton.integration_state import (
    SolverIntegrationStateStore,
    create_solver_integration_state_store,
)
from linkerbot_sim.isaac.physics.runtime import PhysicsCapabilities
from linkerbot_sim.isaac.spec import IsaacNewtonCpuSpec, IsaacNewtonCudaSpec
from linkerbot_sim.isaac.physics.newton.constraints import (
    COLD_STATE_PROJECTION_SCOPE,
    NATIVE_JOINT_EQUALITY_EXECUTOR,
    ExpectedMasterFollowerConstraint,
    MasterFollowerExecutorMetadata,
    NativeMasterFollowerAudit,
    NewtonColdStateProjector,
    NewtonDeviceWorldMasks,
    audit_native_master_follower_constraints,
)
from linkerbot_sim.isaac.physics.newton.replication import (
    NewtonReplicationResult,
    build_replicated_newton_builder,
)
from linkerbot_sim.robots.mimic.mjcf import parse_mjcf_joint_equalities


logger = logging.getLogger(__name__)

_SOLVER_INTEGRATION_SNAPSHOT_SCHEMA = "linkerbot.newton-solver-integration-state.v1"

# Newton 1.2.1 的公开枚举 ABI：GeoType.MESH=8、CONVEX_MESH=10，
# ShapeFlags.COLLIDE_SHAPES=2。把这些冷路径分类常量固定在项目内，可让纯配置/单元测试
# 在没有安装可选 ``newton`` wheel 时审计 mesh metadata；真实 model finalize 仍由 Newton
# 自身创建这些数值，不会由本模块伪造运行时对象。
_NEWTON_MESH_GEO_TYPES = frozenset({8, 10})
_NEWTON_COLLIDE_SHAPES_FLAG = 2


class _NewtonSceneRegistry:
    """仅提供兼容 ``World.scene`` 的对象保活容器，不参与物理所有权。"""

    def __init__(self) -> None:
        self._items: list[object] = []

    def add(self, item: object) -> object:
        self._items.append(item)
        return item

    def clear(self) -> None:
        """在 Kit/native teardown 前断开兼容对象引用，避免析构顺序反转。"""

        self._items.clear()


class NewtonRuntime:
    """独占一个 finalized Newton model，并按 session 规格选择 CPU 或 CUDA。"""

    backend = "newton"

    def __init__(
        self,
        *,
        stage: object,
        physics_spec: IsaacNewtonCpuSpec | IsaacNewtonCudaSpec,
        device: str,
        physics_dt: float,
        rendering_dt: float,
        gravity_z: float,
        add_ground: bool,
        ground_height: float,
        rendering_enabled: bool = False,
        render_callback: object | None = None,
        render_world_indices: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(physics_spec, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec)):
            raise TypeError("physics_spec must be an Isaac Newton specification")
        self.execution = (
            "cpu" if isinstance(physics_spec, IsaacNewtonCpuSpec) else "cuda"
        )
        self.kind = f"newton_{self.execution}"
        if self.execution == "cpu":
            if device != "cpu":
                raise ValueError("Newton CPU runtime requires device='cpu'")
        elif not _is_canonical_cuda_device(device):
            raise ValueError("Newton CUDA runtime requires a canonical cuda:N device")
        self.capabilities = PhysicsCapabilities(
            # Newton 1.2.1 的 MuJoCo C 路径不能按本项目合同提供 independent worlds；
            # CUDA runtime 保留 Kaleidoscope 使用的底层 multi-world 能力。
            supports_multiple_worlds=self.execution == "cuda",
            rendering=True,
            dynamic_chain=True,
            selected_reset=True,
            cuda_graph=self.execution == "cuda",
        )
        if physics_dt <= 0.0 or rendering_dt <= 0.0:
            raise ValueError("Newton timesteps must be positive")

        self.stage = stage
        self.physics_spec = physics_spec
        self.device = device
        self.physics_dt = float(physics_dt)
        self.rendering_dt = float(rendering_dt)
        self.gravity = (0.0, 0.0, float(gravity_z))
        self.scene = _NewtonSceneRegistry()
        if rendering_enabled and not callable(render_callback):
            raise TypeError("Newton rendering requires a callable render callback")
        self._rendering_enabled = bool(rendering_enabled)
        self._render_callback = render_callback
        if render_world_indices is not None:
            if not self._rendering_enabled:
                raise ValueError("Newton render world selection requires rendering")
            if not isinstance(render_world_indices, tuple) or not render_world_indices:
                raise TypeError(
                    "render_world_indices must be a non-empty tuple or None"
                )
            if any(
                type(index) is not int or index < 0 for index in render_world_indices
            ):
                raise ValueError(
                    "render_world_indices must contain non-negative integers"
                )
            if len(set(render_world_indices)) != len(render_world_indices):
                raise ValueError("render_world_indices must be unique")
        # ``None`` 保留 Mirror 的完整 world 映射；Kaleidoscope viewport 则显式选择一个
        # 调试 world，避免为整批训练环境物化 USD/RTX 数据。
        self._render_world_indices = render_world_indices
        # render sync 是物理状态到 USD 镜像的唯一冷边界。Camera/viewport 不属于物理
        # runtime，由 Mirror 的 RenderCoordinator 独立持有并先于 session 关闭。
        self._render_sync: object | None = None
        # 以下对象共同构成唯一的物理 owner。view 不缓存 state/control 的强语义副本，
        # 每次访问都回到 manager 解析当前对象，以兼容未来的 state ping-pong。
        self.model: object | None = None
        self.state: object | None = None
        self.control: object | None = None
        self.solver: object | None = None
        self.stream: object | None = None
        self.replication: NewtonReplicationResult | None = None
        self.native_master_follower_audit: NativeMasterFollowerAudit | None = None
        self.constraint_audit: NativeMasterFollowerAudit | None = None
        self._projector: NewtonColdStateProjector | None = None
        self._initial_state: object | None = None
        self._initial_control: object | None = None
        # SolverMuJoCo 除 Newton State/Control 外还持有 MuJoCo integration state。
        # time、act 与 qacc_warmstart 会影响下一拍；只恢复 joint_q/qd 会让 snapshot/clone
        # 表面相同、后继轨迹却分叉。qpos/qvel、ctrl、applied force 每拍分别由 Newton
        # State/Control 覆盖，mocap 又含 world-frame origin，不能进入跨 world canonical state。
        # 因此这里保存严格的 TIME|ACT|WARMSTART persistent 子集，而不是重复整个 Data。
        self._solver_integration_store: SolverIntegrationStateStore = (
            create_solver_integration_state_store(self.execution)
        )
        # FK dirty 与 equality projection dirty 必须分开：受控关节 reset 只提供 master
        # q/qd，需要按资产关系补齐 follower；而 snapshot/clone 写入的是 solver 产出的完整
        # generalized state，再投影会破坏精确恢复。二者仍在 forward/step/render 边界批量提交。
        self._dirty_worlds: set[int] = set()
        self._projection_worlds: set[int] = set()
        # SAME_STEP 使用固定 N 行 CUDA bool mask，不能用 ``nonzero/any/item`` 变成长 selector。
        # 每次通知只保存零拷贝 mask alias；view-row→world 映射在 view 注册冷路径预先上传。
        self._device_dirty_rows: list[tuple[object, object, object]] = []
        self._device_projection_rows: list[tuple[object, object, object]] = []
        self._world_masks: NewtonDeviceWorldMasks | None = None
        self._step_callbacks: list[object] = []
        self._registered_views: weakref.WeakSet[object] = weakref.WeakSet()
        self._view_world_rows: weakref.WeakKeyDictionary[object, object] = (
            weakref.WeakKeyDictionary()
        )
        self._num_worlds = 0
        self._sim_time = 0.0
        # graph 只捕获纯 GPU physics DAG。首次 model-property 写会使 solver 内部缓存和
        # 已捕获地址/执行图失效，因此 on_newton_view_write 会将状态退回 pending。
        self._graph: object | None = None
        self._graph_state = (
            "pending"
            if self.execution == "cuda"
            and bool(getattr(physics_spec, "use_cuda_graph", False))
            else "disabled"
        )
        self._graph_error: str | None = None
        self._constraint_solver: str | None = None
        self._contact_pipeline_kind: str | None = None
        self._contact_pipeline_trigger_labels: tuple[str, ...] = ()
        self._collision_pipeline: object | None = None
        self._contacts: object | None = None
        self._initialized = False
        self.closed = False
        _configure_newton_stage(
            stage,
            add_ground=add_ground,
            ground_height=ground_height,
            prepare_newton_render_topology=self._rendering_enabled,
        )

    @property
    def current_state(self) -> object | None:
        return self.state

    @property
    def world_count(self) -> int:
        return self._num_worlds

    def assert_single_world(self, *, consumer: str = "Mirror") -> None:
        """由只授权单 world 的 composition root 在 finalize 后调用。"""

        actual = int(self.world_count)
        if actual != 1:
            raise RuntimeError(
                f"{consumer} Newton requires exactly one world; actual={actual}"
            )

    @property
    def simulation_time(self) -> float:
        return self._sim_time

    @property
    def cuda_graph_state(self) -> str:
        return self._graph_state

    @property
    def graph_error(self) -> str | None:
        return self._graph_error

    def initialize_worlds(
        self,
        *,
        env_root_paths: Sequence[str],
        env_origins: object,
        robots: Mapping[str, object],
        object_handles: Sequence[object],
    ) -> None:
        """在 USD 导入完成后 finalize 唯一 Newton model，再允许创建 view。

        初始化顺序是合同的一部分：prototype 单次解析与复制 → builder/model equality 双重
        审计 → 创建权威 state/control → 冷投影与 FK → solver equality 映射审计。任一步失败
        都不能向上层暴露一个“部分可用”的 tensor view。
        """

        self._require_open()
        if self._initialized:
            raise RuntimeError("Newton scene is already initialized")
        # 启动校验之后，资产 importer 仍可能通过依赖闭包加载 physics owner。必须在
        # finalize/分配 Newton model 前，以当前完整闭包和最终 stage 再证明一次排他性。
        validate_newton_exclusivity(
            stage=self.stage,
            phase="pre_finalize",
        )
        roots = tuple(str(path) for path in env_root_paths)
        if not roots:
            raise ValueError("Newton requires at least one environment")
        if len(roots) != self.physics_spec.world_count:
            raise ValueError(
                "Newton environment count differs from the frozen session "
                f"specification: actual={len(roots)}, "
                f"expected={self.physics_spec.world_count}"
            )
        if self._render_world_indices is not None and max(
            self._render_world_indices
        ) >= len(roots):
            raise ValueError(
                "Newton render world index exceeds the finalized world count"
            )
        unsupported = sorted(
            str(getattr(item, "name", "<unnamed>"))
            for item in object_handles
            if str(getattr(item, "kind", "")) not in {"rigid", "dynamic_chain"}
        )
        if unsupported:
            raise RuntimeError(
                "Newton encountered unsupported object kinds; "
                f"unsupported={unsupported}"
            )
        self._constraint_solver = _resolve_constraint_solver(
            self.physics_spec.constraint_solver,
            object_handles=object_handles,
        )
        self._require_mujoco_variants(robots)

        import newton
        import warp as wp

        wp_device = wp.get_device(self.device)
        is_cuda_device = bool(getattr(wp_device, "is_cuda", False))
        if is_cuda_device != (self.execution == "cuda"):
            raise RuntimeError(
                "Newton execution/device mismatch: "
                f"execution={self.execution!r}, device={wp_device}"
            )
        if self.execution == "cpu" and len(roots) != 1:
            raise RuntimeError("Newton CPU runtime supports exactly one world")
        # Warp 不允许为 CPU 创建 Stream。CPU kernel 在 eager default stream 上同步执行；
        # CUDA 则继续独占一条 owner stream，保持 graph 和零拷贝 view 的原有顺序合同。
        self.stream = wp.Stream(wp_device) if self.execution == "cuda" else None
        transforms = _world_transforms(env_origins, device=wp_device)
        if self._rendering_enabled:
            from linkerbot_sim.isaac.physics.newton.render import (
                prepare_newton_render_stage,
            )

            # renderer-facing body 必须在 Newton 解析/finalize 前就固定为最终 matrix
            # op+reset stack。后续 render sync 只绑定属性值，绝不在 Hydra 已 population
            # 之后清空 xformOpOrder；该遍历同时覆盖嵌套机器人 body 与场景对象 body。
            prepare_newton_render_stage(
                stage=self.stage,
                prototype_root=roots[0],
                world_transform=transforms[0],
            )
        # roots[0] 是唯一 USD prototype。其余 world 只存在于 Newton builder 中；这里的
        # world transform 是绝对 env origin，不依赖 renderer 是否物化对应 USD clone。
        self.replication = build_replicated_newton_builder(
            self.stage,
            prototype_root=roots[0],
            destination_roots=roots,
            world_transforms=transforms,
            # USD stage 只 author prototype root，且保留其 stage pose（通常为 identity）；
            # replicated origin 只存在于 Newton clone plan，因此必须按绝对变换应用。
            source_world_transform=wp.transform_identity(),
            environment_root=str(Path(roots[0]).parent),
            up_axis="Z",
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        relation_counts = {
            name: len(parse_mjcf_joint_equalities(getattr(robot, "asset_path", None)))
            for name, robot in robots.items()
        }
        invalid_relation_counts = {
            name: count for name, count in relation_counts.items() if count != 5
        }
        if invalid_relation_counts or not relation_counts:
            raise RuntimeError(
                "Newton production robots must each retain exactly five "
                "asset joint equalities: "
                f"actual={relation_counts}"
            )
        relation_count = sum(relation_counts.values())
        # 先在未复制的 prototype 上确认每个导入机器人各有五条普通 JOINT equality：
        # 单臂 prototype 合计 5 条，双臂合计 10 条。这里不接受 constraint_mimic，因为
        # MuJoCo variant 的动态执行者必须只有 EqType.JOINT。
        _audit_prototype_constraints(
            self.replication.prototype_builder,
            expected_relation_count=relation_count,
        )

        with wp.ScopedDevice(wp_device), self._owner_stream_scope():
            model = self.replication.builder.finalize(device=wp_device)
        if int(getattr(model, "world_count", -1)) != len(roots):
            raise RuntimeError(
                "finalized Newton world count mismatch: "
                f"actual={getattr(model, 'world_count', None)!r}, expected={len(roots)}"
            )
        if str(getattr(model, "device", "")) != str(wp_device):
            raise RuntimeError(
                "finalized Newton device mismatch: "
                f"actual={getattr(model, 'device', None)!r}, expected={wp_device}"
            )
        self.model = model
        self._num_worlds = len(roots)

        # CUDA finalize 与后续分配必须留在 owner stream；CPU 使用同一结构但 scope 是空
        # context，因此不会构造非法的 ``wp.Stream(cpu)``。
        with self._owner_stream_scope():
            trigger_labels = _colliding_planar_mesh_labels(model)
            self._contact_pipeline_trigger_labels = trigger_labels
            self._contact_pipeline_kind = _resolve_contact_pipeline(
                self.physics_spec.contact_pipeline,
                trigger_labels=trigger_labels,
                execution=self.execution,
            )
            model.set_gravity(self.gravity)
            expectations = _asset_expectations(
                model=model,
                robots=robots,
                world_count=self._num_worlds,
            )
            executor_metadata = _executor_metadata(
                stage=self.stage,
                model=model,
                expectations=expectations,
                replication=self.replication,
            )
            audit = audit_native_master_follower_constraints(
                model,
                expectations,
                expected_world_count=self._num_worlds,
                expected_relations_per_world=relation_count,
                executor_metadata=executor_metadata,
                representation="model",
            )
            self.native_master_follower_audit = audit
            self.constraint_audit = audit
            self._projector = NewtonColdStateProjector(
                audit,
                device=wp_device,
                stream=self.stream,
            )
            self._world_masks = NewtonDeviceWorldMasks(
                world_count=self._num_worlds,
                articulation_world=getattr(model, "articulation_world"),
                device=wp_device,
                stream=self.stream,
            )

            state = model.state()
            control = model.control()
            self.state = state
            self.control = control
            # equality 约束的 solver 执行发生在 step 内，但刚创建/刚恢复的 generalized
            # state 还没经过 step。先做一次冷投影，再由 FK 生成一致的 maximal body state。
            self._project_all_worlds()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state, None)

            if self._contact_pipeline_kind == "newton":
                self._collision_pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="explicit",
                )
                self._contacts = self._collision_pipeline.contacts()

            solver = newton.solvers.SolverMuJoCo(
                model,
                **_solver_constructor_kwargs(
                    self.physics_spec,
                    world_count=self._num_worlds,
                    constraint_solver=self._constraint_solver,
                    contact_pipeline=self._contact_pipeline_kind,
                ),
            )
            self.solver = solver
            _audit_solver_equality_mapping(solver, audit)
            self._initialize_solver_integration_state(solver, device=wp_device)
            self._initial_state = model.state()
            self._initial_control = model.control()
            self._initial_state.assign(state)
            _copy_control(self._initial_control, control, stream=self.stream)
        if self._rendering_enabled:
            from linkerbot_sim.isaac.physics.newton.render import (
                NewtonRenderSync,
            )

            self._render_sync = NewtonRenderSync(
                stage=self.stage,
                model=model,
                prototype_root=self.replication.prototype_root,
                destination_roots=self.replication.destination_roots,
                world_transforms=self.replication.world_transforms,
                visible_world_indices=self._render_world_indices,
            )
        self._dirty_worlds.clear()
        self._projection_worlds.clear()
        self._device_dirty_rows.clear()
        self._device_projection_rows.clear()
        self._initialized = True

    def reset(self) -> None:
        """不重建拓扑、不重新分配 buffer，恢复已提交的 generalized 初始状态。"""

        self._require_initialized()
        assert self.state is not None
        assert self.control is not None
        assert self._initial_state is not None
        assert self._initial_control is not None
        with self._owner_stream_scope():
            self.state.assign(self._initial_state)
            _copy_control(self.control, self._initial_control, stream=self.stream)
            self._restore_initial_solver_integration_state()
            # ``_step`` 只控制 update_data_interval；生产合同固定为 1，但全量 reset 仍把
            # host-side bookkeeping 归零，避免诊断信息跨 episode 累积。
            solver = getattr(self, "solver", None)
            if solver is not None and hasattr(solver, "_step"):
                solver._step = 0
        self._dirty_worlds.update(range(self._num_worlds))
        projection_worlds = getattr(self, "_projection_worlds", None)
        if projection_worlds is None:
            projection_worlds = set()
            self._projection_worlds = projection_worlds
        projection_worlds.update(range(self._num_worlds))
        self._flush_cold_state_updates()
        self._sim_time = 0.0

    def commit_initial_state(self) -> None:
        """仅在零时刻初始化阶段替换 reset baseline。"""

        self._require_initialized()
        if self._sim_time != 0.0:
            raise RuntimeError(
                "Newton initial state can only be committed before simulation "
                f"advances; simulation_time={self._sim_time}"
            )
        assert self.state is not None
        assert self.control is not None
        assert self._initial_state is not None
        assert self._initial_control is not None
        # view 可能通过 IK 重建了 generalized coordinates 并标记 dirty world。保存 baseline
        # 前必须先投影 equality、再派生 maximal state，否则 reset 会复现一组内部不一致的
        # q/qd 与 body_q/body_qd。
        self._flush_cold_state_updates()

        with self._owner_stream_scope():
            self._initial_state.assign(self.state)
            _copy_control(self._initial_control, self.control, stream=self.stream)
            self._solver_integration_store.commit()

    @property
    def solver_integration_state_width(self) -> int:
        """返回每个 world 的 MuJoCo persistent state 固定宽度。"""

        return int(self._solver_integration_store.width)

    @property
    def solver_integration_activation_width(self) -> int:
        """返回 persistent layout 中紧随 time 的 actuator activation 列数。"""

        return int(self._solver_integration_store.activation_width)

    @property
    def solver_integration_state_signature(self) -> int:
        """返回 MuJoCo ``TIME|ACT|WARMSTART`` 的稳定 state bitmask。"""

        return int(self._solver_integration_store.signature)

    def borrow_solver_integration_state(self) -> object:
        """借用 manager-owned ``[world, TIME|ACT|WARMSTART]`` canonical buffer。

        CUDA 返回 Warp array，CPU 返回 NumPy float64 array。调用方不得替换其 storage；
        实际写入必须经过 :meth:`set_solver_integration_state`。
        """

        self._require_initialized()
        return self._solver_integration_store.borrow()

    def set_solver_integration_state(
        self,
        values: object,
        *,
        active_world_mask: object | None = None,
    ) -> None:
        """恢复全部或 mask 选中的 MuJoCo integration rows。

        CUDA 路径仍要求 full-N device buffer 与 device mask；CPU 路径严格只有一个 world，
        接受 NumPy/bool selector。两者都只保存不由 Newton State/Control 覆盖的字段。
        """

        self._require_initialized()
        self._solver_integration_store.restore(
            values,
            active_world_mask=active_world_mask,
        )

    def reset_solver_integration_state(self, active_world_mask: object) -> None:
        """把 mask 选中的 solver rows 恢复到 committed baseline。"""

        self._require_initialized()
        self._solver_integration_store.reset(active_world_mask)

    def capture_solver_integration_state_host(self) -> dict[str, object]:
        """在显式冷边界把 solver persistent state 复制为 JSON-compatible payload。

        该接口只供 Mirror snapshot 使用，不属于 Kaleidoscope 热路径。CUDA capture 先在
        owner stream 上更新 canonical buffer，再做一次显式同步与 D2H；CPU 则复制
        MuJoCo C API 所有的 float64 buffer。返回值不共享 manager-owned storage。
        """

        self._require_initialized()
        self._capture_solver_integration_state()
        if self.execution == "cuda":
            self._synchronize_owner_stream()
            borrowed = self._solver_integration_store.borrow()
            to_numpy = getattr(borrowed, "numpy", None)
            if not callable(to_numpy):
                raise RuntimeError(
                    "Newton CUDA solver integration state does not expose a host copy"
                )
            values = np.asarray(to_numpy(), dtype=np.float32)
        else:
            values = np.asarray(
                self._solver_integration_store.borrow(),
                dtype=np.float64,
            )
        return {
            "schema": _SOLVER_INTEGRATION_SNAPSHOT_SCHEMA,
            "source_execution": self.execution,
            "world_count": self._num_worlds,
            "state_signature": self.solver_integration_state_signature,
            "state_width": self.solver_integration_state_width,
            "simulation_time_s": self._sim_time,
            "values": values.copy().tolist(),
        }

    def validate_solver_integration_state_host(
        self,
        payload: Mapping[str, object],
    ) -> None:
        """在任何物理写入前校验一个冷存储 persistent-state payload。"""

        self._require_initialized()
        self._solver_integration_host_values(payload)

    def set_solver_integration_state_host(
        self,
        payload: Mapping[str, object],
    ) -> None:
        """从 Mirror 冷快照恢复 persistent state 与 runtime 仿真时钟。

        source 可以来自 Newton CPU 或 CUDA；只要 MuJoCo state signature 与 shape 一致，
        就按目标 execution 转换 dtype/device。CUDA 上传与 ``set_state`` 都在 owner stream
        排序，并在返回前同步，避免临时 staging buffer 先于异步拷贝释放。
        """

        self._require_initialized()
        values, simulation_time = self._solver_integration_host_values(payload)
        if self.execution == "cuda":
            import warp as wp

            with self._owner_stream_scope():
                device_values = wp.array(
                    values.astype(np.float32, copy=False),
                    dtype=wp.float32,
                    device=self.device,
                )
                self._solver_integration_store.restore(device_values)
            self._synchronize_owner_stream()
        else:
            self._solver_integration_store.restore(
                values.astype(np.float64, copy=False)
            )
        self._sim_time = simulation_time

    def reset_solver_integration_state_host(self) -> None:
        """在 Mirror 冷恢复边界把 solver persistent state 重置到 committed baseline。

        PhysX 快照没有 Newton 专属的 ``TIME|ACT|WARMSTART`` payload。跨引擎恢复到
        Newton 时不能沿用目标 runtime 的旧积分历史，因此显式恢复初始化阶段提交的
        baseline，并把 runtime 仿真时钟归零。CUDA 写入在 owner stream 上完成且返回前
        同步，保证上层补偿事务可以把本调用视为一个已完成的可逆写入。
        """

        self._require_initialized()
        with self._owner_stream_scope():
            self._restore_initial_solver_integration_state()
        self._synchronize_owner_stream()
        self._sim_time = 0.0

    def _solver_integration_host_values(
        self,
        payload: Mapping[str, object],
    ) -> tuple[np.ndarray, float]:
        """严格解析 host payload，并返回 owned float64 矩阵与仿真时间。"""

        if not isinstance(payload, Mapping):
            raise ValueError("Newton solver integration snapshot must be a mapping")
        expected_keys = {
            "schema",
            "source_execution",
            "world_count",
            "state_signature",
            "state_width",
            "simulation_time_s",
            "values",
        }
        unknown = sorted(str(key) for key in set(payload).difference(expected_keys))
        missing = sorted(str(key) for key in expected_keys.difference(payload))
        if unknown or missing:
            raise ValueError(
                "Newton solver integration snapshot fields are invalid: "
                f"missing={missing}, unknown={unknown}"
            )
        if payload["schema"] != _SOLVER_INTEGRATION_SNAPSHOT_SCHEMA:
            raise ValueError(
                "unsupported Newton solver integration snapshot schema: "
                f"{payload['schema']!r}"
            )
        if payload["source_execution"] not in {"cpu", "cuda"}:
            raise ValueError(
                "Newton solver integration source_execution must be 'cpu' or 'cuda'"
            )
        world_count = payload["world_count"]
        width = payload["state_width"]
        signature = payload["state_signature"]
        if type(world_count) is not int or world_count != self._num_worlds:
            raise ValueError(
                "Newton solver integration world_count mismatch: "
                f"snapshot={world_count!r}, runtime={self._num_worlds}"
            )
        if type(width) is not int or width != self.solver_integration_state_width:
            raise ValueError(
                "Newton solver integration state_width mismatch: "
                f"snapshot={width!r}, runtime={self.solver_integration_state_width}"
            )
        if (
            type(signature) is not int
            or signature != self.solver_integration_state_signature
        ):
            raise ValueError(
                "Newton solver integration state_signature mismatch: "
                f"snapshot={signature!r}, "
                f"runtime={self.solver_integration_state_signature}"
            )
        raw_time = payload["simulation_time_s"]
        if type(raw_time) not in {int, float}:
            raise ValueError(
                "Newton solver integration simulation_time_s must be a JSON number"
            )
        simulation_time = float(raw_time)
        try:
            raw_values = np.asarray(payload["values"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Newton solver integration snapshot contains non-numeric values"
            ) from exc
        if raw_values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "Newton solver integration snapshot contains non-numeric values"
            )
        values = raw_values.astype(np.float64, copy=False)
        expected_shape = (self._num_worlds, self.solver_integration_state_width)
        if values.shape != expected_shape:
            raise ValueError(
                "Newton solver integration values shape mismatch: "
                f"snapshot={values.shape}, runtime={expected_shape}"
            )
        if simulation_time < 0.0 or not np.isfinite(simulation_time):
            raise ValueError(
                "Newton solver integration simulation_time_s must be finite and >= 0"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Newton solver integration values must contain only finite numbers"
            )
        return np.ascontiguousarray(values), simulation_time

    def forward(self) -> None:
        """提交冷状态投影并重算运动学，但不推进仿真时间。"""

        self._require_initialized()
        if self._dirty_worlds or getattr(self, "_device_dirty_rows", ()):
            self._flush_cold_state_updates()
            return
        import newton

        assert self.model is not None and self.state is not None
        with self._owner_stream_scope():
            newton.eval_fk(
                self.model,
                self.state.joint_q,
                self.state.joint_qd,
                self.state,
                None,
            )

    def step(self, *, render: bool = False) -> None:
        """推进一次物理 tick，并可选发布同一时刻的渲染快照。

        CUDA Graph capture 记录首个完整 physics DAG，但 Warp capture 本身不执行 DAG，故成功
        后必须立即 launch 一次。若捕获失败，只执行一次 eager step；不能先执行捕获内容再补
        eager，也不能为了“热身”多推进一次可观察的 simulation time。
        """

        self._require_initialized()
        self._flush_cold_state_updates()

        if self._graph_state == "captured":
            import warp as wp

            wp.capture_launch(self._graph, stream=self.stream)
        elif self._graph_state == "pending":
            import warp as wp

            try:
                with wp.ScopedCapture(
                    device=self.device, stream=self.stream
                ) as capture:
                    self._simulate()
                self._graph = capture.graph
                self._graph_state = "captured"
                # capture 只记录首个 physics step，不执行它；立即 launch，保证调用者看到的
                # 第一次 step 确实推进了一次，而不是暴露成“graph warm-up 空帧”。
                wp.capture_launch(self._graph, stream=self.stream)
            except Exception as exc:
                self._graph = None
                self._graph_state = "eager_fallback"
                self._graph_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Newton CUDA graph capture failed; using eager: %s",
                    self._graph_error,
                )
                self._simulate()
        else:
            self._simulate()
        # CUDA packing kernel 不进入 graph capture；CPU 则在 eager step 后调用 MuJoCo C
        # API。store 在各自 execution 下负责正确的顺序与数据驻留。
        self._capture_solver_integration_state()
        self._sim_time += self.physics_dt
        for callback in tuple(self._step_callbacks):
            callback(self.physics_dt)
        if render:
            self.render()

    def render(self) -> None:
        """发布当前 Newton 快照并驱动一次 Kit update，绝不推进物理时间。

        物理 runtime 不知道 camera 数量、隐藏 viewport 或每个产品需要的连续 update 次数；
        这些策略属于 Mirror RenderCoordinator。这里仅提供一次 physics-to-USD 发布和一次
        App update。
        """

        self.pre_render()
        self.render_update()

    def render_update(self) -> None:
        """只推进一次 Kit renderer，不重复同步 CUDA 或重写 USD transform。

        Mirror 先通过 :meth:`pre_render` 发布一个不可变物理快照，再按 camera 声明连续调用
        本方法推进 RTX/SyntheticData history。拆开这两个阶段可避免每个隐藏 render tick
        都执行 D2H 和 stage authoring；该方法不读取 camera，也不推进 simulation time。
        """

        callback = self._render_callback
        if not callable(callback):
            raise RuntimeError("Newton render callback is unavailable")
        callback()

    def pre_render(self) -> None:
        """在唯一显式可见性边界同步 owner stream，并写入一份 renderer 快照。

        renderer/USD 是 CPU 消费者，不能依靠 CUDA stream 间事件直接读取 ``body_q``，所以
        此处允许一次 stream synchronize。同步放在 render 边界而非每个 physics step，纯物理
        benchmark 因而不会支付 host 可见性成本。
        """

        self._require_initialized()
        if not self._rendering_enabled or self._render_sync is None:
            raise RuntimeError(
                "Newton rendering was not enabled for this SimulationApp"
            )
        self._flush_cold_state_updates()
        self._synchronize_owner_stream()
        assert self.state is not None
        self._render_sync.sync(self.state.body_q)

    def get_physics_dt(self) -> float:
        return self.physics_dt

    def get_rendering_dt(self) -> float:
        return self.rendering_dt

    def set_gravity(self, gravity_z: float) -> None:
        self.gravity = (0.0, 0.0, float(gravity_z))
        if self.model is not None:
            with self._owner_stream_scope():
                self.model.set_gravity(self.gravity)
                if self.solver is not None:
                    from newton.solvers import SolverNotifyFlags

                    # SolverMuJoCo 持有独立的 mj_model/mjw_model；仅修改 Newton model
                    # 不会刷新 CPU ``mj_model.opt.gravity`` 或 CUDA solver cache。
                    self.solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)
            if self.solver is not None and bool(
                getattr(self.physics_spec, "use_cuda_graph", False)
            ):
                self._graph = None
                self._graph_state = "pending"

    def on_newton_view_write(
        self,
        *,
        view: object,
        category: str,
        field: str,
        world_indices: tuple[int, ...],
        device_row_mask: object | None = None,
    ) -> None:
        """接收 Newton-view 写通知，但绝不创建 follower target writer。

        state 写只累计受影响 world；control 写天然由下一次 solver step 消费；model 属性写
        还必须让 MuJoCo solver 刷新 DOF 属性缓存，并使旧 CUDA Graph 失效。
        """

        if category == "state":
            worlds = tuple(int(world) for world in world_indices)
            projection_required = field not in {
                "body_q",
                "body_qd",
                "joint_q_full",
                "joint_qd_full",
            }
            if device_row_mask is None:
                self._dirty_worlds.update(worlds)
                if projection_required:
                    self._projection_worlds.update(worlds)
            else:
                if self.execution != "cuda":
                    raise ValueError(
                        "device_row_mask is only available to Newton CUDA consumers"
                    )
                row_world = self._view_world_rows.get(view)
                if row_world is None:
                    raise RuntimeError(
                        "device-masked Newton write came from an unregistered view"
                    )
                import warp as wp

                try:
                    row_mask = wp.from_torch(device_row_mask)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "device_row_mask must be a CUDA torch.bool tensor"
                    ) from exc
                if (
                    row_mask.dtype != wp.bool
                    or len(row_mask.shape) != 1
                    or int(row_mask.shape[0]) != int(row_world.shape[0])
                    or str(row_mask.device) != str(self.device)
                ):
                    raise ValueError(
                        "device_row_mask must match the registered view rows on "
                        f"{self.device}: dtype={row_mask.dtype}, "
                        f"shape={row_mask.shape}, rows={row_world.shape[0]}, "
                        f"device={row_mask.device}"
                    )
                entry = (row_world, row_mask, device_row_mask)
                self._device_dirty_rows.append(entry)
                if projection_required:
                    self._device_projection_rows.append(entry)
            # body_q/body_qd setter 会同时写对应 FREE generalized state；full q/qd 则来自
            # snapshot/clone 的权威 solver state。这四类写入只需 FK。其它 generalized setter
            # 仍是 command/reset 语义，必须先恢复 native equality follower。
        elif category == "model" and self.solver is not None:
            from newton.solvers import SolverNotifyFlags

            # view scatter 与 solver cache invalidation 必须在同一 owner stream 排序。
            # ``sync_enter=False`` 是有意的：scatter 已入此流，无需再与 default stream
            # 建 fence，更不能为一次 gain 写引入 device-wide synchronize。
            with self._owner_stream_scope():
                self.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            if bool(getattr(self.physics_spec, "use_cuda_graph", False)):
                self._graph = None
                self._graph_state = "pending"
        elif category != "control":
            raise ValueError(f"unknown Newton view write category {category!r}")

    def subscribe_physics_step_events(self, callback: object) -> None:
        if not callable(callback):
            raise TypeError("physics step callback must be callable")
        if callback not in self._step_callbacks:
            self._step_callbacks.append(callback)

    def unsubscribe_physics_step_events(self, callback: object) -> None:
        if callback in self._step_callbacks:
            self._step_callbacks.remove(callback)

    def register_newton_view(self, view: object) -> None:
        """弱跟踪 Newton view，便于 teardown 失效化，但不延长其业务生命周期。"""

        self._require_open()
        registry = getattr(self, "_registered_views", None)
        if registry is None:
            # 兼容通过 ``__new__`` 构造的窄测试替身和异常恢复路径；生产构造器会预建 registry。
            registry = weakref.WeakSet()
            self._registered_views = registry
        registry.add(view)
        binding = getattr(view, "binding", None)
        worlds = tuple(int(world) for world in getattr(binding, "world_indices", ()))
        if worlds and self.stream is not None:
            import warp as wp

            view_world_rows = getattr(self, "_view_world_rows", None)
            if view_world_rows is None:
                view_world_rows = weakref.WeakKeyDictionary()
                self._view_world_rows = view_world_rows
            with self._owner_stream_scope():
                view_world_rows[view] = wp.array(
                    worlds,
                    dtype=wp.int32,
                    device=self.device,
                )

    def release_newton_view(self, view: object) -> None:
        """在 manager 仍存活时安全释放一个 view。

        view 的 persistent Warp buffer 可能仍被已入队 kernel 引用，因此先同步 owner stream，
        再清缓存和 manager 引用；仅靠 Python 引用计数无法保证这一 GPU 生命周期顺序。
        """

        if not self.closed:
            self._synchronize_owner_stream()
        registry = getattr(self, "_registered_views", None)
        if registry is not None:
            registry.discard(view)
        view_world_rows = getattr(self, "_view_world_rows", None)
        if view_world_rows is not None:
            view_world_rows.pop(view, None)
        release = getattr(view, "_release_from_manager", None)
        if callable(release):
            release()

    def diagnostics(self) -> dict[str, Any]:
        """返回稳定的初始化/验收 provenance，不泄漏 owner buffer。"""

        audit = self.native_master_follower_audit
        return {
            "backend": self.backend,
            "execution": self.execution,
            "device": self.device,
            "world_count": self._num_worlds,
            "nconmax_per_world": int(self.physics_spec.nconmax_per_world),
            "njmax_per_world": int(self.physics_spec.njmax_per_world),
            "constraint_solver": self._constraint_solver,
            "contact_pipeline": self._contact_pipeline_kind,
            "contact_pipeline_trigger_labels": list(
                self._contact_pipeline_trigger_labels
            ),
            "native_joint_equalities": 0 if audit is None else audit.relation_count,
            "native_joint_equalities_per_world": (
                0 if audit is None else audit.relations_per_world
            ),
            "solver_integration_state_width": self.solver_integration_state_width,
            "solver_integration_activation_width": (
                self.solver_integration_activation_width
            ),
            "solver_persistent_state_fields": [
                "time",
                "act",
                "qacc_warmstart",
            ],
            "cuda_graph": self._graph_state,
            "cuda_graph_error": self._graph_error,
            "rendering_enabled": bool(getattr(self, "_rendering_enabled", False)),
            "render_sync": (
                None
                if getattr(self, "_render_sync", None) is None
                else self._render_sync.diagnostics()
            ),
        }

    def close(self) -> None:
        """按 device → view → renderer → model → stage 引用的依赖顺序幂等销毁。"""

        if self.closed:
            return
        self.closed = True

        # 所有 model/view kernel 都入队到 owner stream。必须先等它们完成，之后才能释放
        # persistent selector、graph、state 等 Warp allocation，否则 CUDA 仍可能解引用旧地址。
        # teardown 尽量继续执行，并在全部资源断开后重抛遇到的第一个异常。
        first_error: BaseException | None = None
        try:
            self._synchronize_owner_stream()
        except BaseException as exc:
            first_error = exc

        registry = getattr(self, "_registered_views", None)
        views = () if registry is None else tuple(registry)
        for view in views:
            release = getattr(view, "_release_from_manager", None)
            if not callable(release):
                continue
            try:
                release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if registry is not None:
            registry.clear()

        # Camera/viewport 已由 Mirror 在进入 session teardown 前关闭。物理 runtime 只释放
        # 自己持有的 USD render sync，随后才清 model/state/solver/stage 引用。
        render_sync = getattr(self, "_render_sync", None)
        if render_sync is not None:
            try:
                render_sync.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._render_sync = None
        self._render_callback = None

        self._graph = None
        self._projector = None
        self._world_masks = None
        self._initial_control = None
        self._initial_state = None
        # store 持有 solver/Data 与 canonical buffer；先替换为空 store，再释放 solver/model。
        self._solver_integration_store = create_solver_integration_state_store(
            self.execution
        )
        self.control = None
        self.state = None
        self._contacts = None
        self._collision_pipeline = None
        self.solver = None
        self.model = None
        self.stream = None
        self.replication = None
        self.native_master_follower_audit = None
        self.constraint_audit = None
        self._dirty_worlds.clear()
        projection_worlds = getattr(self, "_projection_worlds", None)
        if projection_worlds is not None:
            projection_worlds.clear()
        device_dirty_rows = getattr(self, "_device_dirty_rows", None)
        if device_dirty_rows is not None:
            device_dirty_rows.clear()
        device_projection_rows = getattr(self, "_device_projection_rows", None)
        if device_projection_rows is not None:
            device_projection_rows.clear()
        view_world_rows = getattr(self, "_view_world_rows", None)
        if view_world_rows is not None:
            view_world_rows.clear()
        self._step_callbacks.clear()
        self._initialized = False
        scene = getattr(self, "scene", None)
        clear_scene = getattr(scene, "clear", None)
        if callable(clear_scene):
            clear_scene()
        self.stage = None

        if first_error is not None:
            raise first_error

    def _owner_stream_scope(self) -> object:
        """CUDA 返回 owner stream scope，CPU 返回空 context。"""

        if self.stream is None:
            return nullcontext()
        import warp as wp

        return wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False)

    def _synchronize_owner_stream(self) -> None:
        stream = getattr(self, "stream", None)
        if stream is None:
            return
        import warp as wp

        wp.synchronize_stream(stream)

    def _simulate(self) -> None:
        assert self.solver is not None
        assert self.state is not None
        assert self.control is not None
        substep_dt = self.physics_dt / int(self.physics_spec.substeps)
        with self._owner_stream_scope():
            # contacts buffer 与 solver/state 都归 owner。CUDA 可捕获整段 DAG；CPU 在同一
            # 结构下 eager 执行。state 同时作为输入/输出是 Newton 原地积分的预期 ABI。
            contacts = None
            if self._collision_pipeline is not None:
                assert self._contacts is not None
                self._collision_pipeline.collide(self.state, self._contacts)
                contacts = self._contacts
            for _ in range(int(self.physics_spec.substeps)):
                self.solver.step(
                    self.state,
                    self.state,
                    self.control,
                    contacts,
                    substep_dt,
                )
                self.state.clear_forces()

    def _initialize_solver_integration_state(
        self,
        solver: object,
        *,
        device: object,
    ) -> None:
        """初始化 execution 对应的 TIME|ACT|WARMSTART canonical store。"""

        self._solver_integration_store.initialize(
            solver,
            world_count=self._num_worlds,
            device=device,
            stream=self.stream,
        )

    def _capture_solver_integration_state(self) -> None:
        """把当前 MuJoCo persistent Data 捕获到 execution-local buffer。"""

        if self._solver_integration_store.width == 0:
            # ``__new__`` 构造的窄单元测试和初始化失败恢复路径可能尚未分配 solver buffer。
            return
        self._solver_integration_store.capture()

    def _restore_initial_solver_integration_state(self) -> None:
        """恢复全 world solver baseline。"""

        if self._solver_integration_store.width == 0:
            return
        self._solver_integration_store.reset()

    def _project_all_worlds(self) -> None:
        worlds = range(self._num_worlds)
        self._dirty_worlds.update(worlds)
        self._projection_worlds.update(range(self._num_worlds))
        self._flush_cold_state_updates()

    def _flush_cold_state_updates(self) -> None:
        if not self._dirty_worlds and not self._device_dirty_rows:
            return
        import newton

        assert self.model is not None and self.state is not None
        assert self._projector is not None
        assert self._world_masks is not None
        selected_worlds = tuple(sorted(self._dirty_worlds))
        projection_worlds = tuple(sorted(self._projection_worlds))
        device_dirty_rows = tuple(
            (row_world, row_mask)
            for row_world, row_mask, _owner in self._device_dirty_rows
        )
        device_projection_rows = tuple(
            (row_world, row_mask)
            for row_world, row_mask, _owner in self._device_projection_rows
        )
        with self._owner_stream_scope():
            # projection 与 FK 可以覆盖不同 world：full-state restore 只进入 FK 集合，因而
            # solver 动态产生的 follower q/qd 会原样保留。mask 均由初始化时预分配的 device
            # buffer 或 Warp kernel 生成；这里不读取 articulation_world，也不上传 NumPy mask。
            projection_mask = None
            if projection_worlds or device_projection_rows:
                projection_mask = self._world_masks.world_mask(
                    projection_worlds,
                    masked_rows=device_projection_rows,
                )
                self._projector.project(
                    joint_q=self.state.joint_q,
                    joint_qd=self.state.joint_qd,
                    selected_world_mask=projection_mask,
                    stream=self.stream,
                )
            if len(selected_worlds) == self._num_worlds:
                fk_mask = None
            else:
                # 即使 device mask 全 false 也不能在 Python 读取其值来跳过。这里重新构造
                # dirty mask，并让 articulation-map kernel 与 eval_fk(mask) 自然成为 no-op。
                selected_mask = self._world_masks.world_mask(
                    selected_worlds,
                    masked_rows=device_dirty_rows,
                )
                fk_mask = self._world_masks.articulation_mask(selected_mask)
            newton.eval_fk(
                self.model,
                self.state.joint_q,
                self.state.joint_qd,
                self.state,
                fk_mask,
            )
        self._dirty_worlds.clear()
        self._projection_worlds.clear()
        self._device_dirty_rows.clear()
        self._device_projection_rows.clear()

    def _require_mujoco_variants(self, robots: Mapping[str, object]) -> None:
        getter = getattr(self.stage, "GetPrimAtPath", None)
        if not callable(getter):
            raise RuntimeError("Newton requires a USD stage with prim lookup")
        for name, robot in robots.items():
            paths = _robot_imported_root_paths(robot)
            if not paths:
                raise RuntimeError(f"robot {name!r} has no imported root paths")
            prim = getter(paths[0])
            variant = prim.GetVariantSet("Physics")
            selected = str(variant.GetVariantSelection()).strip().lower()
            if selected != "mujoco":
                raise RuntimeError(
                    "Newton requires the importer mujoco Physics variant: "
                    f"robot={name!r}, selected={selected!r}"
                )

    def _require_initialized(self) -> None:
        self._require_open()
        if not self._initialized:
            raise RuntimeError("Newton model has not been initialized")

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Newton manager is closed")


def _resolve_constraint_solver(
    configured: str,
    *,
    object_handles: Sequence[object],
) -> str:
    """在分配 GPU 资源前解析 MuJoCo constraint solver。

    dynamic chain 与 Newton 1.2.1 的接触稳定性要求属于启动期 capability 判定，不能等到
    首次 step 后再降级或热切换 solver。
    """

    if configured not in {"auto", "newton", "cg"}:
        raise ValueError(
            "Newton constraint_solver must be 'auto', 'newton', or 'cg', "
            f"got {configured!r}"
        )
    has_dynamic_chain = any(
        str(getattr(item, "kind", "")) == "dynamic_chain" for item in object_handles
    )
    if not has_dynamic_chain:
        return "newton" if configured == "auto" else configured
    if configured == "newton":
        raise RuntimeError(
            "Newton constraint_solver='newton' is not supported for "
            "dynamic_chain scenes: Newton 1.2.1/MuJoCo-Warp produced non-finite "
            "state when the chain first activated contact; use 'auto' or 'cg'"
        )
    return "cg"


def _resolve_contact_pipeline(
    configured: str,
    *,
    trigger_labels: Sequence[str],
    execution: str,
) -> str:
    """在 model finalize 冷路径选择 MuJoCo 或 Newton 接触检测。

    这里选择的是真实物理接触 pipeline，不是规划碰撞查询；Kaleidoscope 禁用避障并不
    允许跳过该判定。
    """

    if configured not in {"auto", "mujoco", "newton"}:
        raise ValueError(
            "Newton contact_pipeline must be 'auto', 'mujoco', or "
            f"'newton', got {configured!r}"
        )
    labels = tuple(str(label) for label in trigger_labels)
    if execution == "cpu":
        if configured == "newton":
            raise RuntimeError(
                "Newton CPU requires contact_pipeline='mujoco' or 'auto'; "
                "the upstream CPU solver ignores Newton contact buffers"
            )
        if labels:
            raise RuntimeError(
                "Newton CPU cannot simulate colliding planar mesh geometry because "
                "MuJoCo C rejects that collider and the CPU solver cannot consume the "
                f"Newton contact pipeline: labels={list(labels)!r}"
            )
        return "mujoco"
    if execution != "cuda":
        raise ValueError(f"unsupported Newton execution {execution!r}")
    if labels and configured == "mujoco":
        raise RuntimeError(
            "Newton contact_pipeline='mujoco' does not support colliding "
            "planar mesh colliders; use 'auto' or 'newton': "
            f"labels={list(labels)!r}"
        )
    if configured == "auto":
        return "newton" if labels else "mujoco"
    return configured


def _colliding_planar_mesh_labels(model: object) -> tuple[str, ...]:
    """按 Newton 1.2.1 容差返回参与接触的平面 mesh 标签。"""

    shape_type = _host_array(getattr(model, "shape_type"), dtype=np.int32).reshape(-1)
    shape_flags = _host_array(getattr(model, "shape_flags"), dtype=np.int32).reshape(-1)
    shape_group = _host_array(
        getattr(model, "shape_collision_group"), dtype=np.int32
    ).reshape(-1)
    shape_scale = _host_array(getattr(model, "shape_scale"), dtype=np.float64).reshape(
        -1, 3
    )
    shape_sources = tuple(getattr(model, "shape_source"))
    shape_labels = tuple(str(label) for label in getattr(model, "shape_label"))
    count = len(shape_type)
    if not all(
        len(values) == count
        for values in (
            shape_flags,
            shape_group,
            shape_scale,
            shape_sources,
            shape_labels,
        )
    ):
        raise RuntimeError("Newton shape metadata lengths are inconsistent")

    explicit_pair_shapes = _mujoco_pair_shape_indices(model)
    labels: list[str] = []
    for index in range(count):
        if int(shape_type[index]) not in _NEWTON_MESH_GEO_TYPES:
            continue
        uses_contacts = (
            bool(int(shape_flags[index]) & _NEWTON_COLLIDE_SHAPES_FLAG)
            and int(shape_group[index]) != 0
        ) or index in explicit_pair_shapes
        if not uses_contacts:
            continue
        source = shape_sources[index]
        vertices = np.asarray(getattr(source, "vertices", ()), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise RuntimeError(
                "Newton colliding mesh has invalid vertices: "
                f"label={shape_labels[index]!r}, shape={vertices.shape}"
            )
        vertices = vertices * shape_scale[index]
        extent_axis = vertices.max(axis=0) - vertices.min(axis=0)
        if _mujoco_mesh_vertices_are_planar(vertices, extent_axis):
            labels.append(shape_labels[index])
    return tuple(labels)


def _mujoco_pair_shape_indices(model: object) -> set[int]:
    """收集显式 MuJoCo contact pair 引用的 shape 索引。"""

    counts = getattr(model, "custom_frequency_counts", {})
    pair_count = int(counts.get("mujoco:pair", 0))
    attrs = getattr(model, "mujoco", None)
    if attrs is None or pair_count <= 0:
        return set()
    geom1 = getattr(attrs, "pair_geom1", None)
    geom2 = getattr(attrs, "pair_geom2", None)
    if geom1 is None or geom2 is None:
        return set()
    geom1_values = _host_array(geom1, dtype=np.int32).reshape(-1)
    geom2_values = _host_array(geom2, dtype=np.int32).reshape(-1)
    pair_count = min(pair_count, len(geom1_values), len(geom2_values))
    return {
        int(shape)
        for shape in np.concatenate(
            (geom1_values[:pair_count], geom2_values[:pair_count])
        )
        if int(shape) >= 0
    }


def _mujoco_mesh_vertices_are_planar(
    vertices: np.ndarray,
    extent_axis: np.ndarray | None = None,
    eps: float = 1.0e-6,
) -> bool:
    """复现 Newton 1.2.1 的 MuJoCo mesh 共面判定，保持 pipeline 选择一致。"""

    if len(vertices) < 3:
        return False
    vertices = np.asarray(vertices)
    if extent_axis is None:
        extent_axis = vertices.max(axis=0) - vertices.min(axis=0)
    extent_axis = np.asarray(extent_axis)
    tolerance = eps * max(float(np.linalg.norm(extent_axis)), eps)
    if np.all(extent_axis <= tolerance) or np.any(extent_axis <= tolerance):
        return True
    if len(vertices) > 64:
        sample_indices = np.linspace(0, len(vertices) - 1, 64, dtype=np.int32)
        if not _points_are_planar(vertices[sample_indices], tolerance):
            return False
    return _points_are_planar(vertices, tolerance)


def _points_are_planar(points: np.ndarray, tolerance: float) -> bool:
    p0 = points[0]
    offsets = points - p0
    distances_from_p0 = np.linalg.norm(offsets, axis=1)
    p1 = int(np.argmax(distances_from_p0))
    line_length = distances_from_p0[p1]
    if line_length <= tolerance:
        return True
    line = points[p1] - p0
    cross = np.cross(offsets, line)
    cross_lengths = np.linalg.norm(cross, axis=1)
    p2 = int(np.argmax(cross_lengths))
    normal_length = cross_lengths[p2]
    if normal_length <= tolerance * line_length:
        return True
    normal = cross[p2] / normal_length
    return bool(np.all(np.abs(offsets @ normal) <= tolerance))


def _solver_constructor_kwargs(
    settings: IsaacNewtonCpuSpec | IsaacNewtonCudaSpec,
    *,
    world_count: int,
    constraint_solver: str,
    contact_pipeline: str,
) -> dict[str, object]:
    """不导入 Newton 即构造生产 ``SolverMuJoCo`` 参数。

    该纯函数冻结 Mirror 单 world 与 Kaleidoscope per-env independent worlds 的差异，避免
    solver 构造点再从 mode 名称猜测产品语义。
    """

    if type(world_count) is not int or world_count <= 0:
        raise ValueError(f"Newton world_count must be positive, got {world_count!r}")
    if constraint_solver not in {"newton", "cg"}:
        raise ValueError(
            "effective Newton constraint solver must be 'newton' or 'cg', "
            f"got {constraint_solver!r}"
        )
    if contact_pipeline not in {"mujoco", "newton"}:
        raise ValueError(
            "effective Newton contact pipeline must be 'mujoco' or "
            f"'newton', got {contact_pipeline!r}"
        )
    cpu_execution = isinstance(settings, IsaacNewtonCpuSpec)
    if cpu_execution and world_count != 1:
        raise ValueError("Newton CPU solver requires world_count=1")
    if cpu_execution and contact_pipeline != "mujoco":
        raise ValueError("Newton CPU solver requires the MuJoCo contact pipeline")
    return {
        # CPU 使用一个 MuJoCo C Data；CUDA replicated model 为每个环境保留独立 world。
        "separate_worlds": False if cpu_execution else world_count > 1,
        "njmax": int(settings.njmax_per_world),
        "nconmax": int(settings.nconmax_per_world),
        "iterations": int(settings.iterations),
        "ls_iterations": int(settings.line_search_iterations),
        "solver": constraint_solver,
        "use_mujoco_cpu": cpu_execution,
        "use_mujoco_contacts": contact_pipeline == "mujoco",
        # 每一拍都先用 Newton generalized state 刷新 MJWarp qpos/qvel。这样完整
        # mjSTATE_INTEGRATION 可以跨 world clone，同时 target world 的绝对 origin 仍由
        # Newton state 决定；Python-side solver step counter 也不进入 snapshot 合同。
        "update_data_interval": 1,
    }


def _is_canonical_cuda_device(value: object) -> bool:
    if not isinstance(value, str):
        return False
    prefix, separator, index = value.partition(":")
    return (
        prefix == "cuda"
        and separator == ":"
        and index.isdecimal()
        and str(int(index)) == index
    )


def _configure_newton_stage(
    stage: object,
    *,
    add_ground: bool,
    ground_height: float,
    prepare_newton_render_topology: bool = False,
) -> None:
    from pxr import Sdf, UsdGeom

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world_path = Sdf.Path("/World")
    if not stage.GetPrimAtPath(world_path).IsValid():
        UsdGeom.Xform.Define(stage, world_path)
    if prepare_newton_render_topology:
        from linkerbot_sim.assets.root_pose import (
            RootPoseConfig,
            apply_root_pose_transform,
        )

        # Mirror 以 /World 作为 prototype。它必须和新 prim 一起发布最终 topology，
        # 不能等 manager finalize 后再首次添加 matrix op。
        apply_root_pose_transform(
            stage,
            str(world_path),
            RootPoseConfig(),
            prepare_newton_render_topology=True,
        )
    if not add_ground:
        return
    from pxr import Gf, PhysicsSchemaTools, UsdPhysics, UsdShade

    ground_path = "/World/defaultGroundPlane"
    if stage.GetPrimAtPath(ground_path).IsValid():
        raise RuntimeError(f"Newton ground prim already exists: {ground_path}")
    PhysicsSchemaTools.addGroundPlane(
        stage,
        ground_path,
        "Z",
        50.0,
        Gf.Vec3f(0.0, 0.0, float(ground_height)),
        Gf.Vec3f(0.5, 0.5, 0.5),
    )
    material = UsdShade.Material.Define(stage, f"{ground_path}/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(0.5)
    material_api.CreateDynamicFrictionAttr().Set(0.5)
    material_api.CreateRestitutionAttr().Set(0.8)
    collision_prim = stage.GetPrimAtPath(f"{ground_path}/geom")
    if not collision_prim.IsValid():
        raise RuntimeError("Newton ground collision prim was not created")
    UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _world_transforms(origins: object, *, device: object) -> tuple[object, ...]:
    del device
    import warp as wp

    matrix = np.asarray(origins, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"env origins must have shape (N, 3), got {matrix.shape}")
    return tuple(
        wp.transform(tuple(float(value) for value in row), (0.0, 0.0, 0.0, 1.0))
        for row in matrix
    )


def _audit_prototype_constraints(
    prototype: object,
    *,
    expected_relation_count: int,
) -> None:
    # 当前 prototype 期望的全部主从关系在 MuJoCo Physics variant 下都应落为普通 JOINT
    # equality；这里明确拒绝同时存在 mimic 表示，防止一个 follower 被两套求解器重复执行。
    mimic_count = len(getattr(prototype, "constraint_mimic_joint0", ()))
    equality_types = _host_array(
        getattr(prototype, "equality_constraint_type", ()), dtype=np.int32
    ).reshape(-1)
    joint_equality_count = int(np.count_nonzero(equality_types == 2))
    if mimic_count != 0 or joint_equality_count != expected_relation_count:
        raise RuntimeError(
            "Newton prototype must preserve ordinary joint equalities: "
            f"constraint_mimic={mimic_count}, "
            f"joint_equality={joint_equality_count}, "
            f"expected={expected_relation_count}"
        )


def _asset_expectations(
    *,
    model: object,
    robots: Mapping[str, object],
    world_count: int,
) -> tuple[ExpectedMasterFollowerConstraint, ...]:
    labels = tuple(str(value) for value in getattr(model, "joint_label", ()))
    worlds = _host_array(getattr(model, "joint_world"), dtype=np.int32).reshape(-1)
    if len(labels) != worlds.size:
        raise RuntimeError("Newton joint label/world columns have different lengths")
    result: list[ExpectedMasterFollowerConstraint] = []
    for world in range(world_count):
        for robot_name, robot in robots.items():
            roots = _robot_imported_root_paths(robot)
            if len(roots) != world_count:
                raise RuntimeError(
                    f"robot {robot_name!r} root count does not match Newton worlds"
                )
            relations = parse_mjcf_joint_equalities(getattr(robot, "asset_path", None))
            for relation in relations:
                follower = _resolve_joint_label(
                    labels,
                    worlds,
                    world=world,
                    root=roots[world],
                    joint_name=relation.dependent_joint,
                )
                master = _resolve_joint_label(
                    labels,
                    worlds,
                    world=world,
                    root=roots[world],
                    joint_name=relation.master_joint,
                )
                result.append(
                    ExpectedMasterFollowerConstraint(
                        world=world,
                        follower_joint_label=follower,
                        master_joint_label=master,
                        polycoef=relation.polycoef,
                    )
                )
    return tuple(result)


def _robot_imported_root_paths(robot: object) -> tuple[str, ...]:
    """统一 replicated 复数与 Mirror 单数两种 imported-root ABI。"""

    plural = tuple(
        str(path) for path in (getattr(robot, "imported_root_paths", ()) or ())
    )
    if plural:
        return plural
    singular = getattr(robot, "imported_root_path", None)
    if singular is None or not str(singular).strip():
        return ()
    return (str(singular),)


def _resolve_joint_label(
    labels: Sequence[str],
    worlds: np.ndarray,
    *,
    world: int,
    root: str,
    joint_name: str,
) -> str:
    normalized_root = root.rstrip("/")
    candidates = [
        label
        for label, joint_world in zip(labels, worlds, strict=True)
        if int(joint_world) == world
        and (label == normalized_root or label.startswith(normalized_root + "/"))
        and label.rsplit("/", 1)[-1] == joint_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "asset joint must resolve to one exact Newton label: "
            f"world={world}, root={root!r}, joint={joint_name!r}, "
            f"matches={candidates}"
        )
    return candidates[0]


def _executor_metadata(
    *,
    stage: object,
    model: object,
    expectations: Sequence[ExpectedMasterFollowerConstraint],
    replication: NewtonReplicationResult,
) -> MasterFollowerExecutorMetadata:
    # equality 数组只能证明 Newton 内部表示，不能证明 USD DriveAPI 或 importer actuator
    # 已消失。这里把 stage/model 两侧的反证收集成 provenance，交给约束审计 fail closed。
    follower_labels = {item.follower_joint_label for item in expectations}
    follower_drive_paths: set[str] = set()
    replicated_labels_by_prototype: dict[str, set[str]] = {}
    for item in expectations:
        prototype_label = _prototype_label_for_world(
            item.follower_joint_label,
            world=item.world,
            replication=replication,
        )
        replicated_labels_by_prototype.setdefault(prototype_label, set()).add(
            item.follower_joint_label
        )
    for prototype_label, replicated_labels in replicated_labels_by_prototype.items():
        prim = stage.GetPrimAtPath(prototype_label)
        if prim is None or not bool(prim.IsValid()):
            raise RuntimeError(
                "Newton follower joint is absent from the USD prototype: "
                f"{prototype_label}"
            )
        schemas = tuple(str(value) for value in prim.GetAppliedSchemas())
        if any("DriveAPI" in schema for schema in schemas):
            # 即使目标 USD prim 刻意不物化，model 中每个映射 world 仍继承 prototype schema。
            follower_drive_paths.update(replicated_labels)

    labels = tuple(str(value) for value in getattr(model, "joint_label", ()))
    qd_start = _host_array(getattr(model, "joint_qd_start"), dtype=np.int32).reshape(-1)
    stiffness = _host_array(
        getattr(model, "joint_target_ke"), dtype=np.float32
    ).reshape(-1)
    damping = _host_array(getattr(model, "joint_target_kd"), dtype=np.float32).reshape(
        -1
    )
    by_label = {label: index for index, label in enumerate(labels)}
    for label in follower_labels:
        joint = by_label[label]
        dof = int(qd_start[joint])
        if float(stiffness[dof]) != 0.0 or float(damping[dof]) != 0.0:
            follower_drive_paths.add(label)

    follower_actuators: set[str] = set()
    mujoco_attributes = getattr(model, "mujoco", None)
    targets = getattr(mujoco_attributes, "actuator_target_label", ())
    for value in targets or ():
        label = str(value)
        if label in follower_labels:
            follower_actuators.add(label)
    return MasterFollowerExecutorMetadata(
        dynamic_executor=NATIVE_JOINT_EQUALITY_EXECUTOR,
        state_projection_scope=COLD_STATE_PROJECTION_SCOPE,
        runtime_target_writer=None,
        follower_drive_prim_paths=tuple(sorted(follower_drive_paths)),
        follower_actuator_labels=tuple(sorted(follower_actuators)),
    )


def _prototype_label_for_world(
    label: str,
    *,
    world: int,
    replication: NewtonReplicationResult,
) -> str:
    """把 replicated model 标签映射回实际存在的 USD prototype prim。"""

    destinations = tuple(
        str(path).rstrip("/") for path in replication.destination_roots
    )
    selected_world = int(world)
    if selected_world < 0 or selected_world >= len(destinations):
        raise RuntimeError(
            "Newton executor metadata references an invalid world: "
            f"world={selected_world}, world_count={len(destinations)}"
        )
    destination = destinations[selected_world]
    source = str(replication.prototype_root).rstrip("/")
    value = str(label)
    if value == destination:
        suffix = ""
    elif value.startswith(destination + "/"):
        suffix = value[len(destination) :]
    else:
        raise RuntimeError(
            "Newton model label is outside its destination root: "
            f"world={selected_world}, label={value!r}, destination={destination!r}"
        )
    return source + suffix


def _audit_solver_equality_mapping(
    solver: object,
    audit: NativeMasterFollowerAudit,
) -> None:
    mapping = _host_array(getattr(solver, "mjc_eq_to_newton_eq"), dtype=np.int32)
    if mapping.ndim != 2 or mapping.shape[0] != audit.world_count:
        raise RuntimeError(
            "SolverMuJoCo equality mapping has the wrong world shape: "
            f"actual={mapping.shape}, expected_worlds={audit.world_count}"
        )
    all_expected = {binding.equality_index for binding in audit.bindings}
    for world in range(audit.world_count):
        expected = {
            binding.equality_index for binding in audit.bindings_for_world(world)
        }
        mapped = [int(value) for value in mapping[world] if int(value) in all_expected]
        if len(mapped) != len(expected) or set(mapped) != expected:
            raise RuntimeError(
                "SolverMuJoCo did not map all native joint equalities for one world: "
                f"world={world}, mapped={mapped}, expected={sorted(expected)}"
            )


def _copy_control(
    destination: object,
    source: object,
    *,
    stream: object | None = None,
) -> None:
    import warp as wp

    for name in ("joint_f", "joint_target_pos", "joint_target_vel", "joint_act"):
        target = getattr(destination, name, None)
        value = getattr(source, name, None)
        if target is not None and value is not None:
            wp.copy(target, value, stream=stream)


def _host_array(value: object, *, dtype: object) -> np.ndarray:
    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate, dtype=dtype)


__all__ = ["NewtonRuntime"]
