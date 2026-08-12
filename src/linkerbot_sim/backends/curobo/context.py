"""cuRobo 后端共享上下文。

``CuroboContext`` 负责把项目配置转换成 cuRobo 的 DeviceCfg、Kinematics、IK solver 和
planner 实例。第三方导入和 CUDA 资源创建都延迟到构造函数中，保证没有完整 cuRobo 运行时
依赖时，普通配置解析和 fake 单元测试仍可执行。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from linkerbot_sim.backends.curobo.config import CuroboConfig
from linkerbot_sim.backends.curobo.runtime_imports import (
    ensure_torch_device_usable,
    import_curobo_module,
    import_curobo_public,
    import_torch_module,
    require_curobo_kernel_backend,
)
from linkerbot_sim.backends.curobo.robot_model import (
    materialize_curobo_config,
    materialized_robot_mapping,
)
from linkerbot_sim.backends.curobo.resources import curobo_task_resource_path
from linkerbot_sim.backends.curobo.tool_pose import goal_tool_pose_from_arrays
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


_COLLISION_CONSUMER_SOLVER_ATTRIBUTES = {
    "ik": "_ik_solver",
    "planner": "_motion_planner",
}
_COLLISION_CONSUMER_LABELS = {
    "ik": "IK",
    "planner": "MotionPlanner",
}
_MOTION_PLANNER_WARMUP_ITERATIONS = 1


@dataclass(frozen=True)
class CollisionCapability:
    """一个 cuRobo context 的可验证碰撞规划能力。"""

    robot_sphere_count: int
    robot_collision_model_available: bool
    scene_checker_available: bool
    supported_cache_types: tuple[str, ...]
    required_cache: Mapping[str, int]
    configured_cache: Mapping[str, int]
    cache_capacity_sufficient: bool
    synced_scene_version: int | None = None
    materialized_view_fingerprint: str | None = None

    @property
    def available(self) -> bool:
        """是否同时具备 robot、scene checker 和足够 cache。"""

        return bool(
            self.robot_collision_model_available
            and self.scene_checker_available
            and self.configured_cache
            and self.cache_capacity_sufficient
            and self.synced_scene_version is not None
            and self.materialized_view_fingerprint is not None
        )

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        """返回稳定、可直接用于错误响应的缺失项。"""

        missing = []
        if not self.robot_collision_model_available:
            missing.append("robot_collision_spheres")
        if not self.configured_cache:
            missing.append("scene_collision_cache")
        if not self.cache_capacity_sufficient:
            missing.append("scene_collision_cache_capacity")
        if not self.scene_checker_available:
            missing.append("scene_collision_checker")
        if self.synced_scene_version is None:
            missing.append("scene_version_sync")
        if self.materialized_view_fingerprint is None:
            missing.append("materialized_view_fingerprint")
        return tuple(missing)


class CuroboContext:
    """一个机器人模型对应的长期 cuRobo context。"""

    def __init__(
        self,
        config: CuroboConfig,
        *,
        cache_root: str | Path | None = None,
    ) -> None:
        """加载 cuRobo public modules，并创建 kinematics / solvers。"""

        if not isinstance(config, CuroboConfig):
            raise TypeError("config must be CuroboConfig")
        # Context 是最终 runtime 边界；即使调用方绕过正式 profile composition 直接构造
        # typed dataclass，也必须在导入 cuRobo 或创建 CUDA 资源前完成完整校验。
        config.validate()
        source_urdf_path = config.robot.urdf_path
        curobo_module = import_curobo_module()
        config.task_bundle.validate_curobo_version(
            getattr(curobo_module, "__version__", "unknown")
        )
        self.kernel_backend = require_curobo_kernel_backend(expected="cuda_core")
        self.config = materialize_curobo_config(config, cache_root=cache_root)
        self._robot_asset_root_path = (
            None if source_urdf_path is None else source_urdf_path.parent
        )
        self.torch = import_torch_module()
        ensure_torch_device_usable(self.torch, self.config.device.device)
        self.types = import_curobo_public("types")
        self.kinematics_module = import_curobo_public("kinematics")
        self.ik_module = import_curobo_public("inverse_kinematics")
        self.motion_module = import_curobo_public("motion_planner")
        self.scene_module = import_curobo_public("scene")
        self.robot_module = import_curobo_public("_src.types.robot")
        self.device_cfg = self._make_device_cfg()
        self.default_tcp_frame = (
            self.config.robot.default_tcp_frame
            or self.config.robot.resolved_tool_frames[0]
        )
        self.tool_frames = tuple(self.config.robot.resolved_tool_frames)
        self.kinematics = self._make_kinematics()
        # cuRobo planner warmup 会创建求解缓存，因此 solver/planner 不能在 context 构造时
        # 全部创建；按实际调用入口 lazy 创建，让 IK-only/FK-only 调用不承担 planner 显存。
        self._ik_solver = None
        self._motion_planner = None
        self._collision_world = None
        self._synced_scene_version = None
        self._materialized_view_fingerprint = None
        self._local_scene_version = 0

    @property
    def ik_solver(self):
        """按需创建 IK solver。"""

        if self._ik_solver is None:
            self._ik_solver = self._make_ik_solver()
            self._sync_solver_world_if_available(self._ik_solver, consumer="ik")
        return self._ik_solver

    @property
    def motion_planner(self):
        """按需创建单问题 ``MotionPlanner``。"""

        if self._motion_planner is None:
            self._motion_planner = self._make_motion_planner()
            self._sync_solver_world_if_available(
                self._motion_planner,
                consumer="planner",
            )
        return self._motion_planner

    def existing_solvers(self) -> tuple[object, ...]:
        """返回已经创建的 cuRobo solver/planner，不触发 lazy 初始化。"""

        return tuple(
            solver
            for solver in (
                self._ik_solver,
                self._motion_planner,
            )
            if solver is not None
        )

    def close(self) -> None:
        """逐个释放 CUDA graph；失败 solver 保留所有权，允许再次关闭。"""

        first_error: BaseException | None = None
        for name in ("_motion_planner", "_ik_solver"):
            solver = getattr(self, name, None)
            destroy = getattr(solver, "destroy", None)
            try:
                if callable(destroy):
                    destroy()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                continue
            setattr(self, name, None)
        if first_error is not None:
            raise first_error

    @property
    def collision_world(self):
        """返回 context 当前同步的 cuRobo collision world。"""

        if self._collision_world is None:
            return self.sync_collision_world(())
        return self._collision_world

    def sync_collision_world(self, collision_objects: Sequence[CollisionObject] = ()):
        """把项目碰撞对象同步到 cuRobo Scene，并更新 IK/planner。

        cuRobo 的公开 API 以 ``SceneCfg`` 整体更新 world；因此这里不维护单个 obstacle
        handle，而是复用 ``CuroboCollisionWorld`` 重建并推送当前快照。若当前 robot
        只是基础 URDF fallback，没有 collision spheres 或 scene collision cache，world 对象仍
        会保存快照，但不会强行调用不支持 collision 的 solver。
        """

        from linkerbot_sim.backends.curobo.collision_world import (
            CuroboCollisionWorld,
        )

        objects = tuple(collision_objects)
        if self._collision_world is None:
            self._collision_world = CuroboCollisionWorld(self, objects)
        else:
            self._collision_world.sync(objects)
        self._local_scene_version = getattr(self, "_local_scene_version", 0) + 1
        self.record_collision_sync(
            self._local_scene_version,
            _collision_objects_fingerprint(objects),
        )
        return self._collision_world

    def joint_names(self) -> list[str]:
        """返回 cuRobo active C-space 关节名顺序。"""

        names = getattr(self.kinematics, "joint_names")
        return list(names() if callable(names) else names)

    def frame_names(self) -> list[str]:
        """返回当前 context 注册的 tool frames。"""

        return list(self.tool_frames)

    def collision_queries_enabled(self) -> bool:
        """返回当前 context 是否具备真实机器人/环境碰撞查询能力。"""

        return self.collision_capability().available

    def ensure_collision_checker(self, consumer: str) -> CollisionCapability:
        """为一次明确的碰撞请求创建对应 lazy solver，并返回能力。"""

        normalized = _normalize_collision_consumer(consumer)
        if normalized == "ik":
            self.ik_solver
        else:
            self.motion_planner
        return self.collision_capability(consumer=normalized)

    def collision_capability(
        self,
        *,
        consumer: str | None = None,
    ) -> CollisionCapability:
        """返回指定 consumer 的 robot、checker 和 cache 分项诊断。

        未指定 consumer 时只聚合已经创建的 solver，供 status 查询使用且不触发 lazy
        allocation。明确规划请求必须通过 ``ensure_collision_checker`` 选择实际 consumer。
        """

        required = self._required_collision_cache()
        normalized = (
            None if consumer is None else _normalize_collision_consumer(consumer)
        )
        configured = self._configured_collision_cache(normalized)
        solvers = self._collision_solvers(normalized)
        checker_available = bool(solvers) and all(
            getattr(solver, "scene_collision_checker", None) is not None
            for solver in solvers
        )
        return CollisionCapability(
            robot_sphere_count=self.robot_sphere_count(),
            robot_collision_model_available=self._supports_collision_queries(True),
            scene_checker_available=checker_available,
            supported_cache_types=("cuboid", "mesh"),
            required_cache=required,
            configured_cache=configured,
            cache_capacity_sufficient=_collision_cache_capacity_sufficient(
                required,
                configured,
            ),
            synced_scene_version=getattr(self, "_synced_scene_version", None),
            materialized_view_fingerprint=getattr(
                self, "_materialized_view_fingerprint", None
            ),
        )

    def record_collision_sync(
        self,
        scene_version: int,
        materialized_view_fingerprint: str,
    ) -> None:
        """记录 shared scene version 与 materialized view fingerprint diagnostics。"""

        self._synced_scene_version = int(scene_version)
        self._materialized_view_fingerprint = str(materialized_view_fingerprint)

    def robot_sphere_count(self) -> int:
        """读取真实 kinematics robot collision sphere 数量。"""

        value = getattr(getattr(self, "kinematics", None), "total_spheres", 0)
        if callable(value):
            value = value()
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def validate_collision_cache_capacity(
        self,
        required: Mapping[str, int],
        *,
        consumer: str | None = None,
    ) -> None:
        """在 ``update_world`` 前校验实际 consumer 的 cache 容量。"""

        if getattr(
            self, "config", None
        ) is None or not self._supports_collision_queries(True):
            return
        consumers = (
            (_normalize_collision_consumer(consumer),)
            if consumer is not None
            else tuple(
                normalized
                for normalized in self._existing_collision_consumers()
                if any(
                    callable(getattr(solver, "update_world", None))
                    and (
                        not hasattr(solver, "scene_collision_checker")
                        or getattr(solver, "scene_collision_checker") is not None
                    )
                    for solver in self._collision_solvers(normalized)
                )
            )
        )
        for normalized in consumers:
            label = _collision_consumer_label(normalized)
            cache = self._collision_cache_for_consumer(normalized)
            for shape, count in required.items():
                capacity = int(cache.get(shape, 0))
                if int(count) > capacity:
                    raise ValueError(
                        f"cuRobo {label} collision cache capacity is insufficient "
                        f"for {shape}: required={int(count)}, configured={capacity}"
                    )

    def joint_state_from_positions(self, positions: object):
        """把 C-space 数组或 device tensor 转换成 cuRobo ``JointState``。

        已位于 cuRobo device/dtype 的 tensor 保持在设备上；NumPy/序列输入只在这个
        外部边界上传一次。这样顺序 IK 可以直接复用上一 waypoint 的 CUDA 解。
        """

        if self.torch.is_tensor(positions):
            tensor = positions.to(
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
        else:
            array = np.ascontiguousarray(positions, dtype=float)
            tensor = self.torch.as_tensor(
                array,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        return self.types.JointState.from_position(
            tensor,
            joint_names=self.joint_names(),
        )

    def goal_tool_pose_from_arrays(
        self,
        *,
        positions: np.ndarray,
        orientations_wxyz: np.ndarray | None,
        tool_frames,
    ):
        """把项目 batch pose 数组转为 cuRobo ``GoalToolPose``。"""

        return goal_tool_pose_from_arrays(
            positions=positions,
            orientations_wxyz=orientations_wxyz,
            tool_frames=tool_frames,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

    def compute_tcp_poses(
        self,
        joint_positions: np.ndarray,
        *,
        tcp_frame_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量计算指定 TCP frame 的位姿。"""

        frame_name = str(tcp_frame_name or self.default_tcp_frame)
        state = self.kinematics.compute_kinematics(
            self.joint_state_from_positions(joint_positions)
        )
        pose = state.tool_poses.get_link_pose(frame_name)
        return (
            tensor_like_to_numpy(pose.position, dtype=float),
            tensor_like_to_numpy(pose.quaternion, dtype=float),
        )

    def make_forward_kinematics(self):
        """创建正运动学组件。"""

        from linkerbot_sim.backends.curobo.forward_kinematics import (
            CuroboForwardKinematics,
        )

        return CuroboForwardKinematics(self)

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。"""

        from linkerbot_sim.backends.curobo.inverse_kinematics import (
            CuroboInverseKinematics,
        )

        frame_name = self._resolve_tcp_frame_name(tcp_frame_name)
        return CuroboInverseKinematics(
            self,
            tcp_frame_name=frame_name,
        )

    def make_motion_planner(self, *, tcp_frame_name: str | None = None):
        """创建路径级运动规划组件。"""

        from linkerbot_sim.backends.curobo.motion_planner import CuroboMotionPlanner

        frame_name = self._resolve_tcp_frame_name(tcp_frame_name)
        return CuroboMotionPlanner(
            self,
            tcp_frame_name=frame_name,
        )

    def _make_device_cfg(self):
        """创建 cuRobo ``DeviceCfg``。"""

        torch = self.torch
        device = torch.device(self.config.device.device)
        return self.types.DeviceCfg(
            device=device,
            dtype=_torch_dtype(torch, self.config.device.tensor_dtype),
            collision_geometry_dtype=_torch_dtype(
                torch,
                self.config.device.collision_geometry_dtype,
            ),
            collision_gradient_dtype=_torch_dtype(
                torch,
                self.config.device.collision_gradient_dtype,
            ),
            collision_distance_dtype=_torch_dtype(
                torch,
                self.config.device.collision_distance_dtype,
            ),
        )

    def _make_kinematics(self):
        """创建 cuRobo ``Kinematics``。"""

        robot = self.config.robot
        if robot.robot_config_path is not None:
            kin_cfg = self.kinematics_module.KinematicsCfg.from_robot_yaml_file(
                str(robot.robot_config_path),
                tool_frames=list(self.tool_frames),
                device_cfg=self.device_cfg,
                urdf_path=None if robot.urdf_path is None else str(robot.urdf_path),
                load_collision_spheres=bool(robot.load_collision_spheres),
            )
        elif robot.urdf_path is not None and robot.base_link is not None:
            kin_cfg = self.kinematics_module.KinematicsCfg.from_basic_urdf(
                str(robot.urdf_path),
                robot.base_link,
                list(self.tool_frames),
                device_cfg=self.device_cfg,
            )
        else:
            raise ValueError(
                "cuRobo kinematics requires robot_config_path or urdf_path + base_link"
            )
        return self.kinematics_module.Kinematics(kin_cfg)

    def _make_ik_solver(self):
        """创建 cuRobo ``InverseKinematics``。"""

        robot_input = self._robot_input_for_solver()
        task_bundle = self.config.task_bundle
        collision_cache = self._collision_cache_for_solver(
            self.config.ik.collision_cache
        )
        ik_cfg = self.ik_module.InverseKinematicsCfg.create(
            robot=robot_input,
            optimizer_configs=[
                _curobo_task_config_path(path)
                for path in task_bundle.ik_optimizer_configs
            ],
            metrics_rollout=_curobo_task_config_path(task_bundle.ik_metrics_rollout),
            transition_model=_curobo_task_config_path(task_bundle.ik_transition_model),
            device_cfg=self.device_cfg,
            num_seeds=int(self.config.ik.num_seeds),
            position_tolerance=float(self.config.ik.position_tolerance),
            orientation_tolerance=float(self.config.ik.orientation_tolerance),
            use_cuda_graph=bool(self.config.ik.use_cuda_graph),
            random_seed=int(self.config.ik.random_seed),
            optimizer_collision_activation_distance=float(
                self.config.ik.optimizer_collision_activation_distance
            ),
            store_debug=bool(self.config.ik.store_debug),
            override_optimizer_num_iters=dict(
                self.config.ik.override_optimizer_num_iters
            ),
            override_iters_for_multi_link_ik=(
                None
                if self.config.ik.override_iters_for_multi_link_ik is None
                else int(self.config.ik.override_iters_for_multi_link_ik)
            ),
            optimization_dt=(
                None
                if self.config.ik.optimization_dt is None
                else float(self.config.ik.optimization_dt)
            ),
            velocity_regularization_weight=(
                None
                if self.config.ik.velocity_regularization_weight is None
                else float(self.config.ik.velocity_regularization_weight)
            ),
            acceleration_regularization_weight=(
                None
                if self.config.ik.acceleration_regularization_weight is None
                else float(self.config.ik.acceleration_regularization_weight)
            ),
            success_requires_convergence=bool(
                self.config.ik.success_requires_convergence
            ),
            seed_position_weight=float(self.config.ik.seed_position_weight),
            seed_orientation_weight=float(self.config.ik.seed_orientation_weight),
            seed_velocity_weight=float(self.config.ik.seed_velocity_weight),
            seed_acceleration_weight=float(self.config.ik.seed_acceleration_weight),
            seed_solver_num_seeds=int(self.config.ik.seed_solver_num_seeds),
            self_collision_check=self._supports_collision_queries(
                self.config.ik.self_collision_check
            ),
            scene_model={} if collision_cache is not None else None,
            collision_cache=collision_cache,
            max_batch_size=int(self.config.ik.max_batch_size),
            multi_env=bool(self.config.ik.multi_env),
            max_goalset=int(self.config.ik.max_goalset),
            load_collision_spheres=self._supports_collision_queries(
                self.config.robot.load_collision_spheres
            ),
        )
        return self.ik_module.InverseKinematics(ik_cfg)

    def _make_motion_planner(self):
        """创建 Mirror 单请求 ``MotionPlanner``。"""

        robot_input = self._robot_input_for_solver()
        task_bundle = self.config.task_bundle
        planner_cfg = self.motion_module.MotionPlannerCfg.create(
            robot=robot_input,
            ik_optimizer_configs=[
                _curobo_task_config_path(path)
                for path in task_bundle.motion_ik_optimizer_configs
            ],
            ik_transition_model=_curobo_task_config_path(
                task_bundle.motion_ik_transition_model
            ),
            metrics_rollout=_curobo_task_config_path(
                task_bundle.motion_metrics_rollout
            ),
            trajopt_optimizer_configs=[
                _curobo_task_config_path(path)
                for path in task_bundle.trajopt_optimizer_configs
            ],
            trajopt_transition_model=_curobo_task_config_path(
                task_bundle.trajopt_transition_model
            ),
            graph_planner_config=_curobo_task_config_path(
                task_bundle.graph_planner_config
            ),
            graph_planner_rollout=_curobo_task_config_path(
                task_bundle.graph_planner_rollout
            ),
            graph_planner_transition_model=_curobo_task_config_path(
                task_bundle.graph_planner_transition_model
            ),
            device_cfg=self.device_cfg,
            num_ik_seeds=int(self.config.motion_planner.num_ik_seeds),
            num_trajopt_seeds=int(self.config.motion_planner.num_trajopt_seeds),
            position_tolerance=float(self.config.motion_planner.position_tolerance),
            orientation_tolerance=float(
                self.config.motion_planner.orientation_tolerance
            ),
            use_cuda_graph=bool(self.config.motion_planner.use_cuda_graph),
            random_seed=int(self.config.motion_planner.random_seed),
            optimizer_collision_activation_distance=float(
                self.config.motion_planner.optimizer_collision_activation_distance
            ),
            store_debug=bool(self.config.motion_planner.store_debug),
            self_collision_check=self._supports_collision_queries(
                self.config.motion_planner.self_collision_check
            ),
            collision_cache=self._collision_cache_for_solver(
                self.config.motion_planner.collision_cache
            ),
            max_batch_size=1,
            multi_env=False,
            max_goalset=int(self.config.motion_planner.max_goalset),
        )
        planner = self.motion_module.MotionPlanner(planner_cfg)
        warmup = getattr(planner, "warmup", None)
        if self.config.motion_planner.warmup and callable(warmup):
            warmup(
                num_warmup_iterations=_MOTION_PLANNER_WARMUP_ITERATIONS,
            )
        return planner

    def _sync_solver_world_if_available(
        self,
        solver: object,
        *,
        consumer: str,
    ) -> None:
        """把已存在的 collision world 推给刚 lazy 创建的 solver。

        ``CuroboCollisionWorld.update_solvers`` 只更新已经存在的 solver，避免为了同步障碍物而
        反向触发 planner 初始化。这里处理相反方向：如果 world 先创建、solver 后创建，则在
        solver 初始化完成后补一次 ``update_world``。
        """

        if self._collision_world is None:
            return
        update_world = getattr(solver, "update_world", None)
        if not callable(update_world):
            return
        if hasattr(solver, "scene_collision_checker") and (
            getattr(solver, "scene_collision_checker") is None
        ):
            return
        required = getattr(self._collision_world, "materialized_counts", {}) or {}
        if getattr(self, "config", None) is not None:
            configured = self._configured_collision_cache(
                _normalize_collision_consumer(consumer)
            )
            if not _collision_cache_capacity_sufficient(required, configured):
                return
        update_world(self._collision_world.scene_cfg)

    def _robot_input_for_solver(self):
        """返回 cuRobo solver factory 可接受的 robot 参数。"""

        robot = self.config.robot
        if robot.robot_config_path is not None:
            return materialized_robot_mapping(
                robot,
                tool_frames=self.tool_frames,
                asset_root_path=getattr(self, "_robot_asset_root_path", None),
            )
        if robot.urdf_path is not None and robot.base_link:
            return self.robot_module.RobotCfg.from_basic(
                urdf_path=str(robot.urdf_path),
                base_link=str(robot.base_link),
                tool_frames=list(self.tool_frames),
                device_cfg=self.device_cfg,
            )
        raise ValueError("cuRobo IK/planner requires robot_config_path or urdf_path")

    def _resolve_tcp_frame_name(self, tcp_frame_name: str | None) -> str:
        """解析并校验 TCP frame 名。"""

        frame_name = str(tcp_frame_name or self.default_tcp_frame)
        if not frame_name:
            raise ValueError("tcp_frame_name is required")
        if frame_name not in set(self.frame_names()):
            raise ValueError(f"cuRobo frame {frame_name!r} not found")
        return frame_name

    def _supports_collision_queries(self, enabled: bool) -> bool:
        """solver factory 是否应加载 robot collision spheres。"""

        return bool(
            enabled
            and self.config.robot.load_collision_spheres
            and self.robot_sphere_count() > 0
        )

    def _collision_cache_for_solver(
        self, cache: dict[str, int]
    ) -> dict[str, int] | None:
        """缺少 robot collision spheres 时不分配环境 collision cache。"""

        if not self._supports_collision_queries(True) or not cache:
            return None
        return dict(cache)

    def _required_collision_cache(self) -> dict[str, int]:
        """返回当前 materialized world 对各 cache 类型的需求。"""

        world = getattr(self, "_collision_world", None)
        if world is None:
            return {"cuboid": 0, "mesh": 0}
        return dict(getattr(world, "materialized_counts", {}) or {})

    def _configured_collision_cache(
        self,
        consumer: str | None = None,
    ) -> dict[str, int]:
        """返回指定 consumer 的 cache，或已创建 solver 的公共容量。"""

        if consumer is not None:
            return dict(self._collision_cache_for_consumer(consumer))
        consumers = self._existing_collision_consumers()
        if not consumers:
            consumers = ("ik", "planner")
        caches = [self._collision_cache_for_consumer(item) for item in consumers]
        shapes = set().union(*(set(cache) for cache in caches))
        capacities = {
            shape: min(int(cache.get(shape, 0)) for cache in caches)
            for shape in sorted(shapes)
        }
        return {shape: value for shape, value in capacities.items() if value > 0}

    def _collision_solvers(self, consumer: str | None) -> tuple[object, ...]:
        """返回指定 consumer 或全部已创建 consumer 的 solver。"""

        if consumer is None:
            return self.existing_solvers()
        attribute = _collision_consumer_solver_attribute(consumer)
        solver = getattr(self, attribute, None)
        return () if solver is None else (solver,)

    def _existing_collision_consumers(self) -> tuple[str, ...]:
        """返回已经创建 solver 对应的 canonical consumer 名。"""

        return tuple(
            consumer
            for consumer in ("ik", "planner")
            if getattr(
                self,
                _collision_consumer_solver_attribute(consumer),
                None,
            )
            is not None
        )

    def _collision_cache_for_consumer(self, consumer: str) -> Mapping[str, int]:
        """返回 consumer 对应 solver factory 使用的 cache 配置。"""

        normalized = _normalize_collision_consumer(consumer)
        if normalized == "ik":
            return self.config.ik.collision_cache
        return self.config.motion_planner.collision_cache


def _normalize_collision_consumer(consumer: object) -> str:
    """解析并校验 cuRobo collision consumer 名。"""

    normalized = str(consumer).strip().lower()
    if normalized not in _COLLISION_CONSUMER_SOLVER_ATTRIBUTES:
        raise ValueError(f"Unknown cuRobo collision consumer: {consumer!r}")
    return normalized


def _collision_consumer_solver_attribute(consumer: str) -> str:
    """返回 canonical consumer 对应的 lazy solver attribute。"""

    return _COLLISION_CONSUMER_SOLVER_ATTRIBUTES[
        _normalize_collision_consumer(consumer)
    ]


def _collision_consumer_label(consumer: str) -> str:
    """返回 cache capacity 错误使用的稳定 consumer 标签。"""

    return _COLLISION_CONSUMER_LABELS[_normalize_collision_consumer(consumer)]


def _collision_cache_capacity_sufficient(
    required: Mapping[str, int],
    configured: Mapping[str, int],
) -> bool:
    """返回 configured cache 是否覆盖 materialized world 需求。"""

    return all(
        int(required_count) <= int(configured.get(shape, 0))
        for shape, required_count in required.items()
    )


def _curobo_task_config_path(relative_path: str) -> str:
    """返回锁定版本的后端 task 资源绝对路径。"""

    return curobo_task_resource_path(relative_path)


def _torch_dtype(torch, name: str):
    """把 YAML 字符串 dtype 转成 torch dtype。"""

    dtype = getattr(torch, str(name), None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype for cuRobo: {name}")
    return dtype


def _collision_objects_fingerprint(
    collision_objects: Sequence[CollisionObject],
) -> str:
    """为 ad-hoc collision world sync 构造确定性 geometry fingerprint。"""

    payload = []
    for value in collision_objects:
        payload.append(
            {
                "name": str(value.name),
                "shape": str(value.shape),
                "pose": np.asarray(value.pose, dtype=float).round(12).tolist(),
                "size": [float(item) for item in value.size],
                "enabled": bool(value.enabled),
                "padding": float(value.padding),
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
