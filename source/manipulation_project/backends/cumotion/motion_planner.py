"""cuMotion 运动规划 facade。

本模块是任务层进入 cuMotion 运动规划能力的统一入口。它本身不直接调用某一个具体的
cuMotion planner，而是根据 ``MotionPlannerBackendConfig.planning_pipeline`` 把请求分发到
对应 pipeline：

* ``trajectory_optimization``：默认目标式规划路线，直接调用 cuMotion
  ``TrajectoryOptimizer``，成功时主要输出 ``Trajectory``。
* ``graph_search``：显式选择的图搜索路线，先生成 C-space path，再按配置可选做时间参数化。
* ``specified_path``：调用方明确给定路径几何的路线，支持 C-space waypoint、task-space segment
  和 composite path，并统一转成 C-space path 后做可选时间参数化。

这样做的目的是把“任务层选择哪种运动生成策略”和“后端怎样调用 cuMotion API”分开，避免
不同 pipeline 的参数互相影响。
"""

from __future__ import annotations

from manipulation_project.backends.cumotion.graph_motion_planner import (
    plan_graph_search,
)
from manipulation_project.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from manipulation_project.backends.cumotion.specified_path_planner import (
    plan_specified_path,
)
from manipulation_project.backends.cumotion.trajectory_optimizer_planner import (
    plan_trajectory_optimization,
)
from manipulation_project.planning.requests import MotionRequest, SpecifiedPathRequest
from manipulation_project.planning.results import MotionResult


class CuMotionMotionPlanner:
    """按配置把项目规划请求分发给具体 cuMotion pipeline。

    输入/输出关节向量均使用 ``context.joint_names()`` 对应的 cuMotion C-space 顺序。完整
    Isaac articulation DOF 的裁剪、回填和非 C-space DOF 插值仍属于任务层职责。
    """

    def __init__(
        self,
        context,
        *,
        tcp_frame_name: str | None = None,
        config: MotionPlannerBackendConfig | None = None,
    ) -> None:
        self.context = context
        self.cumotion = context.cumotion
        # cuMotion 的 graph planner 和 optimizer 都需要一个 tool frame 来构造后端 config。
        # 关节空间目标虽然不直接约束 TCP，但仍要绑定 robot description 中存在的 frame。
        self.tcp_frame_name = str(
            tcp_frame_name
            or context.config.custom_tcp_frame
            or context.config.flange_frame
        )
        # 优先使用调用方本次传入的配置；否则使用 CuMotionConfig 中的分组配置。
        self.config = config or getattr(context.config, "motion_planner", None)
        if self.config is None:
            self.config = MotionPlannerBackendConfig.from_mapping(None)
        self.config.validate()

    def joint_names(self) -> list[str]:
        """返回 planner 使用的 C-space 关节名。

        调用方构造 ``MotionRequest.current_q`` / ``goal_q``，以及消费 ``MotionResult.joint_path``
        时都必须使用这个顺序。
        """

        return self.context.joint_names()

    def plan(self, request: MotionRequest | SpecifiedPathRequest) -> MotionResult:
        """根据 ``planning_pipeline`` 规划一次运动。

        ``MotionRequest`` 描述“从当前状态到目标”的目标式请求，只适用于 graph search 和
        trajectory optimization。``SpecifiedPathRequest`` 描述“按调用方给定路径几何走”的
        请求，只适用于 specified path。这里显式检查请求类型，能比后端 pybind 抛错更早给出
        清晰边界。
        """

        pipeline = self.config.planning_pipeline
        if pipeline == "graph_search":
            if not isinstance(request, MotionRequest):
                raise ValueError("graph_search requires MotionRequest")
            return plan_graph_search(
                self.context,
                request,
                self.config,
                tcp_frame_name=self.tcp_frame_name,
            )
        if pipeline == "specified_path":
            if not isinstance(request, SpecifiedPathRequest):
                raise ValueError("specified_path requires SpecifiedPathRequest")
            return plan_specified_path(
                self.context,
                request,
                self.config,
                tcp_frame_name=self.tcp_frame_name,
            )
        if pipeline == "trajectory_optimization":
            if not isinstance(request, MotionRequest):
                raise ValueError("trajectory_optimization requires MotionRequest")
            return plan_trajectory_optimization(
                self.context,
                request,
                self.config,
                tcp_frame_name=self.tcp_frame_name,
            )
        raise ValueError(
            "planning_pipeline must be one of: graph_search, specified_path, "
            "trajectory_optimization"
        )
