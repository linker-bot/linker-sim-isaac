# specified_path 最终版代码生成规格

本文档面向后续代码生成模型。目标不是继续讨论方案，而是给出可直接实施的规格。代码生成时请按本文的阶段顺序推进，每一阶段都要保持测试可运行、行为可解释、改动范围可控。

## 0. 给代码生成模型的执行规则

必须遵守：

- 只围绕 `specified_path` 最终版改动，不重构无关模块。
- 保持 `trajectory_optimization` 作为目标式请求默认 pipeline。
- 保持 `graph_search` 作为显式搜索规划或 optimizer fallback pipeline。
- `specified_path` 表示“调用方给定路径几何”，不是避障搜索规划器。
- 不把 `TaskSpacePath` / `CompositePath` 静默 fallback 到 `tcp_line.py` 的逐点 IK。
- 姿态边界统一使用 `wxyz` 四元数；进入 cuMotion 前在 `pose_adapter.py` 转成 `Rotation3` / `Pose3`。
- C-space 向量顺序始终按 `CuMotionContext.joint_names()` / `context.expected_cspace_width`。
- 完整 Isaac articulation DOF、灵巧手、mimic 展开仍属于动作脚本层和控制器层，不要下沉到 cuMotion 后端。
- 新增实现要能被 `tests/test_cumotion_motion_planner.py` 的 fake cuMotion 覆盖，不要求真实 Isaac/cuMotion 环境才能跑单元测试。

推荐工作方式：

1. 先扩展测试 fake 和请求模型测试。
2. 再实现 `path_spec_adapter.py`。
3. 最后把 `specified_path_planner.py` 切到 adapter。
4. 每个阶段运行相关测试。

## 1. 实施前起点

本规格撰写时的首版实现已经存在：

- `source/manipulation_project/planning/requests.py`
  - `SpecifiedPathRequest`
  - `CSpaceWaypointPath`
  - `TaskSpacePath`
  - `CompositePath`
  - `TcpLineSegment`
- `source/manipulation_project/backends/cumotion/specified_path_planner.py`
  - 只支持 `CSpaceWaypointPath`
  - `TaskSpacePath` 抛 `NotImplementedError`
  - `CompositePath` 抛 `NotImplementedError`
- `source/manipulation_project/backends/cumotion/trajectory_generation.py`
  - 已封装 `CSpaceTrajectoryGenerator`
  - 支持 `time_optimal` 和 `time_stamped`
- `source/manipulation_project/backends/cumotion/pose_adapter.py`
  - 已封装 `rotation_from_quat_wxyz`
  - 已封装 `pose_from_position_quat_wxyz`
  - 已封装 `pose_from_matrix`

本任务要把 `specified_path` 从首版推进到最终版：

```text
SpecifiedPathRequest
    -> CSpacePathSpec / TaskSpacePathSpec / CompositePathSpec
    -> LinearCSpacePath
    -> numpy joint_path
    -> generate_cspace_trajectory(...)
    -> MotionResult
```

## 2. 官方 cuMotion API 目标

代码生成时按下列 API 设计 adapter。官方文档入口：

- https://nvidia-isaac.github.io/cumotion/api/python_api.html#rotations-and-poses
- https://nvidia-isaac.github.io/cumotion/api/python_api.html#path-specification
- https://nvidia-isaac.github.io/cumotion/api/python_api.html#path-generation-collision-unaware
- https://nvidia-isaac.github.io/cumotion/api/python_api.html#trajectory-generation-collision-unaware

需要调用的 API 名：

```python
cumotion.Rotation3(w, x, y, z)
cumotion.Rotation3.from_axis_angle(axis, angle)
cumotion.Rotation3.from_scaled_axis(scaled_axis)
cumotion.Rotation3.from_matrix(rotation_matrix)

cumotion.Pose3(rotation, translation)
cumotion.Pose3.from_translation(translation)
cumotion.Pose3.from_rotation(rotation)

cumotion.create_cspace_path_spec(initial_cspace_position)
cspace_path_spec.add_cspace_waypoint(waypoint)
cumotion.create_linear_cspace_path(cspace_path_spec)
linear_cspace_path.waypoints()

cumotion.create_task_space_path_spec(initial_pose)
task_space_path_spec.add_translation(target_position, blend_radius)
task_space_path_spec.add_linear_path(target_pose, blend_radius)
task_space_path_spec.add_rotation(target_rotation)
task_space_path_spec.add_tangent_arc(target_position, constant_orientation)
task_space_path_spec.add_tangent_arc_with_orientation_target(target_pose)
task_space_path_spec.add_three_point_arc(target_position, intermediate_position, constant_orientation)
task_space_path_spec.add_three_point_arc_with_orientation_target(target_pose, intermediate_position)
cumotion.TaskSpacePathConversionConfig()
cumotion.convert_task_space_path_spec_to_cspace(
    task_space_path_spec,
    context.kinematics,
    control_frame,
    conversion_config,
    ik_config,
)

cumotion.create_composite_path_spec(initial_cspace_position)
composite_path_spec.add_cspace_path_spec(path_spec, transition_mode)
composite_path_spec.add_task_space_path_spec(path_spec, transition_mode)
cumotion.convert_composite_path_spec_to_cspace(
    composite_path_spec,
    context.kinematics,
    control_frame,
    conversion_config,
    ik_config,
)
```

## 3. 要实现的行为

### 3.1 CSpaceWaypointPath

输入：

```python
SpecifiedPathRequest(
    current_q=current_q,
    path=CSpaceWaypointPath(waypoints=(current_q, mid_q, goal_q)),
)
```

行为：

```text
validate current_q width
validate at least 2 waypoints
validate every waypoint width
validate first waypoint matches current_q
create CSpacePathSpec from first waypoint
add remaining waypoints
create LinearCSpacePath
extract LinearCSpacePath.waypoints()
return joint_path and optional trajectory
```

First-waypoint rule:

- 默认要求 `waypoints[0]` 与 `request.current_q` 在 `start_match_tolerance` 内一致。
- 不自动插入 `current_q`。
- 不一致时抛 `ValueError`，错误消息包含 `first waypoint` 和 `current_q`。

### 3.2 TaskSpacePath

输入：

```python
SpecifiedPathRequest(
    current_q=current_q,
    tcp_frame_name="pinch_tcp",
    path=TaskSpacePath(segments=(TcpLineSegment(...), ...)),
)
```

行为：

```text
validate current_q width
resolve tcp_frame_name
validate frame exists when context.has_frame is available
compute initial TCP pose from context.kinematics.pose(current_q, tcp_frame_name)
create TaskSpacePathSpec(initial_pose)
append every segment through official TaskSpacePathSpec API
build TaskSpacePathConversionConfig from config
build IkConfig seeded by current_q
convert task-space path spec to LinearCSpacePath
extract LinearCSpacePath.waypoints()
return joint_path and optional trajectory
```

Do not call `tcp_line.py` from this pipeline.

### 3.3 CompositePath

输入：

```python
SpecifiedPathRequest(
    current_q=current_q,
    tcp_frame_name="pinch_tcp",
    path=CompositePath(parts=(...)),
)
```

行为：

```text
validate current_q width
create CompositePathSpec(current_q)
for each part:
    convert CSpaceWaypointPath to CSpacePathSpec, or
    convert TaskSpacePath to TaskSpacePathSpec
    add part with transition mode
convert CompositePathSpec to LinearCSpacePath
extract LinearCSpacePath.waypoints()
return joint_path and optional trajectory
```

Transition modes:

| Project value | cuMotion enum |
|---|---|
| `skip` | `CompositePathSpec.TransitionMode.SKIP` |
| `free` | `CompositePathSpec.TransitionMode.FREE` |
| `linear_task_space` | `CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE` |

Default transition mode is `specified_path.composite.default_transition_mode`, defaulting to `free`.

## 4. File-by-file implementation spec

### 4.1 `planning/requests.py`

Keep existing classes and add only what is needed.

Add literal:

```python
CompositeTransitionMode = Literal["skip", "free", "linear_task_space"]
TaskSpaceArcMode = Literal["tangent", "three_point"]
```

Add dataclass:

```python
@dataclass(frozen=True)
class TcpRotationSegment:
    target_orientation: np.ndarray


@dataclass(frozen=True)
class TcpArcSegment:
    target_position: np.ndarray
    intermediate_position: np.ndarray | None = None
    target_orientation: np.ndarray | None = None
    arc_mode: TaskSpaceArcMode = "tangent"
    constant_orientation: bool = True


@dataclass(frozen=True)
class TcpPoseSequenceSegment:
    poses: tuple[PoseTarget, ...]
    blend_radius: float = 0.0


@dataclass(frozen=True)
class CompositePathPart:
    path: CSpaceWaypointPath | TaskSpacePath
    transition_mode: CompositeTransitionMode | None = None
```

Update union:

```python
TaskSpaceSegment = (
    TcpLineSegment
    | TcpRotationSegment
    | TcpArcSegment
    | TcpPoseSequenceSegment
)
```

Update:

```python
CompositePath.parts: tuple[
    CSpaceWaypointPath | TaskSpacePath | CompositePathPart,
    ...
]
```

Extend validation:

- `TcpLineSegment`:
  - `orientation_mode in {"current", "target", "none"}`
  - exactly one of `target_position` / `target_offset`
  - `orientation_mode="target"` requires `target_orientation`
  - `start_position`, `target_position`, `target_offset` shapes are `(3,)`
  - `target_orientation` shape is `(4,)`
- `TcpRotationSegment`:
  - `target_orientation` shape is `(4,)`
- `TcpArcSegment`:
  - `target_position` shape is `(3,)`
  - `arc_mode in {"tangent", "three_point"}`
  - `arc_mode="three_point"` requires `intermediate_position`
  - `intermediate_position`, when present, shape is `(3,)`
  - `target_orientation`, when present, shape is `(4,)`
- `TcpPoseSequenceSegment`:
  - at least one pose
  - every pose position shape `(3,)`
  - every pose orientation must be present and shape `(4,)`
  - `blend_radius >= 0`
- `CompositePathPart`:
  - transition mode is valid when present

Do not validate frame existence or C-space width here. That remains a backend responsibility.
Do not validate `CSpaceWaypointPath.waypoints[0] == current_q` here either. That check depends on backend config tolerance and belongs in `path_spec_adapter.py`.

### 4.2 `motion_planner_config.py`

Keep `SpecifiedPathConfig` shape compatible, but add typed helper defaults through mapping keys.

Accepted keys:

```yaml
specified_path:
  cspace_waypoints:
    require_start_match: true
    start_match_tolerance: 1.0e-9
  task_space_segments:
    conversion:
      initial_s_step_size: 0.05
      initial_s_step_size_delta: 0.005
      min_s_step_size: 1.0e-5
      min_s_step_size_delta: 1.0e-5
      alpha: 1.4
      max_iterations: 50
      min_position_deviation: 0.001
      max_position_deviation: 0.003
    ik:
      use_current_q_as_seed: true
  composite:
    default_transition_mode: free
```

Implement validation in `MotionPlannerBackendConfig.validate()`:

- `require_start_match` bool if present.
- `start_match_tolerance >= 0`.
- conversion keys must be exactly from the official field list above.
- conversion numeric values obey:
  - step sizes > 0
  - `alpha > 1`
  - `max_iterations > 0`
  - `0 < min_position_deviation < max_position_deviation`
- `ik.use_current_q_as_seed` bool if present.
- `default_transition_mode` in `{"skip", "free", "linear_task_space"}`.

Do not create new dataclass groups unless doing so is very small and does not churn existing config parsing. Mapping-based validation is acceptable.

### 4.3 `pose_adapter.py`

Add helper functions:

```python
def rotation_from_axis_angle(cumotion, axis, angle):
    axis_array = np.asarray(axis, dtype=float).reshape(3)
    return cumotion.Rotation3.from_axis_angle(axis_array, float(angle))


def rotation_from_scaled_axis(cumotion, scaled_axis):
    scaled_axis_array = np.asarray(scaled_axis, dtype=float).reshape(3)
    return cumotion.Rotation3.from_scaled_axis(scaled_axis_array)


def pose_from_rotation_translation(cumotion, rotation, translation):
    return cumotion.Pose3(rotation, np.asarray(translation, dtype=float).reshape(3))
```

Do not change the existing `wxyz` behavior.

### 4.4 New file: `backends/cumotion/path_spec_adapter.py`

Create this module. It should contain only conversion helpers, not facade dispatch.

Required public functions:

```python
def cspace_waypoints_to_joint_path(context, request, config) -> np.ndarray:
    ...


def task_space_path_to_joint_path(context, request, config, *, tcp_frame_name: str) -> np.ndarray:
    ...


def composite_path_to_joint_path(context, request, config, *, tcp_frame_name: str) -> np.ndarray:
    ...
```

Recommended private helpers:

```python
def _cspace_path_spec_from_waypoints(cumotion, waypoints):
    ...


def _task_space_path_spec_from_segments(context, current_q, path, tcp_frame_name):
    ...


def _joint_path_from_linear_cspace_path(linear_path):
    ...


def _task_space_conversion_config(cumotion, config):
    ...


def _ik_config_for_path_conversion(context, current_q, config):
    ...


def _transition_mode(cumotion, value):
    ...
```

Implementation details:

- `_joint_path_from_linear_cspace_path` must support both real pybind and fake objects:
  - prefer `linear_path.waypoints()`
  - if fake exposes `.waypoints`, support that too
  - stack into `(N, dof)` numpy array
- Every `add_*` cuMotion call returning `False` must raise `ValueError` naming the rejected segment.
- `task_space_path_to_joint_path` must pass:
  - `context.kinematics`
  - resolved `tcp_frame_name`
  - conversion config
  - IK config
  to `convert_task_space_path_spec_to_cspace`.
- `composite_path_to_joint_path` must pass the same conversion/IK config to `convert_composite_path_spec_to_cspace`.

IK config:

- Use `context.cumotion.IkConfig()`.
- If `specified_path.task_space_segments.ik.use_current_q_as_seed` is absent or true, set `ik_config.cspace_seeds = [current_q]`.
- Copy these context config values if they exist:
  - `position_tolerance`
  - `orientation_tolerance`
  - `ccd_max_iterations`
  - `bfgs_max_iterations`
  - `orientation_weight` into CCD/BFGS orientation weights.

Initial pose:

- Use `context.kinematics.pose(current_q, tcp_frame_name)`.
- Do not use `TcpLineSegment.start_position` as the official initial pose.
- If `start_position` is present, only validate it is close to the FK pose translation if the fake/real pose exposes translation. Use a tolerance from config if available, otherwise `1.0e-6`.

### 4.5 `specified_path_planner.py`

Replace the current local `_cspace_waypoint_path` implementation with adapter calls.

New dispatch:

```python
if isinstance(request.path, CSpaceWaypointPath):
    family = "cspace_waypoints"
    joint_path = cspace_waypoints_to_joint_path(context, request, config)
elif isinstance(request.path, TaskSpacePath):
    family = "task_space_segments"
    joint_path = task_space_path_to_joint_path(
        context, request, config, tcp_frame_name=resolved_tcp_frame_name
    )
elif isinstance(request.path, CompositePath):
    family = "composite"
    joint_path = composite_path_to_joint_path(
        context, request, config, tcp_frame_name=resolved_tcp_frame_name
    )
else:
    raise ValueError(...)
```

Diagnostics:

```python
PlanningDiagnostics(
    status="SUCCESS",
    message=(
        "pipeline=specified_path "
        f"family={family} "
        "path_conversion=official "
        f"collision_check={config.specified_path.validate_collision_after_generation}"
    ),
    metrics={...},
)
```

Metrics:

- `num_waypoints`
- `path_length`
- `num_collision_objects`

For now, set `num_collision_objects=0.0` unless a collision world is explicitly inspected.

### 4.6 Optional collision validation

If implementing in this pass, keep it small.

Behavior:

- Only run when `config.specified_path.validate_collision_after_generation` is true.
- Check generated `joint_path` waypoints only.
- If collision inspection APIs are not already easy to call, leave a clear `NotImplementedError` or return failed result with diagnostics. Do not invent broad inspector abstractions in this task.

It is acceptable to defer this to a later commit as long as diagnostics continue to report the configured flag.

## 5. Segment mapping rules

### 5.1 TcpLineSegment

Resolve target position:

```python
if target_position is not None:
    target = target_position
else:
    target = current_pose.translation + target_offset
```

Mapping:

| `orientation_mode` | cuMotion call |
|---|---|
| `none` | `add_translation(target, blend_radius)` |
| `current` | `add_linear_path(Pose3(current_rotation, target), blend_radius)` |
| `target` | `add_linear_path(Pose3(target_rotation, target), blend_radius)` |

`TcpLineSegment` currently has no `blend_radius` field. Use `0.0` for it unless you add the field in a backward-compatible way.

### 5.2 TcpRotationSegment

Mapping:

```python
target_rotation = rotation_from_quat_wxyz(cumotion, segment.target_orientation)
task_space_path_spec.add_rotation(target_rotation)
```

### 5.3 TcpArcSegment

If `target_orientation is None`:

- `arc_mode="tangent"` -> `add_tangent_arc(target_position, constant_orientation)`
- `arc_mode="three_point"` -> `add_three_point_arc(target_position, intermediate_position, constant_orientation)`

If `target_orientation is not None`:

- Build `target_pose = Pose3(target_rotation, target_position)`.
- `arc_mode="tangent"` -> `add_tangent_arc_with_orientation_target(target_pose)`
- `arc_mode="three_point"` -> `add_three_point_arc_with_orientation_target(target_pose, intermediate_position)`

### 5.4 TcpPoseSequenceSegment

For every pose:

```python
target_pose = pose_from_position_quat_wxyz(cumotion, pose.position, pose.orientation)
task_space_path_spec.add_linear_path(target_pose, blend_radius)
```

Require every pose to include orientation. If position-only sequences are needed later, add a separate segment type.

## 6. Error handling contract

Use exceptions for programmer/configuration errors:

- invalid request structure
- invalid C-space width
- missing or unknown frame
- unknown transition mode
- unsupported config key
- cuMotion `add_*` returning `False`

Use `MotionResult(success=False)` only for planner-like runtime failures that the cuMotion conversion API represents without throwing. If the real API throws for conversion failure, allow the exception to propagate unless tests establish a fake failure return shape.

Do not swallow exceptions and return an empty successful path.

## 7. Tests to add or update

All tests should run without Isaac Sim.

### 7.1 Request/model tests

Add tests near existing planning request or motion planner tests:

- `test_tcp_line_segment_requires_target_for_target_orientation_mode`
- `test_tcp_arc_segment_requires_intermediate_for_three_point_arc`
- `test_tcp_pose_sequence_requires_orientations`
- `test_composite_path_part_validates_transition_mode`
- `test_specified_path_conversion_config_rejects_unknown_keys`
- `test_specified_path_conversion_config_validates_numeric_ranges`

### 7.2 Fake cuMotion additions

Extend fake classes in `tests/test_cumotion_motion_planner.py`:

- `_FakeCSpacePathSpec`
- `_FakeLinearCSpacePath`
- `_FakeTaskSpacePathSpec`
- `_FakeTaskSpacePathConversionConfig`
- `_FakeCompositePathSpec`

Fake cumotion methods:

```python
create_cspace_path_spec
create_linear_cspace_path
create_task_space_path_spec
convert_task_space_path_spec_to_cspace
create_composite_path_spec
convert_composite_path_spec_to_cspace
TaskSpacePathConversionConfig
IkConfig
```

Fake path specs should record method calls so tests can assert official APIs were used.

### 7.3 Pipeline tests

Add or update:

- `test_specified_path_cspace_uses_official_path_spec`
- `test_specified_path_cspace_requires_start_match`
- `test_specified_path_cspace_waypoints_generates_joint_path`
- `test_specified_path_tcp_line_none_orientation_uses_add_translation`
- `test_specified_path_tcp_line_current_orientation_uses_add_linear_path`
- `test_specified_path_tcp_line_target_orientation_uses_add_linear_path`
- `test_specified_path_tcp_rotation_uses_add_rotation`
- `test_specified_path_three_point_arc_uses_official_arc_api`
- `test_specified_path_task_space_conversion_generates_joint_path_and_trajectory`
- `test_specified_path_composite_converts_to_cspace`
- `test_specified_path_no_silent_fallback_to_tcp_line_helper`
- `test_specified_path_time_stamped_trajectory_uses_duration`

Update existing test:

- Replace `test_specified_path_rejects_task_space_until_conversion_is_implemented` with tests that prove task-space conversion is now implemented.

## 8. Documentation updates after implementation

After code and tests pass, update docs:

- `docs/motion_planner_design.md`
  - Change Phase 5 from C-space-only first version to final path-family version.
  - Remove statements saying `TaskSpacePath` / `CompositePath` are unimplemented.
  - State task-space/composite use official cuMotion path conversion.

- `docs/cumotion_interface.md`
  - Change `PathSpec / path conversion` status to implemented for C-space, task-space, and composite.
  - Document new request segment types and transition mode.
  - Document conversion config keys.

- `README.md`
  - If it still references missing `CUMOTION_PLANNING.md`, replace with actual `docs/` paths.

## 9. Suggested implementation phases

Phase 1: request and config validation

- Edit `planning/requests.py`.
- Edit `motion_planner_config.py`.
- Add validation tests.

Phase 2: official C-space PathSpec

- Add `path_spec_adapter.py`.
- Implement C-space conversion only.
- Update existing C-space specified path tests.

Phase 3: task-space PathSpec conversion

- Implement `TcpLineSegment`, `TcpRotationSegment`, `TcpArcSegment`, and `TcpPoseSequenceSegment` mapping.
- Implement conversion config and IK config helpers.
- Replace the old NotImplemented task-space test.

Phase 4: composite PathSpec conversion

- Implement `CompositePathPart`.
- Implement transition mode mapping.
- Add composite tests.

Phase 5: diagnostics and optional collision validation

- Improve diagnostics metrics.
- Implement waypoint-level collision validation only if existing inspector APIs make it straightforward.

Phase 6: docs

- Update design/interface docs after behavior is verified.

## 10. Final acceptance checklist

The task is complete only when:

- `SpecifiedPathRequest(path=CSpaceWaypointPath(...))` uses official `CSpacePathSpec`.
- `SpecifiedPathRequest(path=TaskSpacePath(...))` no longer raises `NotImplementedError`.
- `SpecifiedPathRequest(path=CompositePath(...))` no longer raises `NotImplementedError`.
- Task-space and composite paths use official cuMotion conversion APIs.
- No specified-path branch calls `tcp_line.py`.
- C-space waypoints still produce the same discrete C-space path shape and values as the first version.
- Successful specified-path planning always produces a trajectory for the generated path.
- `trajectory_generation.mode="time_stamped"` still requires and uses `duration_s`.
- Diagnostics include `pipeline=specified_path`, `family=...`, and `path_conversion=official`.
- Unit tests cover fake cuMotion API calls for all three path families.
- Existing graph-search and trajectory-optimization tests still pass.
