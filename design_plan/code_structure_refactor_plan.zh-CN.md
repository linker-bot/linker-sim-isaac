# 运行形态与目录结构调整计划

**日期:** 2026-07-07
**目标:** 把“单臂、双臂、多 env 单臂、多 env 双臂”理解为四种平行运行形态，并让代码按职责归位，而不是把 multi-env/tiled 相关逻辑全部聚合在单一目录中。

## 1. 当前认知

项目目前大致是这样的调用链：

- `scripts/*.py` 是用户入口，负责解析 CLI、补 `PYTHONPATH`、创建 runtime，然后调用包内逻辑。
- `app/runtime/` 负责创建 Isaac `SimulationApp`、`World`、单臂/双臂 runtime、reset 等运行时对象。
- `app/interactive/` 负责旧双臂交互模式：协议解析、队列、transport、state stream 和主循环。
- `app/motion/`、`execution/`、`planning/`、`backends/cumotion/` 分别处理动作规格、物理 step 执行、规划请求和 cuMotion 后端。
- `telemetry/` 负责 Foxglove/MCAP 等外部状态输出。
- `tiled/` 当前同时包含 tiled config/path/scene/command/IK/state/telemetry/cuMotion adapter，职责偏宽。

结论：用户提出的方向是合理的。多 env 单臂/双臂不应该被理解成“另一个完全独立产品目录”，而是单臂/双臂 runtime 在 env 维度上的批量形态。目录应按职责组织：

- 交互能力放在 `app/interactive/`。
- cuMotion 后端适配放在 `backends/cumotion/`。
- Foxglove/MCAP 输出放在 `telemetry/`。
- `tiled/` 只保留 env 拓扑、批量 command/state、scene builder 等 tiled 核心原语。

## 2. 四种运行形态

四种形态应并列存在：

| 形态 | 机器人数量 | env 数量 | 入口示例 | 核心职责 |
| --- | --- | --- | --- | --- |
| 单臂 single-env | 1 | 1 | `pinch_grasp.py` | 旧单臂 runtime / motion / execution |
| 双臂 single-env | 2 | 1 | `dual_arm_interactive.py` | 旧双臂 runtime / interactive / planner |
| 单臂 tiled-env | 1 | N | tiled scene + command step | 批量 state/action/reset |
| 双臂 tiled-env | 2 | N | `scene3_tiled` | 批量 left/right state/action/reset |

重要边界：

- tiled interactive 不是旧 `dual_arm_interactive.py` 的并行版本。
- tiled command step 不接 `MoveSpec`、planner request、cancel_current/estop。
- tiled 规划属于外部 planner manager + trajectory buffer；不塞进 `TiledCommandAdapter`。

## 3. 目录调整原则

### 3.1 保留在 `tiled/`

保留“tiled 这个概念本身”的核心原语：

- `config.py`: tiled env 配置。
- `paths.py`: env root/path/origin 规则。
- `scene/`: Isaac/PhysX tiled scene builder 子包，按 `types`、`robots`、`objects`、`clone`、`root_pose`、`views`、`builder` 拆分。
- `command.py`: `TiledCommandAction`、同步 command step 适配。
- `ik.py`: backend-agnostic batched IK result/fallback。
- `state.py`: tiled batched state 数据结构。

### 3.2 迁出 `tiled/`

迁出不属于 tiled 核心语义的实现：

- `tiled/telemetry.py` -> `telemetry/tiled.py`
- `tiled/cumotion.py` -> `backends/cumotion/tiled_ik.py`
- `scripts/tiled_env_interactive.py` 的主体 -> `app/interactive/tiled/`
- cuMotion tiled planner adapter -> `backends/cumotion/tiled_planner.py`

脚本只保留薄 CLI wrapper。旧 import 路径不再保留兼容 wrapper，调用方必须使用实现所在的新模块路径。

## 4. 本次落地范围

第一阶段做低风险结构整理：

1. 新增本文件，记录目录策略和迁移边界。
2. 将 tiled interactive 的实现迁入 `src/linkerbot_sim/app/interactive/tiled/` 子包。
3. 将 `scripts/tiled_env_interactive.py` 改成薄入口。
4. 将 tiled telemetry 实现迁入 `src/linkerbot_sim/telemetry/tiled.py`，删除旧路径 re-export。
5. 将 batched cuMotion IK 实现迁入 `src/linkerbot_sim/backends/cumotion/tiled_ik.py`，删除旧路径 re-export。
6. 更新测试 import 到新路径，并验证旧路径已不可导入。

已补充完成：

- `tiled/scene.py` 已拆成 `tiled/scene/` 子包；builder 仍属于 tiled 核心，只是内部按职责分层。
- `tiled/planner.py` 已只保留 backend-agnostic manager/request/result；`CuMotionJointPlannerBackend` 迁入 `backends/cumotion/tiled_planner.py`。
- `tiled.interactive_planning` 已删除，交互规划消息 helper 迁入 `app/interactive/tiled/planning.py`。
- `robot_names` 作为 JSON 输入别名已删除，协议只接受 `robot` / `robots`。

仍暂不做：

- 不引入新的四套 runtime 抽象层。
- 不修改现有单臂/双臂 motion runtime。
- 不处理当前工作区里无关的 `dual_arm_motion_test.py` 删除状态。

## 5. 验收

- `scripts/tiled_env_interactive.py` 仍可按原 CLI 使用。
- `tests/test_tiled_env_interactive.py`、`tests/test_tiled_telemetry.py`、`tests/test_tiled_cumotion.py` 通过。
- 旧 import 路径 `linkerbot_sim.tiled.telemetry` 和 `linkerbot_sim.tiled.cumotion` 已不可导入。
- `git status` 中只提交本次结构调整相关文件，不混入用户本地改动。
