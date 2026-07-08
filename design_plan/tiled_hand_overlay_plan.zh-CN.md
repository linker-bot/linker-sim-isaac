# Tiled 手部 Overlay 支持计划

## 结论

tiled 模式需要支持手部 overlay 和独立 hand motion queue，但实现仍应比旧 motion runtime 更窄。

需要支持的原因：

- 许多 tiled 任务需要在机械臂轨迹执行期间同步改变或保持手部姿态，例如接近时张手、移动时闭合、推动时保持固定手型。
- 手部 overlay 本质是 controller command-space 目标插值，不需要进入 graph search、trajectory optimizer 或 cuMotion path conversion。
- 它可以复用 tiled 已有的 `TiledTrajectoryBuffer` 回放机制：planner 仍只产出关节轨迹，runtime 每个 physics tick 仍只写完整 batched joint target。
- `before` / `after` 和独立 hand motion 可以被表达成同一个 env playback queue 中的 hand-only 段，不需要进入 `TiledCommandAdapter`。

不应该支持的范围：

- 不在 `TiledCommandAdapter.step_action()` 热路径里执行旧 runtime 的 running/pending motion queue。
- 不恢复旧 runtime 的全局 running/pending execution queue；只在 `TiledTrajectoryBuffer` 内为每个 robot/env 保存 playback queue。
- 不支持每个 env 独立推进不同数量的 physics steps；即使某些 env 只在执行手部段，physics step 仍同步推进全体 env。
- 不做无名字的手指列猜测。tiled overlay 必须通过 `joint_positions` mapping 显式写目标关节名，避免误改 arm 关节。

## 目标语义

### 手部 overlay 的位置

overlay 挂在已加载或已规划成功的轨迹上：

- `load_trajectory` 可以带 `overlays`。
- `plan` / `plan_queue` / 旧 MoveSpec-like 消息可以带顶层 `overlays`。
- async planner 成功后，overlay 随 planner result 一起载入 `TiledTrajectoryBuffer`。
- `hand` 和 `dual_hand` 作为独立 hand-only motion，默认追加到对应 robot/env 的 playback queue。

### 支持的 overlay 形式

支持 `before`、`sync` 和 `after`：

```json
{
  "type": "plan",
  "robot": "left",
  "env_ids": [0, 1],
  "duration_s": 0.5,
  "joint_positions": [0.2, 0.0, 0.0],
  "overlays": [
    {
      "timing": "sync",
      "left_hand": {
        "joint_positions": {
          "L6V1_L_hand_index_mcp_pitch": 0.7,
          "L6V1_L_hand_thumb_cmc_pitch": 0.5
        }
      }
    }
  ]
}
```

`left_hand.duration_s` / `right_hand.duration_s` 可选：

- 不写时使用整条轨迹时长。
- 写入时只影响该手部目标从起点插值到目标的时间；之后保持目标。

`joint_positions` 必须是 mapping：

- key 是 tiled robot 的 `command_joint_names` 中存在的关节名。
- value 是目标位置。
- 如果关节名不存在，直接拒绝请求。

### 轨迹采样语义

`TiledTrajectoryBuffer.step()` 的输出仍是完整 batch：

- 有 active playback 的 env：按当前 queue 头采样。`before` 和 `after` 是 hand-only 段；`sync` 覆盖主轨迹采样。
- 没有 active trajectory 的 env：保持传入的 `current_positions`，不会应用 overlay。
- overlay 起点来自载入轨迹时的当前 command target。
- overlay 终点来自 JSON mapping。
- 独立 hand motion 入队时，非手部列在实际执行时使用当时的 current target，避免排队期间 arm 轨迹完成后又被拉回旧姿态。

## 实施步骤

1. 扩展 `src/linkerbot_sim/tiled/trajectory.py`
   - 新增 `TiledTrajectoryOverlay` 数据结构。
   - `TiledTrajectoryBuffer.load(...)` 增加 `overlays` 参数。
   - `_Playback.sample(...)` 在轨迹样本上覆盖 overlay 关节列。
   - 每个 robot/env 从单 playback 扩展为 playback queue，用于 before/main/after 和独立 hand-only motion。
   - `status()` 增加 overlay 关节名、stage 和 queue 长度摘要，便于调试。

2. 扩展 `src/linkerbot_sim/tiled/planner_manager.py`
   - `TiledPlanningRequest` 增加 `trajectory_overlays`。
   - `TiledPlanningResult` 增加 `trajectory_overlays`。
   - linear planner 和 cuMotion tiled planner 成功结果都透传 overlay。

3. 扩展 `src/linkerbot_sim/app/interactive/tiled/planning_messages.py`
   - `load_interactive_trajectory(...)` 解析 `overlays` 并传给 buffer。
   - `planning_request_from_message(...)` 解析顶层 `overlays` 并写入 request。
   - `load_ready_planning_results(...)` 把 result 的 overlay 传给 buffer。
   - 支持 `before`、`sync`、`after` timing；对非 mapping `joint_positions`、未知关节名给出明确错误。
   - 新增 `hand` / `dual_hand` 消息，默认 queue 到 trajectory buffer。

4. 更新文档
   - 更新 tiled 使用说明。
   - 更新 tiled async planner 接入说明，把“不支持手部 overlay”改为“支持 before/sync/after hand overlay 和独立 hand queue”。

5. 补充测试
   - `TiledTrajectoryBuffer`：overlay 只覆盖 selected env 的指定列，并按轨迹时间插值。
   - tiled fake runtime：`load_trajectory` 带 overlay 后 `step_trajectory` 生效。
   - async plan：顶层 overlay 随 planner result 自动载入并回放。
   - before/sync/after 顺序回放。
   - 独立 hand motion queue 在 active arm trajectory 之后执行。

## 验收标准

- 不改变 `TiledCommandAdapter` 的同步 step-control 热路径。
- 不改变 planner worker 访问边界；worker 仍不访问 Isaac runtime。
- `load_trajectory` 和 async planner result 都能驱动 before/sync/after 手部 overlay。
- `hand` / `dual_hand` 能作为独立 hand-only motion 进入 playback queue。
- overlay 只通过显式关节名映射生效，未知关节名报错。
- 原 tiled command、trajectory、planner 单元测试继续通过。

## 执行结果

- 已在 `TiledTrajectoryBuffer` 中实现 per robot/env playback queue，并把 `before`、`sync`、`after` overlay 转换为 before/main/after stage。
- 已实现 `hand` / `dual_hand` 独立 hand-only motion，默认追加到对应 robot/env 的 playback queue。
- 已在 async planner request/result 中透传 `trajectory_overlays`，ready result 载入 buffer 后按同一套回放语义执行。
- 已更新 tiled 使用文档和 async planner 修改说明。
- 已运行 `PYTHONPATH=src env_isaaclab/bin/python -m pytest tests/test_tiled_*.py -q`，结果 `131 passed`。
