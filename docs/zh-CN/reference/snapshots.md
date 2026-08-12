# 状态、Snapshot 与 Clone

语言：[中文](snapshots.md) | [English](../../en/reference/snapshots.md)

项目有三种刻意不同的状态类型。它们的设备、恢复范围和持久化语义不能互换。

| 类型 | 所属 | 存储 | 恢复范围 |
| --- | --- | --- | --- |
| `SceneSnapshot` / `linkerbot.scene-snapshot.v1` | Mirror | 冷 CPU/NumPy/JSON | 一个现实映像的 robot/object 物理状态 |
| `KaleidoscopeEpisodeSnapshot` | Kaleidoscope | owned CUDA tensor | K 个 env 的物理、task/history/counter/RNG episode 状态 |
| Persistent checkpoint | Kaleidoscope 显式冷 API | CPU 文件 | 从 episode snapshot 选择字段进行磁盘 round-trip |

## Mirror scene snapshot

Wire operation：

- `snapshot.get`：无 arguments，返回 owned mapping；
- `snapshot.set`：必需 `snapshot`，可选 `label_map` 与 boolean `strict`。

Snapshot 使用稳定 robot/object label 与 `wxyz` quaternion。Restore 会先校验 schema、目标拓扑、
joint 映射、shape 和有限性，再执行 mutation。setter 中途失败时按逆序 rollback；rollback 也失败则
runtime fail-stop。Snapshot 不包含 transport queue、planner worker 或 camera frame。

Mirror scene snapshot 保留跨 PhysX/Newton 的逻辑恢复能力。Newton-origin snapshot 会额外保存
SolverMuJoCo 的 TIME、ACT、WARMSTART 与仿真时钟；恢复到 Newton 时精确写回这些字段。PhysX-origin
snapshot 没有对应 payload，恢复到 Newton 时会把这些字段重置到已提交 baseline，并把 Newton 时钟
归零，绝不沿用目标 runtime 恢复前的积分历史。该 reset 与 robot/object 写入处于同一补偿事务中。
这保证恢复后的下一步由 snapshot 逻辑状态和明确 baseline 决定，但不承诺延续源物理引擎的私有求解器
轨迹。

Mirror 不提供 clone 操作，因为产品只拥有一个场景。每次 capture 都返回独立 mapping，既不会创建
第二个仿真 World，也不具备批量 env clone 语义。

Capture 在 `metadata.info` 写入 `linkerbot.snapshot.control_mode`，记录 active mode 与来源
generation。Restore 要求 runtime 已处于相同模式，不自动切 mode，也不恢复 generation。legacy
snapshot 继续读取 per-joint target mode metadata；两种 metadata 都没有时只允许 position。

## Kaleidoscope GPU state

```python
state = env.get_state(
    env_ids,
    fields=("robot.q", "object.pose_local_wxyz"),
)
env.set_state(state, env_ids)
```

`env_ids` 必须是 env device 上的一维 `torch.int64` tensor。默认读取返回 owned tensor；所有字段
第一维为 K。Set 会先对全部字段做 unknown/shape/device/dtype/finite 预检，然后调用 engine writer，
最后 `index_copy_` canonical buffer。Writer 失败后 `poisoned=True`，禁止继续 step。

PhysX CUDA 与 Newton 共享以下核心 schema：

| 字段 | 语义 |
| --- | --- |
| `robot.q` | 按场景顺序拼接所有机器人的完整 articulation DOF。 |
| `robot.qd` | 与 `robot.q` 相同的完整 articulation DOF 速度。 |
| `robot.target` | 当前 active mode 的受控 command target。 |
| `robot.position_reference` | 与 active mode 无关、固定使用 rad 的位置参考。 |
| `object.pose_local_wxyz` | env-local XYZ 与 `wxyz` 朝向。 |
| `object.com_velocity` | 物体质心的线速度与角速度。 |

Task/history/counter/RNG bindings 在此核心 schema 上追加。Snapshot 保存所有登记字段；不能用
world-space object pose 代替 `object.pose_local_wxyz`，也不能把仅受控关节向量当作 `robot.q`。

Newton 还会登记后端私有字段 `solver.persistent`。它是每个 env 一行的 CUDA matrix，保存
会影响下一 physics step 的 SolverMuJoCo 积分状态：TIME、ACT 与 WARMSTART
（`qacc_warmstart`）。该字段可写、可 reset、可 clone；默认 `get_state`、snapshot/restore 与
`clone_state` 都会在 GPU 上保留它。PhysX 不暴露此字段。

## Kaleidoscope episode snapshot

```python
snapshot = env.snapshot(env_ids)
env.restore_snapshot(snapshot)
env.restore_snapshot(snapshot, target_env_ids=targets)
```

Snapshot 的 env IDs 和每个 field 都拥有独立 CUDA storage，不能 alias live runtime buffer。恢复到其它
selector 时 K 必须相等，device 必须一致。Snapshot 可以完整恢复 episode；只保存物理 pose 的普通
scene snapshot 不能恢复 reward history、termination counter 或随机数流。

当前 schema 2 记录 `control_mode` 与来源 `control_generation`。Restore 只允许 runtime 已处于同一
mode，不自动切 mode，也不恢复 generation。schema 1 仅作为 position snapshot 兼容；缺少
`robot.position_reference` 时从旧 `robot.target` 派生。

共享核心字段不表示完整 snapshot 可跨后端交换。兼容 fingerprint 同时包含 resolved config 与完整
field schema；Newton 的 `solver.persistent` 等后端私有状态也参与其中。因此 PhysX snapshot 不能恢复
到 Newton，Newton snapshot 也不能恢复到 PhysX；snapshot 必须留在同一后端及同一配置合同内。

## GPU clone_state

```python
env.clone_state(source_env_ids, target_env_ids, include_rng=True)
```

规则：

- source/target 均非空、唯一、范围有效、长度相同；
- 两个 selector 不可重叠；
- 先 clone 所有 source 行，再进入 writer 阶段，避免 field 间看到不同逻辑时刻；
- 默认复制 logical RNG key/counter，使 clone 后首次 rollout 可复现；
- 完成后统一 `physics_runtime.forward()`，让所选 PhysX CUDA 或 Newton 后端在稳定边界
  同步派生状态；
- 不经过 CPU/NumPy，也不创建新 env 或新 session。

## 持久化 checkpoint

磁盘保存/加载是显式冷路径，允许整批 CUDA→CPU 和 CPU→CUDA。它用于训练恢复或离线分析，不能
放入每步 callback。文件必须带 schema/version、字段名、dtype/shape 和 env ID；加载后重新生成 owned
CUDA tensor，不能使用 memory-mapped CPU array 充当 live state。

## 坐标

Kaleidoscope snapshot 中的 object/TCP pose 使用 env-local canonical frame。Clone 到 target env 时，
writer 按 target origin 转成 world pose；直接复制 world pose 会把对象放进 source env，是错误实现。
