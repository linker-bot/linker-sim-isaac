# Snapshot 数据与恢复参考

语言：[中文](snapshots.md) | [English](../../en/reference/snapshots.md)

本文是 `linkerbot.snapshot` 数据结构、目标匹配、恢复结果和事务语义的唯一详细参考。
[Single Scene JSON 参考](single-scene-json.md)与 [Tiled Scene JSON 参考](tiled-scene-json.md)只拥有各自的消息外层、
selector 和响应差异。

Snapshot 描述一个 Single Scene runtime 或一个 Tiled env 的逻辑状态。Tiled env 批次只出现在
`set_snapshot.env_ids` 和 `clone_state.target_env_ids` 中，不进入 snapshot 主体。

## 1. 完整 JSON 结构

以下示例包含 robot、object root、child body、metadata 和全部可选运动状态。所有数值必须有限；
实际使用时优先原样传递 `get_snapshot` 返回的主体。

```json
{
  "schema": "linkerbot.snapshot",
  "metadata": {
    "source_runtime": "tiled_scene",
    "source_env_id": 0,
    "step": 120,
    "time_s": 0.5,
    "coordinate_frame": "env-local",
    "info": {
      "per_env": {
        "replay_id": "case_001"
      }
    }
  },
  "robots": [
    {
      "label": "robot_0",
      "robot_id": 0,
      "robot_profile": "ar5v2_l6v1_l",
      "asset_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "joint_names": ["arm_joint_1", "arm_joint_2"],
      "joint_positions": [0.1, -0.2],
      "joint_velocities": [0.0, 0.05],
      "command_joint_names": ["arm_joint_1", "arm_joint_2"],
      "command_targets": [0.12, -0.18]
    }
  ],
  "objects": {
    "rope": {
      "name": "rope",
      "object_profile": "capsule_rope",
      "positions_local": [0.3, 0.0, 0.1],
      "orientations_wxyz": [1.0, 0.0, 0.0, 0.0],
      "linear_velocities": [0.0, 0.0, 0.0],
      "angular_velocities": [0.0, 0.0, 0.0],
      "body_names": ["segment_0", "segment_1"],
      "body_positions_local": [[0.3, 0.0, 0.1], [0.4, 0.0, 0.1]],
      "body_orientations_wxyz": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      "body_linear_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      "body_angular_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    }
  }
}
```

顶层只支持以下字段：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `schema` | string，必填 | 必须精确为 `linkerbot.snapshot` |
| `metadata` | object，可省略 | 来源和坐标诊断；省略时使用空来源信息 |
| `robots` | array，必填 | 每项一个 robot；可以为空，但不能使用 label-keyed object |
| `objects` | object，可省略 | key 是稳定 object name；采集响应始终输出该字段 |

顶层与 robot entry 拒绝未知字段。metadata 的扩展信息必须放进 `info`；object 也只应使用本文列出的
字段，因为未知内容不会由 `as_dict()` 保留。JSON transport 还会拒绝重复 key、非标准数值和尾随内容。

## 2. Metadata

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `source_runtime` | string | 只用于诊断的 producer 名；常规采集使用 `single_scene` 或 `tiled_scene`，内存 debug adapter 使用 `tiled_scene_debug` |
| `source_env_id` | integer，可选 | Tiled Scene source env；Single Scene 通常省略 |
| `step` | integer，可选 | 采集时 runtime step |
| `time_s` | finite number，可选 | 采集时仿真时间，单位秒 |
| `coordinate_frame` | string | object pose 的局部坐标约定；采集值为 `scene-local` 或 `env-local` |
| `info` | JSON object | profile fingerprint、robot labels 或 per-env metadata 等诊断信息 |

Metadata 不参与 robot/object 身份匹配。Tiled Scene adapter 把 object local pose 相对各目标 env origin
写回；Single Scene adapter 使用 scene-local pose。机器人关节状态不受 `coordinate_frame` 影响。

## 3. Robot Entry

| 字段 | 类型与 shape | 规则 |
| --- | --- | --- |
| `label` | 非空 string，必填 | 稳定 source 身份；同一 snapshot 内唯一 |
| `robot_id` | 非负 integer，必填 | source 会话诊断 ID；不用于恢复目标匹配，且在 snapshot 内唯一 |
| `robot_profile` | string，可选 | 两侧都提供时必须相等 |
| `asset_fingerprint` | string，可选 | 两侧都提供时必须相等 |
| `joint_names` | 非空 string array，必填 | 唯一名字决定位置/速度顺序 |
| `joint_positions` | finite number `[J]`，必填 | 与 `joint_names` 等长 |
| `joint_velocities` | finite number `[J]`，必填 | 与 `joint_names` 等长 |
| `command_joint_names` | string array，可省略 | 唯一名字决定 command target 顺序；采集响应会输出 |
| `command_targets` | finite number `[C]`，可选 | 出现时要求非空 `command_joint_names` 且等长 |

关节值采用 articulation 的原生 DOF 单位：revolute joint 为 rad/rad/s，prismatic joint 为
m/m/s。Snapshot 只保存 controller 管理的 command joints，不保存未控制 DOF。恢复按名字重排，
不能假定 source 和 target 数组列顺序相同。

## 4. Object 与 Body Entry

| 字段 | 类型与 shape | 单位与规则 |
| --- | --- | --- |
| `name` | 非空 string | 应与 `objects` 外层 key 相同；省略时 parser 使用外层 key |
| `object_profile` | string，可选 | 两侧都提供时必须相等 |
| `positions_local` | finite number `[3]` | m，`metadata.coordinate_frame` 标记的局部坐标 |
| `orientations_wxyz` | finite number `[4]` | 非零 `wxyz` 四元数；解析时归一化 |
| `linear_velocities` | finite number `[3]`，可选 | m/s |
| `angular_velocities` | finite number `[3]`，可选 | rad/s |
| `body_names` | string array，可省略 | 唯一 child rigid-body 名；采集响应会输出 |
| `body_positions_local` | finite number `[B,3]` | `body_names` 非空时必填，单位 m |
| `body_orientations_wxyz` | finite number `[B,4]` | `body_names` 非空时必填；逐行归一化 |
| `body_linear_velocities` | finite number `[B,3]`，可选 | m/s |
| `body_angular_velocities` | finite number `[B,3]`，可选 | rad/s |

普通 rigid object 可让 `body_names=[]`，只恢复 root。Dynamic-chain object 必须保留全部 body pose，
否则不能证明 child body 状态完整。速度字段在数据结构中可选，但 linear/angular 应成对提供。
Single Scene 的 live rigid view 会拒绝缺失的必要速度；Tiled Scene object view 会把省略的速度写为零。
零四元数和任何非有限数组都会被拒绝。

## 5. 身份、`label_map`、`strict` 与 `partial`

默认按 robot `label` 精确匹配；source `robot_id` 从不用于目标选择。`label_map` 是显式的
`source label -> target label` JSON object：不得为空，source 和 target 必须存在，target 不得重复。
提供 `label_map` 后只处理其中列出的 source robots。Object 不支持改名，始终按 `name` 匹配。

匹配顺序如下：

1. 解析 robot label 或 `label_map`。
2. 两侧都存在 `robot_profile`/`asset_fingerprint` 时要求相等。
3. 按名字建立 joint 与 command-joint 索引，不按列位置写回。
4. 按 object name 匹配；dynamic object 再按 body name 建立索引。
5. 全部检查和回滚状态采集成功后才进行第一次 runtime 写入。

`strict=true` 要求每个已映射 robot 的 joint 名集合以及每个 dynamic object 的 body 名集合完全相等，
但允许顺序不同。`strict=false` 对这些名字集合只写交集；交集为空仍拒绝。它不会忽略缺失的 object、
profile/fingerprint mismatch 或非法 `label_map`。

Single Scene object restore 会应用生成的 body name index mapping。当前 Tiled Scene writer 直接按 target view 列顺序
消费 body 数组，没有应用该 mapping；因此 Tiled Scene dynamic object 实际恢复还要求 source/target
`body_names` 顺序一致，不能依赖 body 重排或 `strict=false` 的 body 子集写回。

`partial` 只表示 snapshot 中有完整 robot/object entry 没有进入 mapping，例如显式 `label_map` 只选择
部分 source robots。当前结果不会因为一个已映射 robot/body 在 `strict=false` 下只写名字交集而自动
变为 true；调用方不能用 `partial=false` 推断每个 source 数组元素都已写回。

## 6. Capture 差异

| 行为 | Single Scene | Tiled Scene |
| --- | --- | --- |
| Capture 范围 | 当前完整 Single Scene | 必填 `env_id` 指定的单个 env |
| Metadata frame | `scene-local` | `env-local` |
| Metadata clock | adapter 不写 step/time | 写入 source env、runtime step/time |
| Robot 数组 | 每个 runtime robot 一项 | 去掉 env batch 维后每个 robot 一项 |
| Object pose | scene-local root/body | 去掉 batch 维的 env-local root/body |

Single Scene capture 把当前物理关节位置保存为 `command_targets`；Tiled Scene capture 保存实际
`target_positions`。两种 capture 都读取关节位置/速度、object root 和可用 child-body/velocity 状态，
返回的 NumPy 数据在 Python 对象中会复制，JSON 输出为独立 array。Asset fingerprint 同时包含规范化
资产路径和文件内容，因此内容相同但路径不同仍可能不匹配。Single Scene object capture 读取 world transform
而恢复写 local transform；带非恒等父变换的任意 prim 层级不具备严格 round-trip 承诺。

## 7. Restore Result

Python adapter 的统一成功结果为：

```json
{
  "event": "snapshot_restored",
  "accepted": true,
  "robots": ["robot_0"],
  "objects": ["rope"],
  "env_ids": [1, 2],
  "partial": false
}
```

| 字段 | 含义 |
| --- | --- |
| `event` | 普通恢复为 `snapshot_restored`；Tiled Scene clone 外层改为 `state_cloned` |
| `accepted` | adapter 完成恢复时为 true |
| `robots` | Python/Single Scene 中实际写入的 target labels；Tiled Scene JSON 外层转换为 `robot_ids` |
| `objects` | 实际写入的 target object names |
| `env_ids` | Single Scene 为空；Tiled Scene 为全部 target env IDs |
| `partial` | 第 5 节定义的 entry-level mapping 摘要 |
| `message` | 只有非空诊断文本时出现 |

Single Scene 执行异常返回 `snapshot_failed`、`accepted=false`、`id` 和 `error`。Tiled Scene 消息边界把异常收敛为
`{"event":"rejected","error":"..."}`。Python facade 直接抛出解析、匹配或 runtime 异常。

## 8. 补偿事务、Rollback 与 Fail-Stop

Snapshot restore 不是 PhysX 原生原子事务。Adapter 在第一次写入前完成数据解析、目标匹配，并捕获
每个目标的独立 rollback 状态；每个 setter 前注册补偿动作，失败时按写入逆序尽力恢复。

- Single Scene 回滚 robot position/velocity、controller cache 和 object 状态。observer cache reset 与
  collision registry invalidation 被标记为不可逆步骤。
- Tiled Scene 为每个 target env 单独保存 position/velocity、command target、adapter/TCP cache 和 object
  状态；成功写回后清空这些 env 的 trajectory，并取消与之相交的 planner request。
- 所有补偿成功且失败前没有不可逆步骤时，runtime 仍可使用，原始异常返回调用方。
- 任一补偿失败，或异常发生在已标记的不可逆步骤之后，runtime 记录首个 fatal reason、请求退出，
  并拒绝后续状态修改。调用方必须销毁并重建 runtime。

一个 target setter 可能已经产生物理副作用，因此客户端收到不确定终态时不能盲目重发同一恢复。

## 9. Single Scene 消息与 Admission

Single Scene 没有 env selector：

```json
{"type":"get_snapshot","id":"snapshot-1"}
```

```json
{
  "type": "set_snapshot",
  "id": "restore-1",
  "strict": true,
  "snapshot": {
    "schema": "linkerbot.snapshot",
    "robots": [],
    "objects": {}
  }
}
```

`label_map` 可选，`strict` 默认 true。上例只说明 envelope；实际恢复应使用完整 capture 结果。
成功响应增加 `backend="isaac"` 和 request `id`。

Single Scene transport 把 get/set 放入独立的主线程队列。`runtime.interactive.snapshot_timeout_s` 只限制
请求在主线程原子标记 executing 前的等待；默认 profile 为 30 秒。超时请求返回：

```json
{"event":"snapshot_timeout","accepted":false,"id":"snapshot-1"}
```

一旦 executing，调用方会越过 admission deadline 等待确定终态。Shutdown 是唯一提前结束等待的
情况：未执行请求返回 `snapshot_cancelled`；已执行请求返回 `snapshot_running`，它不是成功终态，
也不授权自动重发。

```jsonl
{"event":"snapshot_cancelled","accepted":false,"reason":"shutdown","id":"snapshot-1"}
{"event":"snapshot_running","accepted":true,"state":"running","id":"restore-1"}
```

## 10. Tiled Scene 消息与 Clone

Tiled Scene capture 只接受一个 source env；restore 把单份 snapshot 广播到显式、非空、无重复且范围合法的
target env：

```jsonl
{"type":"get_snapshot","env_id":0}
{"type":"set_snapshot","env_ids":[1,2],"strict":true,"snapshot":{"schema":"linkerbot.snapshot","robots":[],"objects":{}}}
```

成功 get 响应包含 `backend="isaac"`、`env_id`、`step`、`time_s` 和 `snapshot`。成功 set 响应包含
`backend="isaac"`、`robot_ids`、`objects`、`env_ids`、`partial`、`step` 和 `time_s`。

`clone_state` 在主线程执行一次 `get_tiled_scene_snapshot(source)`，再用同一事务语义写入所有 targets；
它不接受 `label_map`：

```json
{"type":"clone_state","source_env_id":0,"target_env_ids":[1,2],"strict":true}
```

响应 event 为 `state_cloned`，并额外返回 `source_env_id` 和 `target_env_ids`。Source env 不会被改写，
但若它也出现在 target 列表中，当前 selector 会接受并按同一 snapshot 写回。

## 11. Python Facade

数据模型与匹配函数是 `pure`；读取或写入实际 runtime 的 adapter 必须在拥有 Isaac runtime 的主线程
调用。公共导入入口是 `linkerbot_sim.snapshots`：

```python
from linkerbot_sim.snapshots import (
    SimulationSnapshot,
    check_snapshot_compatibility,
    clone_tiled_env_state,
    get_snapshot,
    set_snapshot,
)

snapshot = get_snapshot(runtime, env_id=0)
payload = snapshot.as_dict()
parsed = SimulationSnapshot.from_mapping(payload)
result = set_snapshot(runtime, parsed, env_ids=[1, 2], strict=True)
```

`get_snapshot(runtime, env_id=None)` 自动分派 Single Scene/Tiled Scene；Tiled Scene 必须传 `env_id`。
`set_snapshot(runtime, snapshot, env_ids=None, label_map=None, strict=True)` 同样分派，Tiled Scene 必须传
`env_ids`。需要明确 runtime 类型时可使用 `get_single_scene_snapshot`、`set_single_scene_snapshot`、
`get_tiled_scene_snapshot`、`set_tiled_scene_snapshot` 和 `clone_tiled_env_state`。目标描述和纯匹配检查由
`single_scene_target_descriptor`、`tiled_scene_target_descriptor`、`check_snapshot_compatibility` 和
`require_snapshot_compatibility` 提供。

## 12. 使用检查

- 保存完整 `snapshot` object，不要只截取 joint positions。
- 恢复前确认目标 runtime 已停止依赖旧状态的外部工作流。
- 除非明确重命名 robot，否则不传 `label_map`。
- 默认保持 `strict=true`；使用 name intersection 时逐项检查实际目标状态，不只检查 `partial`。
- 不修改 source `robot_id` 来选择目标；使用稳定 label 或显式 `label_map`。
- 把 `snapshot_running`、rollback error 和 fatal mutation 当作需要重建 runtime 的不确定状态。
