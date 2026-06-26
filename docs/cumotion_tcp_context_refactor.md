# cuMotion 自定义 TCP 装配改造设计

## 1. 背景

当前项目已经把 `TcpFrame` 抽象成通用 TCP 描述，把 `write_tcp_urdf(...)` 放在
`source/manipulation_project/backends/cumotion/tcp_urdf_builder.py` 中，用临时 URDF
向 cuMotion 暴露自定义 TCP frame。

cuMotion 的核心约束是：

- FK/IK/planner 只能使用 robot description 中包含的 frame。
- 自定义 TCP frame 必须在 `CuMotionContext` 创建前写入待加载的 URDF。
- `CuMotionContext` 初始化后会加载 URDF/XRDF，并缓存 `robot_description`、`kinematics`、
  `frame_names()` 和 collision world。

因此，自定义 TCP 的 URDF 写入不是任务语义，而是 cuMotion 后端装配细节。

## 2. 当前问题

以 `tasks/pinch_grasp.py` 为例，任务层现在同时承担了两类职责：

1. 任务语义：
   - 根据 MJCF 和闭合手型计算 pinch TCP。
   - 计算 approach、grasp、lift、wiggle 等目标。
   - 处理完整 articulation DOF 映射、手部动作和执行节奏。
2. cuMotion 后端装配：
   - 创建临时目录。
   - 调用 `write_tcp_urdf(...)`。
   - `replace(CuMotionConfig, urdf_path=..., custom_tcp_frame=...)`。
   - 重新创建 `CuMotionContext`。

第二组逻辑属于后端要求。它放在任务层会带来几个问题：

- 任务层需要知道 cuMotion 必须通过 URDF fixed link 暴露 TCP。
- 后续其它任务如果也需要自定义 TCP，会重复 tempdir/URDF/context 装配代码。
- 临时文件生命周期容易被写错；URDF 文件必须至少活到 `CuMotionContext` 和相关后端对象使用结束。
- 文档边界和代码边界不完全一致：`tcp_urdf_builder` 属于 `backends/cumotion`，但调用流程外泄到任务层。

## 3. 目标

改造目标是让任务层只描述“需要哪个 TCP”，让 cuMotion 后端负责“怎样让 cuMotion 认识这个 TCP”。

任务层应该负责：

- 计算或读取 `TcpFrame`。
- 指定本次 IK/planner 使用的 `tcp_frame_name`。
- 同步环境障碍物到 context。
- 做完整 DOF 到 cuMotion C-space 的名称映射。

cuMotion 后端应该负责：

- 判断 TCP 是否已经存在于当前 robot description。
- 对不存在的自定义 TCP 写出临时 URDF。
- 用带 TCP 的 URDF 创建新的 `CuMotionContext`。
- 管理临时 URDF 的生命周期。
- 维持现有 `make_inverse_kinematics(...)`、`make_motion_planner(...)`、`make_forward_kinematics(...)`
  的使用方式。

## 4. 非目标

本次改造不建议做以下事情：

- 不把完整 articulation DOF 映射移入 `backends/cumotion`。cuMotion C-space 通常只覆盖机械臂主动关节，灵巧手、mimic follower 和 controller command space 仍属于任务/controller 层。
- 不在每次 `IK.solve(...)` 或 `MotionPlanner.plan(...)` 时动态写 URDF。TCP 必须在 context 创建前进入 robot description，按请求动态重建 context 会破坏缓存和 seed 连续性，也会增加大量开销。
- 不修改原始仓库 URDF。仍使用临时 URDF 或显式输出目录，避免污染基础资产。
- 不把 pinch TCP 的几何计算放进 cuMotion 后端。pinch TCP 来自灵巧手 MJCF 和闭合手型，是任务/机器人语义，不是 cuMotion 通用能力。

## 5. 推荐设计

### 5.1 新增后端装配入口

建议在 `backends/cumotion/context.py` 或新增 `backends/cumotion/tcp_context.py` 中提供一个
context manager，用来创建带自定义 TCP 的临时 context。

推荐形态：

```python
with CuMotionContext.with_tcp(config, tcp) as context:
    solver = context.make_inverse_kinematics()
    planner = context.make_motion_planner()
```

或者函数式入口：

```python
with make_cumotion_context(config, tcp=tcp) as context:
    solver = context.make_inverse_kinematics()
```

这里推荐 context manager，而不是简单返回 `CuMotionContext`，因为临时 URDF 的目录生命周期必须被显式持有。

实现时建议优先采用新增 `tcp_context.py` 的方式，避免继续膨胀 `context.py`：

```python
with make_cumotion_context(config, tcp=tcp) as context:
    solver = context.make_inverse_kinematics()
```

如果需要额外暴露临时 URDF 路径用于测试或诊断，可以让 context manager 内部使用一个轻量
handle，但对任务层公开的主要入口仍应保持“传入 config/tcp，得到 context”的简单形态。

### 5.2 行为语义

入口接收：

| 参数 | 含义 |
|---|---|
| `config` | 基础 `CuMotionConfig` |
| `tcp` | `TcpFrame | None`，为空时直接创建普通 `CuMotionContext` |
| `output_dir` | 可选；为空时使用临时目录 |

首版不建议提供 `overwrite_existing`。当前 `write_tcp_urdf(...)` 是“追加 fixed link”的纯工具，
不是安全重写已有 link/joint 的 URDF 编辑器。已有 frame 的使用应通过基础 URDF/XRDF 和
`CuMotionConfig.custom_tcp_frame` 表达，避免传入的 `TcpFrame.xyz/rpy` 被静默忽略。

推荐行为：

1. 如果 `tcp is None`：
   - 直接创建 `CuMotionContext(config)`。
2. 如果 `tcp.frame_name == config.flange_frame` 且 offset 为零：
   - 视作使用法兰 TCP，不写临时 URDF。
   - 使用 `replace(config, custom_tcp_frame=None)` 创建 context，保证未显式传 `tcp_frame_name` 的 IK/planner 回到法兰 frame。
  - 零 offset 应使用 `np.allclose(tcp.xyz, 0.0)` 和 `np.allclose(tcp.rpy, 0.0)` 判断，避免浮点精度问题。
3. 如果 `tcp.frame_name` 已经存在于基础 URDF link：
  - 如果不是第 2 条的零 offset 法兰 TCP，应抛出 `ValueError`，不要静默跳过。
  - 如果确实要使用已经存在的工具 link，应通过基础 `CuMotionConfig.custom_tcp_frame` 或显式 frame 名配置表达，而不是传入一个会被忽略 offset 的 `TcpFrame`。
4. 如果 `tcp.frame_name` 不存在：
   - 在临时目录中写出追加 fixed TCP link 的 URDF。
   - 使用 `replace(config, urdf_path=temp_urdf, custom_tcp_frame=tcp.frame_name)` 创建 context。
5. 创建 context 后必须校验：
  - `context.has_frame(tcp.frame_name)` 为真。
  - `context.config.custom_tcp_frame == tcp.frame_name`，除非使用的是零 offset 法兰 TCP。
  - 校验失败时抛出 `ValueError`，提示 URDF/XRDF 或 cuMotion frame 解析问题。
6. context manager 退出时：
   - 释放临时目录。
   - 不尝试修改或清理原始 URDF/XRDF。

注意：装配前的“frame 是否存在”判断建议解析基础 URDF 的 `<link name="...">`，不要为了判断一次 frame
是否存在而先创建一个普通 `CuMotionContext`，再创建第二个带 TCP 的 context。最终仍需要在
`CuMotionContext` 创建后用 `context.has_frame(...)` 做后验校验，因为 cuMotion kinematics 暴露的 frame
才是 IK/planner 真正可用的 frame。

### 5.3 可能的数据结构

如果不希望把 context manager 直接塞进 `CuMotionContext` 类，也可以新增轻量 owner：

```python
@dataclass
class CuMotionContextHandle:
    context: CuMotionContext
    tcp_urdf_path: Path | None
    owns_tempdir: bool
```

但该对象仍应支持：

```python
with CuMotionContextHandle.from_config(config, tcp=tcp) as handle:
    context = handle.context
```

核心点是：不要只返回裸 `CuMotionContext` 后立刻丢掉 tempdir owner。

## 6. 任务层装配边界

### 6.1 后端装配职责

```python
with tempfile.TemporaryDirectory(prefix="pinch_ik_tcp_") as temp_dir:
    base_urdf_path = Path(self.cumotion_config.urdf_path)
    tcp_urdf = Path(temp_dir) / f"{base_urdf_path.stem}_{self.tcp_frame_name}.urdf"
    write_tcp_urdf(base_urdf_path, tcp_urdf, tcp)
    context = CuMotionContext(
        replace(
            self.cumotion_config,
            urdf_path=tcp_urdf,
            custom_tcp_frame=self.tcp_frame_name,
        )
    )
```

### 6.2 任务层调用形态

```python
with CuMotionContext.with_tcp(self.cumotion_config, tcp) as context:
    solver = context.make_inverse_kinematics()
    motion_planner = context.make_motion_planner(...)
```

任务层仍然保留：

- `make_pinch_tcp(...)`
- `context.joint_names()` 到 articulation DOF 的名称映射
- `context.sync_collision_world(...)`
- 手部闭合、lift、wiggle、日志和执行逻辑

后端装配入口负责：

- 管理临时目录生命周期
- 写入带自定义 TCP frame 的临时 URDF
- 构造带 `custom_tcp_frame` 的 `CuMotionConfig`

## 7. 文件组织建议

推荐新增或调整：

```text
source/manipulation_project/backends/cumotion/
├── context.py
├── tcp_context.py          # 可选：自定义 TCP context 装配
├── tcp_urdf_builder.py     # 保留：纯 URDF 写入工具
```

两种组织方式都可以：

1. 小改法：在 `context.py` 上添加 `CuMotionContext.with_tcp(...)`。
2. 更清晰法：新增 `tcp_context.py`，暴露 `make_context_with_tcp(...)` 或 `CuMotionContextHandle`。

如果后续还会支持多个自定义 TCP、工具库 TCP 或缓存临时 URDF，建议采用第二种。

采用第二种时，需要同步更新：

- `source/manipulation_project/backends/cumotion/__init__.py`，重新导出新的 context manager。
- 任务层 import，例如 `tasks/pinch_grasp.py` 应从 cuMotion 后端入口导入 context manager。
- 任务层不应再直接 import `write_tcp_urdf`，除非该任务本身就是 URDF 生成工具或测试。

## 8. 校验和错误处理

后端装配入口应该尽早给出清晰错误：

- `tcp.parent_frame` 不在基础 URDF link 中：抛出 `ValueError`。
- `tcp.frame_name` 已存在且不是零 offset 法兰 TCP：抛出 `ValueError`，不要静默跳过写入。
- 临时 URDF 写入成功但 `CuMotionContext` 加载后找不到 `tcp.frame_name`：抛出 `ValueError`，提示 URDF/XRDF 或 cuMotion 解析问题。
- `custom_tcp_frame` 和 `flange_frame` 相同：保持现有语义，视为无自定义 TCP。

建议保留 `CuMotionContext._validate_model_dependent_config()` 中对 `custom_tcp_frame` 的校验。

## 9. 测试建议

新增或调整测试：

- `test_context_with_tcp_writes_temp_urdf`
  - 输入基础 config 和 `TcpFrame`。
  - 确认创建 context 时使用了追加 TCP 的 URDF。
- `test_context_with_tcp_keeps_tempdir_alive`
  - 在 context manager 内确认临时 URDF 存在。
  - 退出后确认临时目录被清理。
- `test_context_with_tcp_sets_custom_tcp_frame`
  - 确认 `context.config.custom_tcp_frame == tcp.frame_name`。
  - 确认 `context.make_inverse_kinematics()` 和 `context.make_motion_planner()` 默认使用该 TCP，而不是回退到 `flange_frame`。
- `test_context_with_existing_frame_does_not_write_urdf`
  - 使用已经存在的 flange frame。
  - 确认不会重复添加 link。
- `test_context_with_existing_non_flange_tcp_rejects_ignored_offset`
  - 当 `tcp.frame_name` 已存在于基础 URDF，且不是零 offset 法兰 TCP 时，抛出 `ValueError`。
  - 避免传入的 offset 被静默忽略。
- `test_pinch_grasp_no_longer_imports_tcp_urdf_builder`
  - 可选，用更行为化的测试替代直接检查 import。
- 保留 `test_write_tcp_urdf`
  - `tcp_urdf_builder` 仍作为纯函数测试。

如果测试环境没有真实 cuMotion，可以用 fake context factory 或 monkeypatch `cumotion.load_robot_from_file(...)`，
重点验证路径和配置装配逻辑。

## 10. 实施步骤

建议分三步改，降低风险：

1. 新增后端 context manager，并用单元测试覆盖临时 URDF 生命周期和配置替换。
2. 修改 `pinch_grasp.py`，把 tempdir、`write_tcp_urdf(...)`、`replace(...)` 收敛到新入口。
3. 更新 `docs/cumotion_interface.md` 中“自定义 TCP URDF 接口”和“任务层典型数据流”部分，说明任务层只传 `TcpFrame`，后端负责装配带 TCP 的 context。

实现验收清单：

- 新增 `make_cumotion_context(config, tcp=None, ...)` 或等价 context manager。
- context manager 必须持有 `TemporaryDirectory`，不能返回裸 context 后立即释放临时目录。
- `tcp is None` 时保持普通 `CuMotionContext(config)` 语义。
- 自定义 TCP 不存在于基础 URDF 时才写临时 URDF。
- 创建 context 后必须校验目标 TCP frame 已进入 cuMotion frame 集合。
- `pinch_grasp.py` 通过后端 context manager 获取带 TCP 的 `CuMotionContext`。
- 不移动 pinch TCP 几何计算、完整 articulation DOF 映射、手部控制和执行节奏逻辑。
- 保持 `make_inverse_kinematics(...)`、`make_motion_planner(...)` 的调用语义：未显式传 `tcp_frame_name` 时优先使用 `config.custom_tcp_frame`，再回退到 `flange_frame`。

## 11. 结论

`tcp_urdf_builder.py` 放在 `backends/cumotion` 中是合理的；需要调整的是调用边界。

推荐把“临时 URDF + 新 `CuMotionContext`”封装成 cuMotion 后端内部的 context 装配入口。这样任务层只关心任务语义和 DOF 映射，cuMotion 后端统一处理 frame 必须存在于 robot description 的技术约束。
