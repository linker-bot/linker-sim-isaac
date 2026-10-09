# 碰撞边界

语言：[中文](collision-models.md) | [English](../../en/guides/collision-models.md)

“Kaleidoscope 不做碰撞规划”不表示禁用物理接触。需要区分三类能力：

1. **物理碰撞**：所选 PhysX CUDA 或 Newton 后端的接触推动 T-block；
2. **跨环境隔离**：PhysX builder 固定用 env IDs，Newton builder 固定用独立 worlds，均阻止不同 env 接触；
3. **规划碰撞/避障**：cuRobo collision world、cache 和 obstacle sampling，只属于 Mirror。

Mirror 可以从 robot sphere、object primitive/mesh 与 stage provider 构建 planning scene，并在 request
边界刷新 fingerprint。Kaleidoscope config/runtime closure 禁止 import planning/collision backend，也不为
每个 env 分配 collision cache。

Mirror 的 `planning.request_defaults` 只决定默认避障与刷新策略；cuRobo planner 是否具备碰撞能力以及
cache 容量由 `curobo.motion_planner` 独立声明。Kaleidoscope 的可选 cuRobo profile 必须省略整个
`motion_planner`。

资产的物理 collision approximation 仍是共享基础事实，见
[碰撞近似](../development/collision-approximation.md)。

`mirror/scene3` scene（文件 `configs/scenes/mirror/scene3.yaml`，内部 `scene.id: scene3`）的仓库地板
展示了这条边界：原始共面视觉 mesh 只负责渲染，包装资产在同一
局部高度声明不可见解析 Plane 负责物理接触。场景因此保持 `add_ground: false`，避免再叠加一层
默认地面；PhysX 与 Newton CPU/CUDA 都消费同一个解析碰撞体。

### Newton 的明确刚体接缝过滤

当前固定版本的 Newton USD importer 读取 collider 过滤，但不展开 MJCF importer 写出的
body 过滤。Simulator 在 prototype 复制前，将每个明确的 body 对只展开到这两个刚体实际
拥有的 shape；后代刚体仍可碰撞。PhysX 保留原 USD 关系。这样保留 AR5 link5–link7 等
必要接缝，不会关闭整手或相机碰撞，Mirror 与 Kaleidoscope 均使用这一处理。

## 固定手型、工装与携带物

八个带法兰的 L6/O6 profile 声明 `robot.planning_collision.mounted_urdf`。
Mirror 按具名 link 读取每个已导入 collider，并用当前实测关节（包括手部 follower）计算完整
URDF。一次臂规划中手型固定，以覆盖 collider 包围盒体积的球替换旧法兰通用抓手球，随每个
候选臂姿态移动。旧 AR5 YAML 已有粗略 L6 包络，不能称为完全没有手部表示。

每个网格单元的最大边长为 0.03 m，这是保守近似，可能拒绝物理上可通过的狭窄通道。
法兰包络不检查固定手/法兰/相机总成内部自接触；Task 仍需验证手型变化与当前 O6 拇指/相机
限制。携带物则对除显式 `touch_links` 外的所有安装 link 检查接触。这些功能不等于全身手指运动规划。

工作台与顶部相机 profile 使用 `object.planning_collision.source: colliders`，每个 collider
单独生成包围盒，保留零件间空隙。几何名为 `<对象名>/<实例根以下路径>`。Provider 发布世界坐标，
registry 在查询前转换到目标机器人基座。`static_others` 快照包含其他机器人的臂球、手和附件盒体；
`independent` 明确省略其他机器人。

`scene.planning_contacts` 用 `robot_label`、`link_name`、`geometry_name` 明确允许接触对，
示例场景仅允许固定基座安装接触。cuRobo 0.8 没有 link/世界物体级过滤，因此优化时省略指定几何，
返回成功路径前仍检查它对其他所有 link 的碰撞。需要绕开该零件的候选可能被拒绝并需重新规划；
不会全局禁用该部件，也不会修改物理接触过滤。

检查覆盖起终点及中间采样，相邻关节采样最大间隔 0.02 rad，上限 10,000 点；这是采样验证，
不是连续碰撞检测。`MotionResult.diagnostics.coverage` 和 registry 最近一次 coverage 提供
采样数、采样最小间隙、模型指纹、近似与允许接触对。直接关节命令不调用该检查；canonical direct IK
仍不考虑碰撞，避障属于 motion planning。

物理抓持建立后，可调用 Mirror Python facade：

```python
runtime.attach_planning_object("Tblock", 0, touch_links=("hand_lh_index_distal",))
# 保持真实抓持，规划并执行机械臂运动。
runtime.detach_planning_object("Tblock")
```

该声明只改变规划几何，不创建物理焊接或抓取控制器。每个快照根据实时状态冻结物体相对法兰的变换，
抓持或物体位置变化会更新下一次规划。物体从本机器人世界视图移入候选法兰包络，其他机器人仍能看到它；
detach 后恢复为实时世界物。state/snapshot 按机器人 label 保存声明，restore 在物理写入前验证，reset
清空声明。更换几何或附件资产需要重建 runtime。

手型/携带物变化会销毁并延迟重建所有相关 cuRobo solver，包含 graph 与验证模型；单纯臂运动复用
安装模型。Mirror cuboid cache 为 128，覆盖示例中的另一台完整机器人与工装；更大场景需明确配置容量。

`just smoke-mirror-planning-geometry` 验证原生手/相机独有障碍、状态刷新、携带物中间路径碰撞与
声明生命周期。探针使用规划测试物，不声称完成了真实物理抓取。

碰撞感知 timeline 在臂规划结束前不允许命令改变该臂的冻结手型；应先执行手型变化，再从新快照规划。
`static_others` 同样拒绝在规划结束前移动其他机器人的命令。保持/当前位姿目标和规划结束后的变化可以使用；
直接臂手联合控制仍开放。这些检查不证明物理姿态能保持，也不提供动态障碍预测或多机器人联合规划。
`source: colliders` 当前只接受静态对象；动态对象继续使用跟随实时根位姿的显式几何，尚未提供动态链 collider 提取。

任务阶段可临时允许明确 link/geometry 对，并在作用域内**完成规划和执行**：

```python
with runtime.planning_contact_scope(0, (("hand_lh_index_distal", "Tblock"),)):
    # 在这里调用任务的抓取接近规划和执行。
    ...
```

退出作用域（包括异常）会恢复此前策略；嵌套作用域保留外层许可。其他 link 仍检查该对象。
它不修改物理过滤，也不通过 reset/snapshot restore 赋予持久许可；携带物的 `touch_links` 是另一项声明。

当前 AR5 cuRobo YAML 还保留 flange/TCP 对本臂 link4、link6、link7 的排除；整个手、
相机及携带物聚合在同一 flange frame，因此也继承这些排除。mounted coverage 的
`ignored_own_arm_links` 给出实际配置的名称，不代表完整的手对本臂检查。Task 仍需补查，
不能未经细化臂的粗球和安装接触验证就直接删除旧排除。

## 当前 USD 解析器的临时处理

Kit 携带的 OpenUSD 25.11 在一个刚体有多个 mesh 时，`_FinalizeCollisionDescs` 会并行写同一 vector，
本轮分解碰撞资产触发了原生分配器崩溃。Isaac MJCF/URDF 转换和 Newton USD 解析阶段临时使用
一个 USD Work 线程，结束或异常后恢复原值，不改变导入后的物理、渲染或并行采样。此处理仅针对携带的 25.11 版本。

上游 [OpenUSD 修复 ed857d77c95b](https://github.com/PixarAnimationStudios/OpenUSD/commit/ed857d77c95b27dd9cf0919e8ed36366cf3301e6)
已改为线程安全收集再串行合并；携带的 runtime 升级到验证过的修正版后可移除临时处理。
此分配器崩溃与 Hydra `_MarkInstancerDirty` 诊断是不同问题。

Kit 的 `PXR_WORK_THREAD_LIMIT` 会锁住公开 Work setter。本版本的解析保护仅在固定
Linux / USD 25.11 上调用已加载 Kit 扩展导出的 `WorkImpl_SetConcurrencyLimit(unsigned)`，
并核验实际并行度及退出恢复。不会导入第二套 USD、修改二进制或环境变量；符号缺失或限制
未生效就拒绝继续不安全解析。升级到已验证修复版 USD 后移除此版本特定兼容处理。
