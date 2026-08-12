# 命名与身份

语言：[中文](naming.md) | [English](../../en/development/naming.md)

资产、profile、scene instance、session selector 和持久化 identity 是不同层，不能互相推导。

## 产品名

- `mirror`：现实映像产品；类型前缀 `Mirror`；
- `kaleidoscope`：并行强化学习产品；类型前缀 `Kaleidoscope`；
- `isaac`：共享 engine infrastructure，不是第三个产品；
- `training.skrl`：Kaleidoscope 下游 consumer，不拥有 runtime。

配置、脚本、Kit、public module 和文档统一使用上述名称，不保留旧入口 alias。

## Identity 层次

| Identity | Owner | 示例 | 生命周期 |
| --- | --- | --- | --- |
| 资产 | `assets/` 主文件与内部 joint/link 名 | `AR5V2_L6V1_L` | 物理模型稳定身份 |
| Profile selector | mode root 的 `profiles.<slot>` | `mirror/scene3` | catalog 查找键 |
| Profile 文件 | `configs/` 根下的受约束路径 | `configs/scenes/mirror/scene3.yaml` | 磁盘来源/provenance |
| Scene identity | `scene.id` | `scene3` | 加载后的稳定场景身份 |
| Scene robot | `scene.robots[].label` | `left_arm` | snapshot/telemetry 稳定 key |
| Session robot ID | robot 列表零基顺序 | `0` | 仅本次进程 |
| Object instance | `scene.objects[].name` | `Tblock` | scene 内稳定 key |
| Kaleidoscope env ID | mode root `environments` 中的环境行号 | CUDA `int64` selector | 当前 env batch |
| Camera | Mirror `scene.cameras[].id` | `world_rgbd` | frame/output namespace |

Profile 引用是安全相对名称，不是任意路径。禁止 `..`、绝对路径、空 component、反斜杠和任一
component 中的点号（包括把 `.yaml` 全路径当 selector）。示例：

```text
configs/modes/mirror/physx_cpu.yaml            -> mirror/physx_cpu
configs/modes/mirror/newton_cpu.yaml           -> mirror/newton_cpu
configs/modes/kaleidoscope/newton_cuda.yaml    -> kaleidoscope/newton_cuda
configs/scenes/mirror/scene3.yaml              -> mirror/scene3
configs/scenes/kaleidoscope/tblock_push.yaml   -> kaleidoscope/tblock_push
configs/tasks/kaleidoscope/tblock_push_v1.yaml -> kaleidoscope/tblock_push_v1
configs/robots/ar5v2_l6v1_l.yaml               -> ar5v2_l6v1_l
```

Scene selector 的首段必须与 mode 产品一致；Mirror 不能引用 `kaleidoscope/...`，反之亦然。selector
不含 `.yaml`，symlink 解析后也不能越出所选产品 namespace；`scene.id` 只等于文件 basename，
不重复产品命名空间。例如
`mirror/scene3`、`configs/scenes/mirror/scene3.yaml` 和 `scene.id: scene3` 分别是查找键、文件路径和
加载后的身份。

## Joint、link 与 TCP

上游 URDF/MJCF 名称一旦被 mesh、controller、mimic 或 cuRobo 引用就保持不变。控制数组顺序必须由
明确 joint name mapping 得到，不能依赖文件顺序。公开 quaternion 使用 `wxyz`；只有第三方 API
边界可局部重排。

Custom TCP 在 robot profile 中声明 frame、parent、xyz/rpy。Mirror planning 与 Kaleidoscope batch IK
都使用相同稳定 frame identity，但各自拥有 solver context。

## Robot 与 object

Robot label 符合 `[A-Za-z0-9_]+` 且 scene 内唯一。Session ID 稠密但不是持久硬件身份；重排 scene
后必须重新发现。需要扁平 joint namespace 时在输出边界加 label，例如：

```text
left_arm/AR5V2_L_arm_joint_1
right_arm/AR5V2_R_arm_joint_1
```

Object 的 asset/profile/instance/prim path 相互独立。Kaleidoscope snapshot 使用 instance name 匹配
字段，并用 env-local pose；clone writer 再按 target env origin 生成 world pose。

## Camera 与输出

Camera ID 只属于 Mirror scene，不能含 path separator；`prim_path` 是独立绝对 USD identity。
CSV/MCAP/image destination 由 outputs profile 拥有，不从 robot/object 名自动猜测。Kaleidoscope 不生成
per-env camera/topic/file 名称。

## 验证

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile physx_cuda
just test-architecture
```

重命名资产后还要校验 mesh reference、joint/controller/mimic mapping、TCP、snapshot label、prim path 和
cuRobo robot description。不要用兼容 alias 掩盖漏改引用。
