# 资产命名

语言：[中文](naming.md) | [English](../../en/development/naming.md)

本文定义资产、profile、场景实例、运行时协议、传感器和输出所使用的身份。这些名称彼此有关，
但不能互换。

## 通用原则

- 仓库拥有的资产和实体名使用稳定的 ASCII 字符与下划线。避免空格、`new`、`final` 等临时词，
  也避免可能被 importer 归一化为下划线的连字符。
- 上游 URDF/MJCF 名称一旦与 mesh、joint、controller 或 planning 引用形成契约，就应保持不变。
- 物理硬件身份、可复用 profile 身份、场景实例身份和 session 内数值 ID 必须分开。
- Joint、link、body、TCP、camera、path 和 topic 名称都由各自层负责，不能从另一层名称推断。

## 身份层次

| 身份 | 所有者 | 示例 | 契约 |
| --- | --- | --- | --- |
| 资产 | `assets/` 目录和主文件前缀 | `AR5V2_L6V1_L` | 物理模型及变体 |
| Profile | `configs/<group>/` 下的 selector | `ar5v2_l6v1_l` | 可复用且经过校验的配置 |
| Robot 实例 | Env `robots[].label` | `left_arm` | 场景与 Snapshot 的稳定匹配身份 |
| Robot session ID | Env `robots[]` 列表顺序 | `robot_id: 0` | 只在当前进程内有效的稠密公开 selector |
| Object 实例 | Env `objects[].name` | `Tblock` | 稳定的场景物体身份 |
| Camera | Env `sensors.cameras` mapping key | `world_rgbd` | Camera frame 与输出 namespace |

## Profile 名称

普通 profile 的 selector 是 `configs/<group>/` 下 YAML 文件的 stem。目录式 env profile 使用目录名，
并从其中加载 `base.yaml`。Profile selector 是一个安全 stem，不是路径。

示例：

```text
configs/envs/scene1.yaml             -> scene1
configs/envs/scene3_tiled/base.yaml  -> scene3_tiled
configs/robots/ar5v2_l6v1_l.yaml     -> ar5v2_l6v1_l
configs/logging/default_logger.yaml -> default_logger
```

`--runtime-profile`、`--env` 等 CLI 字段和 YAML 内的 profile 引用都使用这些 selector。需要 profile
名称时不要传 `configs/.../*.yaml`。Profile 名称是区分大小写的文件系统身份，应保持 checkout 中
已有的拼写。

## 硬件和资产名称

`AR5V2` 与 `L6V1` 是机械臂和灵巧手的实际硬件系列/版本身份；`L` 与 `R` 是物理变体，不是运行时
selector：

```text
AR5V2_L
AR5V2_R
L6V1_L
L6V1_R
AR5V2_L6V1_L
AR5V2_L6V1_R
```

当前资产树在目录和主文件前缀中保留硬件身份：

```text
assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
assets/single_system/hand/L6V1_L/L6V1_L.xml
assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
assets/single_system/arm/AR5V2_L/AR5V2_L_curobo.yml
```

Robot profile stem 使用当前配置约定，例如 `ar5v2_l`、`l6v1_l` 和 `ar5v2_l6v1_l`。不要添加
仓库不认识的资产修订后缀，也不要把 `V1`、`V2` 当成配置 schema 版本。

## Joint、Link、Body 与 TCP 名称

仓库中的机器人实体保留完整的硬件/类别前缀：

```text
AR5V2_L_arm_joint_1
AR5V2_L_arm_link7
AR5V2_L_arm_flan_link
L6V1_L_hand_base_link
L6V1_L_hand_thumb_cmc_roll
L6V1_L_hand_index_dip
L6V1_L_hand_couple_index
```

`flan_link` 是实际资产身份的一部分，不能静默改写。以下位置的 joint、link、body、actuator、mesh、
mimic/equality 和 TCP 名称必须保持一致：

- MJCF/URDF 文件。
- Mesh 文件名及引用。
- Robot profile 的 joint group 与 rigid-body group。
- Controller 的 active/follower joints。
- cuRobo URDF、robot YAML、collision 配置与 tool frame。
- Motion target、Snapshot、测试和命令示例。

Custom TCP frame 在 robot profile 中命名，例如 `AR5V2_L_tool_tcp` 或
`AR5V2_L_pinch_tcp`，并相对配置的 `flange_frame` 定义。cuRobo 可能在解析后的 cuRobo cache
目录中物化带 fixed link 的派生 URDF。Cache root 依次来自 `runtime.paths.cache_root`、
`LINKERBOT_SIM_CACHE_ROOT`、`XDG_CACHE_HOME` 或用户 cache 目录。派生文件不是主资产，不能替换
仓库中的 URDF。

## Robot 运行时身份

Env 的每个 `robots[]` 条目选择一个可复用 `robot_profile`，并定义一个场景实例。`label` 必须唯一
且符合 `[A-Za-z0-9_]+`；省略时为 `<robot_profile>_<robot_id>`。默认 USD path 为
`/World/Robots/<label>`。

`robot_id` 按 `robots[]` 的零基顺序生成，ID 稠密且不能在 YAML 中配置。Single Scene 和 Tiled Scene 公开控制
协议使用 session `robot_id`；每次启动进程或调整 env 顺序后，都应从 `status` 重新发现。不要把它
缓存为持久硬件身份，也不要用 `L`、`R` 选择 robot。

Label 是内部、telemetry 和 Snapshot 匹配使用的稳定 key。扁平输出需要全局唯一 joint 名称时，
在不修改资产 joint 名的前提下加上 label 前缀：

```text
left_arm/AR5V2_L_arm_joint_1
right_arm/AR5V2_R_arm_joint_1
```

## Object 身份

Object 的资产名、profile 名和场景实例名彼此独立。当前示例包括：

```text
asset:    workstationV1_armbase, capsuleropeV1_default, TblockV1_default
profile:  workstation_armbase, capsule_rope, TblockV1_default
instance: workstation, rope, Tblock
```

Env `objects[].name` 在场景内唯一，Snapshot 与 Tiled per-env pose override 使用该名称。它以字母或
下划线开头，后续只含字母、数字或下划线。`object_profile` 是可复用 profile selector；
`runtime_handle` 是可选交互别名，不会重命名 profile、资产或 USD prim。实例名、runtime handle 和
最终 prim path 都必须唯一，handle 也不能与另一个 object 的 name 冲突。

Env 实例拥有 `prim_path` 与 `root_pose`。省略 `prim_path` 时，runtime 推导
`/World/Objects/<name>`。资产 source/path、导入选项、physics、planning collision 与受支持的
资产内部 `root_path` 仍由 `configs/objects/<object_profile>.yaml` 拥有。

## Camera 与输出名称

`sensors.cameras` 下的 key 是 `CameraFrame`、离线 metadata 与输出路由使用的 camera 名。该名称
不能为空，也不能包含路径分隔符。`prim_path` 是独立的绝对 USD 身份；设置 `parent_prim_path` 时，
camera prim 必须位于该 parent 下。

Tiled camera 的 per-env pose override 引用基础 camera key。Runtime 展开后给每个 camera 分配
`env_NNN_<name>` 身份，并把相同的 `env_NNN` 段追加到已配置的本地 `save_dir` 和 Foxglove topic
前缀：

```text
base camera: world_rgbd
runtime camera: env_000_world_rgbd
local output: logs/cameras/world_rgbd/env_000/
topic prefix: /cameras/world_rgbd/env_000
```

同一 camera 输出目录中的 `metadata.jsonl` 索引确定性 payload 名称，例如 `rgb/000000.png` 和
`depth/000000.npz`。Foxglove camera channel 在已配置前缀后追加 `/rgb`、`/depth` 和 `/info`。

其它输出名称仍由各自配置拥有。Single Scene joint CSV 把配置路径当作模板，追加 session robot ID 和
label，例如 `run.0.left_arm.csv`。State topic 直接来自 `runtime.telemetry.topics`，不会由 robot 或
资产名推导。全部 destination owner 见[输出与持久化](../reference/outputs.md)。

## 验证

不启动 Isaac，校验完整 profile graph：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

重命名后运行对身份敏感的测试：

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_system_configs.py \
  tests/test_robot_profile_schema.py \
  tests/test_object_instances.py \
  tests/test_sensor_camera_config.py \
  tests/test_tiled_cameras.py -q
```

查找残留旧名称：

```bash
rg "AR5-V2|L6-V1|capsule-rope" assets configs src scripts tests README.md docs
```

重命名后还应确认：

- 主 URDF/MJCF/XML 可解析，且所有 mesh 引用存在。
- Asset path、profile 引用、env label/name 与 prim path 仍然唯一。
- Joint group、controller mapping、trajectory target、TCP frame、cuRobo description、Snapshot 和
  示例已同步更新。
- Camera key、Tiled per-env override、输出目录和 topic prefix 仍解析到预期 namespace。
