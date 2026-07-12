# 配置使用说明

语言：[中文](configuration.md) | [English](../../en/guides/configuration.md)

## 配置分层

- `configs/envs/`：场景、物体、传感器和机器人实例列表。
- `configs/robots/`：单个 articulation 的 Isaac 模型、控制分组和 cuRobo 模型绑定。
- `configs/curobo/`：设备、seed、容差、cache 等算法默认值，不保存机器人资源。
- `configs/controllers/<bundle>/`：一套 arm/hand/default 关节控制器参数。
- `configs/objects/`：场景物体资产与物理参数。
- `configs/logging/`：关节 CSV 的采样、刷盘、输出路径和列开关。
- `configs/runtime/`：入口模式、进程资源、telemetry/camera 输出和文件生命周期策略。

本仓库是 workspace 应用，不是可安装 Python 库。runtime profile、脚本、资产和
内置 task 资源都是应用组成部分，因此 `tool.uv.package = false`，本地 PEP 517 backend
也会明确拒绝 wheel、sdist 和 editable build。执行 `uv sync --all-extras` 后，应从
checkout 根目录以 `PYTHONPATH=src` 运行命令；不要只安装或复制 `src/` 目录。

所有项目 profile 只接受当前严格结构。固定 mapping 会严格拒绝未知键并报告完整字段路径；布尔值
不接受 `0/1` 或字符串，周期、采样步数和增益也会执行类型与范围校验。YAML 文档必须解析为
顶层 mapping；空文档、任意嵌套层级的重复 key 和非 mapping 文档都会带源码位置直接报错。

env profile 中只有 World 参数放在顶层 `env:` mapping。`robots`、`objects`、`solver`、
`visuals`、`sensors` 和可选 `tiled` 都是与 `env` 并列的顶层字段，不是 `env` 的子字段。
带 `tiled` 拓扑的目录型 env profile 由 `<name>/base.yaml` 与 `per_env_config_dir` 下的 override 文件合并；有效环境数
最终统一写入 `tiled.num_envs`，CLI 不覆盖。若目录型 base 省略该字段但存在 fragment，loader 会按
最大 `env_id + 1` 派生；没有该派生条件时缺省为 1。

runtime profile 是 Single Scene/Tiled Scene 入口的进程级配置 owner，当前顶层字段如下：

| 字段 | 职责 |
| --- | --- |
| `mode` | 选择 `single_scene` 或 `tiled_scene` 入口契约 |
| `profiles` | 选择 env、cuRobo、logging 和 controller profile |
| `simulation_app` | GUI、GPU、renderer 和 headless 启动参数 |
| `execution` | 控制、空闲步进、decimation 和命令默认值 |
| `interactive` | stdin、snapshot/history 上限、listener 和 transport 资源边界 |
| `planner` | backend、请求默认值、失败策略、worker 和 batch 上限 |
| `playback` | 每个 env 的 trajectory 队列、样本数和时长上限 |
| `camera_output` | 队列、编码、目录生命周期、字节配额和 drain 策略 |
| `telemetry` | env 选择、频率、模态、topic、live endpoint 和 MCAP path |
| `output` | CSV 和 MCAP 已有文件策略 |
| `paths` | 进程 cache root |
| `shutdown` | state publisher、camera publisher 和 transport 关闭超时 |

解析优先级是“代码默认值 < 所选 runtime YAML < 显式 CLI 覆盖”。入口中未传入的可选
CLI 参数保持 `None`，不会意外覆盖 profile。`--dump-effective-config` 会在启动 Isaac 前
输出 effective mapping、fingerprint 和每个叶子字段的来源。完整示例见
`configs/runtime/example.yaml`。

Telemetry 的 rate、模态、buffer/drop/error、exact topics、Foxglove live endpoint、
MCAP path 和 `primary_env_id` 必须写在 `runtime.telemetry`；CSV/MCAP 已有文件策略写在
`runtime.output`。env profile 不接受这些字段。消费语义见
[实时状态流使用说明](telemetry.md)。

所有内置 listener 的 host 只接受 `localhost` 或数值 loopback 地址。这些服务本身不提供认证或
TLS；远程访问必须经过认证 TLS 反向代理或 SSH tunnel，直接绑定非 loopback 地址属于无效配置。

MCAP 文件生命周期由 `runtime.output.mcap_existing_file_policy` 配置。

## 机器人实例

顶层 `robots` 必须是 list。配置禁止填写 `robot_id`；loader 按顺序生成连续 ID。

```yaml
robots:
  - label: robot_0
    robot_profile: ar5v2_l6v1_l
    prim_path: /World/Robots/robot_0  # 可省略
    controller_profile: default      # 可省略
    root_pose:
      xyz: [0.0, 0.09, 0.0]
      rpy: [-1.5707, 0.0, 0.0]
```

`label` 在场景内唯一并建议显式填写；缺省值为 `<robot_profile>_<robot_id>`。`prim_path`
属于场景实例，缺省为 `/World/Robots/<label>`，不放在 robot profile。`robot_id` 是会话内控制索引，
列表重排后可能变化；snapshot 和持久化数据使用 label/profile/fingerprint 匹配。

`controller_profile` 选择 `configs/controllers/<bundle>/`。优先级是 env instance > robot profile >
runtime `profiles.controller_bundle`；同一场景可以为不同机器人选择不同 bundle。

每个 controller bundle 必须包含 `arm_controller.yaml` 和 `hand_controller.yaml`，并可选
`default_controller.yaml`；bundle 不完整会直接报错。

controller 的 `position_control`、`velocity_control`、`effort_control` 各自声明 `method`、
`active_joints` 和 `follower_joints`。增益/限幅可写标量、与选中关节等长的 sequence，或按精确关节名
索引的 mapping；所有值必须非负且有限。

## Logging Profile

Single Scene runtime 通过 `profiles.logging` 选择 `configs/logging/<name>.yaml`；Tiled Scene runtime
不创建关节跟踪 CSV logger，也不消费该 profile。`logging.enabled` 控制是否打开 CSV，
`joint_tracking_path` 可为路径或 `null`，`flush_interval_s` 必须为正有限秒数，`interval_steps` 必须为
正整数。列开关统一使用 `log_actual_position`、`log_actual_velocity`、`log_command_position`、
`log_command_velocity`、`log_command_effort`、`log_action_effort`、`log_measured_effort` 和
`log_applied_effort`。后两项需要读取较重的 PhysX effort 数据，默认关闭。

## Robot Profile

```yaml
robot:
  name: ar5v2_l6v1_l
  kind: arm_hand
  asset_type: mjcf
  asset_path: assets/combined_system/AR5V2_L6V1_L/AR5V2_L6V1_L.xml
  controller_profile: default  # 可选

curobo:
  enabled: true
  planning_joint_group: arm
  robot:
    urdf_path: assets/single_system/arm/AR5V2_L/AR5V2_L.urdf
    robot_config_path: assets/single_system/arm/AR5V2_L/AR5V2_L_curobo.yml
    load_collision_spheres: true
    flange_frame: AR5V2_L_arm_flan_link
    default_tcp_frame: AR5V2_L_pinch_tcp

joint_groups:
  arm: [AR5V2_L_arm_joint_1]
  hand: [L6V1_L_hand_index_mcp_pitch]
```

规则：

- `kind` 只能是 `arm|hand|arm_hand`。
- `hand` 必须设置 `curobo.enabled: false`。
- `arm`/`arm_hand` 启用 cuRobo planning 时，planning joint 必须完整映射到 arm group。
- MJCF 只用于 Isaac；cuRobo v0.8.0 使用 planning URDF/robot YAML。
- `joint_groups.arm` 与 `joint_groups.hand` 不得重叠。

Importer 字段按资产格式严格校验，不适用字段会直接报错：

| 字段 | MJCF robot | URDF robot/object |
| --- | --- | --- |
| `collision_approximation`、`fix_base`、`import_inertia_tensor` | 是 | 是 |
| `self_collision` | 仅 robot | 仅 robot |
| `merge_fixed_joints` | 是（默认 `false`） | 是（默认 `true`） |
| `import_sites` | 是 | 否 |
| `collision_from_visuals` | 否 | 是 |

已有 USD object 不经过 importer，因此不能声明这些 import 字段。URDF mimic 显式交给 PhysX 原生
MimicJoint 约束；MJCF equality follower 由 runtime 根据 master 实际状态驱动。

## Object 实例

对象资产和物理属性放在 `configs/objects/`，USD 实例路径和位姿放在 env：

```yaml
objects:
  - name: fixture
    object_profile: workstation_armbase
    prim_path: /World/Fixtures/fixture  # 可省略
    root_pose:
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
```

省略 `prim_path` 时使用 `/World/Objects/<name>`。name、runtime handle 和最终 prim path 在场景内
必须唯一；同一 object profile 可以实例化多次。

Object profile 使用以下项目 schema 边界：

```yaml
object:
  name: fixture
  kind: rigid                 # rigid | dynamic_chain
  source: urdf                # usd | urdf
  asset_path: assets/fixture.urdf
  physics:
    static: true
```

校验器会立即进入 kind-specific importer、physics、material、state summary 和 planning collision
解析器，因此深层拼写错误会在 `validate-config` 阶段报告完整路径。资产来源属于
`object.source`，实例放置路径属于 env-owned `env.objects[].prim_path`。

## Planner 选择

普通 Single Scene 的 planner 默认来自 `runtime.planner.backend`，CLI 可对本次启动显式覆盖：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --env scene2 --planner-backend curobo --curobo-profile default
```

`--planner-backend` 只接受 `curobo|linear`。`linear` 仅执行
`plan_cspace_goal/plan_cspace_delta` 的关节空间插值，不创建 cuRobo solver，不提供 IK、避碰、
关节限位校验或受约束优化。Task-space 请求必须选择 `curobo`。

Tiled Scene 异步 planner 不提供 CLI backend 覆盖，由 runtime profile 统一选择 backend、cuRobo profile 和
joint batch 模式：

```yaml
runtime:
  profiles:
    curobo: default
  planner:
    backend: curobo        # curobo | linear
    joint_batch_mode: auto # auto | per_env | batch_only
```

`joint_batch_mode` 只影响 cuRobo joint-space 调度；`linear` 后端不调用
`BatchMotionPlanner`。同步 tiled `ee_*` step-control 另行使用机器人 cuRobo IK binding，不受异步
planner backend 选择替代。

## 校验

```bash
.venv/bin/python -m pytest -q tests/test_system_configs.py \
  tests/test_robot_instances.py tests/test_robot_capabilities.py
```

涉及真实 Isaac importer、USD/PhysX 或 cuRobo GPU 时，再用
`.venv/bin/python` 运行相同测试或对应专项测试。
