# LinkerHand Simulation

这是一个用于在 Isaac Sim / Isaac Lab 环境中验证机械臂、灵巧手和绳体操作算法的仿真工程。当前项目围绕 AR5 机械臂、LinkerHand L6 灵巧手和 capsule/box 近似绳体搭建，重点覆盖资产导入、PhysX 参数覆盖、TCP 定义、IK 求解、轨迹生成、关节驱动、mimic 关节展开、日志记录和基础测试。

`tem-file/` 是历史脚本和临时日志目录，不属于当前工程结构，已加入 `.gitignore`。

## 当前状态

- 机器人资产：左侧 AR5、L6、AR5+L6 MJCF、AR5 URDF、cuMotion XRDF、Lula 描述文件；右侧 AR5、L6 已按同一规范加入 URDF 和 mesh。
- IK 后端：优先支持 cuMotion，保留 Lula 作为兼容后端。
- TCP：支持法兰 TCP、自定义固定偏移 TCP、拇指/食指闭合夹捏中心 TCP。
- 轨迹：支持关节目标轨迹、笛卡尔点到点采样、笛卡尔直线采样。
- 控制：当前核心为 Isaac/PhysX implicit position drive。
- 任务：已有关节目标执行辅助和 AR5+L6 对绳端 box 的脚本化 pinch grasp 流程。
- 配置：机器人、环境、控制器、轨迹和日志均通过 `configs/` 下 YAML 管理。
- 可视化：Isaac viewport 负责仿真画面，Foxglove 可选用于 MCAP/WebSocket 数据可视化。
- 注释：`source/` 和 `configs/` 已补充中文说明，类/函数包含参数和返回值说明。
- 命名：资产目录和文件命名遵循 `ASSET_NAMING_CONVENTIONS.md`。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # 各设备统一 mesh，按 arm/hand 和单体系统名分目录
│   ├── single_system/        # 单体 URDF/MJCF/XRDF/USD 资产，按 arm/hand 分目录
│   ├── combined_system/      # 复合体 URDF/MJCF/XRDF/USD 资产
│   ├── static_env_objects/   # 静态环境对象资产
│   └── dynamic_env_objects/  # 动态环境对象资产，例如 capsule rope
├── configs/
│   ├── controllers/          # 控制器参数
│   ├── envs/                 # 空场景、桌面、绳体场景
│   ├── logging/              # CSV 日志配置
│   ├── objects/              # capsule rope 等对象资产生成参数
│   ├── robots/               # 机器人资产、关节组、IK 资源
│   └── trajectories/         # 关节、笛卡尔、pinch grasp 任务配置
├── source/manipulation_project/
│   ├── app/                  # Isaac SimulationApp 启动和 CLI 小工具
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 迭代设置
│   ├── controllers/          # implicit drive 和手部目标展开
│   ├── envs/                 # World 和场景构建
│   ├── ik/                   # cuMotion/Lula IK、请求结果类型、临时 TCP URDF
│   ├── logging/              # CSV 和关节跟踪日志
│   ├── objects/              # 可复用对象资产生成和引用
│   ├── robots/               # 关节组、MJCF mimic/equality、状态容器
│   ├── tasks/                # 任务流程，包括 pinch grasp
│   ├── tcp/                  # TCP frame、法兰/自定义/夹捏中心 TCP
│   ├── trajectories/         # 插值和轨迹采样
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # 相机、debug draw/marker、Foxglove 日志封装
├── scripts/                  # Isaac Sim 实际运动 demo/smoke 入口
├── tests/                    # 轻量单元测试
├── pyproject.toml
└── README.md
```

## 关键约定

### 资产命名和左右侧

- 单体设备目录使用 `<device-family><device-version>_<side>`，例如 `AR5V2_L`、`AR5V2_R`、`L6V1_L`、`L6V1_R`；正式资产名和关节名不使用连字符，避免 Isaac importer 把 `-` 自动改成 `_` 后与配置不一致。
- mesh 统一放在 `assets/mesh/<arm-or-hand>/<single-system-name>/`，URDF/MJCF/XRDF/USD 描述文件放在 `assets/single_system/<arm-or-hand>/<single-system-name>/`。
- 当前运行配置默认仍使用左侧 `AR5V2_L`、`L6V1_L` 和组合 `AR5V2_L6V1_L`；右侧 `AR5V2_R`、`L6V1_R` 目前提供单体 URDF 与 mesh，后续如需右臂 IK 或右手 MJCF 运行资产，需要再生成对应 XRDF/Lula/MJCF 描述。

### 坐标和姿态

- 项目对外统一使用 wxyz 四元数，即 `[w, x, y, z]`。
- SciPy 内部使用 xyzw，转换封装在 `utils/math_utils.py` 和 `utils/rotations.py`。
- 配置中的 RPY 使用固定轴 XYZ 顺序（外旋 XYZ 顺序），外部输入单位通常为 degree；在 SciPy 中对应小写 `"xyz"`。
- 距离单位为 m，关节位置单位通常为 rad，关节速度单位通常为 rad/s。

### Mimic 关节

LinkerHand L6 的从动关节通过 MJCF `equality/joint` 描述。`robots/mimic.py` 会解析 `polycoef` 多项式关系：

```text
dependent = a0 + a1 * master + a2 * master^2 + ...
```

控制器和夹捏 TCP 计算都会使用这套关系补齐 follower 关节目标，避免只控制 master 关节时几何中心或 drive target 不一致。

当前实现没有依赖 PhysX/MJCF importer 的硬约束来实时约束 mimic 关节，而是在软件层按上述多项式关系同步 follower 的 position drive 目标；速度目标会通过多项式导数计算。线性 mimic 只是 `a0 + a1 * master` 这一特例。

### TCP

当前支持三类 TCP：

- `flange_tcp.py`：直接使用 AR5 法兰 link。
- `custom_tcp.py`：在任意父 frame 下添加固定 `xyz/rpy` 偏移。
- `pinch_tcp.py`：读取 MJCF body 链，在闭合手型下计算 thumb tip 和 index tip 中点。

IK 后端通常要求目标 frame 已经在 URDF 中存在。`ik/tcp_urdf_builder.py` 会复制基础 URDF，并临时追加一个 fixed TCP link/joint，供 cuMotion 或 Lula 使用。

## IK 后端

统一入口是 `manipulation_project.ik.solver_factory.make_ik_solver()`。

- `backend="auto"`：能 import `cumotion` 时优先使用 cuMotion，否则回退到 Lula。
- `backend="cumotion"`：使用 `assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf` + URDF。
- `backend="lula"`：使用 Isaac Sim 自带 Lula motion generation 扩展。

项目内部使用统一数据类型：

- `IKRequest`：目标 TCP 位置、可选 wxyz 姿态、warm start 和容差。
- `IKResult`：关节解、成功标志、位置误差、可选姿态误差和诊断信息。

## 配置入口

常用配置文件：

- `configs/robots/ar5_l6.yaml`：AR5+L6 资产、关节组、IK 资源和 PhysX 覆盖。
- `configs/envs/rope_scene.yaml`：绳体抓取场景的世界、步频和机器人 solver 参数。
- `configs/objects/capsule_rope.yaml`：绳体几何、质量、D6 joint、材质和 USD 输出路径；生成到 `assets/dynamic_env_objects/capsuleropeV1_default/capsuleropeV1_default.usda`。
- `configs/controllers/implicit_position_drive.yaml`：默认聚合入口，分别引用 `arm_controller.yaml` 和 `hand_controller.yaml`。
- `configs/controllers/arm_controller.yaml`、`configs/controllers/hand_controller.yaml`：机械臂/灵巧手各自的 implicit position drive、速度控制、effort 控制、材料和刚体参数。
- `configs/trajectories/pinch_grasp.yaml`：夹捏抓取阶段、IK 容差、手型和摆动参数。
- `configs/logging/default_logger.yaml`：关节跟踪 CSV 日志设置。

配置读取工具在 `utils/config.py`，相对路径默认按仓库根目录解析。

## 实际运动脚本

`tests/` 里的测试默认不启动 Isaac Sim；真正跑仿真运动请使用 `scripts/` 下的入口。

关节目标 smoke：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_joint_target.py
```

带 GUI 观察：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_joint_target.py --gui --hold
```

绳端夹捏抓取 demo：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --gui
```

修改绳体对象参数后，先重新生成 USD 资产：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

快速 headless smoke：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --short-smoke
```

两个脚本都会读取 `configs/` 中的默认配置，并把关节跟踪日志写到 `logs/joint_tracking/`。

## Foxglove 可视化

Foxglove 用于调试和回放仿真数据，不替代 Isaac Sim 的真实场景渲染。当前封装在 `visualization/foxglove_logger.py`，支持：

- 离线写 MCAP；
- 本地 WebSocket live server；
- `JointStates` 曲线；
- TCP、轨迹点和其它调试点的 `SceneUpdate` marker。

安装可选依赖：

```bash
env_isaaclab/bin/python -m pip install foxglove-sdk
```

最小示例：

```python
from manipulation_project.visualization.foxglove_logger import FoxgloveLogger

with FoxgloveLogger.open_mcap("logs/debug.mcap") as logger:
    logger.log_joint_state(
        joint_names=["joint_1"],
        positions=[0.1],
        velocities=[0.0],
        time_s=0.0,
    )
    logger.log_scene_line_strip(
        entity_id="tcp_plan",
        points=[[0.2, -0.5, 0.3], [0.25, -0.55, 0.45]],
        time_s=0.0,
    )
```

## 开发和验证

使用当前 Isaac/Isaac Lab Python 环境时，建议从仓库根目录运行：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m compileall -q source
```

验证 YAML 配置可解析：

```bash
PYTHONPATH=source env_isaaclab/bin/python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("configs").rglob("*.yaml")):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("yaml ok")
PY
```

如果环境安装了 `pytest`，可以运行：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m pytest -q
```

当前测试覆盖插值、轨迹采样、MJCF mimic 解析、配置加载、pinch TCP 计算和临时 TCP URDF 写入。

## 后续扩展方向

- 将历史脚本中仍有价值的入口从 `tem-file/` 迁移为正式 `scripts/` 或 CLI。
- 补齐 `MoveTcpToPoseTask` 和 `MoveTcpLineTask` 的完整 Isaac 执行流程。
- 增加 cuMotion 实机环境 smoke test，并记录安装版本要求。
- 将 visualization 中的 marker/debug draw 占位模块接入 TCP 目标、IK 误差和轨迹可视化。
