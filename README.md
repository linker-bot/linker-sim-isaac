# LinkerHand Simulation

这是一个基于 Isaac Sim / Isaac Lab 的机械臂、灵巧手和绳体操作仿真工程。项目当前围绕 AR5 机械臂、LinkerHand L6 灵巧手和 capsule/box 近似绳体搭建，用于验证运动算法、IK、TCP 定义、mimic 关节同步、PhysX 参数和基础抓取流程。

`tem-file/` 只保留历史脚本和临时输出，不属于当前工程结构，已加入 `.gitignore`。

## 当前能力

- 资产导入：支持 AR5、L6、AR5+L6 组合 MJCF/URDF/XRDF 资产。
- IK 后端：优先使用 cuMotion，保留 Isaac Sim Lula 兼容后端。
- 控制器：核心为 Isaac/PhysX implicit position drive，机械臂和灵巧手参数分文件配置。
- Mimic 关节：软件层解析 MJCF `equality/joint` 的 `polycoef` 多项式并同步 follower drive 目标。
- TCP：支持 AR5 法兰 TCP、自定义固定 TCP、thumb/index 闭合夹捏中心 TCP。
- 任务脚本：提供关节目标 smoke 和 AR5+L6 绳端夹捏抓取 demo。
- 日志和可视化：支持关节跟踪 CSV，Foxglove MCAP/WebSocket 可选用于调试回放。

## 目录结构

```text
.
├── assets/
│   ├── mesh/                 # 各设备 mesh，按 arm/hand 和单体系统名分目录
│   ├── single_system/        # 单体 URDF/MJCF/XRDF/USD 资产
│   ├── combined_system/      # 复合系统资产
│   ├── static_env_objects/   # 静态环境对象资产
│   └── dynamic_env_objects/  # 动态环境对象资产，例如 capsule rope
├── configs/
│   ├── controllers/          # 控制器、材料、刚体参数
│   ├── envs/                 # 空场景、桌面、绳体场景
│   ├── logging/              # CSV 日志配置
│   ├── objects/              # 可生成对象的参数，例如 capsule rope
│   ├── robots/               # 机器人资产、关节组、IK 资源
│   └── trajectories/         # 关节、笛卡尔、pinch grasp 任务配置
├── scripts/                  # Isaac Sim 实际运行入口
├── source/manipulation_project/
│   ├── app/                  # SimulationApp 启动和 CLI 工具
│   ├── assets/               # 资产导入、USD/PhysX 覆盖、solver 设置
│   ├── controllers/          # implicit drive 和控制器配置解析
│   ├── envs/                 # World 和场景构建
│   ├── ik/                   # cuMotion/Lula IK、临时 TCP URDF
│   ├── logging/              # CSV 和关节跟踪日志
│   ├── objects/              # 对象资产生成和引用
│   ├── robots/               # 关节组、mimic/equality、状态容器
│   ├── tasks/                # 任务流程
│   ├── tcp/                  # TCP frame 和夹捏中心计算
│   ├── trajectories/         # 插值和轨迹采样
│   ├── utils/                # 配置、路径、旋转、数学、计时工具
│   └── visualization/        # 相机、marker、Foxglove 日志封装
├── tests/                    # 不启动 Isaac Sim 的轻量测试
├── ASSET_NAMING_CONVENTIONS.md
├── pyproject.toml
└── README.md
```

## 环境约定

- 使用 Isaac Sim / Isaac Lab 对应的 Python 环境运行脚本，当前示例默认环境目录为 `env_isaaclab/`。
- Python 包源码位于 `source/`，从仓库根目录运行脚本时建议设置 `PYTHONPATH=source`。
- 基础 Python 依赖见 `pyproject.toml`：`numpy`、`pyyaml`、`scipy`。
- Foxglove 可视化是可选能力，需要额外安装 `foxglove-sdk`。

## 快速运行

生成或更新 capsule rope USD 资产：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/build_capsule_rope_asset.py
```

运行关节目标 smoke：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_joint_target.py
```

带 GUI 观察关节目标：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_joint_target.py --gui --hold
```

运行绳端夹捏抓取 demo：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --gui
```

快速 headless smoke：

```bash
PYTHONPATH=source env_isaaclab/bin/python scripts/run_pinch_grasp.py --no-grasp --short-smoke
```

脚本默认读取 `configs/` 中的配置，并将关节跟踪日志写到 `logs/joint_tracking/`。

## 关键配置

- `configs/robots/ar5_l6.yaml`：AR5+L6 组合资产、关节组、IK 资源和 TCP frame。
- `configs/robots/ar5_arm.yaml`：单独 AR5 机械臂资产和 IK 资源。
- `configs/controllers/implicit_position_drive.yaml`：默认控制器聚合入口。
- `configs/controllers/arm_controller.yaml`：机械臂 drive、速度、effort、材料和刚体参数。
- `configs/controllers/hand_controller.yaml`：灵巧手 drive、速度、effort、材料和刚体参数。
- `configs/envs/rope_scene.yaml`：绳体抓取场景、步频、重力和 PhysX solver 设置。
- `configs/objects/capsule_rope.yaml`：绳体资产生成参数和 USD 输出路径。
- `configs/trajectories/joint_target.yaml`：关节目标 smoke 的稀疏目标。
- `configs/trajectories/pinch_grasp.yaml`：夹捏抓取阶段、手型、IK 容差和摆动参数。
- `configs/logging/default_logger.yaml`：关节跟踪 CSV 日志配置。

配置读取工具在 `source/manipulation_project/utils/config.py`，相对路径默认按仓库根目录解析。

## 资产和命名

当前正式命名不使用连字符 `-`，避免 Isaac importer 把 `-` 自动转换为 `_` 后造成配置关节名和 runtime DOF 名不一致。更多规则见 `ASSET_NAMING_CONVENTIONS.md`。

当前默认运行资产：

- 左臂：`assets/single_system/arm/AR5V2_L/`
- 右臂：`assets/single_system/arm/AR5V2_R/`
- 左手：`assets/single_system/hand/L6V1_L/`
- 右手：`assets/single_system/hand/L6V1_R/`
- 左臂+左手组合：`assets/combined_system/AR5V2_L6V1_L/`
- 绳体对象：`assets/dynamic_env_objects/capsuleropeV1_default/`

右侧 AR5/L6 当前提供单体 URDF 和 mesh。若要运行右臂 IK 或右手 MJCF 组合资产，需要继续生成对应 XRDF/Lula/MJCF 描述。

## 坐标、姿态和单位

- 项目对外统一使用 wxyz 四元数，即 `[w, x, y, z]`。
- SciPy 内部使用 xyzw，转换封装在 `utils/math_utils.py` 和 `utils/rotations.py`。
- 配置中的 RPY 使用固定轴 XYZ 顺序，即外旋 XYZ；在 SciPy 中对应小写 `"xyz"`。
- 距离单位为 m，角度配置通常为 degree，关节位置为 rad，关节速度为 rad/s。

## Mimic 关节

LinkerHand L6 的从动关节通过 MJCF `equality/joint` 描述。`robots/mimic.py` 会解析 `polycoef` 多项式关系：

```text
dependent = a0 + a1 * master + a2 * master^2 + ...
```

当前实现不依赖 PhysX/MJCF importer 的硬约束实时约束 mimic 关节，而是在软件层同步 follower 的 position drive 目标；速度目标通过多项式导数计算。线性 mimic 只是该多项式的一阶特例。

## TCP 和 IK

TCP 实现在 `source/manipulation_project/tcp/`：

- `flange_tcp.py`：直接使用 AR5 法兰 link。
- `custom_tcp.py`：在任意父 frame 下添加固定 `xyz/rpy` 偏移。
- `pinch_tcp.py`：读取 MJCF body 链，在闭合手型下计算 thumb tip 和 index tip 中点。

IK 统一入口是 `manipulation_project.ik.solver_factory.make_ik_solver()`：

- `backend="auto"`：能 import `cumotion` 时优先使用 cuMotion，否则回退到 Lula。
- `backend="cumotion"`：使用 `assets/single_system/arm/AR5V2_L/AR5V2_L.xrdf` 和 URDF。
- `backend="lula"`：使用 Isaac Sim 自带 Lula motion generation 扩展。

IK 后端通常要求目标 frame 已经在 URDF 中存在。`ik/tcp_urdf_builder.py` 会复制基础 URDF，并临时追加 fixed TCP link/joint。

## Foxglove 可视化

Foxglove 用于调试和回放仿真数据，不替代 Isaac Sim 的真实场景渲染。封装位于 `visualization/foxglove_logger.py`，支持：

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
```

## 验证

语法检查：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m compileall -q source scripts tests
```

检查 YAML 配置：

```bash
PYTHONPATH=source env_isaaclab/bin/python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path("configs").rglob("*.yaml")):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("yaml ok")
PY
```

如果环境安装了 `pytest`：

```bash
PYTHONPATH=source env_isaaclab/bin/python -m pytest -q
```

当前轻量测试覆盖插值、轨迹采样、MJCF mimic 解析、配置加载、pinch TCP 计算、临时 TCP URDF 写入和 Foxglove logger 基本行为。

## 后续方向

- 补齐右臂/右手组合运行资产和对应 IK 描述。
- 补齐 `MoveTcpToPoseTask` 和 `MoveTcpLineTask` 的完整 Isaac 执行流程。
- 增加 cuMotion 安装版本记录和自动 smoke check。
- 将 visualization 中的 marker/debug draw 接入 TCP 目标、IK 误差和轨迹可视化。
