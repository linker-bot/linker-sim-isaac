# linker-lab 快速入门

linker-lab 是 Isaac Lab 的扩展，提供一台 Linker 工作站 **LW-v0**（两台 AR5 机械臂、两只 O6 灵巧手、桌子和三台相机），在 NVIDIA GPU 上仿真，行为与真机一致：机器人模型文件、驱动模型、真实控制器的运动限位和物理参数都已内置。

它有三种用法：用 **playground** 手动操作机器人；用 **gallery** 运行预训练策略；用 **任务模板** 编写自己的任务，并用 Isaac Lab 的训练器训练。

本页是中文快速入门，内容与英文文档对应：安装、检查、playground、gallery、任务模板。完整说明见英文的 [getting started](../getting-started.md)、[LW-v0 环境](../environment.md) 和 [API 参考](../api.md)。英文文档是准确的版本；两者不一致时以英文为准。

## 1. 安装

需要：Linux（x86_64 或 aarch64，如 Jetson Thor），glibc 2.34 或更新（Ubuntu 22.04 或更新）；`git`；NVIDIA GPU，驱动 580 或更新；约 10 GB 可用磁盘空间；网络连接。详见 [requirements](../requirements.md)。

```bash
git clone https://github.com/linker-bot/linker-lab
cd linker-lab
./install.sh
```

`install.sh` 先检查上述要求，在需要时把 uv（Python 包管理器）安装到 `~/.local/bin`，再把 Python 3.12、Isaac Lab 3.0 和物理引擎按 `uv.lock` 中锁定的版本安装到本目录的 `.venv` 中（首次需下载约 4 GB），最后运行 `linker-lab doctor`。

如果安装程序安装了 uv，当前终端还找不到它（`uv: command not found`）：打开一个新终端，或运行 `source ~/.local/bin/env`。

LW-v0 的机器人模型文件随安装一起装好：来自 [linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) 的软件包 `linker-lw-v0`，版本标签 `v1.0.0`。

## 2. 检查

```bash
uv run linker-lab doctor
```

它检查已安装的版本、GPU、引擎补丁，并仿真 LW-v0 一秒钟。最后一行是 `All checks passed.`。第一次运行还会编译 GPU 内核（约一分钟，之后有缓存）。

所有命令都通过 `uv run` 在本目录的环境中运行；也可以在每个终端中先执行一次 `source .venv/bin/activate`，之后省去 `uv run`。

## 3. Playground：手动操作机器人

```bash
uv run linker-lab play
```

窗口中出现工作站，桌上有一个方块。你移动一只手臂的 **掌心点**（掌面中部前方 3.5 cm）的目标，机械臂像真机在 xCore SDK 跟踪模式下一样，在真实控制器的限位内跟随。

| 输入 | 作用 |
| --- | --- |
| I / K、J / L、U / O | 把目标向前 / 后、左 / 右、上 / 下移动（相对于视角） |
| Shift + 上述按键 | 改为绕掌心点旋转目标 |
| 鼠标 | 移动视角；右键拖动物体；拖动手柄移动或旋转目标 |
| Tab | 切换机械臂 |
| 1 / 2 / 3 / 4 | 手：张开 / 握紧 / 捏取 / 拇指对掌 |
| R | 复位 |

W/A/S/D、Q/E 和方向键移动视角，空格暂停，H 隐藏面板，Esc 退出。面板显示机械臂的工具（TCP）位姿（与控制器报告的一致）、掌心点离目标多远、接触和仿真速度。机械臂的速度受真实控制器限位约束，掌心点最快约 0.15 m/s。

`uv run linker-lab play --selftest` 运行一段脚本化的操作并报告机械臂和手是否跟随。

## 4. Gallery：预训练策略

```bash
uv run linker-lab gallery list               # 所有条目
uv run linker-lab gallery info ycb-grasp     # 说明、许可证和文件
uv run linker-lab gallery run ycb-grasp      # 打开
```

`run` 把条目的文件（约 12 MB）下载到 `~/.cache/linker-lab/gallery/`，校验后打开场景：一台工作站和一个面板。选择一个物体（31 个 YCB 物体，训练时都没有见过），按 **Drop**：物体落在桌面左半部分的随机位置，姿态和朝向随机。物体静止后按 **Grasp**：策略看一次物体，左手把它抓起并在 15 cm 高处保持。状态行显示是否抓住。**Reset** 清空桌面。策略权重仅供研究使用（非商业用途），许可证详见 `gallery info`。详见 [gallery](../gallery.md)。

## 5. 任务模板

[templates/task](../../templates/task/README.md) 是一个独立项目形式的任务：复制这个文件夹，改名，修改任务。例子任务是左手抓起方块：场景、动作、复位、观测、奖励和终止条件都已写好，抓取的控制器或策略由你来写（`run.py` 只让机器人保持初始姿态）。任务配置 `MyTaskEnvCfg` 是 linker-lab 默认环境 `LWv0EnvCfg` 的子类，按 Isaac Lab 的方式逐项修改场景、动作、事件、观测、奖励和终止条件。

```bash
cp -r templates/task ~/my-task && cd ~/my-task
uv sync
uv run python run.py --viewer
```

加入学习器配置后，可以直接用 Isaac Lab 自带的 `isaaclab train` / `isaaclab play` 训练和回放，见模板的 README。
