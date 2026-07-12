# Tiled Scene 快速开始

语言：[中文](tiled-scene-quickstart.md) | [English](../../en/getting-started/tiled-scene-quickstart.md)

本流程启动独立 Tiled Scene runtime，发现 env 与机器人 ID，在 env 0 同步执行一次 hold step，检查
最终 global step，然后关闭进程。Tiled Scene 使用克隆 env 行，不通过 `SingleSceneRuntime` 运行。

## 准备 Checkout

在 Linux x86-64、Python 3.11、Isaac Sim 5.1 和兼容的 NVIDIA 环境中执行：

```bash
git clone https://gitea.linkerhub.work/LinkerOS/scene-replay-sim-Isaac.git
cd scene-replay-sim-Isaac
uv sync --all-extras
```

所有项目命令都从 checkout 根目录运行。Workspace 依赖本地 profile、asset、script 和
cuRobo task 资源。

## 启动 Isaac 前校验

下面的命令不会创建 Isaac，用它校验内置 Tiled Scene 的完整配置依赖图：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene
```

成功时输出包含 `"event": "config_validated"` 和配置 fingerprint 的 JSON。遇到
`CONFIG_INVALID` 时应先修复配置，再启动 runtime。

## 启动 Tiled Scene Service

阅读并接受适用的 NVIDIA/Kit EULA，然后在部署环境记录该选择；项目不会自动代替用户接受。

在终端 1 执行：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene \
  --no-stdin \
  --tcp-jsonl-port 8765
```

等待 `TILED_SCENE_INTERACTIVE_READY`。`num_envs` 由所选 `scene3_tiled` env profile
决定，不由 CLI count 决定。内置 runtime 为 headless，空闲时暂停 physics；只有需要连续可视
刷新时才加入 `--gui --idle-physics-policy hold_step`。

## 发现、Step、结果检查与退出

在终端 2 的 checkout 根目录直接运行下面这个仅使用标准库的 TCP JSONL 客户端：

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import socket


def request(stream, payload):
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    stream.flush()
    line = stream.readline()
    if not line:
        raise ConnectionError("Tiled Scene service closed before responding")
    return json.loads(line)


with socket.create_connection(("127.0.0.1", 8765), timeout=5.0) as sock:
    sock.settimeout(30.0)
    stream = sock.makefile("rwb")
    try:
        discovery = request(stream, {"type": "status"})
        assert discovery["event"] == "status", discovery
        assert discovery["num_envs"] > 0, discovery
        assert discovery["robots"], discovery

        robot = discovery["robots"][0]
        robot_id = robot["robot_id"]
        print("discovered", discovery["num_envs"], robot_id, robot["label"])

        step = request(
            stream,
            {
                "type": "step",
                "kind": "hold",
                "env_ids": [0],
                "robot_id": robot_id,
            },
        )
        assert step["event"] == "step" and step["accepted"] is True, step
        assert step["ticks"] > 0, step

        after = request(stream, {"type": "status"})
        assert after["step"] >= step["step"], (step, after)
        print("terminal", step)
    finally:
        print("quit", request(stream, {"type": "quit"}))
PY
```

`status` 是 session discovery：`env_id` 选择 clone 行，`robot_id` 选择在各行重复的
机器人定义。进程重启或 profile 变化后，应重新发现两个维度。与 Single Scene motion command 不同，
`step` 是同步操作；所有请求的 physics tick 完成后，直接响应就是终态结果。

## 检查关闭结果

客户端收到 `{"event":"quit","accepted":true}` 后，终端 1 应输出：

```text
TILED_SCENE_INTERACTIVE_EXIT
```

随后进程应以状态 0 退出。Tiled Scene 没有额外成功标记。出现 `TILED_SCENE_INTERACTIVE_FAILED`、非零退出
状态或任意 shutdown timeout 诊断，都表示本次运行失败。

全部启动参数见 [Tiled Scene CLI 参考](../reference/tiled-scene-cli.md)；selector、state、trajectory 和
异步 planning 协议见 [Tiled Scene JSON 参考](../reference/tiled-scene-json.md)。
