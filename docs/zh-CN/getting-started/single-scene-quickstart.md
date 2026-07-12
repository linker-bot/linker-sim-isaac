# Single Scene 快速开始

语言：[中文](single-scene-quickstart.md) | [English](../../en/getting-started/single-scene-quickstart.md)

本流程启动一个 Single Scene runtime，发现当前 session 的机器人，提交最小 hold 命令，轮询其终态，
最后关闭进程。Single Scene 表示一个物理 World，可以包含所选 env profile 声明的任意数量机器人；
每次启动后都必须重新发现 `robot_id`。

## 准备 Checkout

在 Linux x86-64、Python 3.11、Isaac Sim 5.1 和兼容的 NVIDIA 环境中执行：

```bash
git clone https://gitea.linkerhub.work/LinkerOS/scene-replay-sim-Isaac.git
cd scene-replay-sim-Isaac
uv sync --all-extras
```

所有项目命令都从 checkout 根目录运行。本项目是 workspace 应用，本地 `configs/`、
`assets/` 和 `scripts/` 都是 runtime 的组成部分。

## 启动 Isaac 前校验

下面的命令不会导入或创建 Isaac，用它校验内置 Single Scene 的完整配置依赖图：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_single_scene
```

成功时输出包含 `"event": "config_validated"`、所选 profile 和配置 fingerprint 的
JSON。遇到 `CONFIG_INVALID` 时应先修复配置，再启动 runtime。

## 启动 Single Scene Service

阅读并接受适用的 NVIDIA/Kit EULA，然后在同一部署环境记录该选择；项目不会替用户设置它。

在终端 1 执行：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene \
  --tcp-jsonl-port 8765
```

等待 `SINGLE_SCENE_INTERACTIVE_READY`。它表示 TCP 和共享命令队列已经可以接收请求。
该 headless 流程不需要 GUI；只有需要可视检查时才加入 `--gui`。

## 发现、Hold、终态检查与退出

在终端 2 的 checkout 根目录直接运行下面这个仅使用标准库的 TCP JSONL 客户端：

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import socket
import time


def request(stream, payload):
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    stream.flush()
    line = stream.readline()
    if not line:
        raise ConnectionError("Single Scene service closed before responding")
    return json.loads(line)


with socket.create_connection(("127.0.0.1", 8765), timeout=5.0) as sock:
    sock.settimeout(30.0)
    stream = sock.makefile("rwb")
    try:
        discovery = request(stream, {"type": "status"})
        assert discovery["event"] == "status", discovery
        assert discovery["robots"], discovery

        robot = discovery["robots"][0]
        robot_id = robot["robot_id"]
        groups = robot["joint_groups"]
        group = "arm" if groups.get("arm") else "hand"
        assert groups.get(group), robot
        print("discovered", robot_id, robot["label"], group)

        accepted = request(
            stream,
            {
                "type": "hold",
                "id": "quickstart-hold",
                "robot_id": robot_id,
                "group": group,
                "duration_s": 0.2,
            },
        )
        assert accepted["event"] == "accepted", accepted

        deadline = time.monotonic() + 30.0
        while True:
            status = request(
                stream,
                {"type": "status", "id": "quickstart-hold"},
            )
            assert status["commands"], status
            command = status["commands"][0]
            if command["state"] in {"done", "failed", "cancelled"}:
                assert command["state"] == "done", command
                print("terminal", command)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("quickstart-hold did not reach a terminal state")
            time.sleep(0.05)
    finally:
        print("quit", request(stream, {"type": "quit"}))
PY
```

第一次 `status` 响应就是 session discovery 契约。进程重启、env 中机器人顺序变化或增删
机器人后，不能继续使用缓存的 `robot_id`。hold 提交是异步的，因此客户端按命令 ID 轮询到
`done`、`failed` 或 `cancelled`；TCP 不会在直接响应之间插入生命周期事件。

## 检查关闭结果

客户端收到 `{"event":"quit"}` 后，终端 1 应依次输出：

```text
SINGLE_SCENE_INTERACTIVE_EXIT
SINGLE_SCENE_INTERACTIVE_OK steps=<n>
```

随后进程应以状态 0 退出。出现 `SINGLE_SCENE_INTERACTIVE_FAILED`、非零退出状态或任意
shutdown timeout 诊断，都表示本次运行失败。

全部启动参数见 [Single Scene CLI 参考](../reference/single-scene-cli.md)；timeline、planning、reset 和
snapshot 协议见 [Single Scene JSON 参考](../reference/single-scene-json.md)。
