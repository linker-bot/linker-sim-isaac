# Mirror CLI 参考

语言：[中文](mirror-cli.md) | [English](../../en/reference/mirror-cli.md)

入口：

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py [options]
```

## 参数

| 参数 | 默认值 | 语义 |
| --- | --- | --- |
| `--version` | - | 不启动 Isaac，输出 workspace 兼容版本后退出 |
| `--profile NAME` | `physx_cpu` | 加载 `configs/modes/mirror/NAME.yaml`；只接受 `physx_cpu`、`physx_cpu_hybrid`、`newton_cpu` 或 `newton_cuda` |
| `--stdin` / `--no-stdin` | enabled | 启用或禁用 stdin JSONL ingress |
| `--tcp-jsonl HOST:PORT` | disabled | 开启 loopback TCP JSONL server |
| `--websocket HOST:PORT` | disabled | 开启 loopback WebSocket text server |
| `--response-timeout-s SECONDS` | `30.0` | ingress 等待 owner-thread response 的上限 |
| `--poll-timeout-s SECONDS` | `0.05` | owner loop 等待下一条 admission 的间隔 |

Timeout 必须大于零；port 必须在 1–65535。TCP/WebSocket host 只接受数值 loopback 或
`localhost`。未知 option 由 argparse 拒绝，旧 option spelling 不提供 alias。

## Profile

- `physx_cpu`：scene selector `mirror/scene3` + PhysX CPU；
- `physx_cpu_hybrid`：240 Hz `mirror/scene3_hybrid` + PhysX CPU + 显式 hybrid controller；
- `newton_cpu`：相同 Mirror 业务能力，Newton CPU 物理，根 CUDA 供 cuRobo/RTX，一个 world；
- `newton_cuda`：相同业务能力，Newton CUDA 物理，一个 world。

Mode profile 引用 scene/physics/control/curobo/planning/outputs 六个必选 leaf；hybrid composition 另加
可选 `hybrid_control`。PhysX/Newton 都引用统一的
`control: mirror`，controller bundle 由 physics 派生。CLI 不覆盖 leaf
内部字段，防止命令行产生第二份配置事实。`curobo` 拥有数值能力，`planning` 只拥有后端中立的请求
默认策略。

## 生命周期标记

- `MIRROR_INTERACTIVE_READY`：runtime 和所选 ingress 已可用；
- `MIRROR_INTERACTIVE_EXIT`：正常关闭完成；
- `MIRROR_INTERACTIVE_FAILED <type>: <message>`：启动或运行失败，进程快速退出。

stdin EOF 会调用 `request_quit`，不会从 reader 线程直接关闭 Isaac。关闭报告如果仍有 live
resource，CLI 以失败退出，不会伪装成成功。

消息格式见 [Mirror v1/v2/v3 JSON](mirror-json.md)。
