# 故障排查

语言：[中文](troubleshooting.md) | [English](../../en/operations/troubleshooting.md)

## 配置在启动前失败

先运行：

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py --mode kaleidoscope --profile newton_cuda
```

检查 unknown/duplicate key、profile path、未消费 compute、重复 device、backend 与产品不匹配，以及
Kaleidoscope closure 中误加的 camera/planning/telemetry 字段。

## Isaac 未启动

- 确认 `OMNI_KIT_ACCEPT_EULA=Y`；
- 确认从 checkout root 运行并使用 simulation `.venv`；
- 不要在该环境安装 PyPI `usd-core`；
- 检查 Kit：Kaleidoscope `physx_cuda` 应选择
  `linkerbot_sim.kaleidoscope.physx_cuda.python.kit`，`newton_cuda` 应选择
  `linkerbot_sim.kaleidoscope.newton.python.kit`；显式 viewer 分别选择对应的
  `physx_cuda_viewport` 或 `newton_cuda_viewport` Kit，不要手工拼接闭包。

## Mirror 仓库背景缺失

默认 `mirror/scene3` 的包装资产会引用下面的外部 NVIDIA 文件：

```text
usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd
```

该素材受 NVIDIA 许可约束，不随仓库分发。配置图校验不会下载或补全它；如果启动日志提示
未解析 payload，或场景中仓库视觉为空，请确认文件放在上述精确路径。Kaleidoscope 不依赖该素材。

## Kaleidoscope backend 启动失败

- PhysX CUDA：检查 Fabric/GPU pipeline 诊断和进程显存门禁；
- Newton：检查 session spec 是否从最终 `environments.num_envs` 派生 `world_count`，以及
  Newton/MuJoCo-Warp Python multi-world runtime 与 CUDA/Warp device；
- Newton 机器人在小位置目标下受重力漂移：确认 robot profile 的 `gravity=false` 已在 model finalize
  前投影为 `mjc:gravcomp=1`；不要尝试在 reset 后调用不受支持的逐 link 重力 setter；
- Newton Kit 不应加载 `isaacsim.physics.newton`、`isaacsim.physics.newton.tensors` 或 PhysX extension；
- 用同一 smoke 入口分别传 `--profile physx_cuda` 与 `--profile newton_cuda`，不要用一个后端
  的错误去推断另一个后端也失败。

## Kaleidoscope device/shape 错误

- `torch.cuda.is_available()` 必须为真；
- action 是 `(N,A)` float32 CUDA；selector 是 `(K,)` int64 CUDA；
- 所有 tensor 的 device 与 `env.device` 相同；
- source/target clone selector 等长、不重叠；
- native done 行在下一步前 reset。

出现 `poisoned`/fail-stop 表示 engine writer 失败后状态一致性无法证明，应关闭并重建 env，不要捕获
异常后继续 step。

## Kaleidoscope viewport 无画面

- 必须用 `make_viewport_env()`、Gymnasium `render_mode="human"` 或
  `scripts/kaleidoscope_viewer.py` 构造；headless `make_torch_env()` 的 `render()` 会明确失败；
- `selected_env` 必须小于最终 `num_envs`，且只有该环境会进入 renderer-facing USD；
- 训练 `step()` 不自动渲染，调用方必须按 `render_every_n_steps` 显式调用 `env.render()`；
- PhysX 检查 Fabric transformation 输出，Newton 检查 selected-world render binding；
- 不要尝试用 camera、SyntheticData 或 Replicator 取图，这些扩展仍被 viewport Kit 排除。

## Mirror request 失败

- Envelope 必须含精确的 `protocol/request_id/operation/arguments`；
- request ID 不能重复；JSON 不允许 NaN/Infinity/重复字段；
- estop 后先成功 reset；
- queue capacity/error code 见 [Mirror JSON](../reference/mirror-json.md)。

## Camera/telemetry 无输出

确认使用 Mirror，scene 声明 camera，outputs profile 启用对应 sink，目标路径 preflight 成功且 queue
未按 policy drop。Kaleidoscope 没有这些 service。

## 关闭超时

读取 close report 的 `live_resources`/`errors`。不要提前关闭仍被 worker 使用的 sink/session；先停止
ingress，再重试产品资源 close，最后释放 SimulationApp。

## 架构或文档门禁失败

源码移动后运行：

```bash
just update-architecture
just test-architecture
just check-docs
```

最终门禁拒绝旧产品名、physics shim import、额外 Kit、module-map/count/hash drift 和未冻结 facade。
