# C3-A：轻量观测与积分状态打包

依据 Framework #119 的测量与用户 2026-11-06 授权，A/B/C 分别提交 PR，由用户逐个合并。
本 PR 只处理 A，不调整相机或依赖版本，不修改已冻结 Task。

## 合同与实现

1. 增加 owner-thread `MirrorRuntime.get_observation()`，返回独立 CPU/JSON mapping，包含
   command joints 的名称、位置、速度和当前模式目标，以及场景物体 pose/velocity。
   使用现有状态读取语义，不计算指纹、不捕获 solver/controller history，不可用于 restore。
2. `get_state()` 和 snapshot/restore 的公开合同保持不变；不变配置指纹按 runtime 缓存。
3. Newton store 以模块级 kernel 打包固定 TIME/ACT/WARMSTART，继续使用 owner stream、
   固定 canonical buffer 和 active-world mask。保留 post-step capture：批量环境借用此 buffer，
   延迟更新会改变其可见性；本轮去掉 kernel 重建成本，不改变状态生命周期。
4. 不改安装的 MuJoCo Warp 或 v3 使用的 source checkout；复用已锁定依赖。

## 验证

- 单元测试：轻量字段与完整快照一致、结果独立、owner/close 限制、current mode targets、
  不触发恢复数据读取；配置指纹稳定；kernel 的 CPU/CUDA、activation、mask 与完整状态等价。
- `just quality`；已有快照/恢复、Newton、Mirror 相关回归。
- 真机：串行运行相同 AR5/O6 场景，比较 observation、完整状态、step+同步；观测读取目标≤1 ms。
- #119 的 600 决策公开 default 基准使用独立副本做前后对照；不改变冻结 Task 或私有集。
- 相关 Mirror / Kaleidoscope Newton smoke；保留 JSON、profile 和日志；记录未跑的大矩阵。
- 同步双语 API 文档与 changelog；提交 Issue/PR，由用户合并后再进行 B。

基线：Simulator 0f6eda5；预期收益须由实测确认，不能把主机端 kernel 构造误写为 CUDA 重编译。

## 验收结果（2026-11-06）

跟踪：[Simulator #75](https://github.com/linker-bot/linker-sim-isaac/issues/75)。
同一 RTX 5090 D v2、Newton CUDA、AR5/O6、顶部 RGB/depth 场景，使用 #119 的独立
600 决策 harness 副本串行对照，保留原有每周期三次 `get_state()` 和仅开局取图的行为。
未替换正式 Task 的任何调用，未改变依赖、物理配置、相机路径或已冻结 Simulator 版本。
这是一次前后配对测量，数值不构成所有场景的性能上界。

| 测量 | 0f6eda5 | 本次实现 |
| --- | ---: | ---: |
| `get_state()`，600 次均值 | 5.308 ms | 1.369 ms |
| `get_observation()`，600 次均值 / P95 | 无 | 0.780 / 0.925 ms |
| `step()` + Warp 同步，600 次均值 | 6.328 ms | 5.725 ms |
| `step()` + Warp 同步，中位数 | 5.678 ms | 5.702 ms |
| `render()`，100 次均值 | 40.074 ms | 41.349 ms |
| 600 决策循环墙钟（首请求至末响应） | 14.373 s | 7.045 s |

两侧 CUDA graph 都为 captured；新观测的关节/物体字段与同一时刻完整快照一致。
`get_state` 均值下降约 74%，600 决策循环下降约 51%。物理推进中位耗时基本不变：
cProfile 显示 post-step capture 的 Python 构造成本消失，但原来的 CPU 工作与 GPU 物理计算
有重叠，所以不能把主机端节省等同于 step 墙钟节省。相机优化仍属于 B。

验证通过：

- `just quality`：Ruff format/lint、Pyright 必检范围、配置/文档/架构检查；185 项架构测试，
  1730 项 pure 测试、65 项跳过；pure production 覆盖率 85.49%（门槛 84%）。
- 仿真环境 `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_newton_integration_state.py`：
  8 项通过，含真实 CPU/CUDA、activation、selected-world 恢复以及未选行保留。
- `scripts/smoke_mirror_physics.py --profile newton_cuda --steps 8`：完整场景、恢复、目标回读、
  恢复后再推进、输出排空及 supervisor 零退出码/成功 marker。
- `scripts/smoke_kaleidoscope_physics.py --profile newton_cuda --num-envs 2 --steps 2 --exercise-training-adapters`：
  snapshot、clone successor、局部 reset 隔离、多控制模式、Gymnasium/skrl SAME_STEP。

新 worktree 最初缺少 LFS 实体和外部 Warehouse 素材，已通过已有本地资产存储/链接补齐后重跑；
这两次装配失败不纳入性能数据。原始 JSON、profile 和日志在本机验收归档中保留。
本轮未运行完整 `just test-simulation` 大矩阵（其余 PhysX/CPU profile、256-world 容量、IK、
视频及 teardown stress）；使用以上与本次状态读取/积分恢复变更直接相关的窄门禁。
由用户审阅合并 A 后，再开始 B；首帧等待 C 和 Task 接入仍按后续计划分别推进。
