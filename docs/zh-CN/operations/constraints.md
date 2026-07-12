# 已知风险与设计约束

语言：[中文](constraints.md) | [English](../../en/operations/constraints.md)

本文定义当前 runtime、资源、配置和安全边界。

## Runtime 与资源边界

- Execution 层只播放已经生成的 target 或 trajectory，不在 physics-step playback 中执行 IK 或 planning。
- cuRobo backend 只处理 cuRobo C-space；完整 articulation command-space 映射属于 runtime/controller 层。
- Mimic follower 展开属于 controller/runtime logic，不属于 cuRobo backend。
- 在 Tiled Scene 中，`TiledCommandAdapter` 是同步 command-step adapter；graph search 和 trajectory optimization 属于异步 planner worker 或 backend planning 层。
- Reset、`set_state` 和 snapshot restore 在第一次写入前捕获 rollback state。Rollback 不完整，或不可逆 cache/queue reset 后发生失败时，runtime 进入 fail-stop；后续操作需要重建 runtime，不能继续提交修改。
- 关闭时先停止 transport 和 publisher 线程，再关闭 planner、camera、IK 资源，最后关闭 `SimulationApp`。Timeout 表示关闭未完成；仍存活的子资源保留 sink 或 runtime dependency，以便再次执行关闭。

## 线程边界

- Isaac stage、articulation、PhysX view 和 camera wrapper 只能在仿真主线程访问。
- 后台线程可以发布已经序列化的 snapshot、写文件或处理 transport response。
- Foxglove 和 camera publisher 只消费主线程捕获的数据，不直接访问 Isaac object。

## 配置边界

- Robot placement 属于 env profile，不属于 robot profile。
- cuRobo robot model resource 属于 robot profile。
- Planner 算法默认值属于 `configs/curobo/`。
- Object 资产身份、import option、physics 和 planning collision 属于 object profile；逐场景 placement 属于 env profile。
- Runtime process、resource、transport、telemetry、output 和 shutdown policy 属于 runtime profile；env profile 不接受这些字段。
- 当前 checkout 是 workspace application。Runtime 需要 `configs/`、`scripts/`、`assets/` 和 vendored task resource，因此明确拒绝 distribution build 和 editable build。

## Fixed-Base 边界

- URDF rigid object 未设置 `import.fix_base` 时，其有效值跟随 `physics.static`：static object 使用 fixed import，dynamic object 使用 floating import。
- Fixed URDF object 不叠加 kinematic rigid-body freeze。Static object 显式设置 `import.fix_base: false` 时先 floating import，再通过 kinematic body 和关闭重力实现冻结。
- `physics.static: false` 与 `import.fix_base: true` 语义冲突，配置会被拒绝。
- Static USD object 通过 kinematic body 和关闭重力实现冻结；USD object reference 不接受 `import` 段。

## Tiled Scene 约束

- 全部 env 同步推进 physics。
- Env-specific command 只更新选中 env 的 target row，不暂停其他 env。
- 同一个启用 `tiled` 的 env profile 中，全部 env 共享相同的 robot/object 集合。
- Per-env object 差异只允许覆盖同名 object 的 pose。

## 网络与 Telemetry 安全

- Foxglove state stream 只用于观测，不接受控制命令。
- 内置 control、state 和 camera listener 只接受 `localhost` 或数值 loopback 地址，不提供认证或 TLS。远程访问必须通过以 loopback endpoint 为上游的认证 TLS proxy 或 SSH tunnel。
- Command port、Foxglove state live port 和 camera live port 是不同服务，必须使用不同端口。
- Camera output port 在 env profile 中配置，不属于 interactive state-stream CLI 参数。

## 验证

常用检查：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_tiled_*.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_env_profile_directory.py tests/test_controller_configs.py tests/test_robot_loader_import_config.py -q
git diff --check
```
