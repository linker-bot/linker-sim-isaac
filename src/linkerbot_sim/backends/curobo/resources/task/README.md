# cuRobo 0.8 task 配置资源

本目录中的 YML 文件原样取自 NVlabs/cuRobo 的 `v0.8.0` tag（commit
`4ea77366ca48ee453e7df139e39fa6532af49f3b`），许可证为 Apache-2.0。

官方 `nvidia-curobo==0.8.0` wheel 没有打包 `curobo/content/configs/task`，但 cuRobo
的 solver factory 在创建 IK、轨迹优化和图搜索组件时仍会读取这些资源。项目固定加载本目录，
从而允许普通的只读 wheel 安装正常使用规划功能，无需修改 site-packages。

## 目录职责

- `ik/`：IK 的 LBFGS、particle optimizer 和 transition model 参数。
- `trajopt/`：B-spline 轨迹优化器及其 transition model 参数。
- `graph_planner/`：图搜索 planner 的搜索和 transition 参数。
- `metrics_base.yml`：IK、轨迹优化和图搜索共同使用的 rollout cost/constraint 基线。

后端固定把这些文件作为已验证的 cuRobo 0.8.0 bundle 整体加载；它们不是独立的项目 profile，
runtime YAML 也没有 bundle 名称或单文件路径 selector。机器人模型由 `configs/robots/` 管理；
四条数值路径的 dtype 固定为 `float32`。mode 引用的 `configs/curobo/*.yaml` 只声明
batch/seed/CUDA graph 和碰撞能力。`collision_check: true` 时必须声明并分配 `collision_cache`；
关闭碰撞时可省略或保留合法 cache，runtime 都会将其投影为空且不分配碰撞缓存。
Mirror 的 duration、采样周期、timeout、默认避障、刷新和 coordination 属于
`configs/planning/mirror.yaml` 的请求默认策略，不进入本第三方 bundle，也不进入 cuRobo 数值 profile。

## 完整性约束

八份 YML 是按原始字节做 SHA-256 校验的第三方 bundle。即使只增加注释也会改变哈希，因此
不要直接编辑、翻译或重新格式化这些文件。需要升级时，必须同时更新锁定的 cuRobo release、
上游 commit、整组资源和测试中的 bundle hash gate，并重新执行完整规划测试。
