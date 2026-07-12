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

这些文件由 `CuroboTaskBundle.named("curobo_v0_8_default")` 作为一个整体引用，不是独立的
项目 profile，也不能从 runtime YAML 任意选择其中某个路径。机器人模型、碰撞 cache、batch
容量和用户可调的 IK/规划参数仍分别由 `configs/robots/` 与 `configs/curobo/*.yaml` 管理。

## 完整性约束

八份 YML 是按原始字节做 SHA-256 校验的第三方 bundle。即使只增加注释也会改变哈希，因此
不要直接编辑、翻译或重新格式化这些文件。需要升级时，必须同时更新锁定的 cuRobo release、
上游 commit、整组资源和测试中的 bundle hash gate，并重新执行完整规划测试。
