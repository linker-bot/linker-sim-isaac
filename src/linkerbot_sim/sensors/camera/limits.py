"""配置层与 recorder 层共享的相机输出资源上限。

默认值必须是有限正整数，使未显式配置的数据采集任务也不会无限占用磁盘。配置解析和实际
落盘复用同一常量，避免 YAML 默认值与 runtime 守卫发生漂移。
"""

# 10 GiB 是“单相机目录”上限，包含 metadata 与所有 modality payload；多相机任务的总
# 上限是各相机配额之和。需要更大数据集的任务必须在 runtime profile 中显式提高该值。
DEFAULT_MAX_BYTES_PER_CAMERA = 10 * 1024**3


__all__ = ["DEFAULT_MAX_BYTES_PER_CAMERA"]
