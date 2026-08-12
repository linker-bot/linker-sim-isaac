"""与 ``configs/training`` 对应的训练 profile schema 命名空间。"""

# 具体 schema 从 ``configuration`` facade 或对应子模块显式导入。保持 package initializer
# 无转发导入，既避免把内部文件伪装成第二套 public API，也不会把外部 skrl 运行时误带入
# configuration 的 pure import 闭包。

__all__: list[str] = []
