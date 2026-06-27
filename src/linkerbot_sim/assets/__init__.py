"""机器人资产路径、导入和 USD/PhysX 参数覆盖工具。

assets 层负责把仓库中的 MJCF/URDF/USD 资产转换成 Isaac stage 中可用的
articulation，并在导入后统一修正 PhysX 材料、solver iteration、gravity 和 drive
参数。该入口文件不重新导出重型 Isaac API，目的是让配置解析和单元测试可以安全
导入子包；需要实际操作 stage 时再从具体模块调用延迟导入的函数。
"""
