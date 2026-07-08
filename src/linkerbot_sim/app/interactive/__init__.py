"""交互式仿真入口、协议、队列、transport 和 runtime loop。

单臂/双臂 single-env 交互与 tiled-env step-control 都放在本包下；它们共享“交互入口”
这个职责，但 tiled runtime 不复用 single-env motion queue / async planner 语义。
"""
